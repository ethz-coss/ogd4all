from typing_extensions import List, Any, Union
from e2b_code_interpreter import Sandbox
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
import sys
import base64
import gradio as gr
import time
import structlog;log=structlog.get_logger()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # not the nicest way of handling this, but oh well...
from retrieval.retriever import Metadata
from generation.analyzer import Analyzer, CodeAct
from utils import get_path_from_title, get_file_from_title, generate_system_prompt


class SimpleAnalyzer(Analyzer):
    """
    Uploads datasets to E2B sandbox, then runs AI-generated code in the sandbox to analyze the data.
    Is able to fix errors in the code and retry the analysis.
    """
    def __init__(self, groupOwner, metadata_docs: List[Metadata], coding_client=None, timeout=60, max_steps=5):
        super().__init__(groupOwner, metadata_docs, coding_client)

        self.coding_llm = self.coding_client.with_structured_output(CodeAct, include_raw=True)
        self.sbx = Sandbox(template="ck9xjl2f121b2synr4f5", api_key=os.environ.get("E2B_API_KEY"), timeout=timeout)
        self.max_steps = max_steps

        setup_code, logs, file_names = self.init_sandbox()
        self.init_conversation(setup_code, logs, file_names)


    def finalize(self):
        log.info("Killing the sandbox...")
        self.sbx.kill()


    def run_ai_generated_code(self, ai_generated_code: str) -> tuple[bool, List[Any]]:
        """
        Run the AI-generated code in the sandbox and return the results.
        
        :param ai_generated_code: The AI-generated code to run.
        :return: A tuple containing a boolean indicating success and a list of results.
        """

        log.info("Will run following code in the sandbox:")
        log.info("========================== Code ==========================")
        log.info(ai_generated_code)
        log.info("==========================================================\n")
        execution = self.sbx.run_code(ai_generated_code)
        log.info('Code execution finished!')

        response = []

        print_log = "\n".join(execution.logs.stdout)
        response.append(print_log)
        log.info(print_log)

        # First let's check if the code ran successfully.
        if execution.error:
            log.info('AI-generated code had an error.')
            log.info(execution.error.traceback)
            response.append(execution.error.traceback)
            return False, response # stdout, stderr


        # Iterate over all the results and specifically check for png files that will represent the chart.
        result_idx = 0
        log.info(f"Number of results: {len(execution.results)}")
        for result in execution.results:
            if result.text:
                response.append(result.text)
            if result.png:
                # Save the png to a file
                # The png is in base64 format.
                with open(f'chart-{result_idx}.png', 'wb') as f:
                    f.write(base64.b64decode(result.png))
                response.append(gr.Image(f'chart-{result_idx}.png', width=800))
                log.info(f'Chart saved to chart-{result_idx}.png')
                result_idx += 1

        return True, response


    def init_sandbox(self) -> tuple[str, List[Any], List[str]]:
        """
        Initialize the E2B sandbox by uploading relevant datasets and running setup code.

        :return: A tuple containing the setup code, the logs from the setup code, and the names of the uploaded datasets
        """

        try:
            file_names = [m.title for m in self.metadata_docs]

            # Upload datasets to sandbox
            for file_name in file_names:
                file_path = get_path_from_title(self.groupOwner, file_name)
                with open(file_path, "rb") as f:
                    self.sbx.files.write(get_file_from_title(self.groupOwner, file_name), f)
                log.info(f"Dataset {get_file_from_title(self.groupOwner, file_name)} uploaded to sandbox!")

            setup_code = f"""
            import geopandas as gpd
            import matplotlib.pyplot as plt
            import contextily as cx
            from geopy.geocoders import Nominatim
            
            geolocator = Nominatim(user_agent="ogd4all")

            """

            cleaned_file_names = [get_file_from_title(self.groupOwner, file_name) for file_name in file_names]

            # Initial code for LLM to understand data:
            # - For every dataset, print layer names
            for idx, file_name in enumerate(cleaned_file_names):
                name = file_name.split(".")[0] # remove suffix
                setup_code += f"""
                # Get layers included in dataset {name}
                path_{idx} = "{file_name}"
                print(f"Layers in dataset {name}: {{gpd.list_layers(path_{idx})['name'].to_list()}}")
                
                """

            # Fix indentation of the code
            setup_code = "\n".join([line.strip() for line in setup_code.split("\n")])

            # Run code on sandbox
            success, logs = self.run_ai_generated_code(setup_code)
            if not success:
                log.error('Setup code had an error.')
                log.error(logs)
                self.finalize()
                sys.exit(1)
            else:
                log.info("Setup code ran successfully!")
                log.info(logs)


            return setup_code, logs, cleaned_file_names
        except Exception as e:
            log.error("Caught an exception in sandbox init: %s", e, exc_info=True, backtrace=True, diagnose=True)
            self.finalize()
            sys.exit(1)
    
    
    def init_conversation(self, setup_code, logs, file_names):
        """
        Initialize the conversation with the analyzer

        :param logs: The logs from the setup code.
        :param file_names: The names of the uploaded datasets.
        """

        system_prompt = generate_system_prompt(file_names, self.metadata_docs, setup_code, logs)

        log.info("System prompt:")
        log.info(system_prompt)

        self.messages.append(SystemMessage(system_prompt))


    def analyze(self, query: Union[str, dict]):
        # Extract text from multimodal input for backward compatibility
        if isinstance(query, dict) and "text" in query:
            text_query = query["text"]
        elif isinstance(query, str):
            text_query = query
        else:
            text_query = str(query)
            
        self.messages.append(HumanMessage(f"Request/Question: {text_query}\n")),
        start_time = time.time()

        log.info(f"User question: {text_query}")

        thought_msg = gr.ChatMessage(
            role="assistant",
            content="",
            metadata={"title": "Analyzing the data...",
                      "status": "pending"},
        )

        yield thought_msg

        steps = 0

        while steps < self.max_steps:
            steps += 1
            response = self.coding_llm.invoke(self.messages)
            codeAct = response["parsed"]
            self.total_tokens += response.usage_metadata["total_tokens"]

            # If the LLM wrapped the code with ```python ``` or ``` ```, remove it
            codeAct.code = codeAct.code.replace("```python", "").replace("```", "").strip()

            thought_msg.content += f"**Thought**: {codeAct.thought}\n"
            thought_msg.content += f"**Code**: \n```python\n{codeAct.code}\n``` \n"
            yield thought_msg

            success, response = self.run_ai_generated_code(codeAct.code)
            if not success:
                thought_msg.content += f"**Output**:\n```\n{response[0].replace("```", "")}\n```\n**Error**:\n```\n{response[1].replace("```", "")}\n```\n"
                new_msgs = [AIMessage("Thought: " + codeAct.thought), AIMessage("**Code**: " + f"```python\n{codeAct.code}\n```"), AIMessage(f"**Output**: \n{response[0]}\nError: \n{response[1]}"), HumanMessage("Please fix the error and try again.")]
                self.messages = self.messages + new_msgs
                yield thought_msg
            else:
                thought_msg.metadata["status"] = "done"
                thought_msg.metadata["title"] = "Analysis completed"
                thought_msg.metadata["duration"] = time.time() - start_time

                # Concat all text responses
                output_texts = [r for r in response if isinstance(r, str)]
                output_text = "\n".join(output_texts)

                new_msgs = [AIMessage("Thought: " + codeAct.thought), AIMessage("**Code**: " + f"```python\n{codeAct.code}\n```"), AIMessage(f"**Output**: \n{output_text}")]
                self.messages = self.messages + new_msgs

                yield [thought_msg] + response
                return
            
        thought_msg.metadata["status"] = "done"
        thought_msg.metadata["title"] = f"Maximum thinking steps ({self.max_steps}) exceeded"
        thought_msg.metadata["duration"] = time.time() - start_time
        yield [thought_msg, "Unfortunately, I was not able to find a solution to your request."]
        return
        
