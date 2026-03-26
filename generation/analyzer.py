from abc import ABC, abstractmethod
from typing import List, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from retrieval.retriever import Metadata
from utils import get_llm_client


class CodeAct(BaseModel):
    """Thought and python code for data analysis"""
    thought: str = Field(..., description="The thought process for creating the code")
    code: str = Field(..., description="The Python code to run as plain text.")

class CodeAction(TypedDict):
    """Thought and python code for data analysis"""
    thought: Annotated[str, ..., "The thought process for creating the code"]
    code: Annotated[str, ..., "The Python code to run as plain text."]


class Analyzer(ABC):
    """
    Abstract base class for data analyzers.

    An analyzer is responsible for analyzing given datasets based on user queries.
    """

    def __init__(self, groupOwner, metadata_docs: List[Metadata], coding_client=None):
        # Initialize coding client if not provided
        if coding_client is None:
            coding_client = get_llm_client("gpt-4.1")
        
        self.groupOwner = groupOwner
        self.metadata_docs = metadata_docs
        self.coding_client = coding_client
        self.messages = []
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_reasoning_tokens = 0
        self.dataset_idx_offset = 0 # only used for setting dataset paths


    def _is_cached(self, messages) -> bool:
        """
        Return True if invoking the coding LLM with these messages would be a cache hit.
        Used to decide whether to skip streaming and use invoke() instead.
        """
        from langchain_core.globals import get_llm_cache
        from langchain_core.load import dumps
        from langchain_core.runnables import RunnableBinding, RunnableParallel, RunnableSequence
        try:
            cache = get_llm_cache()
            if cache is None:
                return False

            # with_structured_output returns a RunnableSequence. We need the kwargs
            # bound to the underlying LLM (tools / response_format) to reproduce the
            # exact llm_string the cache uses.
            #
            # Two observed structures depending on include_raw:
            #   without include_raw: RunnableSequence([RunnableBinding(llm, **kw), parser])
            #   with include_raw:    RunnableSequence([RunnableParallel({"raw": RunnableBinding(llm, **kw)}), ...])
            bound_kwargs = {}
            if isinstance(self.coding_llm, RunnableSequence):
                first = self.coding_llm.first
                if isinstance(first, RunnableBinding):
                    bound_kwargs = first.kwargs
                elif isinstance(first, RunnableParallel):
                    for step in first.steps.values():
                        if isinstance(step, RunnableBinding):
                            bound_kwargs = step.kwargs
                            break

            # Strip LangSmith metadata keys — they are excluded from the actual cache key
            bound_kwargs = {k: v for k, v in bound_kwargs.items() if not k.startswith("ls_")}

            prompt = dumps(messages)
            llm_string = self.coding_client._get_llm_string(**bound_kwargs)
            return cache.lookup(prompt, llm_string) is not None
        except Exception:
            return False

    def _write_cache(self, messages, thought: str, code: str) -> None:
        """
        Write the streaming result to the LangChain cache so the next identical call
        gets a cache hit (stream() does not write to cache; this bridges the gap).
        """
        import json
        from langchain_core.globals import get_llm_cache
        from langchain_core.load import dumps
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration
        from langchain_core.runnables import RunnableBinding, RunnableParallel, RunnableSequence
        try:
            cache = get_llm_cache()
            if cache is None:
                return
            bound_kwargs = {}
            if isinstance(self.coding_llm, RunnableSequence):
                first = self.coding_llm.first
                if isinstance(first, RunnableBinding):
                    bound_kwargs = first.kwargs
                elif isinstance(first, RunnableParallel):
                    for step in first.steps.values():
                        if isinstance(step, RunnableBinding):
                            bound_kwargs = step.kwargs
                            break
            bound_kwargs = {k: v for k, v in bound_kwargs.items() if not k.startswith("ls_")}
            prompt = dumps(messages)
            llm_string = self.coding_client._get_llm_string(**bound_kwargs)
            # Reconstruct the raw AIMessage content as JSON (what the model would have returned)
            raw_content = json.dumps({"thought": thought, "code": code})
            cache.update(prompt, llm_string, [ChatGeneration(message=AIMessage(content=raw_content))])
        except Exception:
            pass  # cache write failure is non-fatal

    @abstractmethod
    def analyze(self, query: Union[str, dict]):
        """
        Analyze the given datasets based on the user query.

        :param query: The user query to analyze the datasets for. Can be a string or dict with text and files.
        :return: The analysis result.
        """
        pass

    @abstractmethod
    def extend_sandbox(self, file_names):
        """
        Extend the sandbox with additional datasets.
        :param file_names: The names of the additional datasets.
        """
        pass


    @abstractmethod
    def finalize(self):
        """
        Clean up resources
        """
        pass
