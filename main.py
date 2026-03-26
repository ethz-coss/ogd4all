from dotenv import load_dotenv

load_dotenv()

import gradio as gr
import argparse
import time
import os
import sys
import logging
import structlog;log=structlog.get_logger()
import copy
import datetime

import folium
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

import utils
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

_script_dir = os.path.dirname(os.path.abspath(__file__))
set_llm_cache(SQLiteCache(database_path=os.path.join(_script_dir, "cache", ".langchain.db")))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # not the nicest way of handling this, but oh well...
from generation.iterative_local_analyzer import IterativeLocalAnalyzer
from generation.simple_analyzer import SimpleAnalyzer
from generation.simple_local_analyzer_v2 import SimpleLocalAnalyzerV2
from retrieval.agentic_retriever import AgenticRetriever
from retrieval.knn_retriever import KNNRetriever
from retrieval.retriever import Retriever
from retrieval.verified_retriever import VerifiedRetriever
from utils import SUPPORTED_LLMS, get_llm_client, init_mappings, download_dataset_file, get_file_from_title, get_path_from_title


class RetrievalCheck(BaseModel):
    """Results of whether an additional dataset retrieval is required"""
    thought: str = Field(..., description="Short explanation whether additional retrieval is required.")
    retrievalRequired: bool = Field(..., description="Whether an additional dataset retrieval is required or not.")

class OGD4All():
    def __init__(self, groupOwner, retriever, analyzer_type, coding_llm, retrieval_check_client, streaming, lazy_download=False, timeout=60):
        self.groupOwner = groupOwner
        self.retriever = retriever
        self.analyzer_type = analyzer_type.lower()
        self.analyzer = None
        self.reset = True
        self.timeout = timeout
        self.added_datasets = False
        self.coding_llm = coding_llm
        self.streaming = streaming
        self.lazy_download = lazy_download
        self.retrieval_check_client = retrieval_check_client.with_structured_output(RetrievalCheck)


    def chat_fn(self, query, history):
        """
        Function to handle the chat interaction with the user.
        It takes a query and history of messages, processes them, and returns the response.

        :param retriever: The retriever object to fetch relevant metadata documents.
        :param analyzer: The analyzer object to analyze the data based on the query.
        :param query: The user query.
        :param history: The history of messages exchanged in the chat. Currently ignored, as the Analyzer keeps track of the context
        :return: The response to the user's query.
        """
        try:
            updated_map = gr.update()
            thought_msg_download = None
            _dl = []
            if self.reset:
                # Kill previous sandbox
                if self.analyzer is not None:
                    self.analyzer.finalize()
                    self.analyzer = None


                start_time = time.time()
                
                # We start by retrieving the relevant metadata documents based on the user's query.
                thought_msg_retrieval = gr.ChatMessage(
                    role="assistant",
                    content="",
                    metadata={"title": "Retrieving relevant open data...",
                            "status": "pending"},
                )

                yield thought_msg_retrieval, updated_map

                try:
                    search_count = 0
                    
                    for result in self.retriever.retrieve(query):
                        if isinstance(result, str):
                            # Intermediate search query string
                            search_count += 1
                            thought_msg_retrieval.content += f"**Query {search_count}**: {result}\n"
                            yield thought_msg_retrieval, updated_map
                        else:
                            # Final result tuple
                            metadata_docs, explanation = result
                            break
                        
                except Exception as e:
                    log.error("Error during dataset retrieval:", exc_info=True)
                    thought_msg_retrieval.content = "An error occurred while retrieving datasets."
                    thought_msg_retrieval.metadata["status"] = "done"
                    thought_msg_retrieval.metadata["title"] = "Retrieval failed"
                    error_msg = "I'm sorry, an error occurred during the retrieval of relevant datasets. Please try again."
                    self.reset = True
                    yield [thought_msg_retrieval, error_msg], updated_map
                    return
                thought_msg_retrieval.metadata["status"] = "done"
                thought_msg_retrieval.metadata["title"] = "Retrieval completed"
                thought_msg_retrieval.metadata["duration"] = time.time() - start_time

                if len(metadata_docs) == 0:
                    thought_msg_retrieval.content += "\nNo suitable datasets found."
                    yield [thought_msg_retrieval, explanation], updated_map
                    return

                _q = "query" if search_count == 1 else "queries"
                _s = "" if len(metadata_docs) == 1 else "s"
                thought_msg_retrieval.content += f"\nBased on datasets retrieved with the above {_q}, I will be using the following dataset{_s} to answer your question:\n"
                thought_msg_retrieval.content += "\n".join([f"- [{doc.title}]({doc.downloadURL})" for doc in metadata_docs])
                #thought_msg_retrieval.content += f"\n\n{explanation}" # Explanation does not seem very helpful and is currently not optimized for display towards the user

                if self.lazy_download:
                    dl_start = time.time()
                    missing = [
                        (doc, get_file_from_title(self.groupOwner, doc.title))
                        for doc in metadata_docs
                        if not os.path.exists(get_path_from_title(self.groupOwner, doc.title))
                    ]
                    if missing:
                        thought_msg_download = gr.ChatMessage(
                            role="assistant",
                            content="",
                            metadata={"title": "Downloading open datasets...", "status": "pending"},
                        )
                        _dl = [thought_msg_download]
                        yield [thought_msg_retrieval, thought_msg_download], updated_map
                        for doc, filename in missing:
                            thought_msg_download.content += f"- **{doc.title}** ({filename})\n"
                            yield [thought_msg_retrieval, thought_msg_download], updated_map
                            download_dataset_file(self.groupOwner, filename)
                        thought_msg_download.metadata["status"] = "done"
                        thought_msg_download.metadata["title"] = "Datasets downloaded"
                        thought_msg_download.metadata["duration"] = time.time() - dl_start
                        yield [thought_msg_retrieval, thought_msg_download], updated_map

                start_time = time.time()
                thought_msg_coding_init = gr.ChatMessage(
                    role="assistant",
                    content="",
                    metadata={"title": "Initializing coding environment...",
                              "status": "pending"},
                )

                yield [thought_msg_retrieval] + _dl + [thought_msg_coding_init], updated_map

                # Initialize the analyzer based on the specified type
                if self.analyzer_type == "simple":
                    self.analyzer = SimpleAnalyzer(retriever.groupOwner, metadata_docs, timeout=self.timeout, coding_client=self.coding_llm)
                elif self.analyzer_type == "simple_local_v2":
                    self.analyzer = SimpleLocalAnalyzerV2(retriever.groupOwner, metadata_docs, coding_client=self.coding_llm, streaming=self.streaming)
                elif self.analyzer_type == "iterative_local":
                    self.analyzer = IterativeLocalAnalyzer(retriever.groupOwner, metadata_docs, coding_client=self.coding_llm, streaming=self.streaming)
                else:
                    log.error(f"Unknown analyzer type: {self.analyzer_type}. Exiting...")
                    sys.exit(1)

                thought_msg_coding_init.metadata["status"] = "done"
                thought_msg_coding_init.metadata["title"] = "Coding environment initialized"
                thought_msg_coding_init.metadata["duration"] = time.time() - start_time
                thought_msg_coding_init.content += f"I have initialized a persistent Python environment that allows me to analyze the user's question. "
                thought_msg_coding_init.content += f"I have loaded all required datasets and imported required libraries. "
                thought_msg_coding_init.content += f"The following code has been executed to print context about the datasets:\n```python{self.analyzer.setup_code}\n```"

                yield [thought_msg_retrieval] + _dl + [thought_msg_coding_init], updated_map
            else:
                # Run a check to see whether new datasets need to be retrieved
                prompt = f"""
                Are additional datasets required to answer the following question? If so, please provide a short explanation of why they are needed.
                You currently have access to the following datasets:
                {", ".join([str(doc) for doc in self.analyzer.metadata_docs])}

                Question: {query}
                """
                log.debug(f"Retrieval check prompt: {prompt}")
                messages = self.analyzer.messages.copy()
                messages.append(HumanMessage(prompt))
                try:
                    retrieval_check = self.retrieval_check_client.invoke(messages)
                except Exception as e:
                    log.error("Error during retrieval check:", exc_info=True)
                    yield "An error occurred while checking whether additional datasets are required.", updated_map
                    return

                if retrieval_check.retrievalRequired:
                    thought_msg_retrieval = gr.ChatMessage(
                        role="assistant",
                        content="",
                        metadata={"title": "Retrieving additional datasets...",
                                  "status": "pending"},
                    )
                    yield thought_msg_retrieval, updated_map

                    # Retrieve additional datasets
                    try:
                        search_count = 0
                        
                        for result in self.retriever.retrieve(query):
                            if isinstance(result, str):
                                # Intermediate search query string
                                search_count += 1
                                thought_msg_retrieval.content += f"**Query {search_count}**: {result}\n"
                                yield thought_msg_retrieval, updated_map
                            else:
                                # Final result tuple
                                metadata_docs, explanation = result
                                break
                                
                        if metadata_docs is None:
                            # Something went wrong, no final result was yielded
                            raise Exception("No final result received from retriever")
                            
                    except Exception as e:
                        log.error("Error during additional dataset retrieval:", exc_info=True)
                        thought_msg_retrieval.content = "An error occurred while retrieving additional datasets."
                        thought_msg_retrieval.metadata["status"] = "done"
                        thought_msg_retrieval.metadata["title"] = "Retrieval failed"
                        error_msg = "I'm sorry, an error occurred during the retrieval of relevant datasets. Please try again."
                        yield [thought_msg_retrieval, error_msg], updated_map
                        return
                    thought_msg_retrieval.metadata["status"] = "done"
                    thought_msg_retrieval.metadata["title"] = "Retrieval completed"

                    # Filter out already existing metadata documents based on title
                    existing_titles = {doc.title for doc in self.analyzer.metadata_docs}
                    extra_docs = [doc for doc in metadata_docs if doc.title not in existing_titles]

                    if len(extra_docs) == 0 and len(metadata_docs) == 0:
                        # Nothing relevant found
                        thought_msg_retrieval.content += "\nNo suitable additional datasets found. "
                        yield [thought_msg_retrieval, explanation], updated_map
                        return
                    elif len(extra_docs) == 0:
                        # Only existing datasets were found. Difference to previous case is that we still let analyzer run.
                        thought_msg_retrieval.content += "\nNo suitable additional datasets found. "
                        yield [thought_msg_retrieval, explanation], updated_map
                    else:
                        self.added_datasets = True
                        _q = "query" if search_count == 1 else "queries"
                        _s = "" if len(extra_docs) == 1 else "s"
                        thought_msg_retrieval.content += f"\nBased on datasets retrieved with the above {_q}, I will be using the following additional dataset{_s} to answer your question:\n"
                        thought_msg_retrieval.content += "\n".join([f"- [{doc.title}]({doc.downloadURL})" for doc in extra_docs])

                        yield thought_msg_retrieval, updated_map

                        if self.lazy_download:
                            dl_start = time.time()
                            missing = [
                                (doc, get_file_from_title(self.groupOwner, doc.title))
                                for doc in extra_docs
                                if not os.path.exists(get_path_from_title(self.groupOwner, doc.title))
                            ]
                            if missing:
                                thought_msg_download = gr.ChatMessage(
                                    role="assistant",
                                    content="",
                                    metadata={"title": "Downloading open datasets...", "status": "pending"},
                                )
                                _dl = [thought_msg_download]
                                yield [thought_msg_retrieval, thought_msg_download], updated_map
                                for doc, filename in missing:
                                    thought_msg_download.content += f"- **{doc.title}** ({filename})\n"
                                    yield [thought_msg_retrieval, thought_msg_download], updated_map
                                    download_dataset_file(self.groupOwner, filename)
                                thought_msg_download.metadata["status"] = "done"
                                thought_msg_download.metadata["title"] = "Datasets downloaded"
                                thought_msg_download.metadata["duration"] = time.time() - dl_start
                                yield [thought_msg_retrieval, thought_msg_download], updated_map

                        start_time = time.time()
                        thought_msg_coding_extend = gr.ChatMessage(
                            role="assistant",
                            content="",
                            metadata={"title": "Updating coding environment...",
                                    "status": "pending"},
                        )

                        if isinstance(self.analyzer, SimpleLocalAnalyzerV2) or isinstance(self.analyzer, IterativeLocalAnalyzer):
                            yield [thought_msg_retrieval] + _dl + [thought_msg_coding_extend], updated_map

                            self.analyzer.metadata_docs.extend(extra_docs)
                            self.analyzer.extend_sandbox([m.title for m in extra_docs])

                            thought_msg_coding_extend.metadata["status"] = "done"
                            thought_msg_coding_extend.metadata["title"] = "Coding environment updated"
                            thought_msg_coding_extend.metadata["duration"] = time.time() - start_time
                            thought_msg_coding_extend.content += f"I have updated the persistent Python environment with new datasets. "
                            thought_msg_coding_extend.content += f"The following code has been executed to print context about the new datasets:\n```python{self.analyzer.setup_code}\n```"
                            yield [thought_msg_retrieval] + _dl + [thought_msg_coding_extend], updated_map
                        else:
                            thought_msg_coding_extend.metadata["status"] = "done"
                            thought_msg_coding_extend.metadata["title"] = "Coding environment not updated"
                            thought_msg_coding_extend.metadata["duration"] = time.time() - start_time
                            thought_msg_coding_extend.content = "I am not able to extend the sandbox with additional datasets with the current analyzer type. Please reset the system to start a new analysis."
                            yield [thought_msg_retrieval] + _dl + [thought_msg_coding_extend], updated_map


            for out in self.analyzer.analyze(query):
                if not isinstance(out, list):
                    out = [out]

                produced_new_map = False
                new_map = None
                for i, item in enumerate(out[:]): # iterate over copy
                    if isinstance(item, folium.Map):
                        new_map = item
                        out.pop(i)
                        produced_new_map = True

                if produced_new_map:
                    copied_map = copy.deepcopy(new_map)
                    folium.LayerControl().add_to(copied_map)
                    updated_map = gr.update(value=copied_map._repr_html_())

                if self.reset:
                    yield [thought_msg_retrieval] + _dl + [thought_msg_coding_init] + out, updated_map
                elif self.added_datasets:
                    yield [thought_msg_retrieval] + _dl + [thought_msg_coding_extend] + out, updated_map
                elif retrieval_check is not None and retrieval_check.retrievalRequired:
                    # Retrieval was performed, but no new datasets were added
                    yield [thought_msg_retrieval] + _dl + out, updated_map
                else:
                    yield out, updated_map

            self.added_datasets = False
            self.reset = False

        except Exception as e:
            log.error("Caught an exception in chat_fn: %s", e, exc_info=True, backtrace=True, diagnose=True)
            self.finalize()
            yield gr.ChatMessage(role="assistant", content="I am sorry, there has been an error processing your request. Please try again."), updated_map
            self.reset = True
            return
    

    def finalize(self):
        log.info("Finalizing OGD4All...")
        if self.analyzer is not None:
            self.analyzer.finalize()


def start_frontend(retriever: Retriever, analyzer_type: str, coding_llm, retrieval_check_client, streaming: bool = True, lazy_download: bool = False):
    """Starts an interactive Gradio interface for OGD4All"""
    log.info("Starting OGD4All...")

    def create_session():
        return OGD4All(retriever.groupOwner, retriever, analyzer_type, coding_llm, retrieval_check_client, streaming, lazy_download=lazy_download, timeout=360)

    _static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

    with open(os.path.join(_static, "style.css")) as f:
        custom_css = f.read()

    with open(os.path.join(_static, "map_placeholder.html")) as f:
        map_placeholder = f.read()

    with open(os.path.join(_static, "map.js")) as f:
        map_js_head = f"<script>\n{f.read()}\n</script>"

    i18n = gr.I18n(
        en={"placeholder": "Ask me anything about Zurich's open data..."},
        de={"placeholder": "Frag mich etwas über die offenen Daten der Stadt Zürich..."},
    )

    with gr.Blocks(title="OGD4All", fill_height=True) as demo:
        map = gr.HTML(value=map_placeholder, render=False, elem_classes="map-panel")
        session_state = gr.State(create_session)
        with gr.Row(scale=1, elem_id="title-row"):
            with gr.Column(scale=1):
                gr.HTML("""
                <div id="title-area">
                    <div class="title-big">
                        <div class="title-big-name">OGD4ALL</div>
                        <div class="title-big-sub">Accessible Retrieval &amp; Analysis of Geospatial Open Government Data</div>
                    </div>
                    <div class="title-small">
                        <h1><b>OGD4All</b>: Accessible Retrieval &amp; Analysis of Geospatial Open Government Data</h1>
                    </div>
                </div>
                """)
        with gr.Row(elem_classes="full-height", scale=4):
            with gr.Column(scale=1, elem_classes="full-height", elem_id="chat-col-inner"):
                chatbot = gr.Chatbot(scale=1, show_label=False, elem_id="main-chatbot", allow_tags=False, buttons=[])

                def clear_all(session):
                    session.reset = True # reset analyzer state
                    return gr.update(value=map_placeholder) # restore placeholder

                chatbot.clear(fn=clear_all, inputs=[session_state], outputs=[map])

                def chat_fn(query, history, session):
                    yield from session.chat_fn(query, history)

                gr.ChatInterface(
                    fn=chat_fn,
                    multimodal=False, # we manually handle multimodal input
                    textbox=gr.MultimodalTextbox(file_types=["image", ".pdf"], placeholder=i18n("placeholder"), file_count='multiple'),
                    examples=[
                        ["Wo plant die Stadt Zürich neue Bäume zu pflanzen?", None],
                        ["Welche Hundefreilaufzone liegt am nächsten beim Kunsthaus Zürich?", None],
                        ["Welches Quartier in Zürich hat die höchste Dichte an Spielplätzen?", None],
                        ["Wie hat sich die Anzahl der Elektrofahrzeuge in Zürich in den letzten 20 Jahren entwickelt?", None],
                    ],
                    chatbot=chatbot,
                    additional_inputs=[session_state],
                    additional_outputs=[map],
                )
            with gr.Column(scale=1, elem_classes="full-height", elem_id="map-col"):
                map.render()
        with gr.Row(scale=1, elem_id="footer-row"):
            gr.HTML("""
            <div id="footer-bar">
                <p class="footer-attribution">OGD4All hat Zugriff auf 430 tabellarische und geografische Datensätze der Stadt Zürich (Datenstand März–Mai 2025). | Quelle: Stadt Zürich | © OpenStreetMap-Mitwirkende, Tiles © Esri — Quelle: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, und die GIS User Community, © OpenStreetMap-Mitwirkende © CARTO | <a href="https://github.com/ethz-coss/ogd4all" target="_blank" rel="noopener">GitHub Repo</a>|<a href="https://arxiv.org/abs/2602.00012" target="_blank" rel="noopener">Paper</a></p>
                <div class="footer-mobile-btns">
                    <button id="footer-info-btn" aria-label="Info">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="8"/><line x1="12" y1="12" x2="12" y2="16"/></svg>
                        Info
                    </button>
                </div>
            </div>
            <div id="info-modal-backdrop">
                <div id="info-modal" role="dialog" aria-modal="true" aria-label="Info">
                    <p>OGD4All hat Zugriff auf 430 tabellarische und geografische Datensätze der Stadt Zürich (Datenstand März–Mai 2025). Quelle: Stadt Zürich.</p>
                    <p>© OpenStreetMap-Mitwirkende, Tiles © Esri — Quelle: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, und die GIS User Community, © OpenStreetMap-Mitwirkende © CARTO</p>
                    <p><a href="https://github.com/ethz-coss/ogd4all" target="_blank" rel="noopener">GitHub Repo</a>|<a href="https://arxiv.org/abs/2602.00012" target="_blank" rel="noopener">Paper</a></p></div>
            </div>
            """)

    demo.launch(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Roboto"), "Arial", "sans-serif"]), css=custom_css, head=map_js_head, footer_links=[], i18n=i18n)


if __name__ == "__main__":
    print(utils.CONSOLE_LOGO)

    parser = argparse.ArgumentParser(description="Chat with Geospatial Open Government Data.")
    parser.add_argument("--groupOwner", type=str, default="50000006", help="The groupOwner id whose metadata should be queried (default: 50000006).")
    parser.add_argument("--top_n", type=int, default=10, help="The number of documents to retrieve for a single KNN search (default: 10).")
    parser.add_argument("--retriever", type=str, choices=["agentic", "knn", "verified"], default="agentic", help="The retrieval strategy to use")
    parser.add_argument("--analyzer", type=str, choices=["simple_local_v2", "simple", "simple_local", "iterative_local"], default="iterative_local", help="The analyzer type to use")
    parser.add_argument("--retrieval_llm", choices=SUPPORTED_LLMS, default='gpt-4.1', help="The LLM to use for retrieval tasks.")
    parser.add_argument("--retrieval_check_llm", choices=SUPPORTED_LLMS, default='gpt-4o', help="The LLM to use when checking whether a follow-up retrieval is needed. Ideally quite fast.")
    parser.add_argument("--coding_llm", choices=[llm for llm in SUPPORTED_LLMS], default='gpt-4.1', help="The LLM to use for coding tasks/analysis.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--hybrid_search", action="store_true", help="Enable hybrid search with Milvus.")
    group.add_argument("--bm25_search", action="store_true", help="Enable BM25 search with Milvus.")
    parser.add_argument("--no_streaming", action="store_true", help="Disable streaming for the coding LLM. This enables validation of LLM responses and token counting, but makes the system feel less responsive.")
    parser.add_argument("--lazy-download", action="store_true", dest="lazy_download", help="Enable lazy downloading of dataset files from HuggingFace Datasets on demand. Use in deployment.")
    args = parser.parse_args()
    
    log_name = f"main_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.retriever}_{args.analyzer}"
    utils.setup_logging(level=logging.INFO, log_filename=f"{log_name}.log")

    # Initialize utils
    init_mappings(args.groupOwner, lazy_download=args.lazy_download)

    # Initialize the retriever
    retriever = None
    if args.retriever.lower() == "verified":
        retriever = VerifiedRetriever(args.groupOwner, args.top_n, hybrid_search=args.hybrid_search, llm_client=get_llm_client(args.retrieval_llm), bm25_search=args.bm25_search)
    elif args.retriever.lower() == "knn":
        retriever = KNNRetriever(args.groupOwner, args.top_n, hybrid_search=args.hybrid_search, llm_client=get_llm_client(args.retrieval_llm), bm25_search=args.bm25_search)
    elif args.retriever.lower() == "agentic":
        retriever = AgenticRetriever(args.groupOwner, args.top_n, hybrid_search=args.hybrid_search, llm_client=get_llm_client(args.retrieval_llm), bm25_search=args.bm25_search)
    else:
        log.error(f"Unknown retriever type: {args.retriever}. Exiting...")
        sys.exit(1)

    coding_llm = get_llm_client(args.coding_llm)
    retrieval_check_client = get_llm_client(args.retrieval_check_llm)

    _SENSITIVE_ENV_VARS = [
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_ENDPOINT_EMBEDDING_LARGE",
        "MILVUS_CLUSTER_TOKEN",
        "GOOGLE_GEOCODING_API_KEY",
        "E2B_API_KEY",
        "OPENROUTER_API_KEY"
    ]
    for key in _SENSITIVE_ENV_VARS:
        os.environ.pop(key, None)
    log.info("Cleared sensitive environment variables from process environment.")

    start_frontend(retriever, args.analyzer, coding_llm=coding_llm, retrieval_check_client=retrieval_check_client, streaming=not args.no_streaming, lazy_download=args.lazy_download)