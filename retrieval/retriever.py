from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Union, Generator
from openai import AzureOpenAI, OpenAI
import os
import numpy as np
import pandas as pd
import ast
import json
import sys
import structlog;log=structlog.get_logger()
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # not the nicest way of handling this, but oh well...
from utils import does_title_exist

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute the cosine similarity between two vectors.

    :param a: First vector.
    :param b: Second vector.
    :return: Cosine similarity between the two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_embedding(embedding_client: Any, text: str, model: str = "text-embedding-3-large"):
    """
    Get the embedding for a given text using the specified model.
    
    :param embedding_client: The client to use for generating embeddings.
    :param text: The text to embed.
    :param model: The model to use for embedding. Default is "text-embedding-3-large".
    :return: tuple with the embedding vector for the text and the total token usage
    """
    embedding_request = embedding_client.embeddings.create(input=[text], model=model)
    return embedding_request.data[0].embedding, embedding_request.usage.total_tokens


class Metadata():
    """
    Class representing metadata for a dataset in geocat.ch format.
    """

    def __init__(self, groupOwner, metadata_id):
        self.metadata_id = metadata_id

        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, f"../data/metadata/{groupOwner}/processed/{metadata_id}.json")
        with open(json_path, "r", encoding='utf-8') as f:
            data = json.load(f)

        self.title = data["title"]
        self.description = data["abstract"]
        self.classes = data["contentInfo"]["classes"]
        self.downloadURL = data["downloadURL"]
        self.date = data.get("identificationDate", None)

        # Iterate over classes and attributes and set all the values to lower case
        for cls in self.classes:
            cls["name"] = cls["name"].lower()
            for attr in cls["attributes"]:
                attr["name"] = attr["name"].lower()
                if attr["name"] == "geometrie":
                    attr["name"] = "geometry"

    def __str__(self):
        metadata_str = ""
        metadata_str += f"**Titel: {self.title}**\n"
        if self.date:
            try:
                parsed_date = pd.to_datetime(self.date).strftime("%d.%m.%Y")
                metadata_str += f"Datum: {parsed_date}\n"
            except Exception:
                # For the city of Zurich, in some rarer cases dates are filled in with e.g. "Jeweils Montags"
                metadata_str += f"Datum: {self.date}\n"
        metadata_str += f"Beschreibung: {self.description}\n"
        metadata_str += f"Klassen: \n"
        for cls in self.classes:
            metadata_str += f"Klasse: {cls["name"]}\n"
            for attr in cls["attributes"]:
                if "description" in attr and attr["description"]:
                    metadata_str += f" - {attr['name']}: {attr['description']}\n"
                else:
                    metadata_str += f" - {attr['name']}\n"
        return metadata_str


class Retriever(ABC):
    """
    Abstract base class for data retrievers.

    A retriever is responsible for retrieving the set of relevant (geo)data to answer a user query
    """

    def __init__(self, groupOwner, top_n, embedding_client=None, embedding_model=None, hybrid_search=False, bm25_search=False):
        self.groupOwner = groupOwner
        self.top_n = top_n
        
        # Initialize embeddings dataframe
        script_dir = os.path.dirname(os.path.abspath(__file__))
        embeddings_path = os.path.join(script_dir, f"../data/metadata/{groupOwner}/processed_metadata_embeddings.csv")
        df = pd.read_csv(embeddings_path)
        df["text-embedding-3-large"] = df["text-embedding-3-large"].apply(ast.literal_eval)

        prev_len = len(df) # Pre-filtering length for logging
        df = df[df["title"].apply(lambda title: does_title_exist(groupOwner, title))]

        log.info(f"{len(df)}/{prev_len} metadata embeddings remaining after filtering for existing data.")

        # Initialize embedding client if not provided
        # Use AzureOpenAI as default, and if not set, try native OpenAI
        if embedding_client is None:
            if os.getenv("AZURE_OPENAI_API_KEY"):
                embedding_client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version="2024-10-21",
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDING_LARGE")
                )
            elif os.getenv("OPENAI_API_KEY"):
                embedding_client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                )
            else:
                raise ValueError("No embedding client provided and neither AZURE_OPENAI_API_KEY nor OPENAI_API_KEY environment variables are set.")

        # Init Milvus client if hybrid search is enabled
        if hybrid_search or bm25_search:
            self.milvus_client = MilvusClient(
                uri=os.getenv("MILVUS_CLUSTER_ENDPOINT"),
                token=os.getenv("MILVUS_CLUSTER_TOKEN"), 
            )

        self.embeddings_df = df
        self.embedding_client = embedding_client
        self.embedding_model = embedding_model if embedding_model else "text-embedding-3-large"
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_reasoning_tokens = 0
        self.performed_searches = 0
        self.hybrid_search = hybrid_search
        self.bm25_search = bm25_search


    def get_top_n_similar_docs(self, user_query: str, top_n: int) -> List[Metadata]:
        """
        Search for the top n most similar datasets to the user query and return them as metadata objects.

        :param user_query: The user query to search for.
        :param top_n: The number of top similar datasets to retrieve.
        :param hybrid_search: Whether to perform a hybrid search (dense vector + sparse BM25 vector).
        :return: A list of metadata objects corresponding to the top n similar datasets.
        """
        query = user_query if isinstance(user_query, str) else user_query["text"]
        log.info("Performing KNN", query=query, top_n=top_n)

        if not self.hybrid_search and not self.bm25_search:
            embedding, total_tokens = get_embedding(self.embedding_client, query)
            self.total_tokens += total_tokens
            # Regular semantic search based on cosine similarity
            self.embeddings_df["similarities"] = self.embeddings_df[self.embedding_model].apply(lambda x: cosine_similarity(x, embedding))

            res = (
                self.embeddings_df
                .sort_values("similarities", ascending=False)
                .head(top_n)
            )

            log.info("Vector search results", datasets=res["title"].tolist())
            return [Metadata(self.groupOwner, row["metadata_id"]) for _, row in res.iterrows()]
        elif self.bm25_search:
            if self.milvus_client is None:
                raise ValueError("Milvus client is not initialized.")
            
            search_params = {
                "param": {"drop_ratio_search": 0.2}, # between 0 and 1, but unsure what makes most sense, also not documented well
                "limit": top_n
            }

            res = self.milvus_client.search(
                collection_name="zurich_ogd_metadata",
                data=[query],
                anns_field="sparseVector",
                search_params=search_params,
                output_fields=["title"],
            )

            potential_datasets = [(hit["primary_key"], hit["title"]) for hit in res[0]]

            # Filter out datasets based on title with does_title_exist
            potential_datasets = [dataset for dataset in potential_datasets if does_title_exist(self.groupOwner, dataset[1])]

            metadata_ids = [dataset[0] for dataset in potential_datasets]
            titles = [dataset[1] for dataset in potential_datasets]

            log.info("BM25 search results", datasets=titles)
            return [Metadata(self.groupOwner, metadata_id) for metadata_id in metadata_ids]
        else:
            # Hybrid search based on both dense vector and sparse BM25 vector (keyword-based)
            # Currently based on cloud-deployed Milvus
            embedding, total_tokens = get_embedding(self.embedding_client, query)
            self.total_tokens += total_tokens

            if self.milvus_client is None:
                raise ValueError("Milvus client is not initialized.")
            
            # Reranks 2*top_n results to only return top_n results
            ranker = RRFRanker(100) # not sure if this object could be re-used

            search_param_1 = {
                "data": [embedding],
                "anns_field": "vector",
                "param": {
                    "metric_type": "COSINE"
                },
                "limit": top_n
            }
            request_1 = AnnSearchRequest(**search_param_1)

            search_param_2 = {
                "data": [query],
                "anns_field": "sparseVector",
                "param": {"drop_ratio_search": 0.2}, # between 0 and 1, but unsure what makes most sense, also not documented well
                "limit": top_n
            }
            request_2 = AnnSearchRequest(**search_param_2)

            res = self.milvus_client.hybrid_search(
                collection_name="zurich_ogd_metadata",
                reqs = [request_1, request_2],
                ranker=ranker,
                limit=top_n,
                output_fields=["title"]
            )

            potential_datasets = [(hit["primary_key"], hit["title"]) for hit in res[0]]

            # Filter out datasets based on title with does_title_exist
            potential_datasets = [dataset for dataset in potential_datasets if does_title_exist(self.groupOwner, dataset[1])]

            metadata_ids = [dataset[0] for dataset in potential_datasets]
            titles = [dataset[1] for dataset in potential_datasets]

            log.info("Hybrid search results", datasets=titles)
            return [Metadata(self.groupOwner, metadata_id) for metadata_id in metadata_ids]


    def reset_tracking(self):
        """
        Reset the total tokens count and searches count for the retriever.
        """
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_reasoning_tokens = 0
        self.performed_searches = 0


    @abstractmethod
    def retrieve(self, query: str) -> Generator[Union[str, Tuple[List[Metadata], str]], None, None]:
        """
        Retrieve relevant (geo)datasets for answering the user query.
        
        :param query: The user query to retrieve data for.
        :return: generator potentially yielding search query strings and a final tuple (relevant metadata list, explanation)
        """
        pass