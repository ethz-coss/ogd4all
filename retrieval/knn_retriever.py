import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # not the nicest way of handling this, but oh well...
from retrieval.retriever import Retriever, Metadata
from typing import List, Tuple, Generator

class KNNRetriever(Retriever):
    """
    KNNRetriever class for retrieving documents using a KNN approach.
    Super simple, intended as a baseline for more complex retrieval methods.
    """
    def __init__(self, groupOwner, top_n, embedding_client=None, embedding_model=None, hybrid_search=False, bm25_search=False):
        super().__init__(groupOwner, top_n, embedding_client, embedding_model, hybrid_search=hybrid_search, bm25_search=bm25_search)

    def retrieve(self, query: str) -> Generator[Tuple[List[Metadata], str], None, None]:
        metadata_list = super().get_top_n_similar_docs(query, self.top_n)
        yield metadata_list, f"Retrieved {len(metadata_list)} datasets using KNN similarity search."