from abc import ABC, abstractmethod
from typing import List, Union
from retrieval.retriever import Metadata
from utils import get_llm_client
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated


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
