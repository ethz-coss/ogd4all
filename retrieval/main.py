from retriever import Metadata
from knn_retriever import KNNRetriever
from verified_retriever import VerifiedRetriever
from agentic_retriever import AgenticRetriever
import structlog;log=structlog.get_logger()

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve top k metadata documents for a query using a simple KNN approach.")
    parser.add_argument("--groupOwner", type=str, default="50000006",help="The groupOwner id whose metadata should be queried (default: 50000006).")
    parser.add_argument("--top_n", type=int, default=10, help="The number of documents to retrieve for a single KNN search (default: 10).")
    parser.add_argument("--retriever", type=str, default="knn", help="knn (default) | verified | agentic")
    parser.add_argument("--testTrue", action="store_true", help="Run the retrieval with a positive test query.")
    parser.add_argument("--testFalse", action="store_true", help="Run the retrieval with a negative test query.")
    args = parser.parse_args()

    # Initialize the retriever
    retriever = None
    if args.retriever.lower() == "knn":
        retriever = KNNRetriever(args.groupOwner, args.top_n)
    elif args.retriever.lower() == "verified":
        retriever = VerifiedRetriever(args.groupOwner, args.top_n)
    elif args.retriever.lower() == "agentic":
        retriever = AgenticRetriever(args.groupOwner, args.top_n)
    else:
        raise ValueError(f"Invalid retriever type {args.retriever}. Choose one of knn | verified | agentic")

    if args.testTrue:
        metadata_list = retriever.retrieve("Nester von welchen Vögeln wurden im Umkreis des Toni-Areals gesichtet?")
        print(f"Relevant datasets:")
        for metadata in metadata_list:
            print(f"- {metadata.title}")
    elif args.testFalse:
        metadata_list = retriever.retrieve("How many bees are there in Zurich?")
        print(f"Relevant datasets:")
        for metadata in metadata_list:
            print(f"- {metadata.title}")
    else:
        # Allow users to repeatedly input queries, or quit by typing "q"
        # Print explanation
        print("Enter a question to retrieve the top n most similar documents.")
        print("Type 'q' to quit.")

        while True:
            user_query = input("Question: ")
            if user_query == "q":
                break

            metadata_list = retriever.retrieve(user_query)
            print(f"Relevant datasets:")
            for metadata in metadata_list:
                print(f"- {metadata.title}")
            print("")