import sys
import os
import json
from typing import List
import datetime
import tqdm
import logging
import structlog;log=structlog.get_logger()
import time
import argparse
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # not the nicest way of handling this, but oh well...

from utils import init_mappings, setup_logging, get_llm_client, get_llm_pricing, LLM_JUDGE_SYSTEM_PROMPT, SUPPORTED_LLMS

from retrieval.retriever import Retriever, Metadata
from retrieval.verified_retriever import VerifiedRetriever
from retrieval.agentic_retriever import AgenticRetriever

from generation.analyzer import Analyzer
from generation.simple_analyzer import SimpleAnalyzer
from generation.simple_local_analyzer_v2 import SimpleLocalAnalyzerV2
from generation.iterative_local_analyzer import IterativeLocalAnalyzer


class Eval(BaseModel):
    """Thought and evaluation of the answer"""
    thought: str = Field(..., description="The thought process for the evaluation.")
    correct: bool = Field(..., description="Whether the answer is correct or not.")


"""
Binary evaluator to check if LLM correctly determines whether datasets can be used to answer the question.
"""
def validation_accuracy(outputs: List[str], reference_outputs: List[str]) -> bool:
    canBeAnswered = len(reference_outputs) > 0
    if canBeAnswered:
        return float(len(outputs) > 0)
    else:
        return float(len(outputs) == 0)


"""
Evaluate the precision of the retrieved datasets.
Precision = #relevant retrieved instances / #total retrieved instances

Note: if no datasets are retrieved, precision is 1 if no datasets are relevant, and 0 if datasets are relevant
"""
def precision(outputs: List[str], reference_outputs: List[str]) -> bool:
    relevant_datasets = set(reference_outputs)

    # Special case: no datasets retrieved
    if len(outputs) == 0:
        # Return 0 precision if no datasets were relevant, otherwise 1
        # Method inspired by https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
        return float(len(relevant_datasets) == 0)

    retrieved_datasets = set(outputs)
    intersection = retrieved_datasets.intersection(relevant_datasets)
    precision = len(intersection) / len(retrieved_datasets)
    return precision


"""
Evaluate the recall of the retrieved datasets.
Recall = #relevant retrieved instances / #total relevant instances

Note: if no datasets are relevant, recall is 1 if no datasets are retrieved, and 0 if datasets are retrieved
"""
def recall(outputs: List[str], reference_outputs: List[str]) -> bool:
    relevant_datasets = set(reference_outputs)

    # Special case: no datasets retrieved
    if len(outputs) == 0:
        # Return 0 precision if no datasets were relevant, otherwise 1
        # Method inspired by https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
        return float(len(relevant_datasets) == 0)

    # Special case: datasets retrieved, but no datasets are relevant
    if len(relevant_datasets) == 0:
        return 0.0 # avoids division by zero

    retrieved_datasets = set(outputs)
    intersection = retrieved_datasets.intersection(relevant_datasets)
    recall = len(intersection) / len(relevant_datasets)
    return recall


"""
Run retriever on examples, evaluate with evaluators and persist results to JSON lines file.
"""
def evaluate(retriever: Retriever, analyzer_type: str, data: List[dict], evaluators, experiment_prefix: str, language: str = "de", 
             only_retrieval: bool = False, only_analysis: bool = False, retrieval_llm: str = "gpt-4.1", coding_llm: str = "gpt-4.1") -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_dir = os.path.join(script_dir, "../data/evaluation/50000006")
    os.makedirs(eval_dir, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(eval_dir, f"{experiment_prefix}_{ts}.json")

    meta = {
        "timestamp": ts,
        "retriever": retriever.__class__.__name__,
        "analyzer": analyzer_type,
        "language": language,
        "experiment_prefix": experiment_prefix,
        "top_n": retriever.top_n,
        "only_retrieval": only_retrieval,
        "only_analysis": only_analysis,
        "retrieval_llm": retrieval_llm,
        "coding_llm": coding_llm,
    }

    all_results = {evaluator.__name__: [] for evaluator in evaluators}
    all_results["retrieval_latency"] = []
    all_results["coding_latency"] = []
    all_results["correct"] = []
    all_results["retrieval_tokens"] = []
    all_results["retrieval_input_tokens"] = []
    all_results["retrieval_output_tokens"] = []
    all_results["retrieval_reasoning_tokens"] = []
    all_results["coding_tokens"] = []
    all_results["coding_input_tokens"] = []
    all_results["coding_output_tokens"] = []
    all_results["coding_reasoning_tokens"] = []
    all_results["retrieval_nr_searches"] = []
    results_data = []

    llm_judge = get_llm_client("gpt-4o").with_structured_output(Eval)
    embedding_price, _ = get_llm_pricing('text-embedding-3-large') # prices are per million tokens

    with open(outfile, "w", encoding="utf-8") as f_out:
        for example in tqdm.tqdm(data, desc=f"Running {experiment_prefix} evaluation"):
            retriever.reset_tracking()
            question = example["inputs"]["question"]
            id = example["id"]
            reference_titles = example["outputs"]["relevant_datasets"]
            if not only_analysis:
                start_time = time.time()
                try:
                    retriever_result = retriever.retrieve(question)
                    for result in retriever_result:
                        if isinstance(result, tuple):
                            # We only care about final result in eval, not intermediate search queries
                            metadata_list, _ = result
                            break   
                except Exception as e:
                    log.error("Caught an exception retrieving metadata: %s", e, question=question, exc_info=True, backtrace=True, diagnose=True)
                    metadata_list = []

                retrieval_latency = time.time() - start_time
                retrieval_tokens = retriever.total_tokens
                retrieval_input_tokens = retriever.total_input_tokens
                retrieval_output_tokens = retriever.total_output_tokens
                retrieval_reasoning_tokens = retriever.total_reasoning_tokens
                predicted_titles = [metadata.title for metadata in metadata_list]

                results = {"retrieval_latency": retrieval_latency, "retrieval_tokens": retrieval_tokens, "retrieval_input_tokens": retrieval_input_tokens,
                           "retrieval_output_tokens": retrieval_output_tokens, "retrieval_reasoning_tokens": retrieval_reasoning_tokens,
                           "retrieval_nr_searches": retriever.performed_searches}
                all_results["retrieval_latency"].append(retrieval_latency)
                all_results["retrieval_tokens"].append(retrieval_tokens)
                all_results["retrieval_input_tokens"].append(retrieval_input_tokens)
                all_results["retrieval_output_tokens"].append(retrieval_output_tokens)
                all_results["retrieval_reasoning_tokens"].append(retrieval_reasoning_tokens)
                all_results["retrieval_nr_searches"].append(retriever.performed_searches)

                # Compute cost of retrieval call
                input_price, output_price = get_llm_pricing(retrieval_llm) # prices are per million tokens
                retrieval_cost_usd = (retrieval_input_tokens * input_price + retrieval_output_tokens * output_price) / 1E6
                # Embedding tokens are added to total, but not counted in input/output tokens
                retrieval_cost_usd += (retrieval_tokens - retrieval_input_tokens - retrieval_output_tokens) * embedding_price / 1E6
                results["retrieval_cost_usd"] = retrieval_cost_usd
            else:
                results = {}
                predicted_titles = reference_titles # Use GT if only analysis is run
                # Open up processed_metadata_embeddings.csv and init metadata_list by looking up ID with title
                metadata_df = pd.read_csv(os.path.join(script_dir, "../data/metadata/50000006/processed_metadata_embeddings.csv"))
                metadata_list = []
                for title in predicted_titles:
                    metadata_id = metadata_df.loc[metadata_df['title'] == title, 'metadata_id'].values[0]
                    metadata_list.append(Metadata("50000006", metadata_id))

            # Run other evaluators
            for evaluator in evaluators:
                result = evaluator(predicted_titles, reference_titles)
                if "alternative_relevant_datasets" in example["outputs"]:
                    # Also evaluate against potential alternative relevant datasets, and take best result
                    for alt_reference_titles in example["outputs"]["alternative_relevant_datasets"]:
                        alt_result = evaluator(predicted_titles, alt_reference_titles)
                        result = max(result, alt_result) # Note that this requires all evaluators to have an output where higher is better
                        
                results[evaluator.__name__] = result
                all_results[evaluator.__name__].append(result)

            # Write results to file
            result_dict = {
                "question": question,
                "id": id,
                "predicted_titles": predicted_titles,
                "reference_titles": reference_titles,
                "results": results,
            }

            if len(predicted_titles) == 0 and len(reference_titles) > 0:
                # If no titles are predicted, but there are relevant datasets, set correct to False
                result_dict["results"]["correct"] = False
                all_results["correct"].append(False)
            elif len(predicted_titles) == 0 and len(reference_titles) == 0:
                # If no titles are predicted, and there are no relevant datasets, set correct to True
                result_dict["results"]["correct"] = True
                all_results["correct"].append(True)

            # Note that we don't set correct to False if there are predicted titles, but no relevant datasets, as we give the Analyzer the chance
            # to realize that the datasets are not relevant and produce an answer stating that.

            # If the LLM predicted a set of titles and the "only_retrieval" flag is not set, we additionally run the analyzer
            if len(predicted_titles) > 0 and not only_retrieval:
                result_dict["reference_answer"] = example["outputs"].get("answer", "This question cannot be answered with the available data")

                start_time = time.time()
                if analyzer_type == "simple":
                    analyzer = SimpleAnalyzer(retriever.groupOwner, metadata_list, timeout=60, coding_client=get_llm_client(coding_llm))
                elif analyzer_type == "simple_local_v2":
                    analyzer = SimpleLocalAnalyzerV2(retriever.groupOwner, metadata_list, coding_client=get_llm_client(coding_llm), streaming=False)
                elif analyzer_type == "iterative_local":
                    analyzer = IterativeLocalAnalyzer(retriever.groupOwner, metadata_list, coding_client=get_llm_client(coding_llm), streaming=False)
                else:
                    raise ValueError(f"Invalid analyzer type {analyzer_type}. Choose one of: simple | simple_local_v2 | iterative_local")
                
                try:
                    *_, final_messages = analyzer.analyze(question)
                    analyzer.finalize()
                except Exception as e:
                    log.error("Caught an exception analyzing question: %s", e, question=question, exc_info=True, backtrace=True, diagnose=True)
                    final_messages = ["An error occurred while analyzing the question."]
                coding_latency = time.time() - start_time
                result_dict["results"]["coding_latency"] = coding_latency
                result_dict["results"]["coding_tokens"] = analyzer.total_tokens
                all_results["coding_latency"].append(coding_latency)
                all_results["coding_tokens"].append(analyzer.total_tokens)
                all_results["coding_input_tokens"].append(analyzer.total_input_tokens)
                all_results["coding_output_tokens"].append(analyzer.total_output_tokens)
                all_results["coding_reasoning_tokens"].append(analyzer.total_reasoning_tokens)

                # Compute cost of coding call
                input_price, output_price = get_llm_pricing(coding_llm) # prices are per million tokens
                coding_cost_usd = (analyzer.total_input_tokens * input_price + analyzer.total_output_tokens * output_price) / 1E6
                result_dict["results"]["coding_cost_usd"] = coding_cost_usd

                # Search for predicted answer (in case there are additional elements, e.g. figures or maps)
                # Search backwards through final_messages for the first string message to use as predicted_answer
                result_dict["predicted_answer"] = None
                for msg in reversed(final_messages):
                    if isinstance(msg, str) and msg.strip():
                        result_dict["predicted_answer"] = msg
                        break

                # If there is a predicted answer, use the LLM judge to evaluate it.
                # If not, it means the analyzer failed to generate an answer.
                if result_dict["predicted_answer"] is not None:
                    prompt = f"Question: {question}\nReference Answer: {result_dict["reference_answer"]}\nPredicted Answer: {result_dict["predicted_answer"]}"
                    try:
                        eval_result = llm_judge.invoke([SystemMessage(LLM_JUDGE_SYSTEM_PROMPT), HumanMessage(content=prompt)])
                        result_dict["results"]["correct"] = eval_result.correct
                        all_results["correct"].append(eval_result.correct)
                    except Exception as e:
                        log.error("Caught an exception evaluating answer: %s", e, question=question, exc_info=True, backtrace=True, diagnose=True)
                        result_dict["results"]["correct"] = False
                        all_results["correct"].append(False)
                else:
                    result_dict["results"]["correct"] = False
                    all_results["correct"].append(False)

            results_data.append(result_dict)

        # Log average of evaluator results
        avg_results = {}
        for name, values in all_results.items():
            # Only average numeric results (skip bools for accuracy, treat as float)
            if all(isinstance(v, bool) for v in values):
                avg = sum(float(v) for v in values) / len(values) if values else 0.0
            else:
                avg = sum(values) / len(values) if values else 0.0
            avg_results[name] = avg

        # Print average results
        print(f"Average results for {experiment_prefix}:")
        for name, avg in avg_results.items():
            print(f"{name}: {avg:.4f}")

        # Save results and metadata to JSON file
        with open(outfile, "w", encoding="utf-8") as f_out:
            json.dump({"meta": meta, "data": results_data}, f_out, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval models.")
    parser.add_argument("--groupOwner", type=str, default="50000006", help="The groupOwner id whose metadata should be queried (default: 50000006).")
    parser.add_argument("--top_n", type=int, default=10, help="The number of documents to retrieve for a single KNN search (default: 10).")
    parser.add_argument("--retriever", type=str, choices=["agentic", "verified"], default="agentic", help="The retrieval strategy to use")
    parser.add_argument("--analyzer", type=str, choices=["simple_local_v2", "simple", "iterative_local"], default="simple_local_v2", help="The analyzer type to use")
    parser.add_argument("--retrieval_llm", choices=SUPPORTED_LLMS, default='gpt-4.1', help="The LLM to use for retrieval tasks.")
    parser.add_argument("--coding_llm", choices=SUPPORTED_LLMS, default='gpt-4.1', help="The LLM to use for coding tasks/analysis.")
    parser.add_argument("--experiment_name", type=str, help="Custom experiment name suffix.")
    parser.add_argument("--file_name", type=str, default="50000006_german.jsonl", help="The filename of the dataset to evaluate.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--only_retrieval", action="store_true", help="Only run the retrieval test, without the analyzer (skipped).")
    group.add_argument("--only_analysis", action="store_true", help="Only run the analysis test, without the retrieval (analyzer gets GT retrieval results).")
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument("--hybrid_search", action="store_true", help="Enable hybrid search with Milvus.")
    group2.add_argument("--bm25_search", action="store_true", help="Enable BM25 search with Milvus.")
    args = parser.parse_args()

    # Init utils
    init_mappings("50000006", excluded_extensions=[".tif"])
    log_name = f"eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.retriever}_{args.analyzer}_{args.file_name.split('.')[0]}"
    if args.experiment_name:
        log_name += f"_{args.experiment_name}"
    setup_logging(level=logging.INFO, log_filename=f"{log_name}.log")

    # Initialize the selected retriever
    if args.retriever == "verified":
        retriever = VerifiedRetriever("50000006", args.top_n, hybrid_search=args.hybrid_search, llm_client=get_llm_client(args.retrieval_llm), bm25_search=args.bm25_search)
        prefix = "verified_retrieval"
    else:
        retriever = AgenticRetriever("50000006", args.top_n, hybrid_search=args.hybrid_search, llm_client=get_llm_client(args.retrieval_llm), bm25_search=args.bm25_search)
        prefix = "agentic_retrieval"

    if args.hybrid_search:
        prefix += "_hybrid"
    elif args.bm25_search:
        prefix += "_bm25"

    if args.only_retrieval:
        prefix += "_only_retrieval"
    elif args.only_analysis:
        prefix += "_only_analysis"

    # Load JSON lines dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../data/evaluation", args.file_name)
    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    experiment_prefix = f"{prefix}_{args.experiment_name}" if args.experiment_name is not None else prefix
    language = os.path.splitext(args.file_name)[0].split("_")[-1]

    evaluate(
        retriever,
        args.analyzer,
        data=data,
        evaluators=[validation_accuracy, precision, recall],
        experiment_prefix=experiment_prefix,
        language=language,
        only_retrieval=args.only_retrieval,
        only_analysis=args.only_analysis,
        retrieval_llm=args.retrieval_llm,
        coding_llm=args.coding_llm
    )