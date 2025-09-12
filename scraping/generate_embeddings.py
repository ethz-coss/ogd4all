import os
import re
import os
import pandas as pd
import tiktoken
from openai import AzureOpenAI
import json
import argparse
import logging
from tqdm import tqdm

pd.options.mode.chained_assignment = None

"""
Prepare data for tokenization (removes consecutive whitespaces and cleans up punctuation)

:param s: input text
:return: cleaned text
"""
def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s


"""
Generate a dataframe with metadata records used for retrieval and storage of embeddings

:param json_dir: The directory containing the processed metadata records
:param max_files: The maximum number of metadata records to process
:param groupOwner: The groupOwner id to filter metadata records by (e.g. "50000006" for the city of Zurich)
:param overwrite: If True, overwrite existing embeddings
:return: DataFrame
"""
def generate_dataframe(json_dir, max_files, groupOwner, overwrite=False):
    # If overwrite is False, load existing embeddings and collect metadata_ids
    existing_ids = set()
    if not overwrite and groupOwner is not None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        embeddings_path = os.path.join(script_dir, f"../data/metadata/{groupOwner}/processed_metadata_embeddings.csv")
        if os.path.exists(embeddings_path):
            existing_df = pd.read_csv(embeddings_path)
            existing_ids = set(existing_df["metadata_id"].astype(str))

    # Create dataframe with text columns title, purpose, abstract, identification date
    df = pd.DataFrame(columns=["title", "purpose", "abstract", "identification date", "metadata_id"])

    file_count = 0
    skip_count = 0
    for root, _, files in os.walk(json_dir):
        for file in files:
            if file.endswith(".json"):
                metadata_id = file.split(".")[0]
                if not overwrite and metadata_id in existing_ids:
                    # Skip files that already have embeddings if overwrite is False
                    skip_count += 1
                    continue
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding='utf-8') as f:
                    data = json.load(f)
                file_count += 1
                if file_count > max_files:
                    break

                df = pd.concat([df, pd.DataFrame([{
                    "metadata_id": metadata_id,
                    "title": data["title"],
                    "purpose": data["purpose"],
                    "abstract": data["abstract"],
                    "identification date": data["identificationDate"]
                }])], ignore_index=True)
        if file_count > max_files:
            break

    df['purpose'] = df["purpose"].apply(lambda x : normalize_text(x))
    df['abstract'] = df["abstract"].apply(lambda x : normalize_text(x))

    tokenizer = tiktoken.get_encoding("cl100k_base")
    # Identification date is None sometimes
    df['text'] = df["title"] + ((" (" + df["identification date"] + ")") if pd.notnull(df["identification date"]).all() else "") + " " + df["purpose"] + " " + df["abstract"]
    df['n_tokens'] = df["text"].apply(lambda x: len(tokenizer.encode(x)))

    logging.info(f"Token Statistics:\nsum: {df['n_tokens'].sum()}\n{df['n_tokens'].describe()}")
    logging.info(f"Processed {file_count} files, skipped {skip_count} files with existing embeddings. Set overwrite=True to reprocess all files.")

    return df

"""
Generate an embedding for a single text using the Azure OpenAI API

:param client: AzureOpenAI client
:param text: The text to generate an embedding for
:param model: The embedding model to use
:return: Embedding
"""
def generate_embedding(client, text, model):
    return client.embeddings.create(input = [text], model=model).data[0].embedding


"""
Generate embeddings for all df["text"] fields using the text-embedding-3-large model by Azure OpenAI and save them to a csv file

:param groupOwner: The groupOwner id to filter metadata records by (e.g. "50000006" for the city of Zurich)
:param df: DataFrame containing (new) metadata records
:param overwrite: If True, overwrite existing embeddings, otherwise append to existing embeddings
"""
def generate_embeddings(groupOwner, df, overwrite=False):
    deployment_name = "text-embedding-3-large"

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-10-21",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDING_LARGE")
    )
    
    tqdm.pandas(desc="Generating embeddings")
    df[deployment_name] = df["text"].progress_apply(lambda x: generate_embedding(client, x, deployment_name))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, f"../data/metadata/{groupOwner}/processed_metadata_embeddings.csv")

    if not overwrite and os.path.exists(save_path):
        existing_df = pd.read_csv(save_path)
        # Ensure columns match
        required_cols = ["title", "metadata_id", deployment_name]
        for col in required_cols:
            if col not in existing_df.columns:
                raise ValueError(f"Column '{col}' is missing in the existing DataFrame. Please check the CSV file.")

        combined_df = pd.concat([existing_df[required_cols], df[required_cols]], ignore_index=True)
        
        # Check for duplicates and if there are, print a warning but keep only last occurrence
        if combined_df.duplicated(subset=["metadata_id"]).any():
            logging.warning("Duplicates found in metadata_id. Keeping the last occurrence.")
            combined_df = combined_df.drop_duplicates(subset=["metadata_id"], keep='last')

        combined_df.to_csv(save_path, index=False)
        logging.info(f"Appended embeddings to {save_path}. Total records: {len(combined_df)}")
    else:
        df[["title", "metadata_id", deployment_name]].to_csv(save_path, index=False)
        logging.info(f"Saved embeddings to {save_path}. Total records: {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings from metadata records using the Azure OpenAI API.")
    parser.add_argument("--groupOwner", type=str, default="50000006",help="The groupOwner id to filter metadata records by (default: 50000006).")
    parser.add_argument("--max_files", type=int, default=10000, help="The maximum number files to generate embeddings for (default: 10000).")
    parser.add_argument("--verbose", action="store_true", help="Print more information during processing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing embeddings")
    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO if args.verbose else logging.WARNING)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_dir = os.path.join(script_dir, f"../data/metadata/{args.groupOwner}/processed")

    df = generate_dataframe(json_dir, max_files=args.max_files, groupOwner=args.groupOwner, overwrite=args.overwrite)
    generate_embeddings(args.groupOwner, df, overwrite=args.overwrite)