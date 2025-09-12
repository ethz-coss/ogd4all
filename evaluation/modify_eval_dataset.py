import argparse
import json
import os
import sys
from tqdm import tqdm
from deep_translator import DeeplTranslator
import re
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import random
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # not the nicest way of handling this, but oh well...
from utils import generate_rephrasal_prompt


class Rephrasal(BaseModel):
    """Rephrasals of a question with the same meaning."""
    rephrasals: list[str] = Field(default_factory=list, description="Multiple rephrasals of the same question, always with exactly the same meaning.")


def rephrase_dataset(lines, num_rephrasings: int, root: str, dataset_name: str):
    """
    Rephrase the questions using GPT-4o num_rephrasing times and save num_rephrasings new files.
    Also adds two typos.
    """
    llm_rephraser = AzureChatOpenAI(
                azure_deployment="gpt-4o",
                api_version="2024-10-21", # low temperature removed to get more diverse rephrasals
                temperature=0.5, # mixture of adherence to the original and creativity
                max_tokens=None,
                timeout=None,
                max_retries=2,
    ).with_structured_output(Rephrasal)

    log_path = os.path.join(root, f"{dataset_name}_rephrasing_log.txt")
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("Original and Rephrasals Log\n")
        log_file.write("=" * 50 + "\n")

    modified_lines = [] # list of lists
    for i in range(num_rephrasings):
        modified_lines.append([])

    for line in tqdm(lines, desc="Rephrasing", unit="line"):
        obj = json.loads(line)
        q = obj["inputs"]["question"]
        rephrasals = llm_rephraser.invoke([
            SystemMessage(generate_rephrasal_prompt(num_rephrasings)),
            HumanMessage(q)
        ])

        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"Original: {q}\n")
            for i, rephrasal in enumerate(rephrasals.rephrasals):
                new_q =  add_typos(rephrasal, 2)
                log_file.write(f"Rephrasal {i + 1}: {new_q}\n")
                obj["inputs"]["question"] = new_q
                modified_lines[i].append(json.dumps(obj, ensure_ascii=False))
            log_file.write("=" * 50 + "\n")

    for i, modified in enumerate(modified_lines):
        out_name = f"{dataset_name}_rephrased_{i}.jsonl"
        out_path = os.path.join(root, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(modified))
        print(f"Rephrased file written to {out_path}")
    print(f"Log file written to {log_path}")


def add_typos(q, num_typos: int):
    # Randomly pick num_typos words to add a random typo to (either swap, delete or insert characters)
    words = q.split()
    for _ in range(num_typos):
        word_idx = random.randint(0, len(words) - 1)
        word = words[word_idx]

        typo_type = random.choice(["swap", "delete", "insert"])
        if typo_type == "swap":
            if len(word) > 1:
                char_idx = random.randint(0, len(word) - 2)
                chars = list(word)
                chars[char_idx], chars[char_idx + 1] = chars[char_idx + 1], chars[char_idx]
                word = "".join(chars)
        elif typo_type == "delete":
            char_idx = random.randint(0, len(word) - 1)
            word = word[:char_idx] + word[char_idx + 1:]
        elif typo_type == "insert":
            char_idx = random.randint(0, len(word))
            insert_char = random.choice("abcdefghijklmnopqrstuvwxyz")
            word = word[:char_idx] + insert_char + word[char_idx:]

        words[word_idx] = word
    return " ".join(words)


def translate_dataset(lines, source_lang: str, target_lang: str):
    log_path = os.path.join(root, f"translation_log.txt")
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("Original and Translated Log\n")
        log_file.write("=" * 50 + "\n")


    translator = DeeplTranslator(api_key=os.environ.get("DEEPL_API_KEY"), source=source_lang, target=target_lang)
    request_count = 0
    for line in tqdm(lines, desc="Translating", unit="line"):
        obj = json.loads(line)
        q = obj["inputs"]["question"]
        obj["inputs"]["question"] = translator.translate(q)

        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"Original: {q}\n")
            log_file.write(f"Translated: {obj['inputs']['question']}\n")
            log_file.write("=" * 50 + "\n")

        yield json.dumps(obj, ensure_ascii=False)

        request_count += 1
        if request_count % 50 == 0:
            print("Sleeping for 20 seconds to avoid hitting rate limits...")
            time.sleep(20)


"""
Update ground truth outputs in translated files if there has been a change in the original dataset.
Prints the number of changes performed for each file.
"""
def update_outputs(orig_lines, dir_path, base_name):
    patt = re.compile(rf"^{re.escape(base_name)}_translated_.+\.jsonl$")
    files = [f for f in os.listdir(dir_path) if patt.match(f)]
    if not files:
        raise FileNotFoundError("No translated files found to update.")
    for fname in files:
        t_path = os.path.join(dir_path, fname)
        with open(t_path, encoding="utf-8") as f:
            t_lines = f.readlines()
        if len(t_lines) != len(orig_lines):
            raise ValueError(f"Line count mismatch in {fname}")
        out_lines = []
        changes = 0
        for o_line, t_line in zip(orig_lines, t_lines):
            o_obj, t_obj = json.loads(o_line), json.loads(t_line)
            if t_obj.get("outputs") != o_obj.get("outputs"):
                changes += 1
            t_obj["outputs"] = o_obj["outputs"]
            out_lines.append(json.dumps(t_obj, ensure_ascii=False))
        with open(t_path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines))
        print(f"Updated outputs in {fname} ({changes} change{'s' if changes != 1 else ''})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify a given evaluation dataset, e.g. translating questions or rephrasing them.")
    parser.add_argument("--dataset_name", type=str, required=True, help="The filename of the dataset to modify.")
    parser.add_argument("--modification", type=str, required=True, choices=["translate", "rephrase", "update_outputs"], help="The modification to apply to the dataset.")
    parser.add_argument("--target_language", type=str, help="The target language (e.g. en, fr, it).")
    parser.add_argument("--source_language", type=str, default="de", help="The source language (default: de).")
    args = parser.parse_args()

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/evaluation")
    src_path = os.path.join(root, args.dataset_name)

    if not os.path.exists(src_path):
        raise FileNotFoundError(src_path)

    with open(src_path, encoding="utf-8") as f:
        orig_lines = f.readlines()

    base = args.dataset_name.rsplit(".jsonl", 1)[0]

    if args.modification == "update_outputs":
        update_outputs(orig_lines, root, base)
    elif args.modification == "translate":
        if not args.target_language:
            raise ValueError("Missing --target_language for translation.")
        out_name = f"{base}_translated_{args.target_language}.jsonl"
        out_path = os.path.join(root, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            for t in translate_dataset(orig_lines, args.source_language, args.target_language):
                f.write(t + "\n")
        print(f"Translated file written to {out_path}")
    elif args.modification == "rephrase":
        rephrase_dataset(orig_lines, num_rephrasings=5, root=root, dataset_name=base)
