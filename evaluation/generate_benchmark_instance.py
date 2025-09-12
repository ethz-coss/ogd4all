import os
import json
import subprocess
import sys
import tqdm
import re
import copy  # Import copy module for deep copying
import random

"""
This script executes the GT scripts for the benchmark questions and saves their outputs.
It further handles templated benchmark questions by supplying the necessary parameter.

NOTE: If you want to generate a benchmark instance, setting the `GOOGLE_GEOCODING_API_KEY` environment variable is recommended.
"""

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "../data/evaluation", "50000006_german_template.jsonl")

# Iterate over each line in the file and add an "id" field
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Create a new list to store the modified lines
modified_lines = []
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8" # as sometimes prints will print with special characters

for i, line in enumerate(tqdm.tqdm(lines, desc="Processing lines")):
    data = json.loads(line.strip())
    # Skip tests where no datasets are relevant
    if not data["outputs"]["relevant_datasets"]:
        modified_lines.append(json.dumps(data, ensure_ascii=False))
        continue

    filename = f"{data['id']}.py"
    path = os.path.join(script_dir, "../data/evaluation/50000006_gt_scripts", filename)

    print(f"Processing question: {data["inputs"]["question"]} (ID: {data['id']})")

    if "param_type" in data:
        # There are templated parameters, so we will execute the script multiple times with varying parameters
        if data["param_type"] in ["ordinal", "nominal"]:
            options = data["param_options"] # exhaust all options
        else:
            raise ValueError(f"Unsupported param_type: {data['param_type']}")

        # Constrain number of options to at most 4 (randomly chosen) to avoid raising importance of question too much
        if len(options) > 4:
            options = random.sample(options, 4)

        for i, option in enumerate(options):
            # Create a deep copy of the data dictionary to avoid modifying the original
            data_copy = copy.deepcopy(data)

            # Update question_id with idx of option
            data_copy["id"] = f"{data_copy['id']}_{i}"
            # Strip {{ ... }} (with regex) from question and insert chosen option
            data_copy["inputs"]["question"] = re.sub(r"\{\{.*?\}\}", str(option), data_copy["inputs"]["question"], count=1)

            # Run script similar as if no parameters were present, except that we pass parameter
            result = subprocess.run(
                [sys.executable, path, str(option)],
                capture_output=True,
                text=True,
                check=False,  # Don't raise exception on non-zero exit
                cwd=os.path.join(script_dir, ".."),
                encoding="utf-8",
                env=env
            )
            stdout_output = result.stdout
            data_copy["outputs"]["answer"] = stdout_output.strip()
            modified_lines.append(json.dumps(data_copy, ensure_ascii=False))

    else:
        # No templated parameters, so we just execute the script as is
        # Execute the python script and capture the stdout output, save it to data
        # Run the script with the current Python interpreter and set the working directory to the parent directory
        

        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.join(script_dir, ".."),
            encoding="utf-8",
            env=env
        )
        stdout_output = result.stdout
        data["outputs"]["answer"] = stdout_output.strip()
        modified_lines.append(json.dumps(data, ensure_ascii=False))


# Write the modified lines to file
output_file_path = os.path.join(script_dir, "../data/evaluation", "50000006_german.jsonl")
with open(output_file_path, "w", encoding="utf-8") as output_file:
    for modified_line in modified_lines:
        output_file.write(modified_line + "\n")