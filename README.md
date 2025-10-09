# OGD4All ![Python Version](https://img.shields.io/badge/Python-3.12-blue) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OGD4All is a prototype system enabling an easy-to-use, transparent interaction with Geospatial Open Government Data through natural language. It was developed and tested with 430 tabular and geospatial datasets of the city of Zurich.

## 💡 Architecture
<img width="2948" height="869" alt="highLevelOverview" src="https://github.com/user-attachments/assets/828f68c2-470b-4c59-b80f-8b371cac0d29" />


## ✨ Demo
https://github.com/user-attachments/assets/b3b6c24a-a1f7-4d31-b16a-a9a23a658f77

<details>
<summary>

## ⚙️ Setup

</summary>

### 1. Code Environment Setup
Run the following commands with Python 3.12 to create a virtual environment that contains all required dependencies:
```bash
python -m venv .venv # Create new virtual environment named .venv (or whatever you wanna call it)

# activate virtual environment depending on your platform #
.venv\Scripts\activate.bat # Windows, cmd.exe
.venv\Scripts\Activate.ps1 # Windows, Powershell
source .venv/bin/activate  # Linux/MacOS
# ------------------------------------------------------- #

pip install -r requirements.txt
```
Configure the following environment variables:
AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_ENDPOINT_EMBEDDING_LARGE

### 2. Obtaining Zurich OGD
The system was tested with 430 tabular and geospatial datasets of the city of Zurich, but could also be used with other types of data (with some additional steps).
While the associated metadata is already contained in this repository, you need to perform some additional steps for getting the actual data.
1. Download the datasets [here](https://drive.google.com/file/d/1MdqwSZW0i__oRXNFUJAqlwYcFqypGpia/view?usp=sharing). Unfortunately, Zurich's open data platform does not offer an API for downloading datasets, meaning these datasets were manually exported. Note that this was performed in March and May of 2025, meaning some datasets may be out-of-date. All extraction timestamp, dataset titles and filenames are listed in `data\opendata\50000006\downloads.csv`, feel free to replace datasets with more recent versions.
2. Extract the ZIP folder into the directory `data\opendata\50000006\extracted` (make sure it is not further nested).
3. If you added the data correctly, you should see the following messages at startup:
```
[info     ] Will use 430 files for group owner 50000006.
[info     ] 430/548 metadata embeddings remaining after filtering for existing data.
```

### 3. Vector Store with Embeddings
By default, OGD4All uses OpenAI's `text-embedding-3-large` for semantic search. Corresponding embeddings were generated for the metadata of all datasets, and is stored in `data\metadata\50000006\processed_metadata_embeddings.csv`. You have two options:
- **Keep text-embedding-3-large**: You'll need to set either the environment variable `OPENAI_API_KEY` (with your OpenAI API key) or both `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT_EMBEDDING_LARGE` (with your Azure OpenAI API key and endpoint).
- **Use custom embeddings**: First, update the `processed_metadata_embeddings.csv` file by modifying the `scraping\generate_embeddings.py` script with your own method. Then, update the `get_embedding` function in `retrieval\retriever.py`.

### 4. LLM Setup
Support for LLMs provided by Azure OpenAI, Azure, OpenAI, OpenRouter, and Ollama is implemented.
Depending on which one you use, set the environment variable `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_API_KEY`, `OPENAI_API_KEY`, or `OPENROUTER_API_KEY`.

If your desired model is not listed in `SUPPORTED_LLMS` in `utils.py`, just add it to the list and adapt the `get_llm_client` function.
</details>

  
## 🗺️ Usage
```
python main.py
```
This will start a local gradio server where you can interact with 430 geospatial and tabular datasets of the city of Zurich through a simple interface.

### Command-Line Arguments
| Argument                | Default           | Choices                                                        | Description                                                                                                                                               |
| ----------------------- | ----------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--groupOwner`          | `50000006`        | –                                                              | The `groupOwner` ID whose metadata should be queried (e.g., the city of Zurich).                                                                          |
| `--top_n`               | `10`              | Any number > 0                                                 | The number of documents to retrieve for a single KNN search. Cost considerations and your model's context window should determine how high you push this. |
| `--retriever`           | `agentic`         | `agentic`, `knn`, `verified`                                   | The retrieval strategy to use.                                                                                                                            |
| `--analyzer`            | `iterative_local` | `simple_local_v2`, `simple`, `iterative_local`                 | The analyzer type to use.                                                                                                                                 |
| `--retrieval_llm`       | `gpt-4.1`         | `SUPPORTED_LLMS`                                               | The LLM to use for retrieval tasks.                                                                                                                       |
| `--retrieval_check_llm` | `gpt-4o`          | `SUPPORTED_LLMS`                                               | The LLM to use for checking whether a follow-up retrieval is needed. Should ideally be fast.                                                              |
| `--coding_llm`          | `gpt-4.1`         | `SUPPORTED_LLMS`                                               | The LLM to use for coding tasks/analysis.                                                                                                                 |
| `--hybrid_search`       | `Not set`         | –                                                              | Enable hybrid search with Milvus (default is semantic search).                                                                                            |
| `--bm25_search`         | `Not set`         | –                                                              | Enable BM25 search with Milvus (default is semantic search).                                                                                              |
| `--no_streaming`        | `Not set`         | –                                                              | Disable streaming for the coding LLM. This enables validation of LLM responses and token counting, but makes the system feel less responsive.             |

```
SUPPORTED_LLMS = ['gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4o', 'gpt-o1', 'gpt-o3-mini-preview', 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17', 'meta-llama-70B-instruct-latest', 'mistral-codestral', 'mistral-large', 'openai/gpt-oss-120b', 'meta-llama/llama-4-maverick']
```

## Acknowledgement
This prototype was implemented as part of Michael Siebenmann's Master's thesis at the Professorship of Computational Social Science and was further supported by the Esri R&D Center Zurich.
