FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    proj-bin \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv --no-cache-dir

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    GRADIO_SERVER_NAME=0.0.0.0

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

RUN mkdir -p data/opendata/50000006/extracted

RUN uv sync --frozen --no-dev

EXPOSE 7860

CMD ["uv", "run", "python", "main.py", \
     "--lazy-download", \
     "--retriever", "agentic", \
     "--analyzer", "iterative_local", \
     "--retrieval_llm", "gpt-4.1-mini", \
     "--retrieval_check_llm", "gpt-4.1-mini", \
     "--coding_llm", "gpt-4.1"]
