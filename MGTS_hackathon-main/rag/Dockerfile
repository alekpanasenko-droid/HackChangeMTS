FROM python:3.12-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    HF_HOME=/opt/huggingface \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 \
    && python -m pip install --no-cache-dir -r requirements.txt
COPY rag_common.py build_index.py rag_pipeline.py ./
RUN python -c "from rag_common import load_embedder; load_embedder('cpu')"
ENV HF_HUB_OFFLINE=1

FROM builder AS runtime
COPY artifacts/index.zip /app/artifacts/index.zip
ENTRYPOINT ["python", "rag_pipeline.py"]
CMD ["--help"]
