FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY rag_master/ rag_master/
COPY server/ server/
COPY domains/ domains/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Pre-warm embedding model so first /reset doesn't time out
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY inference.py main.py openenv.yaml ./

RUN mkdir -p /app/data/faiss_indices

ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "main.py"]
