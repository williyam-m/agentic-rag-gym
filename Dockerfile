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

COPY . .

RUN mkdir -p /app/data/faiss_indices

ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=7860
ENV API_BASE_URL=https://router.huggingface.co/novita/v3/openai
ENV MODEL_NAME=meta-llama/llama-3.1-8b-instruct
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "main.py"]
