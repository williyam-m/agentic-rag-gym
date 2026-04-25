FROM python:3.11-slim

# Build args
ARG PORT=8080

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml ./
COPY agentic_rag_os/requirements.txt ./agentic_rag_os_requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        fastapi>=0.115.0 \
        uvicorn[standard]>=0.30.0 \
        pydantic>=2.9.0 \
        pydantic-settings>=2.5.0 \
        python-dotenv>=1.0.0 \
        sqlalchemy>=2.0.0 \
        aiosqlite>=0.20.0 \
        httpx>=0.27.0 \
        python-jose[cryptography]>=3.3.0 \
        python-multipart>=0.0.12 \
        aiofiles>=23.0.0 \
        faiss-cpu>=1.8.0 \
        sentence-transformers>=3.0.0 \
        numpy>=1.26.0 \
        structlog>=24.0.0 \
        tenacity>=9.0.0

# Copy full project
COPY . .

# Create required directories
RUN mkdir -p /app/data/faiss_indices /app/agentic_rag_os/uploads

# Non-root user for security
RUN useradd -m -u 1000 ragos && chown -R ragos:ragos /app
USER ragos

ENV PORT=${PORT}
ENV RAGAS_PORT=${PORT}
ENV RAGAS_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

CMD ["python", "-m", "agentic_rag_os.run"]
