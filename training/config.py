"""
Configuration for GRPO training of Agentic RAG Gym.
All secrets and tunables are loaded from environment variables / .env file.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Environment / secrets
# ---------------------------------------------------------------------------
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
HF_USERNAME: str = os.getenv("HF_USERNAME", "williyam")

# Environment server (our FastAPI gym)
ENV_URL: str = os.getenv("ENV_URL", "http://localhost:7860")

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
BASE_MODEL_ID: str = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
FINETUNED_MODEL_ID: str = os.getenv(
    "FINETUNED_MODEL_ID",
    f"{HF_USERNAME}/agentic-rag-aerospace-grpo",
)
