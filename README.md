---
title: Agentic RAG Gym
emoji: ⚜
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
tags:
  - openenv
  - rag
  - reinforcement-learning
  - multi-agent
  - aerospace
  - legal-research
pinned: true
---

[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/williyam/agentic-rag-gym)
[![Fine-Tuned Model](https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Model-Qwen2.5--GRPO-orange)](https://huggingface.co/williyam/agentic-rag-aerospace-grpo)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/FastAPI-OpenEnv%20API-009688?logo=fastapi&logoColor=white)](server/app.py)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-4CAF50)](openenv.yaml)


# ⚜ Agentic RAG Gym

| Resource | Link |
|---|---|
| **HF Space** | [Agentic RAG Gym](https://huggingface.co/spaces/williyam/agentic-rag-gym) |
| **Fine-Tuned Model** | [Qwen2.5 GRPO LoRA Adapter](https://huggingface.co/williyam/agentic-rag-aerospace-grpo) |
| **Training Notebook** | [Google Colab — GRPO Fine-Tuning](https://colab.research.google.com/drive/14il2JQmy9-id_fSGpmYbssp-j975DSDo?usp=sharing) |
| **YouTube Demo** | [Watch the Demo](https://www.youtube.com/watch?v=M65DHY8za6M) |
| **GitHub Repo** | [williyam-m/agentic-rag-gym](https://github.com/williyam-m/agentic-rag-gym) |
| **Agentic RAG OS** | [Live Platform](https://rag-8000.zcodecorp.in/) — see demo below |

**Agentic RAG Gym** is an open-source reinforcement learning framework that empowers the entire RAG technology stack — retrieval, reasoning, critique, verification, and synthesis — with RL-driven process supervision. We designed a new orchestrator from scratch, **RAG Master**, where autonomous agents don't just retrieve and generate; they *learn to research like domain experts* through multi-agent collaboration, per-step reward signals from real domain graders, and adversarial self-improvement.

The framework is **domain-agnostic by design**: plug in any knowledge domain through the adapter pattern, define tasks, build deterministic graders, and the RL training loop handles everything else — embedding, FAISS indexing, composite reward computation, anti-reward-hacking, and policy optimization via GRPO.

The first two domains? **Aerospace Research** (scramjet propulsion, UHTC thermal protection, Mars EDL architecture) and **Legal Research** (cross-border IP disputes, GDPR/CCPA compliance, ICC arbitration) — because if your agent can design a hypersonic vehicle or navigate a multi-jurisdictional patent dispute, it can handle anything.

---

## Why This Matters

Traditional RAG systems are static pipelines: retrieve → generate → done. They have no feedback loop, no process awareness, no ability to learn from mistakes. **Agentic RAG Gym** changes this by:

1. **Process Supervision** — Per-step rewards teach agents *how* to research, not just *what* to output
2. **Multi-Agent Collaboration** — Five specialized agents (Retriever, Reasoner, Critic, Planner, Verifier) work together through message-passing
3. **RL-Driven Improvement** — GRPO fine-tuning uses real domain graders as reward signals — no proxy rewards
4. **Anti-Reward-Hacking** — Adversarial guards prevent the system from gaming its own rewards
5. **Domain Extensibility** — Plug in any domain (aerospace, legal, medical, finance) through the adapter pattern

---

## Themes Implemented

| Theme | Implementation |
|---|---|
| **#1 Multi-Agent Interactions** | Cooperative 5-agent system: Retriever, Reasoner, Critic, Planner, Verifier with structured message-passing |
| **#2 Long-Horizon Planning** | Tasks require 10-20 step trajectories with planning, iterative retrieval, reasoning, critique, and verification |
| **#3.1 World Modeling (Professional)** | Dynamic RAG environment with FAISS vector store, LLM reasoning, and tool orchestration |
| **#4 Self-Improvement** | Adversarial critique loops, iterative refinement, GRPO fine-tuning with real graders |
| **#5 Wild Card** | **Agentic RAG OS** — An RL B2B Rewards-as-a-Service (RaaS) startup platform built as a full-stack operating system for agentic RAG. **RAG Master** — A new orchestrator/framework designed from scratch (like LangChain/LangGraph) as a domain-agnostic engine for multi-agent RAG with RL-driven process supervision. |

### Agentic RAG OS — Live Demo

![Agentic RAG OS Demo](assets/demo/agentic-rag-os-demo.gif)

---

## Architecture

```
┌────────────── HF Space Client ──────────────────────────┐
│  Interactive │ Auto Pilot │ Tasks │ About                │
│           [Domain Selector: Aerospace / Legal]           │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              FastAPI Server (OpenEnv API)                 │
│  /reset │ /step │ /state │ /grade │ /domain/switch       │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              RAG Master Orchestrator                      │
│                                                          │
│  Retriever ←→ Reasoner ←→ Critic ←→ Verifier           │
│       ↕            ↕           ↕                         │
│  FAISS Store   LLM Client    Composite Rewards           │
├──────────────────────────────────────────────────────────┤
│  Domains: [Aerospace] [Legal Research] [Your Domain]     │
└──────────────────────────────────────────────────────────┘
```

## Key Features

- **RAG Master Framework** — Domain-agnostic orchestrator designed from scratch, extensible to any domain through clean adapters
- **Multi-Domain Support** — Aerospace Research (5 tasks, 16 documents) + Legal Research (5 tasks, 16 documents) with runtime domain switching
- **10 Research Tasks** — Easy → Medium → Hard difficulty across domains with deterministic graders
- **Multi-Agent System** — Retriever, Reasoner, Critic, Planner, Verifier agents with structured message-passing
- **Process Supervision** — Per-step rewards for retrieval quality, reasoning depth, and answer completeness
- **Anti-Reward-Hacking** — Repetition detection, degenerate output penalties, keyword stuffing guards, copy-paste detection
- **GRPO Fine-Tuning** — Real domain graders as reward signal, LoRA adapters, published on HF Hub
- **OpenAI-Compatible** — Works with any OpenAI-compatible API: Ollama, vLLM, OpenAI, HuggingFace Inference, and more
- **[Live HF Space](https://huggingface.co/spaces/williyam/agentic-rag-gym)** — Interactive demo with domain switching and real-time feedback

---

## Quick Start

### Prerequisites

- Python 3.10+
- Any OpenAI-compatible LLM endpoint (Ollama, OpenAI, HF Inference, vLLM, etc.)

### Install & Run

```bash
# Clone the repository
git clone https://github.com/williyam-m/agentic-rag-gym.git
cd agentic-rag-gym

# Create environment and install dependencies
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your API endpoint and keys

# Start the server
python main.py
```

### Docker

```bash
docker compose up --build
```

The server starts at `http://localhost:7860`.

---

## Configuration

All configuration is via environment variables (`.env` file):

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `http://localhost:11434/v1` | Any OpenAI-compatible LLM endpoint |
| `MODEL_NAME` | `qwen2.5:7b` | Model identifier |
| `HF_TOKEN` | — | HuggingFace token |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `FAISS_INDEX_DIR` | `./data/faiss_indices` | FAISS storage path |
| `SERVER_PORT` | `7860` | Server port |

---

## API Usage

### Switch Domain

```bash
curl -X POST http://localhost:7860/domain/switch \
  -H "Content-Type: application/json" \
  -d '{"domain": "legal_research"}'
```

### Reset Environment

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "aero_easy_propulsion_comparison"}'
```

### Take a Step

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"type": "retrieve", "query": "ion propulsion specific impulse"}'
```

### Get State

```bash
curl -X GET http://localhost:7860/state
```

### Grade Episode

```bash
curl -X POST http://localhost:7860/grade
```

### List Tasks (for active domain)

```bash
curl -X GET http://localhost:7860/tasks
```

### List Available Domains

```bash
curl -X GET http://localhost:7860/domains
```

---

## Domains

### Aerospace Research

| Task | Difficulty | Max Steps | Description |
|---|---|---|---|
| `aero_easy_propulsion_comparison` | Easy | 12 | Compare ion vs nuclear thermal propulsion for Mars |
| `aero_easy_debris_mitigation` | Easy | 12 | Analyze space debris challenges and ADR technologies |
| `aero_medium_mars_edl` | Medium | 16 | Design Mars Entry/Descent/Landing architecture |
| `aero_medium_life_support` | Medium | 16 | Design integrated life support for Mars mission |
| `aero_hard_hypersonic_vehicle` | Hard | 20 | Design reusable hypersonic space access vehicle |

### Legal Research

| Task | Difficulty | Max Steps | Description |
|---|---|---|---|
| `legal_easy_contract_review` | Easy | 12 | Analyze liability, indemnification, and termination clauses |
| `legal_easy_privacy_compliance` | Easy | 12 | GDPR & CCPA/CPRA compliance gap analysis |
| `legal_medium_ip_analysis` | Medium | 16 | Patent eligibility, prior art, and trade secret strategy |
| `legal_medium_ma_due_diligence` | Medium | 16 | Technology company acquisition due diligence |
| `legal_hard_cross_border_dispute` | Hard | 20 | Multi-jurisdictional technology dispute resolution |

---

## Running Inference

```bash
# Ensure the server is running
python main.py &

# Set environment variables
export API_BASE_URL=http://localhost:11434/v1
export MODEL_NAME=qwen2.5:7b

# Run inference
python inference.py
```

---

## Action & Observation Spaces

### Action Space

| Action | Parameters | Description |
|---|---|---|
| `retrieve` | `query: str` | Search knowledge base |
| `reason` | — | Analyze retrieved documents |
| `answer` | `answer: str` | Submit final answer |
| `plan` | — | Create execution strategy |
| `critique` | — | Evaluate current reasoning |
| `verify` | — | Check answer grounding |

### Observation Space

Each observation contains:
- **task** — Current task info (id, name, description, difficulty, max_steps)
- **step** — Current step number
- **retrieved_docs** — Recently retrieved documents with scores
- **query_history** — Recent queries
- **current_answer** — Answer draft
- **last_reward** — Previous step reward
- **done** — Episode completion flag

---

## Reward Design

| Component | Weight | Signal |
|---|---|---|
| Retrieval Relevance | 25% | Average document relevance score |
| Reasoning Quality | 20% | Trace analysis (evidence, logic markers) |
| Answer Completeness | 30% | Coverage, length, rubric alignment |
| Efficiency | 15% | Step usage ratio penalty |
| Anti-Hacking | 10% | Deduction for detected exploits |

All scores strictly within **[0.01, 0.99]**.

---

## Extending to Any Domain

The `rag_master` framework is designed to be **domain-agnostic**. Adding a new domain takes four steps:

### 1. Create Domain Directory

```
domains/your_domain/
├── __init__.py
├── config.py          # Implements BaseDomainConfig
├── knowledge_base.py  # Your documents
├── tasks.py           # Your tasks
└── graders.py         # Your graders
```

### 2. Implement BaseDomainConfig

```python
from rag_master.adapters import BaseDomainConfig

class YourDomainConfig(BaseDomainConfig):
    def get_tasks(self) -> List[TaskDefinition]: ...
    def get_documents(self) -> List[Document]: ...
    def get_grader(self, task_id: str) -> BaseGrader: ...
    def get_reward_function(self) -> BaseRewardFunction: ...
    def get_system_prompt(self) -> str: ...
```

### 3. Register in Server

Add your domain to `DOMAIN_REGISTRY` in `server/app.py`:
```python
DOMAIN_REGISTRY = {
    "aerospace": AerospaceDomainConfig,
    "legal_research": LegalResearchDomainConfig,
    "your_domain": YourDomainConfig,
}
```

### 4. Choose Evaluation Method

- **Rule-based**: Use `KeywordCoverageGrader` with domain-specific keywords
- **LLM Judge**: Use `LLMJudgeRewardFunction` for nuanced evaluation
- **Hybrid**: Combine both approaches

---

## GRPO Fine-Tuning (Reinforcement Learning)

We fine-tune **Qwen2.5-0.5B-Instruct** using **Group Relative Policy Optimization (GRPO)** from TRL,
with LoRA adapters and the **real domain graders** as the reward signal — no proxy rewards.

### Training Results

| Metric | Baseline | GRPO-Trained | Improvement |
|--------|----------|-------------|-------------|
| **Mean Score** | 0.5580 | 0.5860 | **+0.0280** |
| Propulsion Comparison | 0.508 | 0.562 | +0.053 |
| Debris Mitigation | 0.633 | 0.689 | +0.056 |
| Hypersonic Vehicle | 0.482 | 0.521 | +0.039 |
| Mars EDL | 0.574 | 0.568 | -0.006 |
| Life Support | 0.592 | 0.590 | -0.002 |

### Training Curves

![Training Curves](assets/qwen-finetuning-plots/training_curves.png)

### Baseline vs. GRPO-Trained

![Baseline vs Trained](assets/qwen-finetuning-plots/baseline_vs_trained.png)

### Score Distribution

![Score Distribution](assets/qwen-finetuning-plots/score_distribution.png)

### Run Training

```bash
# Primary: Jupyter notebook
jupyter notebook agentic-rag-for-aerospace-research.ipynb

# Headless/CI
python train.py
```

### Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Method | GRPO (Group Relative Policy Optimization) |
| LoRA | r=16, α=32, targets=q/k/v/o_proj |
| Optimizer | AdamW (torch) |
| Learning Rate | 5e-6 |
| Epochs | 2 |
| Group Size (G) | 4 |
| Max Completion | 512 tokens |

### Fine-Tuned Model

The GRPO-trained model is available on Hugging Face:
**[williyam/agentic-rag-aerospace-grpo](https://huggingface.co/williyam/agentic-rag-aerospace-grpo)**

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("williyam/agentic-rag-aerospace-grpo")
tokenizer = AutoTokenizer.from_pretrained("williyam/agentic-rag-aerospace-grpo")
```

---

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=rag_master --cov=server --cov-report=term-missing

# Specific test file
pytest tests/test_rewards.py -v
```

---

## Reproduce in 60 Seconds

```bash
git clone https://github.com/williyam-m/agentic-rag-gym.git
cd agentic-rag-gym
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # Edit with your API keys
python main.py &
python inference.py    # Results saved to baseline_results.json
```

---

## Project Structure

```
agentic-rag-gym/
├── rag_master/              # Generic agentic RAG framework (domain-agnostic)
├── server/                  # FastAPI + Gradio server with domain switching
├── domains/aerospace/       # Aerospace research domain (5 tasks, 16 docs)
├── domains/legal_research/  # Legal research domain (5 tasks, 16 docs)
├── training/                # GRPO training package
├── tests/                   # Unit & integration tests
├── documents/               # Architecture & design docs
├── assets/qwen-finetuning-plots/  # Training curves & evaluation plots
├── agentic-rag-for-aerospace-research.ipynb  # GRPO training notebook
├── train.py                 # Standalone training script
├── inference.py             # Baseline inference script
├── openenv.yaml             # OpenEnv specification
├── Dockerfile               # Container definition
├── docker-compose.yml       # Service orchestration
└── main.py                  # Entry point
```

---

*Built for the Meta × OpenEnv × Hugging Face × PyTorch Hackathon*
