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
pinned: true
---

# ⚜ Agentic RAG Gym

> *Where autonomous agents learn to research like experts.*

**Agentic RAG Gym** is a reinforcement learning environment that trains AI agents on complex **Retrieval-Augmented Generation** tasks. Built on the OpenEnv specification, it simulates real-world research workflows where agents must search, reason, critique, and synthesize information from a domain-specific knowledge base.

The first domain? **Aerospace Research** — because if your agent can design a hypersonic vehicle by cross-referencing scramjet propulsion, UHTC materials, and autonomous flight systems, it can handle anything.

---

## Themes Implemented

| Theme | Implementation |
|---|---|
| **#1 Multi-Agent Interactions** | Cooperative multi-agent system: Retriever, Reasoner, Critic, Planner, Verifier agents with message-passing |
| **#2 Long-Horizon Planning** | Tasks require 10-20 step trajectories with planning, iterative retrieval, reasoning, critique, and verification |
| **#3.1 World Modeling (Professional)** | Dynamic RAG environment with FAISS vector store, LLM reasoning, and tool orchestration |
| **#4 Self-Improvement** | Adversarial critique loops, iterative refinement, process supervision rewards |
| **#5 Wild Card** | Aerospace research domain — novel, technically deep, genuinely useful for evaluating agent capabilities |

---

## Architecture

```
┌──────────── Gradio UI (Royal Glassmorphism) ────────────┐
│  Interactive │ Auto Pilot │ Tasks │ About                │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              FastAPI Server (OpenEnv API)                 │
│  POST /reset │ POST /step │ GET /state │ POST /grade     │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              RAG Master Orchestrator                      │
│                                                          │
│  Retriever ←→ Reasoner ←→ Critic ←→ Verifier           │
│       ↕            ↕                                     │
│  FAISS Store   LLM Client    Composite Rewards           │
└──────────────────────────────────────────────────────────┘
```

## Key Features

- **Generic `rag_master` Framework** — Domain-agnostic orchestrator (like LangChain/LangGraph) configurable for any domain
- **5 Research Tasks** — Easy → Medium → Hard difficulty with deterministic graders
- **Multi-Agent System** — Retriever, Reasoner, Critic, Planner, Verifier agents
- **Process Supervision** — Per-step rewards, not just final outcomes
- **Anti-Reward-Hacking** — Repetition detection, degenerate output penalties, copy-paste detection
- **16 Curated Documents** — Real aerospace research covering propulsion, materials, orbital mechanics, life support, hypersonics, and more
- **Royal Glassmorphism UI** — Black/gold/white theme with real-time feedback

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- Local LLM (Ollama with `qwen2.5:7b`) or API key (GROQ / HuggingFace)

### Quick Start with `uv`

```bash
# Clone the repository
git clone https://github.com/williyam/agentic-rag-gym.git
cd agentic-rag-gym

# Create environment and install dependencies
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your API keys and model settings

# Start the server
python main.py
```

### Quick Start with pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
python main.py
```

### Docker

```bash
# Single command
docker compose up --build

# Or manually
docker build -t agentic-rag-gym .
docker run -p 7860:7860 --env-file .env agentic-rag-gym
```

The server starts at `http://localhost:7860`.

---

## Configuration

All configuration is via environment variables (`.env` file):

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `http://localhost:11434/v1` | LLM API endpoint |
| `MODEL_NAME` | `qwen2.5:7b` | Model identifier |
| `HF_TOKEN` | — | HuggingFace token |
| `GROQ_API_KEY` | — | GROQ API key |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `FAISS_INDEX_DIR` | `./data/faiss_indices` | FAISS storage path |
| `SERVER_PORT` | `7860` | Server port |

---

## API Usage

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

### List Tasks

```bash
curl -X GET http://localhost:7860/tasks
```

---

## Tasks

| Task | Difficulty | Max Steps | Description |
|---|---|---|---|
| `aero_easy_propulsion_comparison` | Easy | 12 | Compare ion vs nuclear thermal propulsion for Mars |
| `aero_easy_debris_mitigation` | Easy | 12 | Analyze space debris challenges and ADR technologies |
| `aero_medium_mars_edl` | Medium | 16 | Design Mars Entry/Descent/Landing architecture |
| `aero_medium_life_support` | Medium | 16 | Design integrated life support for Mars mission |
| `aero_hard_hypersonic_vehicle` | Hard | 20 | Design reusable hypersonic space access vehicle |

---

## Running Inference

```bash
# Ensure the server is running
python main.py &

# Set environment variables
export API_BASE_URL=http://localhost:11434/v1
export MODEL_NAME=qwen2.5:7b
export HF_TOKEN=your_token_here

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

## Using RAG Master for Any Domain

The `rag_master` framework is designed to be **domain-agnostic**. To create a new domain:

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
    def get_tasks(self) -> List[TaskDefinition]:
        return [...]  # Your tasks

    def get_documents(self) -> List[Document]:
        return [...]  # Your knowledge base

    def get_grader(self, task_id: str) -> BaseGrader:
        return YourGrader(task_id)  # Your grading logic

    def get_reward_function(self) -> BaseRewardFunction:
        return CompositeRewardFunction()  # Or custom

    def get_system_prompt(self) -> str:
        return "You are an expert in..."
```

### 3. Register in Server

Update `server/app.py` to use your domain config instead of `AerospaceDomainConfig`.

### 4. Choose Evaluation Method

- **Rule-based**: Use `KeywordCoverageGrader` with domain-specific keywords
- **LLM Judge**: Use `LLMJudgeRewardFunction` for nuanced evaluation
- **Hybrid**: Combine both with `CompositeRewardFunction`

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

![Training Curves](plots/training_curves.png)

### Baseline vs. GRPO-Trained

![Baseline vs Trained](plots/baseline_vs_trained.png)

### Score Distribution

![Score Distribution](plots/score_distribution.png)

### Run Training (Notebook)

The primary training interface is the Jupyter notebook:

```bash
jupyter notebook agentic-rag-for-aerospace-research.ipynb
```

### Run Training (Script)

For headless/CI environments:

```bash
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
| Hardware | Apple M1 Pro (MPS) |
| Training Time | ~116 min |

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

## Baseline Scores

Scored with `meta-llama/Llama-3.2-3B-Instruct` via HF Inference API, `TEMPERATURE=0.0`:

| Task | Difficulty | Score |
|---|---|---|
| Propulsion Comparison | Easy | 0.40 – 0.65 |
| Debris Mitigation | Easy | 0.35 – 0.60 |
| Mars EDL | Medium | 0.30 – 0.50 |
| Life Support | Medium | 0.25 – 0.45 |
| Hypersonic Vehicle | Hard | 0.15 – 0.35 |

*Run `python inference.py` to reproduce. Results saved to `baseline_results.json`.*

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
├── rag_master/              # Generic agentic RAG framework
├── server/                  # FastAPI + Gradio server
├── domains/aerospace/       # Aerospace research domain
├── domains/legal_research/  # Legal research domain (stub)
├── training/                # GRPO training package
├── tests/                   # Unit & integration tests (102+)
├── .github/workflows/       # CI pipeline
├── documents/               # Architecture & design docs
├── plots/                   # Training curves & evaluation plots
├── agentic-rag-for-aerospace-research.ipynb  # GRPO training notebook
├── train.py                 # Standalone training script
├── inference.py             # Baseline inference script
├── openenv.yaml             # OpenEnv specification
├── Dockerfile               # Container definition
├── docker-compose.yml       # Service orchestration
└── main.py                  # Entry point
```

---

## License

MIT

---

*Built for the Meta × OpenEnv × Hugging Face × PyTorch Hackathon*
