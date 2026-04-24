# Architecture Document — Agentic RAG Gym

## High-Level System Design

Agentic RAG Gym is a **reinforcement learning environment** built on the OpenEnv specification that trains AI agents on complex **Retrieval-Augmented Generation** tasks. The system uses a **multi-agent architecture** with cooperative agents, process-aware reward functions, and anti-reward-hacking mechanisms.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Gradio UI (Royal Theme)                  │
│  Interactive Mode │ Auto Pilot │ Tasks │ About               │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────┐
│                    FastAPI Server                             │
│  /reset  /step  /state  /grade  /health  /tasks              │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    RAG Master Orchestrator                    │
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Retriever│ │ Reasoner │ │  Critic  │ │ Verifier │       │
│  │  Agent   │ │  Agent   │ │  Agent   │ │  Agent   │       │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │
│       │             │             │             │             │
│  ┌────▼─────────────▼─────────────▼─────────────▼───────┐   │
│  │              Multi-Agent Message Bus                   │   │
│  └───────────────────────┬──────────────────────────────┘   │
│                          │                                   │
│  ┌───────────────────────▼──────────────────────────────┐   │
│  │            Composite Reward Function                   │   │
│  │  • Retrieval Relevance  • Reasoning Quality           │   │
│  │  • Answer Completeness  • Efficiency                  │   │
│  │  • Anti-Hacking Guards                                │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
│  FAISS Vector │ │ OpenAI-compat │ │    MySQL      │
│    Store      │ │   LLM API     │ │   Database    │
│  (Retrieval)  │ │ (Ollama/GROQ) │ │ (Persistence) │
└───────────────┘ └───────────────┘ └───────────────┘
```

## Component Architecture

### 1. RAG Master Framework (`rag_master/`)

The core orchestration layer, designed to be **domain-agnostic** and reusable:

- **`config.py`** — Pydantic Settings for env-based configuration
- **`models.py`** — Core domain models (Document, Trajectory, StepRecord, etc.)
- **`adapters.py`** — Abstract base classes (Adapter pattern) for all pluggable components
- **`orchestrator.py`** — Central engine managing episode lifecycle and agent coordination
- **`agents.py`** — Multi-agent implementations (Retriever, Reasoner, Critic, Planner, Verifier)
- **`rewards.py`** — Composite reward functions with process supervision
- **`retriever.py`** — FAISS-based vector retrieval
- **`llm_client.py`** — OpenAI-compatible LLM client
- **`database.py`** — SQLAlchemy database adapter

### 2. Server Layer (`server/`)

- **`app.py`** — FastAPI application with OpenEnv endpoints
- **`models.py`** — API request/response Pydantic models
- **`ui.py`** — Gradio UI with royal glassmorphism theme

### 3. Domain Layer (`domains/`)

Pluggable domain implementations:

- **`aerospace/`** — Aerospace Research domain with:
  - `knowledge_base.py` — 16 curated research documents
  - `tasks.py` — 5 tasks across easy/medium/hard difficulty
  - `graders.py` — Deterministic keyword-coverage graders with process evaluation
  - `config.py` — Domain configuration implementing `BaseDomainConfig`

## Design Patterns

| Pattern | Usage |
|---|---|
| **Adapter** | All external dependencies behind abstract interfaces |
| **Strategy** | Pluggable reward functions, graders, retrievers |
| **Observer** | Agent message passing via pub/sub-style messages |
| **Factory** | Domain config creates graders and reward functions |
| **Composite** | Reward function combines multiple signals |

## Data Flow

1. **Reset** → Orchestrator selects task → Initializes episode state → Returns observation
2. **Step** → Agent selects action → Orchestrator dispatches to handler → Multi-agent processing → Reward computation → State update → Returns (observation, reward, done, info)
3. **Grade** → Domain grader evaluates trajectory → Returns clamped score [0.01, 0.99]

## Security Measures

- All user input sanitized via `bleach`
- API keys stored as `SecretStr` in Pydantic
- No secrets in code — all from environment variables
- CORS middleware configured
- Input validation on all endpoints
