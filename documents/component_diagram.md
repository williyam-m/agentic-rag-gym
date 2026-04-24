# Component Diagram — Agentic RAG Gym

## Package Structure

```
agentic-rag-gym/
│
├── rag_master/                    # Core Framework (domain-agnostic)
│   ├── __init__.py               # Package metadata
│   ├── config.py                 # Pydantic Settings configuration
│   ├── models.py                 # Core domain models
│   ├── adapters.py               # Abstract base classes (Adapter pattern)
│   ├── orchestrator.py           # Central episode engine
│   ├── agents.py                 # Multi-agent implementations
│   ├── rewards.py                # Reward functions with anti-hacking
│   ├── retriever.py              # FAISS vector retriever
│   ├── llm_client.py             # OpenAI-compatible LLM client
│   ├── database.py               # SQLAlchemy persistence
│   └── logging_config.py         # Structured logging
│
├── server/                        # API & UI Layer
│   ├── __init__.py
│   ├── app.py                    # FastAPI application (OpenEnv endpoints)
│   ├── models.py                 # API Pydantic models
│   └── ui.py                     # Gradio UI (royal glassmorphism)
│
├── domains/                       # Domain Implementations
│   ├── __init__.py
│   └── aerospace/                # Aerospace Research Domain
│       ├── __init__.py
│       ├── config.py             # Domain config (implements BaseDomainConfig)
│       ├── knowledge_base.py     # 16 curated research documents
│       ├── tasks.py              # 5 tasks (easy → hard)
│       └── graders.py            # Deterministic graders
│
├── tests/                         # Test Suite
│   ├── test_config.py
│   ├── test_models.py
│   ├── test_rewards.py
│   ├── test_orchestrator.py
│   ├── test_graders.py
│   ├── test_server.py
│   └── test_domain.py
│
├── documents/                     # Documentation
│   ├── architecture.md
│   ├── component_diagram.md
│   ├── data_flow.md
│   └── reward_function.md
│
├── data/                          # Runtime Data
│   └── faiss_indices/            # FAISS index files
│
├── main.py                        # Entry point
├── inference.py                   # Baseline inference script
├── openenv.yaml                   # OpenEnv specification
├── Dockerfile                     # Container definition
├── docker-compose.yml             # Multi-service orchestration
├── pyproject.toml                 # Python project config
├── .env.example                   # Environment template
└── README.md                      # Project documentation
```

## Component Dependencies

```
┌─────────────┐
│   main.py   │──────┐
└──────┬──────┘      │
       │             │
       ▼             ▼
┌──────────┐  ┌───────────┐
│server/app│  │ server/ui │
└─────┬────┘  └───────────┘
      │
      ▼
┌──────────────────┐
│   Orchestrator   │
└─┬──┬──┬──┬──┬───┘
  │  │  │  │  │
  │  │  │  │  └──▶ RewardFunction
  │  │  │  │
  │  │  │  └─────▶ Agents (Retriever, Reasoner, Critic, Planner, Verifier)
  │  │  │
  │  │  └────────▶ LLMClient
  │  │
  │  └───────────▶ FAISSRetriever
  │
  └──────────────▶ DomainConfig
                      │
                      └──▶ Tasks, Documents, Graders
```

## Interface Contracts

### BaseDomainConfig → Orchestrator
- `get_tasks()` → `List[TaskDefinition]`
- `get_documents()` → `List[Document]`
- `get_grader(task_id)` → `BaseGrader`
- `get_reward_function()` → `BaseRewardFunction`
- `get_system_prompt()` → `str`

### BaseRetriever → Orchestrator
- `index_documents(docs)` → `int`
- `retrieve(query, top_k)` → `List[RetrievalResult]`
- `clear_index()` → `None`

### BaseLLMClient → Agents
- `generate(messages, temp, max_tokens)` → `str`
- `generate_with_metadata(...)` → `Dict`

### BaseRewardFunction → Orchestrator
- `compute_step_reward(state, step)` → `float`
- `compute_episode_reward(trajectory, state)` → `float`
- `get_reward_bounds()` → `Tuple[float, float]`

### BaseGrader → Orchestrator
- `grade(state, trajectory)` → `float` in [0.01, 0.99]
