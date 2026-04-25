# Agentic RAG OS — Architecture & Design Document

## Overview

**Agentic RAG OS** is a production-grade, API-first, full-stack web application that serves as an operating system for Agentic RAG workflows. It provides:

1. **RAG Pipeline** — Multi-tenant document ingestion, FAISS indexing, and semantic retrieval
2. **Rewards-as-a-Service (RaaS)** — Dynamic reward function generation for RL fine-tuning (GRPO, PPO, DPO)
3. **GitHub OAuth** — One-click sign-in and user management
4. **Tiered Access** — Free / Pro / Enterprise tiers with storage limits

Built on top of **RAG Master**, the domain-agnostic orchestration framework from `agentic-rag-gym`.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Browser                           │
│  Landing Page (/index.html) │ Dashboard (/dashboard.html)       │
│  Vanilla JS + CSS animations │ Alpine.js-style JS interactions  │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP/REST
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FastAPI Application (:8080)                   │
│  agentic_rag_os/app.py — independent from gym server (:7860)   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    API Router (v1)                        │   │
│  │  /auth     GitHub OAuth 2.0 + JWT issuance               │   │
│  │  /users    User profile & storage info                    │   │
│  │  /projects RAG knowledge base CRUD + document upload     │   │
│  │  /rewards  Reward config CRUD + batch compute + export   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────────────────────┐   │
│  │  Auth Service     │    │  RAG Service                     │   │
│  │  - OAuth flow     │    │  - FAISSRetriever (rag_master)   │   │
│  │  - JWT create/    │    │  - Text chunking + embedding     │   │
│  │    verify         │    │  - Per-project indices           │   │
│  └──────────────────┘    └──────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Rewards Service                                          │   │
│  │  - 7 built-in rule types (keyword, format, citations…)   │   │
│  │  - GRPO / PPO / DPO reward normalization                 │   │
│  │  - Batch compute (up to 500 pairs/call)                  │   │
│  │  - Export: JSON, CSV, HuggingFace dataset                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Database (SQLAlchemy async)                              │   │
│  │  SQLite (dev) | PostgreSQL (prod)                        │   │
│  │  Tables: users, projects, project_documents,             │   │
│  │          reward_configs, reward_computations             │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
              │ reuses                        │ stores
              ▼                              ▼
┌────────────────────┐         ┌─────────────────────────┐
│   RAG Master       │         │  FAISS Indices           │
│   (rag_master/)    │         │  data/faiss_indices/     │
│   FAISSRetriever   │         │  os_projects/u{id}/p{id} │
│   Document model   │         └─────────────────────────┘
│   Adapters         │
└────────────────────┘
```

---

## Directory Structure

```
agentic_rag_os/
├── __init__.py           # Package metadata
├── app.py                # FastAPI app factory
├── config.py             # Pydantic settings (RAGAS_ env prefix)
├── database.py           # SQLAlchemy async engine + session
├── run.py                # uvicorn entrypoint
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container definition (port 8080)
├── docker-compose.yml    # Local dev orchestration
├── cloudbuild.yaml       # Google Cloud Build + Cloud Run deployment
├── .env.example          # Environment variable template
│
├── models/               # SQLAlchemy ORM models
│   ├── user.py           # User (github_id, tier, ...)
│   ├── project.py        # Project + ProjectDocument
│   └── reward_config.py  # RewardConfig + RewardComputation
│
├── api/                  # FastAPI route handlers
│   ├── deps.py           # JWT auth dependency
│   ├── auth.py           # GitHub OAuth + /me
│   ├── users.py          # User profile + storage
│   ├── projects.py       # RAG projects CRUD + query
│   └── rewards.py        # RaaS CRUD + compute + export
│
├── services/             # Business logic
│   ├── auth_service.py   # OAuth exchange, JWT create/decode
│   ├── rag_service.py    # File ingest, chunking, FAISS retrieval
│   └── rewards_service.py# Reward rule evaluation engine
│
├── frontend/             # Static frontend assets
│   ├── templates/
│   │   ├── index.html    # Landing page (animated dark UI)
│   │   ├── dashboard.html# SPA dashboard
│   │   └── auth_callback.html # OAuth redirect handler
│   └── static/
│       ├── css/styles.css
│       └── js/
│           ├── landing.js  # Particles, typewriter, counters
│           └── dashboard.js# Dashboard SPA logic
│
└── docs/                 # This documentation
    ├── architecture.md   # (this file)
    ├── api_reference.md  # API endpoint reference
    └── deployment.md     # GCP deployment guide
```

---

## Data Flow

### RAG Pipeline

```
User uploads file
       │
       ▼
File validation (type, size limit)
       │
       ▼
Text extraction & sanitization
       │
       ▼
Chunking (1000 chars, 100 overlap)
       │
       ▼
Sentence-transformer embedding (all-MiniLM-L6-v2)
       │
       ▼
FAISS IndexFlatIP (inner product, normalized)
       │   stored at: data/faiss_indices/os_projects/u{uid}/p{pid}/
       ▼
User queries → semantic search → ranked results
```

### Rewards-as-a-Service

```
User defines RewardConfig
  name, algorithm (GRPO|PPO|DPO), rules [{name, type, weight, ...}]
       │
       ▼
POST /rewards/{id}/compute  { inputs: [{prompt, response}] }
       │
       ▼
For each input pair:
  For each rule:
    _apply_rule(rule, prompt, response) → [0, 1] score
    weighted_score += score * weight
  base_reward = weighted_score / total_weight
  final_reward = _normalize(base_reward, algorithm)
       │
       ▼
RewardComputation stored in DB
       │
       ▼
GET /rewards/{id}/computations/{cid}/export?format=json|csv|hf_dataset
```

---

## Security

- **Authentication**: GitHub OAuth 2.0 → JWT (HS256, 7-day expiry)
- **State validation**: Cryptographically random OAuth state (10-min expiry)
- **Input sanitization**: HTML tags and control characters stripped on all user inputs
- **File validation**: Extension allowlist, size limits enforced per tier
- **SQL injection**: SQLAlchemy ORM with parameterized queries only
- **CORS**: Configurable allowed origins
- **No secrets in code**: All secrets via environment variables

---

## Reward Rule Types

| Type | Description |
|------|-------------|
| `keyword_match` | Reward proportion of keywords present in response |
| `length_range` | Reward responses within min/max character bounds |
| `format_check` | Reward specific formats: JSON, markdown, numbered list |
| `no_hallucination` | Penalize forbidden phrases / hallucination markers |
| `citation_present` | Reward presence of citation patterns |
| `reasoning_steps` | Reward step-by-step reasoning structure |
| `custom_regex` | Reward custom regex pattern match |

---

## Storage Tiers

| Tier | Projects | Storage | Batch/day |
|------|----------|---------|-----------|
| Free | 3 | 5 MB | 100 |
| Pro | Unlimited | 500 MB | 10,000 |
| Enterprise | Unlimited | Unlimited | Unlimited |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + uvicorn (async) |
| ORM | SQLAlchemy 2.0 (async) |
| Database | SQLite (dev), PostgreSQL (prod) |
| Auth | GitHub OAuth 2.0 + JWT (python-jose) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | FAISS (faiss-cpu) |
| Frontend | HTML5 + CSS3 + Vanilla JS |
| Containerization | Docker |
| Cloud | Google Cloud Run + Container Registry |
| CI/CD | Google Cloud Build |
