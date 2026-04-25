# Agentic RAG OS — Architecture & Design Document

## 1. Overview

**Agentic RAG OS** is an RL B2B Rewards-as-a-Service (RaaS) platform — the operating system for agentic Retrieval-Augmented Generation. It provides a complete pipeline from user data ingestion → embedding → retrieval → reward signal generation, enabling teams to build, evaluate, and fine-tune LLMs using production-grade reward functions.

### 1.1 Mission Statement

Democratize access to RL-driven RAG improvement by providing infrastructure-as-a-service for reward signal generation, eliminating the need for teams to build custom evaluation pipelines from scratch.

### 1.2 Core Value Proposition

| Capability | Description |
|---|---|
| **Data → Embeddings** | Upload text documents, auto-embed with sentence-transformers, store in FAISS |
| **Rewards-as-a-Service** | Generate reward signals for GRPO, PPO, DPO, REINFORCE via API |
| **Multi-Algorithm Support** | Choose reward algorithm dynamically per request |
| **Anti-Reward Hacking** | Built-in guards against gaming, repetition, degenerate outputs |
| **API-First** | Every capability exposed as RESTful endpoint |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Layer                               │
│  ┌──────────────┐  ┌─────────────┐  ┌───────────────────┐   │
│  │  Web App SPA │  │  API Client  │  │  SDK / CLI        │   │
│  │  (Dark Theme) │  │  (REST/JWT)  │  │  (API Key Auth)   │   │
│  └──────┬───────┘  └──────┬──────┘  └────────┬──────────┘   │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                 FastAPI Application Layer                     │
│                                                              │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────────┐     │
│  │  Auth Routes  │  │ RAG Routes │  │  Reward Routes    │     │
│  │  /auth/*      │  │ /rag/*     │  │  /rewards/*       │     │
│  └──────┬───────┘  └─────┬──────┘  └────────┬─────────┘     │
│         │                │                   │               │
│  ┌──────▼───────┐  ┌─────▼──────┐  ┌────────▼─────────┐     │
│  │ User Routes  │  │  Deps      │  │  Health / Docs    │     │
│  │ /user/*      │  │  (Auth DI) │  │  /api/docs|redoc  │     │
│  └──────────────┘  └────────────┘  └──────────────────┘     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Service Layer                              │
│                                                              │
│  ┌──────────────────┐  ┌───────────────────────────────┐    │
│  │  Auth Service     │  │  RAG Service                   │    │
│  │  - JWT tokens     │  │  - Domain CRUD                 │    │
│  │  - Password hash  │  │  - Document upload/embed       │    │
│  │  - GitHub OAuth   │  │  - FAISS indexing              │    │
│  │  - API keys       │  │  - Semantic retrieval          │    │
│  └──────────────────┘  └───────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Reward Service                                         │  │
│  │  - Multi-algorithm compute (GRPO, PPO, DPO, REINFORCE) │  │
│  │  - Config management                                    │  │
│  │  - Job execution                                        │  │
│  │  - Anti-hack detection                                  │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Data / Storage Layer                        │
│                                                              │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │  SQLite DB    │  │  FAISS Indices  │  │  File Storage   │   │
│  │  - Users      │  │  - Per-domain   │  │  - Uploads      │   │
│  │  - API Keys   │  │  - MiniLM-L6    │  │  - 2MB free     │   │
│  │  - Domains    │  │  - 384-dim      │  │  - Premium      │   │
│  │  - Documents  │  │  - Inner product │  │    coming soon  │   │
│  │  - Configs    │  │                 │  │                 │   │
│  │  - Jobs       │  │                 │  │                 │   │
│  └──────────────┘  └────────────────┘  └────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   RAG Master Core                            │
│  (Shared library from agentic-rag-gym)                      │
│                                                              │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │  Retriever    │  │  Reward Funcs   │  │  Models         │   │
│  │  FAISSRetriever│ │  Composite      │  │  Document       │   │
│  │  SentenceTransf│ │  LLM Judge      │  │  Trajectory     │   │
│  └──────────────┘  └────────────────┘  └────────────────┘   │
│                                                              │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │  Agents       │  │  Orchestrator   │  │  Adapters       │   │
│  │  5 specialized│  │  Multi-agent    │  │  Base classes   │   │
│  └──────────────┘  └────────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Authentication Architecture

### 3.1 Dual Auth Strategy

```
┌──────────────────────────────────────────┐
│           Authentication Flow             │
│                                          │
│  Browser ─→ JWT Bearer Token             │
│    1. POST /auth/register (username/pw)  │
│    2. POST /auth/login (username/pw)     │
│    3. GET  /auth/github/callback (OAuth) │
│    ↓                                     │
│    JWT in Authorization: Bearer <token>  │
│                                          │
│  API Client ─→ API Key                   │
│    1. POST /user/api-keys (generate)     │
│    ↓                                     │
│    X-API-Key: ragos_<hex>                │
└──────────────────────────────────────────┘
```

### 3.2 Security Measures

- **Password Hashing**: PBKDF2-HMAC-SHA256, 310,000 iterations, random salt
- **JWT**: HS256 signing, 24h expiry, rotating secret keys
- **API Keys**: SHA-256 hashed storage, prefix-only display, revocable
- **GitHub OAuth**: Standard OAuth 2.0 code exchange flow

---

## 4. Data Flow

### 4.1 Document Ingestion Pipeline

```
User Upload (text file)
    │
    ▼
Storage Limit Check (2MB free / 100MB premium)
    │
    ▼
SQLite Record (id, content, metadata)
    │
    ▼
Sentence-Transformer Encoding (all-MiniLM-L6-v2, 384-dim)
    │
    ▼
FAISS IndexFlatIP (inner product similarity)
    │
    ▼
Domain-Isolated Index (per-user, per-domain)
```

### 4.2 RAG Query Pipeline

```
User Query (text)
    │
    ▼
Embedding (same model as indexing)
    │
    ▼
FAISS Similarity Search (top-k, cosine via normalized IP)
    │
    ▼
Score Normalization [0, 1]
    │
    ▼
Results + Metadata → Response
    │
    ▼
Query Logged (for analytics)
```

### 4.3 Reward Computation Pipeline

```
Input: {algorithm, query, answer, retrieved_docs, config}
    │
    ├──→ Retrieval Relevance (query-doc overlap)
    ├──→ Reasoning Quality (marker detection)
    ├──→ Answer Completeness (length + coverage)
    ├──→ Efficiency (length bounds)
    └──→ Anti-Hack Detection (degenerate check)
    │
    ▼
Weighted Sum → Clamp [0.01, 0.99]
    │
    ▼
Algorithm-Specific Adjustments
    │
    ▼
{total_reward, breakdown, algorithm, metadata}
```

---

## 5. API Design

### 5.1 Endpoint Map

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/auth/register` | No | Create account |
| POST | `/api/v1/auth/login` | No | Login, get JWT |
| GET | `/api/v1/auth/github/callback` | No | GitHub OAuth |
| GET | `/api/v1/user/me` | Yes | Current user profile |
| GET | `/api/v1/user/dashboard` | Yes | Dashboard statistics |
| POST | `/api/v1/user/api-keys` | Yes | Create API key |
| GET | `/api/v1/user/api-keys` | Yes | List API keys |
| DELETE | `/api/v1/user/api-keys/:id` | Yes | Revoke API key |
| POST | `/api/v1/rag/domains` | Yes | Create domain |
| GET | `/api/v1/rag/domains` | Yes | List domains |
| DELETE | `/api/v1/rag/domains/:id` | Yes | Delete domain |
| POST | `/api/v1/rag/domains/:id/documents` | Yes | Upload document |
| GET | `/api/v1/rag/domains/:id/documents` | Yes | List documents |
| DELETE | `/api/v1/rag/domains/:id/documents/:docId` | Yes | Delete document |
| POST | `/api/v1/rag/domains/:id/query` | Yes | RAG query |
| POST | `/api/v1/rewards/compute` | Yes | Compute reward |
| POST | `/api/v1/rewards/configs` | Yes | Create config |
| GET | `/api/v1/rewards/configs` | Yes | List configs |
| DELETE | `/api/v1/rewards/configs/:id` | Yes | Delete config |
| POST | `/api/v1/rewards/jobs` | Yes | Create job |
| GET | `/api/v1/rewards/jobs` | Yes | List jobs |
| GET | `/api/v1/rewards/algorithms` | No | List algorithms |
| GET | `/api/v1/health` | No | Health check |

### 5.2 Error Responses

All errors follow a consistent format:
```json
{
  "detail": "Human-readable error message"
}
```

Standard HTTP status codes: 400 (bad request), 401 (unauthorized), 404 (not found), 500 (server error).

---

## 6. Reward Algorithms

### 6.1 Supported Algorithms

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **GRPO** | Group Relative Policy Optimization | Stable RL training with relative rewards |
| **PPO** | Proximal Policy Optimization | Clipped surrogate for monotonic improvement |
| **DPO** | Direct Preference Optimization | Preference-based, no reward model needed |
| **REINFORCE** | Policy gradient with baseline | Simple gradient estimation |
| **Custom** | User-defined weights | Domain-specific scoring |

### 6.2 Reward Signal Components

| Component | Weight | Signal Source |
|-----------|--------|---------------|
| Retrieval Relevance | 0.25 | Query-document token overlap |
| Reasoning Quality | 0.20 | Reasoning marker detection |
| Answer Completeness | 0.30 | Length + query coverage |
| Efficiency | 0.15 | Answer length bounds |
| Anti-Hack Penalty | 0.10 | Degenerate content detection |

All weights are configurable per reward config.

---

## 7. Deployment Architecture

### 7.1 Google Cloud Run

```
┌─────────────────────────────────────┐
│          Google Cloud Run            │
│                                     │
│  ┌────────────────────────────────┐ │
│  │  Container: Agentic RAG OS     │ │
│  │                                │ │
│  │  FastAPI App (:8080)           │ │
│  │  ├── Auth Service              │ │
│  │  ├── RAG Service               │ │
│  │  ├── Reward Service            │ │
│  │  └── Static Frontend           │ │
│  │                                │ │
│  │  SQLite + FAISS (ephemeral)    │ │
│  └────────────────────────────────┘ │
│                                     │
│  Auto-scaling: 0-10 instances       │
│  Memory: 2Gi                        │
│  CPU: 2                             │
└─────────────────────────────────────┘
```

### 7.2 Infrastructure

| Component | Service | Details |
|-----------|---------|---------|
| Compute | Cloud Run | Serverless containers, auto-scaling |
| Container Registry | Artifact Registry | Docker image storage |
| Networking | Cloud Run default | HTTPS with managed TLS |
| Storage | Ephemeral (container FS) | SQLite + FAISS per instance |
| Monitoring | Cloud Logging | Automatic log collection |

---

## 8. Security Architecture

### 8.1 OWASP Compliance

| Risk | Mitigation |
|------|------------|
| Injection | Parameterized SQL queries (aiosqlite), input sanitization |
| Auth Failure | PBKDF2 password hashing, JWT with expiry, API key rotation |
| Data Exposure | Hashed passwords, hashed API keys, prefix-only display |
| SSRF | No user-controlled URL fetching (except GitHub OAuth) |
| Security Logging | All auth events logged |

### 8.2 Data Isolation

- Each user has isolated domains
- Each domain has isolated FAISS index
- SQL queries filter by `user_id` at every layer
- File uploads restricted to text content only

---

## 9. Frontend Architecture

### 9.1 SPA Design

- **Framework**: Vanilla JavaScript (zero dependencies)
- **Theme**: Dark mode with purple/cyan accent gradient
- **Routing**: Hash-based client-side navigation
- **State**: In-memory with localStorage for JWT persistence
- **Animations**: CSS keyframes + IntersectionObserver

### 9.2 Views

| View | Path | Auth Required |
|------|------|---------------|
| Landing Page | `/` | No |
| Dashboard | `/dashboard` | Yes |
| Domains | `/domains` | Yes |
| Rewards | `/rewards` | Yes |
| API Keys | `/apikeys` | Yes |

---

## 10. Integration with RAG Master

Agentic RAG OS reuses core components from the `rag_master` library:

| Component | Usage |
|-----------|-------|
| `FAISSRetriever` | Document indexing and retrieval |
| `Document` model | Document data structure |
| `CompositeRewardFunction` | Reference reward implementation |
| `SentenceTransformer` | Embedding generation |

The RAG OS operates independently of the Gym server — it does **not** require the gym to be running.

---

## 11. Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| API Framework | FastAPI |
| Database | SQLite (aiosqlite) |
| Vector Store | FAISS (faiss-cpu) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Auth | PyJWT + PBKDF2 |
| Frontend | HTML5 + CSS3 + Vanilla JS |
| Container | Docker |
| Cloud | Google Cloud Run |
| CI/CD | GitHub → Cloud Build → Cloud Run |
