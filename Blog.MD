# Building Agentic RAG Gym: Empowering the Entire RAG Stack with Reinforcement Learning

> **We are not optimizing a single use case — we are re-engineering the entire Retrieval-Augmented Generation technology stack from first principles, using reinforcement learning as the foundational training signal across every layer: retrieval, reasoning, critique, verification, and synthesis.**

## The Story Behind This Project

It started with a frustration every AI engineer has felt: watching a RAG system retrieve perfectly relevant documents, only to generate a shallow, incomplete answer that ignores half the evidence. The retriever did its job. The generator didn't know what to do with it.

The real problem isn't retrieval — it's the *process*. Traditional RAG has no awareness of its own quality. It can't tell when it needs more information, when its reasoning has gaps, or when it should stop and verify its work. It's a pipeline, not a researcher.

We asked: **What if we could build an environment where AI agents learn to research the way experts do?** Not just retrieve-and-generate, but plan their investigation, critique their own reasoning, verify claims against evidence, and iterate until the answer is genuinely comprehensive.

That question became **Agentic RAG Gym** — and eventually led us to build an entire ecosystem around it.

## The Problem: Why Static RAG Fails at Scale

Traditional Retrieval-Augmented Generation systems are fundamentally constrained by their architecture: a single-pass pipeline of retrieve → generate → done. This design has no feedback loop, no process awareness, no mechanism for self-correction or iterative refinement. The agent has zero visibility into whether its retrieval was sufficient, whether its reasoning is coherent, or whether its answer actually addresses the question.

At scale, these limitations compound. When the knowledge base grows to thousands of documents spanning multiple technical domains, a single retrieval pass is almost never sufficient. The agent needs the capacity to **reformulate queries**, **triangulate across sources**, **identify contradictions**, and **synthesize multi-document evidence** — capabilities that require a fundamentally different architecture.

What if we could teach the agent to *research* — not just retrieve?

## Our Approach: An RL Gym for Agentic RAG

**Agentic RAG Gym** is an open-source reinforcement learning framework that empowers the complete RAG technology stack with RL-driven optimization. We designed a new orchestrator from scratch — **RAG Master** — where autonomous agents don't just retrieve and generate; they *learn to research like domain experts* through RL-driven process supervision, multi-agent collaboration, and adversarial self-improvement.

This is not a wrapper around existing tools. This is a **ground-up reimagining** of how RAG systems should work — one where every component (retrieval strategy, reasoning depth, critique quality, verification rigor) is individually trainable through reinforcement learning signals grounded in real domain-expert evaluation.

**🔗 Explore the project:**

| Resource | Link |
|---|---|
| **Live Demo** | [Agentic RAG Gym — HF Space](https://huggingface.co/spaces/williyam/agentic-rag-gym) |
| **YouTube Demo** | [Watch It in Action](https://www.youtube.com/watch?v=M65DHY8za6M) |
| **Fine-Tuned Model** | [Qwen2.5 GRPO LoRA Adapter](https://huggingface.co/williyam/agentic-rag-aerospace-grpo) |
| **Source Code** | [GitHub — agentic-rag-gym](https://github.com/williyam-m/agentic-rag-gym) |
| **Training Notebook** | [Google Colab — GRPO Fine-Tuning](https://colab.research.google.com/drive/14il2JQmy9-id_fSGpmYbssp-j975DSDo?usp=sharing) |
| **Agentic RAG OS** | [Live Platform](https://rag-8000.zcodecorp.in/) |

### What Makes This Different

| Capability | Traditional RAG | Agentic RAG Gym |
|---|---|---|
| Pipeline | Static single-pass (retrieve → generate) | Dynamic multi-turn (plan → retrieve → reason → critique → verify → answer) |
| Feedback | None — open-loop execution | Per-step composite reward signals with bounded scoring |
| Agents | Monolithic single-model pipeline | 5 cooperating specialized agents with structured message-passing |
| Learning | No training loop — frozen at deployment | GRPO fine-tuning with real domain-expert graders as reward signals |
| Robustness | Vulnerable to hallucination and junk retrieval | Adversarial anti-reward-hacking guards with degenerate output detection |
| Extensibility | Hardcoded to one domain | Domain-agnostic adapter pattern — plug any knowledge domain in 4 steps |

## The HF Space: A Live Research Lab

The **[Agentic RAG Gym HF Space](https://huggingface.co/spaces/williyam/agentic-rag-gym)** is more than a demo — it's a live research lab where you can watch AI agents think in real-time:

- **Interactive Mode** — Take the wheel and guide the agent step-by-step. Choose when to retrieve, reason, critique, or answer. See exactly how each action affects the reward signal.
- **Auto Pilot** — Sit back and watch the agent research autonomously. It plans its approach, retrieves evidence, reasons over documents, critiques its own work, and delivers a final answer — all while you see the reward breakdown at every step.
- **Multi-Domain** — Switch between Aerospace Research (scramjet propulsion, Mars EDL, hypersonic vehicles) and Legal Research (IP disputes, M&A due diligence, privacy compliance) with one click.
- **Real-Time Reward Visualization** — Every step shows the composite reward decomposition: retrieval relevance, reasoning quality, answer completeness, efficiency, and anti-hacking penalties.

The Space runs as a Docker container on HF infrastructure, exposing a full OpenEnv-compliant API. It uses the **[GRPO fine-tuned Qwen2.5 model](https://huggingface.co/williyam/agentic-rag-aerospace-grpo)** as its default LLM for aerospace tasks.

## Deep Architecture: How the System Works End-to-End

The architecture is a layered stack designed for composability, observability, and RL-native training:

```
┌─────────────────── Presentation Layer ──────────────────┐
│  Gradio UI (6 tabs)                                      │
│  Interactive │ Auto Pilot │ Tasks │ Blog │ README │ About│
│           [Domain Selector: Aerospace / Legal]           │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP/SSE
┌────────────────────────▼────────────────────────────────┐
│           API Gateway (FastAPI + Uvicorn)                 │
│  POST /reset │ /step │ /grade │ /domain/switch           │
│  GET  /state │ /tasks │ /domains │ /health               │
│  OpenEnv-compliant: reset() → step() → grade()          │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│         RAG Master Orchestrator (Core Engine)             │
│                                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ Planner  │→│ Retriever│→│ Reasoner │→│  Critic  │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
│       ↕            ↕            ↕            ↕          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ Verifier │ │FAISS VDB │ │LLM Client│ │  Reward  │   │
│  │          │ │per-domain│ │(any API) │ │ Computer │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
├──────────────────────────────────────────────────────────┤
│  Domain Adapters (pluggable via BaseDomainConfig)        │
│  [Aerospace] [Legal Research] [Your Domain Here]         │
└──────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              RL Training Loop (Offline)                   │
│  GRPO Policy Optimization ← Composite Reward Signal      │
│  LoRA Adapters (r=16, α=32) → q/k/v/o_proj targets      │
│  Real domain graders as reward (no proxy models)         │
└──────────────────────────────────────────────────────────┘
```

### Data Flow for a Single Episode

1. **Reset** — The orchestrator selects a task, clears state, and presents the agent with a task description and access to the domain's FAISS vector store
2. **Plan** — The Planner agent generates an execution strategy: which queries to run, what aspects to cover, how many retrieval rounds are needed
3. **Retrieve** — The Retriever agent queries the FAISS index using `sentence-transformers/all-MiniLM-L6-v2` embeddings. Top-k documents (with cosine similarity scores) are added to the agent's working memory
4. **Reason** — The Reasoner agent synthesizes across all retrieved documents, building structured arguments with evidence citations
5. **Critique** — The Critic agent evaluates the current reasoning: Are there gaps? Contradictions? Missing evidence? Unsupported claims?
6. **Verify** — The Verifier agent checks that every claim in the draft answer is grounded in retrieved evidence
7. **Answer** — The agent submits a final answer, and the composite reward function scores the entire trajectory
8. **Grade** — Domain-specific deterministic graders evaluate the final answer on keyword coverage, technical depth, cross-reference accuracy, and structural completeness

Each step produces a **per-step reward signal** that decomposes into five weighted components — this is what makes the system RL-trainable.

## The Five Agents: Specialized Collaboration

Our multi-agent system uses five specialized agents that collaborate through structured message-passing:

1. **Retriever** — Searches the FAISS vector store with learned query strategies. Evaluates document relevance and decides whether to retrieve more
2. **Reasoner** — Analyzes retrieved documents, draws cross-document connections, builds structured technical arguments with evidence chains
3. **Critic** — Adversarially evaluates reasoning quality. Identifies logical gaps, unsupported claims, missing perspectives, and contradictions
4. **Planner** — Creates execution strategies for multi-step research tasks. Allocates the step budget across retrieval, reasoning, and verification phases
5. **Verifier** — Performs factual grounding verification. Checks every claim against the retrieved evidence base, flags hallucinations and unsupported assertions

Each agent receives per-step reward signals that teach it *how* to do its job better — this is fundamentally different from outcome-only supervision.

## Why We Built RAG Master From Scratch

We evaluated existing orchestration frameworks — LangChain, LangGraph, LlamaIndex — and found they weren't designed for RL-native agentic research. They're excellent for building RAG pipelines, but they lack:

- **RL-native reward computation** — No built-in support for per-step rewards that can drive policy optimization via GRPO, PPO, or DPO
- **Process-aware grading** — They evaluate final outputs, not the research *process* (how many retrieval rounds, what reasoning strategies, which critique patterns)
- **Anti-reward-hacking** — No adversarial guards against the degenerate strategies agents inevitably discover during RL training (keyword stuffing, repetition loops, shallow pattern matching)
- **Domain-agnostic adapter pattern** — Adding a new knowledge domain shouldn't require rewriting the orchestrator core

So we built **RAG Master** — a domain-agnostic orchestrator designed from the ground up for agentic RAG with reinforcement learning. Think of it as a purpose-built framework (in the same category as LangChain/LangGraph) but architecturally optimized for RL-driven research agents:

```python
from rag_master.adapters import BaseDomainConfig

class YourDomainConfig(BaseDomainConfig):
    def get_tasks(self) -> List[TaskDefinition]: ...
    def get_documents(self) -> List[Document]: ...
    def get_grader(self, task_id: str) -> BaseGrader: ...
    def get_reward_function(self) -> BaseRewardFunction: ...
    def get_system_prompt(self) -> str: ...
```

Four steps to add any domain: create config, define tasks, build graders, register. The framework handles embedding, indexing, retrieval, reward computation, and agent orchestration.

The key architectural decisions:
- **Composite rewards with bounded scores [0.01, 0.99]** — Prevents degenerate training signals (vanishing/exploding gradients)
- **Deterministic graders** — Real domain-expert grading criteria, not learned reward models that can be gamed
- **Pluggable LLM backend** — Works with any OpenAI-compatible API: Ollama, vLLM, OpenAI, HuggingFace Inference, and more
- **FAISS vector store with per-domain indices** — Automatic document ingestion, embedding, and index management per domain

## Reward Design: The Core Innovation

The reward function is where the RL magic happens. We decompose the composite reward into five orthogonal components, each measuring a distinct aspect of research quality:

| Component | Weight | What It Measures |
|---|---|---|
| **Retrieval Relevance** | 25% | Cosine similarity between query embeddings and retrieved document embeddings. Penalizes irrelevant retrieval |
| **Reasoning Quality** | 20% | Presence of logical connectives, evidence citations, structured argumentation, and multi-document synthesis |
| **Answer Completeness** | 30% | Coverage of all required technical aspects as defined by the task rubric. Keyword coverage + structural depth |
| **Efficiency** | 15% | Step usage ratio — rewards solving tasks in fewer steps. Penalizes unnecessary retrieval/reasoning loops |
| **Anti-Hacking** | 10% | Deduction for detected reward gaming: repetition, keyword stuffing, degenerate outputs, copy-paste patterns |

All scores are strictly bounded within **[0.01, 0.99]** — this clamping is critical for stable GRPO training, preventing both vanishing rewards (all-zero trajectories) and exploding rewards (degenerate high-reward exploits).

### Anti-Reward-Hacking: Adversarial Robustness

During early RL experiments, we observed agents discovering creative exploits:

- **Repetition loops** — Repeating the same high-reward sentence 50 times to inflate scores
- **Keyword stuffing** — Cramming domain keywords into answers without coherent reasoning
- **Degenerate outputs** — Submitting empty strings, single-word answers, or nonsensical token sequences
- **Query manipulation** — Crafting retrieval queries designed to game relevance scores rather than find useful information

We built adversarial guards that detect and penalize all of these patterns. The anti-hacking component uses n-gram overlap analysis, entropy measurement, and structural coherence checks to identify and suppress gaming strategies.

## GRPO Fine-Tuning: Real Graders as Reward Signals

We fine-tuned **Qwen2.5-0.5B-Instruct** using **Group Relative Policy Optimization (GRPO)** from the TRL library. The critical innovation: we use **real domain-expert graders** (deterministic, rule-based) as the reward signal — not proxy reward models that can be gamed or hallucinate.

GRPO compares groups of completions for the same prompt and optimizes the policy to prefer higher-scoring completions relative to the group mean. This avoids the need for a separate critic/value network (unlike PPO) and produces stable training dynamics even with small models.

### Training Configuration

| Parameter | Value |
|---|---|
| Base Model | Qwen/Qwen2.5-0.5B-Instruct |
| Method | GRPO (Group Relative Policy Optimization) |
| LoRA | r=16, α=32, targets=q_proj, k_proj, v_proj, o_proj |
| Optimizer | AdamW (8-bit) |
| Learning Rate | 5e-6 with cosine annealing |
| Epochs | 2 |
| Group Size (G) | 4 completions per prompt |
| Max Completion | 512 tokens |
| Training Time | ~116 minutes on Colab T4 |

### Results

| Metric | Baseline | GRPO-Trained | Improvement |
|---|---|---|---|
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

The GRPO-trained model shows consistent improvement on easy and hard tasks (+5.3% to +5.6% on easy, +3.9% on hard), while medium-difficulty tasks remained stable. This pattern indicates the model learned genuine domain-specific retrieval and reasoning strategies rather than shallow surface-level pattern matching.

## Two Domains: Aerospace & Legal

We built the framework to be **domain-agnostic**, and proved it with two radically different technical domains:

### Aerospace Research
- **5 tasks** ranging from easy (propulsion comparison) to hard (hypersonic vehicle design)
- **16 documents** covering propulsion systems (ion, nuclear thermal, scramjet), debris mitigation (ADR technologies), Mars Entry/Descent/Landing, life support systems (ECLSS), and hypersonic aerothermodynamics (UHTC materials, thermal protection)
- **Deterministic graders** with keyword coverage matrices, technical depth assessment (multi-document cross-referencing), and structural completeness scoring

### Legal Research
- **5 tasks** from contract review to cross-border dispute resolution
- **16 documents** spanning contract law (liability, indemnification, IP assignment), privacy regulations (GDPR Article 6/17, CCPA/CPRA), patent law (35 USC §101/§103), M&A due diligence, and international dispute resolution (ICC, LCIA, SIAC arbitration)
- **Legal-specific graders** evaluating citation accuracy, jurisdictional awareness, regulatory framework coverage, and risk assessment completeness

The key insight: if your agent can design a hypersonic vehicle by cross-referencing scramjet propulsion with UHTC materials, *or* navigate a cross-border IP dispute across three jurisdictions referencing ICC arbitration rules, it can handle anything.

## From Research to Production: The Motivation Behind Agentic RAG OS

After building the RL gym, we realized something: **the reward computation engine we built is valuable on its own**. Every team fine-tuning LLMs with RL needs reward signals, but building domain-specific graders, anti-hacking guards, and composite reward functions from scratch is painful.

That realization led us to build **Agentic RAG OS** — a full-stack **Rewards-as-a-Service (RaaS)** platform that packages our reward computation engine as an API anyone can use:

- **Upload any data** → automatically embed and index with FAISS
- **Configure reward functions** → choose algorithm (GRPO/PPO/DPO/REINFORCE), set component weights
- **Compute rewards via API** → integrate into any LLM training pipeline with a single HTTP call
- **Dashboard & monitoring** → track usage, storage (1 GB per user), API keys, and reward distributions
- **Multi-domain support** — Upload documents for any domain and get calibrated rewards immediately

The idea is simple: if you're fine-tuning an LLM and need reward signals grounded in real documents, you shouldn't have to build the entire retrieval + grading + anti-hacking infrastructure yourself. Upload your data, configure your weights, and call the API.

### Agentic RAG OS — Live Demo

![Agentic RAG OS Demo](assets/demo/agentic-rag-os-demo.gif)

**Live at:** [https://rag-8000.zcodecorp.in/](https://rag-8000.zcodecorp.in/)

## What We Learned

1. **Process supervision beats outcome supervision.** Per-step rewards teach better research strategies than just grading the final answer. The agent learns *when* to retrieve more vs. *when* to stop and synthesize.
2. **Anti-reward-hacking is non-negotiable.** Without adversarial guards, agents discover keyword stuffing and repetition exploits within the first 20 training steps. Every RL-for-LLM system needs these defenses.
3. **Real graders > proxy rewards.** Using deterministic domain-expert grading criteria (even if rule-based) produces more aligned behavior than learned reward models. Proxy models can be gamed; deterministic graders cannot.
4. **Small models can learn research skills.** Even Qwen2.5-0.5B (500M parameters) shows measurable improvement with GRPO fine-tuning — you don't need 70B models to benefit from RL-driven RAG.
5. **Domain agnosticism requires careful abstraction.** The adapter pattern works well for different knowledge domains while keeping the core RL loop domain-independent. Four files is all it takes to add a new domain.

---

## References

1. Shao, Z., et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." *arXiv preprint arXiv:2402.03300* (2024). — GRPO algorithm foundation
2. Lewis, P., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*. — RAG foundational paper
3. Hu, E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*. — LoRA fine-tuning methodology
4. Schulman, J., et al. "Proximal Policy Optimization Algorithms." *arXiv preprint arXiv:1707.06347* (2017). — PPO baseline for RL policy optimization
5. Rafailov, R., et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." *NeurIPS 2023*. — DPO alternative alignment method
6. Johnson, J., et al. "Billion-scale similarity search with GPUs." *IEEE Transactions on Big Data* (2019). — FAISS vector search engine
7. Qwen Team. "Qwen2.5 Technical Report." *arXiv preprint arXiv:2412.15115* (2024). — Base model architecture

---

*Built for the Meta × OpenEnv × Hugging Face × PyTorch Hackathon*
