# Building Agentic RAG Gym: Teaching AI Agents to Research Like Experts

> **What happens when you stop treating RAG as a pipeline and start treating it as a learning system? You get agents that don't just retrieve — they investigate, critique, and refine until the answer is right.**

## The Story Behind This Project

It started with a frustration every AI engineer has felt: watching a RAG system retrieve perfectly relevant documents, only to generate a shallow, incomplete answer that ignores half the evidence. The retriever did its job. The generator didn't know what to do with it.

The real problem isn't retrieval — it's the *process*. Traditional RAG has no awareness of its own quality. It can't tell when it needs more information, when its reasoning has gaps, or when it should stop and verify its work. It's a pipeline, not a researcher.

We asked: **What if we could build an environment where AI agents learn to research the way experts do?** Not just retrieve-and-generate, but plan their investigation, critique their own reasoning, verify claims against evidence, and iterate until the answer is genuinely comprehensive.

That question became **Agentic RAG Gym** — and eventually led us to build an entire ecosystem around it.

## The Problem

Traditional Retrieval-Augmented Generation (RAG) systems are static pipelines: retrieve → generate → done. They have no feedback loop, no process awareness, no ability to learn from mistakes. The agent doesn't know when it has retrieved enough, when its reasoning is flawed, or when its answer is incomplete.

What if we could teach the agent to *research* — not just retrieve?

## Our Approach: An RL Gym for Agentic RAG

**Agentic RAG Gym** is an open-source reinforcement learning framework that fundamentally transforms RAG. We designed a new orchestrator from scratch — **RAG Master** — where autonomous agents don't just retrieve and generate; they *learn to research like experts* through RL-driven process supervision, multi-agent collaboration, and adversarial self-improvement.

### What Makes This Different

| Capability | Traditional RAG | Agentic RAG Gym |
|---|---|---|
| Pipeline | Static (retrieve → generate) | Dynamic (plan → retrieve → reason → critique → verify → answer) |
| Feedback | None | Per-step reward signals |
| Agents | Single | 5 cooperating agents (Retriever, Reasoner, Critic, Planner, Verifier) |
| Learning | No training loop | GRPO fine-tuning with real domain graders |
| Robustness | Vulnerable to junk | Anti-reward-hacking guards |

## The HF Space: A Live Research Lab

The **[Agentic RAG Gym HF Space](https://huggingface.co/spaces/williyam/agentic-rag-gym)** is more than a demo — it's a live research lab where you can watch AI agents think in real-time:

- **Interactive Mode** — Take the wheel and guide the agent step-by-step. Choose when to retrieve, reason, critique, or answer. See exactly how each action affects the reward signal.
- **Auto Pilot** — Sit back and watch the agent research autonomously. It plans its approach, retrieves evidence, reasons over documents, critiques its own work, and delivers a final answer — all while you see the reward breakdown at every step.
- **Multi-Domain** — Switch between Aerospace Research (scramjet propulsion, Mars EDL, hypersonic vehicles) and Legal Research (IP disputes, M&A due diligence, privacy compliance) with one click.
- **Real-Time Reward Visualization** — Every step shows the composite reward decomposition: retrieval relevance, reasoning quality, answer completeness, efficiency, and anti-hacking penalties.

The Space runs as a Docker container on HF infrastructure, exposing a full OpenEnv-compliant API. It uses the **[GRPO fine-tuned Qwen2.5 model](https://huggingface.co/williyam/agentic-rag-aerospace-grpo)** as its default LLM for aerospace tasks.

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

## The Five Agents

Our multi-agent system uses five specialized agents that collaborate through structured message-passing:

1. **Retriever** — Searches the FAISS vector store with learned query strategies
2. **Reasoner** — Analyzes retrieved documents, draws connections, builds arguments
3. **Critic** — Evaluates reasoning quality, identifies gaps and logical flaws
4. **Planner** — Creates execution strategies for multi-step research tasks
5. **Verifier** — Checks answer grounding against retrieved evidence

Each agent receives per-step reward signals that teach it *how* to do its job better.

## Why We Built RAG Master From Scratch

We looked at existing orchestration frameworks — LangChain, LangGraph, LlamaIndex — and found they weren't designed for what we needed. They're great for building RAG pipelines, but they lack:

- **RL-native reward computation** — No built-in support for per-step rewards that can drive policy optimization
- **Process-aware grading** — They evaluate final outputs, not the research *process*
- **Anti-reward-hacking** — No adversarial guards against the degenerate strategies agents discover during RL training
- **Domain-agnostic adapter pattern** — Adding a new knowledge domain shouldn't require rewriting the orchestrator

So we built **RAG Master** — a domain-agnostic orchestrator designed from the ground up for agentic RAG with reinforcement learning. Think of it as a specialized framework (like LangChain/LangGraph) but purpose-built for RL-driven research agents:

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
- **Composite rewards with bounded scores [0.01, 0.99]** — Prevents degenerate training signals
- **Deterministic graders** — Real domain-expert grading criteria, not learned reward models
- **Pluggable LLM backend** — Works with any model via a unified client interface
- **FAISS vector store** — Per-domain embedding indices with automatic document ingestion

## Reward Design: The Core Innovation

The reward function is where the magic happens. We decompose the reward into five components:

| Component | Weight | What It Measures |
|---|---|---|
| **Retrieval Relevance** | 25% | Are the retrieved documents relevant to the query? |
| **Reasoning Quality** | 20% | Does the reasoning show evidence of logical analysis? |
| **Answer Completeness** | 30% | Does the answer cover all required aspects? |
| **Efficiency** | 15% | Did the agent solve it in a reasonable number of steps? |
| **Anti-Hacking** | 10% | Penalty for detected reward gaming attempts |

All scores are strictly bounded within **[0.01, 0.99]** to prevent degenerate training signals.

### Anti-Reward-Hacking

We built adversarial guards that detect and penalize:
- **Repetition** — Copy-pasting the same sentences or phrases
- **Keyword stuffing** — Loading answers with domain keywords without coherent reasoning
- **Degenerate outputs** — Empty, trivially short, or nonsensical content
- **Query manipulation** — Submitting queries designed to game relevance scores

## GRPO Fine-Tuning: Real Graders as Reward Signals

We fine-tuned **Qwen2.5-0.5B-Instruct** using **Group Relative Policy Optimization (GRPO)** from the TRL library. The key innovation: we use **real domain graders** (not proxy rewards) as the training signal.

### Training Configuration

| Parameter | Value |
|---|---|
| Base Model | Qwen/Qwen2.5-0.5B-Instruct |
| Method | GRPO (Group Relative Policy Optimization) |
| LoRA | r=16, α=32, targets=q/k/v/o_proj |
| Learning Rate | 5e-6 |
| Epochs | 2 |
| Group Size | 4 |
| Max Completion | 512 tokens |

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

The GRPO-trained model shows consistent improvement on easy and hard tasks, with the largest gains on Debris Mitigation (+0.056) and Propulsion Comparison (+0.053). Medium-difficulty tasks remained stable, suggesting that the model learned domain-specific retrieval and reasoning strategies rather than shallow pattern matching.

## Two Domains: Aerospace & Legal

We built the framework to be **domain-agnostic**, and proved it with two very different domains:

### Aerospace Research
- **5 tasks** ranging from easy (propulsion comparison) to hard (hypersonic vehicle design)
- **16 documents** covering propulsion, debris mitigation, Mars EDL, life support systems, and hypersonic aerothermodynamics
- **Deterministic graders** with keyword coverage, technical depth assessment, and cross-reference validation

### Legal Research
- **5 tasks** from contract review to cross-border dispute resolution
- **16 documents** spanning contract law, privacy regulations (GDPR/CCPA), IP law, M&A due diligence, and international dispute resolution
- **Legal-specific graders** evaluating citation accuracy, jurisdictional awareness, and regulatory framework coverage

The key insight: if your agent can design a hypersonic vehicle by cross-referencing scramjet propulsion with UHTC materials, *or* navigate a cross-border IP dispute across three jurisdictions, it can handle anything.

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

1. **Process supervision beats outcome supervision.** Per-step rewards teach better research strategies than just grading the final answer.
2. **Anti-reward-hacking is essential.** Without it, agents quickly learn to stuff keywords and repeat phrases to inflate scores.
3. **Real graders > proxy rewards.** Using actual domain-expert grading criteria (even if rule-based) produces more aligned behavior than learned reward models.
4. **Small models can learn research skills.** Even Qwen2.5-0.5B shows measurable improvement with GRPO fine-tuning on domain tasks.
5. **Domain agnosticism requires careful abstraction.** The adapter pattern works well for different knowledge domains while keeping the core RL loop domain-independent.

## Try It Yourself

- **Live Demo:** [Agentic RAG Gym on HF Spaces](https://huggingface.co/spaces/williyam/agentic-rag-gym)
- **YouTube Demo:** [Watch It in Action](https://www.youtube.com/watch?v=M65DHY8za6M)
- **Fine-Tuned Model:** [Qwen2.5 GRPO LoRA Adapter](https://huggingface.co/williyam/agentic-rag-aerospace-grpo)
- **Source Code:** [GitHub Repository](https://github.com/williyam-m/agentic-rag-gym)
- **Training Notebook:** [Google Colab](https://colab.research.google.com/drive/14il2JQmy9-id_fSGpmYbssp-j975DSDo?usp=sharing)
- **Agentic RAG OS:** [Live Platform](https://rag-8000.zcodecorp.in/)

---

*Built for the Meta × OpenEnv × Hugging Face × PyTorch Hackathon*
