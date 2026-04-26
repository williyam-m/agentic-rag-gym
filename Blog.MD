# Building Agentic RAG Gym: Teaching AI Agents to Research Like Experts

> **We didn't just build another RAG pipeline — we redesigned the foundation of how agentic RAG systems learn, decide, and improve.**

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

## Agentic RAG OS: From Research to Production

Beyond the RL gym, we built **Agentic RAG OS** — a full-stack Rewards-as-a-Service (RaaS) platform that packages the reward computation engine as an API:

- **Upload any data** → automatically embed and index with FAISS
- **Configure reward functions** → choose algorithm (GRPO/PPO/DPO/REINFORCE), set weights
- **Compute rewards via API** → integrate into any LLM training pipeline
- **Dashboard & monitoring** → track usage, storage, API keys

**Live at:** [https://rag-8000.zcodecorp.in/](https://rag-8000.zcodecorp.in/)

## The RAG Master Framework

We designed **RAG Master** from scratch as a domain-agnostic orchestrator — think of it as a specialized framework (like LangChain/LangGraph) built specifically for agentic RAG with RL:

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

## What We Learned

1. **Process supervision beats outcome supervision.** Per-step rewards teach better research strategies than just grading the final answer.
2. **Anti-reward-hacking is essential.** Without it, agents quickly learn to stuff keywords and repeat phrases to inflate scores.
3. **Real graders > proxy rewards.** Using actual domain-expert grading criteria (even if rule-based) produces more aligned behavior than learned reward models.
4. **Small models can learn research skills.** Even Qwen2.5-0.5B shows measurable improvement with GRPO fine-tuning on domain tasks.
5. **Domain agnosticism requires careful abstraction.** The adapter pattern works well for different knowledge domains while keeping the core RL loop domain-independent.

## Try It Yourself

- **Live Demo:** [Agentic RAG Gym on HF Spaces](https://huggingface.co/spaces/williyam/agentic-rag-gym)
- **Agentic RAG OS:** [Live Platform](https://rag-8000.zcodecorp.in/)
- **Fine-Tuned Model:** [Qwen2.5 GRPO LoRA Adapter](https://huggingface.co/williyam/agentic-rag-aerospace-grpo)
- **Source Code:** [GitHub Repository](https://github.com/williyam-m/agentic-rag-gym)
- **Training Notebook:** [Google Colab](https://colab.research.google.com/drive/14il2JQmy9-id_fSGpmYbssp-j975DSDo?usp=sharing)

---

*Built for the Meta × OpenEnv × Hugging Face × PyTorch Hackathon*
