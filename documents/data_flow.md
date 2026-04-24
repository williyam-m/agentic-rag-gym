# Data Flow — Agentic RAG Gym

## Episode Lifecycle

```
Agent                    Environment                    Backend
  │                          │                            │
  │──── POST /reset ────────▶│                            │
  │                          │── Select Task              │
  │                          │── Initialize State          │
  │                          │── Clear Trajectory          │
  │◀──── Observation ────────│                            │
  │                          │                            │
  │──── POST /step ─────────▶│                            │
  │    {type: "retrieve"}    │── Dispatch to Handler      │
  │                          │── Multi-Agent Processing ──▶│── FAISS Search
  │                          │◀── Retrieval Results ──────│
  │                          │── Compute Step Reward       │
  │                          │── Record to Trajectory      │
  │◀──── (obs, reward) ─────│                            │
  │                          │                            │
  │──── POST /step ─────────▶│                            │
  │    {type: "reason"}      │── Dispatch to Reasoner     │
  │                          │── Agent Reasoning ─────────▶│── LLM API Call
  │                          │◀── Reasoning Output ───────│
  │                          │── Compute Step Reward       │
  │◀──── (obs, reward) ─────│                            │
  │                          │                            │
  │──── POST /step ─────────▶│                            │
  │    {type: "answer"}      │── Dispatch to Answer       │
  │                          │── Check Termination         │
  │                          │── Compute Episode Reward    │
  │◀──── (obs, reward, done)│                            │
  │                          │                            │
  │──── POST /grade ────────▶│                            │
  │                          │── Domain Grader Evaluation  │
  │◀──── Score [0.01-0.99] ─│                            │
```

## Multi-Agent Message Flow

```
┌──────────┐     query      ┌──────────┐   retrieval    ┌──────────┐
│ Planner  │────────────────▶│ Retriever│───results─────▶│ Reasoner │
│  Agent   │                 │  Agent   │               │  Agent   │
└──────────┘                 └──────────┘               └────┬─────┘
                                  ▲                          │
                                  │                     reasoning
                            refinement                   output
                              query                         │
                                  │                    ┌────▼─────┐
                             ┌────┴─────┐              │  Critic  │
                             │  Critic  │◀─────────────│  Agent   │
                             │  (loop)  │              └──────────┘
                             └──────────┘
                                                       ┌──────────┐
                                                       │ Verifier │
                                                       │  Agent   │
                                                       └──────────┘
```

## Reward Computation Pipeline

```
Step Record
    │
    ├──▶ Retrieval Quality Signal ──────────┐
    │    (doc relevance scores)              │
    │                                        │
    ├──▶ Reasoning Quality Signal ──────────┤
    │    (trace analysis: evidence, logic)   │     ┌──────────────────┐
    │                                        ├────▶│ Weighted          │
    ├──▶ Answer Quality Signal ─────────────┤     │ Combination       │──▶ Step Reward
    │    (length, grounding, coverage)       │     └──────────────────┘
    │                                        │            │
    ├──▶ Efficiency Signal ─────────────────┤            │
    │    (step_ratio penalty)                │            ▼
    │                                        │     Anti-Hacking
    └──▶ Anti-Hacking Penalty ──────────────┘     Verification
         (repetition, degenerate output)          │
                                                  ▼
                                            Clamped [0.01, 0.99]
```

## Database Persistence Flow

```
Episode Complete
    │
    ├──▶ Save EpisodeRecord (episode_id, task_id, score, answer)
    │
    └──▶ Save StepLogs (per-step action, reward, reasoning trace)
            │
            └──▶ Available for:
                 • Training data extraction
                 • Performance analytics
                 • Self-improvement curriculum
```
