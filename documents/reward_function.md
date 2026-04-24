# Reward Function & Training Loop — Agentic RAG Gym

## Reward Function Design

### Philosophy

The reward function follows three core principles:
1. **Process Supervision** — Reward intermediate steps, not just final outcomes
2. **Anti-Hacking** — Detect and penalize degenerate strategies
3. **Composite Signals** — Multiple evaluation dimensions prevent reward hacking

### Composite Reward Architecture

The `CompositeRewardFunction` combines five weighted signals:

| Signal | Weight | Description |
|---|---|---|
| Retrieval Relevance | 0.25 | Average relevance score of retrieved documents |
| Reasoning Quality | 0.20 | Depth and logic of reasoning traces |
| Answer Completeness | 0.30 | Coverage of task requirements in final answer |
| Efficiency | 0.15 | Penalty for excessive steps |
| Anti-Hack Penalty | 0.10 | Deductions for detected hacking patterns |

### Step-Level Rewards (Process Supervision)

Each action type receives a specialized evaluation:

#### Retrieval Steps
- Base reward proportional to average document relevance scores
- Bonus for retrieving diverse sources
- No reward if no documents found

#### Reasoning Steps
- Trace length (>50 chars baseline)
- Evidence markers: "because", "therefore", "based on"
- Critical thinking markers: "however", "alternatively", "caveat"
- Scaled by step position (earlier = higher efficiency multiplier)

#### Answer Steps
- Length heuristic (word count thresholds)
- Source grounding: overlap between answer terms and retrieved document terms
- Rubric alignment: presence of task-specific criteria keywords

### Anti-Reward-Hacking Measures

Four detection mechanisms:

1. **Repetition Detection** — If last 3 queries are identical → 0.3 penalty
2. **Monotonic Action Exploit** — If all steps use same action type → 0.3 penalty
3. **Copy-Paste Detection** — If answer equals a previous query → 0.4 penalty
4. **Degenerate Output** — If unique word ratio < 0.3 → 0.3 penalty

### Score Clamping

All scores are strictly clamped to `[0.01, 0.99]` to ensure:
- No task returns exactly 0.0 (which would indicate a broken grader)
- No task returns exactly 1.0 (which would indicate a trivial task)

### LLM Judge Mode

An alternative `LLMJudgeRewardFunction` uses the LLM itself as an evaluator:
- Per-step: Prompts the LLM to rate step quality 0.0-1.0
- Per-episode: Prompts the LLM to rate overall performance

This is used when rule-based evaluation is insufficient for the domain.

## Training Loop

### Episode Structure

```
reset(task_id)
    │
    for step in range(max_steps):
    │   │
    │   ├── Agent observes state
    │   ├── Agent selects action (retrieve/reason/answer/plan/critique/verify)
    │   ├── Environment processes action
    │   ├── Step reward computed (process supervision)
    │   ├── State updated
    │   │
    │   └── if done: break
    │
    ├── Episode reward computed
    ├── Grader evaluates final performance
    └── Trajectory saved for training
```

### Self-Improvement Loop

The environment supports self-improvement through:

1. **Adversarial Critique** — The Critic agent identifies weaknesses in reasoning
2. **Iterative Refinement** — Retriever can be redirected based on critique
3. **Verification Gate** — Verifier checks answer grounding before submission
4. **Curriculum Difficulty** — Tasks range easy → hard, challenging frontier models

### Trajectory Analysis

Each trajectory records:
- Per-step actions and reasoning traces
- Per-step intermediate rewards
- Final score and episode metadata
- Agent communication history

This data enables:
- Offline RL training on collected trajectories
- Process reward model training
- Failure mode analysis
- Self-improvement curriculum generation

## Grading System

### Deterministic Graders

Each task has a `KeywordCoverageGrader` that:
1. Checks keyword presence across rubric categories
2. Weights coverage by category importance
3. Adds process quality bonus (20% weight)
4. Clamps final score to [0.01, 0.99]

### Process Quality Evaluation

The process component (20% of final grade) rewards:
- Diverse action types (3+ types → bonus)
- Proper sequencing (retrieve before answer)
- Reasoning steps present
- Not using excessive steps
