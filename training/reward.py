"""
Reward functions for GRPO training.

Uses the real domain graders from the Agentic RAG Gym so the RL signal
matches the actual evaluation rubric — not a proxy.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List

from domains.aerospace.graders import GRADER_REGISTRY
from rag_master.models import EpisodeState, StepRecord, Trajectory
from rag_master.rewards import _SCORE_MIN


def _make_dummy_state(task_id: str, answer: str) -> EpisodeState:
    """Minimal EpisodeState for offline grading."""
    from domains.aerospace.config import AerospaceDomainConfig

    domain = AerospaceDomainConfig()
    tasks = {t.task_id: t for t in domain.get_tasks()}
    task = tasks.get(task_id)
    if task is None:
        raise ValueError(f"Unknown task_id: {task_id}")

    return EpisodeState(
        episode_id="grpo-eval",
        task=task,
        current_step=5,
        query_history=["query"],
        retrieved_docs=[],
        agent_messages=[],
        generated_answer=answer,
        intermediate_rewards=[0.5] * 5,
        done=True,
        info={},
    )


def _make_dummy_trajectory(task_id: str) -> Trajectory:
    """Minimal trajectory with a good action sequence for process scoring."""
    now = datetime.now(timezone.utc)
    steps = [
        StepRecord(step_index=0, action_type="plan", action_payload={},
                   observation_summary="planned", intermediate_reward=0.5,
                   reasoning_trace="Planning approach.", timestamp=now),
        StepRecord(step_index=1, action_type="retrieve", action_payload={},
                   observation_summary="retrieved", intermediate_reward=0.6,
                   reasoning_trace="Retrieving.", timestamp=now),
        StepRecord(step_index=2, action_type="reason", action_payload={},
                   observation_summary="reasoned", intermediate_reward=0.5,
                   reasoning_trace="Analyzing because data is relevant.", timestamp=now),
        StepRecord(step_index=3, action_type="answer", action_payload={},
                   observation_summary="answered", intermediate_reward=0.6,
                   reasoning_trace="Final answer.", timestamp=now),
        StepRecord(step_index=4, action_type="verify", action_payload={},
                   observation_summary="verified", intermediate_reward=0.5,
                   reasoning_trace="Verifying.", timestamp=now),
    ]
    return Trajectory(
        episode_id="grpo-eval", task_id=task_id, steps=steps,
        total_reward=0.0, final_score=0.0, completed=True, metadata={},
    )


def grade_answer_sync(task_id: str, answer: str) -> float:
    """Grade a single answer using the domain grader (synchronous)."""
    grader_cls = GRADER_REGISTRY.get(task_id)
    if grader_cls is None:
        return float(_SCORE_MIN)
    grader = grader_cls()
    state = _make_dummy_state(task_id, answer)
    trajectory = _make_dummy_trajectory(task_id)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, grader.grade(state, trajectory)).result()
    return asyncio.run(grader.grade(state, trajectory))
