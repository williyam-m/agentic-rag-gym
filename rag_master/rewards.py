"""Reward functions with process supervision and anti-hacking measures."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from rag_master.adapters import BaseLLMClient, BaseRewardFunction
from rag_master.logging_config import get_logger
from rag_master.models import EpisodeState, StepRecord, Trajectory

logger = get_logger(__name__)

_SCORE_MIN: float = 0.01
_SCORE_MAX: float = 0.99


def clamp_score(score: float) -> float:
    """Clamp score strictly within [_SCORE_MIN, _SCORE_MAX]."""
    return max(_SCORE_MIN, min(_SCORE_MAX, score))


class CompositeRewardFunction(BaseRewardFunction):
    """
    Multi-signal reward function with process supervision.

    Combines:
    - Retrieval quality reward (relevance of retrieved documents)
    - Reasoning quality reward (step-level verifier)
    - Answer quality reward (final output evaluation)
    - Efficiency penalty (excessive steps)
    - Anti-hacking guards (repetition detection, degenerate output)
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._llm = llm_client
        self._weights = weights or {
            "retrieval_relevance": 0.25,
            "reasoning_quality": 0.20,
            "answer_completeness": 0.30,
            "efficiency": 0.15,
            "anti_hack_penalty": 0.10,
        }

    def get_reward_bounds(self) -> Tuple[float, float]:
        """Return reward bounds within [0.01, 0.99]."""
        return (_SCORE_MIN, _SCORE_MAX)

    async def compute_step_reward(
        self,
        state: EpisodeState,
        step: StepRecord,
    ) -> float:
        """Compute per-step reward using process supervision signals."""
        reward = 0.0

        # Retrieval step reward
        if step.action_type == "retrieve":
            reward += self._retrieval_step_reward(state, step)

        # Reasoning step reward
        if step.action_type == "reason":
            reward += self._reasoning_step_reward(state, step)

        # Answer step reward
        if step.action_type == "answer":
            reward += self._answer_step_reward(state, step)

        # Efficiency: penalize late steps
        step_ratio = step.step_index / max(state.task.max_steps, 1)
        efficiency = max(0.0, 1.0 - step_ratio * 0.5)
        reward *= efficiency

        # Anti-hacking: detect repetitive actions
        penalty = self._detect_repetition(state, step)
        reward -= penalty

        return clamp_score(reward)

    async def compute_episode_reward(
        self,
        trajectory: Trajectory,
        state: EpisodeState,
    ) -> float:
        """Compute final episode reward with anti-hacking checks."""
        if not trajectory.steps:
            return _SCORE_MIN

        # Aggregate step rewards
        step_rewards = [s.intermediate_reward for s in trajectory.steps]
        avg_step_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0

        # Answer quality signal
        answer_score = self._evaluate_answer_quality(state)

        # Retrieval coverage
        retrieval_score = self._evaluate_retrieval_coverage(state)

        # Efficiency bonus/penalty
        steps_used = len(trajectory.steps)
        max_steps = state.task.max_steps
        efficiency = 1.0 - (steps_used / max_steps) * 0.3 if max_steps > 0 else 0.5

        # Anti-hacking: check for degenerate patterns
        hack_penalty = self._detect_hacking_patterns(trajectory, state)

        # Weighted combination
        w = self._weights
        score = (
            w["retrieval_relevance"] * retrieval_score
            + w["reasoning_quality"] * avg_step_reward
            + w["answer_completeness"] * answer_score
            + w["efficiency"] * efficiency
            - w["anti_hack_penalty"] * hack_penalty
        )

        return clamp_score(score)

    def _retrieval_step_reward(self, state: EpisodeState, step: StepRecord) -> float:
        """Reward for retrieval quality based on retrieved document scores."""
        if not state.retrieved_docs:
            return 0.1
        avg_score = sum(r.score for r in state.retrieved_docs) / len(state.retrieved_docs)
        return avg_score * 0.8

    def _reasoning_step_reward(self, state: EpisodeState, step: StepRecord) -> float:
        """Reward for reasoning quality based on trace analysis."""
        trace = step.reasoning_trace
        if not trace:
            return 0.1
        # Process supervision signals
        score = 0.3
        if len(trace) > 50:
            score += 0.2
        if any(kw in trace.lower() for kw in ["because", "therefore", "evidence", "based on"]):
            score += 0.2
        if any(kw in trace.lower() for kw in ["however", "but", "alternatively", "caveat"]):
            score += 0.1
        return min(score, 0.9)

    def _answer_step_reward(self, state: EpisodeState, step: StepRecord) -> float:
        """Reward for answer quality."""
        answer = state.generated_answer
        if not answer:
            return 0.05
        score = 0.3
        if len(answer) > 100:
            score += 0.15
        if len(answer) > 300:
            score += 0.1
        if state.retrieved_docs:
            ref_terms = set()
            for r in state.retrieved_docs[:3]:
                ref_terms.update(r.document.content.lower().split()[:20])
            answer_terms = set(answer.lower().split())
            overlap = len(ref_terms & answer_terms) / max(len(ref_terms), 1)
            score += overlap * 0.3
        return min(score, 0.9)

    def _evaluate_answer_quality(self, state: EpisodeState) -> float:
        """Evaluate final answer quality against task criteria."""
        answer = state.generated_answer
        if not answer:
            return 0.05

        score = 0.3
        # Length heuristic
        word_count = len(answer.split())
        if word_count >= 50:
            score += 0.15
        if word_count >= 150:
            score += 0.1

        # Grading rubric alignment
        rubric = state.task.grading_rubric
        for criterion, weight in rubric.items():
            if criterion.lower() in answer.lower():
                score += weight * 0.3

        return min(score, 0.95)

    def _evaluate_retrieval_coverage(self, state: EpisodeState) -> float:
        """Evaluate how well retrieved docs cover the task."""
        if not state.retrieved_docs:
            return 0.1
        avg_relevance = sum(r.score for r in state.retrieved_docs) / len(state.retrieved_docs)
        coverage = min(len(state.retrieved_docs) / 3.0, 1.0)
        return avg_relevance * 0.6 + coverage * 0.4

    def _detect_repetition(self, state: EpisodeState, step: StepRecord) -> float:
        """Detect and penalize repetitive actions."""
        if len(state.query_history) < 2:
            return 0.0
        last_queries = state.query_history[-3:]
        if len(set(last_queries)) == 1 and len(last_queries) >= 2:
            return 0.3
        return 0.0

    def _detect_hacking_patterns(self, trajectory: Trajectory, state: EpisodeState) -> float:
        """Detect reward hacking patterns."""
        penalty = 0.0

        # Pattern 1: All steps are the same action
        action_types = [s.action_type for s in trajectory.steps]
        if len(set(action_types)) == 1 and len(action_types) > 3:
            penalty += 0.3

        # Pattern 2: Answer is suspiciously short or copy-pasted from query
        answer = state.generated_answer
        if answer and state.query_history:
            for q in state.query_history:
                if answer.strip() == q.strip():
                    penalty += 0.4

        # Pattern 3: Excessive steps without progress
        if len(trajectory.steps) > state.task.max_steps * 0.8:
            no_progress_count = sum(
                1 for s in trajectory.steps[-5:] if s.intermediate_reward < 0.1
            )
            if no_progress_count >= 4:
                penalty += 0.2

        # Pattern 4: Degenerate repeated content
        if answer:
            words = answer.split()
            if len(words) > 10:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.3:
                    penalty += 0.3

        return min(penalty, 0.8)


class LLMJudgeRewardFunction(BaseRewardFunction):
    """Reward function that uses an LLM as a judge for evaluation."""

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self._llm = llm_client

    def get_reward_bounds(self) -> Tuple[float, float]:
        return (_SCORE_MIN, _SCORE_MAX)

    async def compute_step_reward(
        self,
        state: EpisodeState,
        step: StepRecord,
    ) -> float:
        """Use LLM to evaluate step quality."""
        prompt = (
            f"Evaluate this reasoning step on a scale of 0.0 to 1.0.\n"
            f"Task: {state.task.description}\n"
            f"Step {step.step_index}: {step.action_type}\n"
            f"Reasoning: {step.reasoning_trace}\n"
            f"Respond with ONLY a decimal number between 0.0 and 1.0."
        )
        try:
            response = await self._llm.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            score = float(response.strip())
            return clamp_score(score)
        except (ValueError, Exception):
            return 0.5

    async def compute_episode_reward(
        self,
        trajectory: Trajectory,
        state: EpisodeState,
    ) -> float:
        """Use LLM to evaluate overall episode quality."""
        steps_summary = "\n".join(
            f"Step {s.step_index}: [{s.action_type}] {s.reasoning_trace[:100]}"
            for s in trajectory.steps[-10:]
        )
        prompt = (
            f"Evaluate this agent's performance on a scale of 0.0 to 1.0.\n"
            f"Task: {state.task.description}\n"
            f"Steps taken:\n{steps_summary}\n"
            f"Final answer: {state.generated_answer[:500]}\n"
            f"Criteria: accuracy, completeness, efficiency, reasoning quality.\n"
            f"Respond with ONLY a decimal number between 0.0 and 1.0."
        )
        try:
            response = await self._llm.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            score = float(response.strip())
            return clamp_score(score)
        except (ValueError, Exception):
            return 0.5
