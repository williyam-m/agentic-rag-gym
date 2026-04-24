"""Tests for reward functions."""

from __future__ import annotations

import pytest

from rag_master.models import (
    DifficultyLevel,
    Document,
    EpisodeState,
    RetrievalResult,
    StepRecord,
    TaskDefinition,
    Trajectory,
)
from rag_master.rewards import (
    CompositeRewardFunction,
    _SCORE_MAX,
    _SCORE_MIN,
    clamp_score,
)


class TestClampScore:
    """Test suite for score clamping."""

    def test_clamp_within_bounds(self) -> None:
        assert clamp_score(0.5) == 0.5

    def test_clamp_below_min(self) -> None:
        assert clamp_score(-1.0) == _SCORE_MIN
        assert clamp_score(0.0) == _SCORE_MIN

    def test_clamp_above_max(self) -> None:
        assert clamp_score(1.5) == _SCORE_MAX
        assert clamp_score(1.0) == _SCORE_MAX

    def test_clamp_at_bounds(self) -> None:
        assert clamp_score(_SCORE_MIN) == _SCORE_MIN
        assert clamp_score(_SCORE_MAX) == _SCORE_MAX

    def test_bounds_values(self) -> None:
        assert _SCORE_MIN == 0.01
        assert _SCORE_MAX == 0.99


def _make_task() -> TaskDefinition:
    return TaskDefinition(
        task_id="test", name="Test", description="Test task",
        difficulty=DifficultyLevel.EASY, max_steps=10,
    )


def _make_state(answer: str = "", docs: int = 0) -> EpisodeState:
    task = _make_task()
    state = EpisodeState(task=task, generated_answer=answer)
    for i in range(docs):
        state.retrieved_docs.append(
            RetrievalResult(
                document=Document(content=f"Document {i} about propulsion systems", source="test"),
                score=0.7,
                rank=i,
            )
        )
    return state


class TestCompositeRewardFunction:
    """Test suite for CompositeRewardFunction."""

    @pytest.fixture
    def reward_fn(self) -> CompositeRewardFunction:
        return CompositeRewardFunction()

    @pytest.mark.asyncio
    async def test_step_reward_retrieve(self, reward_fn: CompositeRewardFunction) -> None:
        state = _make_state(docs=3)
        step = StepRecord(step_index=1, action_type="retrieve")
        reward = await reward_fn.compute_step_reward(state, step)
        assert _SCORE_MIN <= reward <= _SCORE_MAX

    @pytest.mark.asyncio
    async def test_step_reward_reason(self, reward_fn: CompositeRewardFunction) -> None:
        state = _make_state()
        step = StepRecord(
            step_index=2,
            action_type="reason",
            reasoning_trace="Based on the evidence, ion propulsion offers higher specific impulse because of efficient ionization.",
        )
        reward = await reward_fn.compute_step_reward(state, step)
        assert _SCORE_MIN <= reward <= _SCORE_MAX

    @pytest.mark.asyncio
    async def test_step_reward_answer(self, reward_fn: CompositeRewardFunction) -> None:
        state = _make_state(answer="Ion propulsion systems use electrically charged atoms.", docs=2)
        step = StepRecord(step_index=3, action_type="answer")
        reward = await reward_fn.compute_step_reward(state, step)
        assert _SCORE_MIN <= reward <= _SCORE_MAX

    @pytest.mark.asyncio
    async def test_step_reward_unknown_action(self, reward_fn: CompositeRewardFunction) -> None:
        state = _make_state()
        step = StepRecord(step_index=1, action_type="unknown")
        reward = await reward_fn.compute_step_reward(state, step)
        assert reward == _SCORE_MIN

    @pytest.mark.asyncio
    async def test_episode_reward_empty(self, reward_fn: CompositeRewardFunction) -> None:
        state = _make_state()
        traj = Trajectory(task_id="test")
        reward = await reward_fn.compute_episode_reward(traj, state)
        assert reward == _SCORE_MIN

    @pytest.mark.asyncio
    async def test_episode_reward_with_steps(self, reward_fn: CompositeRewardFunction) -> None:
        state = _make_state(
            answer="Comprehensive analysis of ion propulsion and nuclear thermal propulsion systems.",
            docs=3,
        )
        traj = Trajectory(
            task_id="test",
            steps=[
                StepRecord(step_index=1, action_type="retrieve", intermediate_reward=0.4),
                StepRecord(step_index=2, action_type="reason", intermediate_reward=0.5),
                StepRecord(step_index=3, action_type="answer", intermediate_reward=0.6),
            ],
        )
        reward = await reward_fn.compute_episode_reward(traj, state)
        assert _SCORE_MIN <= reward <= _SCORE_MAX

    @pytest.mark.asyncio
    async def test_repetition_penalty(self, reward_fn: CompositeRewardFunction) -> None:
        state = _make_state()
        state.query_history = ["same query", "same query", "same query"]
        step = StepRecord(step_index=4, action_type="retrieve")
        reward = await reward_fn.compute_step_reward(state, step)
        assert reward <= 0.5

    @pytest.mark.asyncio
    async def test_anti_hacking_same_actions(self, reward_fn: CompositeRewardFunction) -> None:
        state = _make_state(answer="test")
        traj = Trajectory(
            task_id="test",
            steps=[
                StepRecord(step_index=i, action_type="retrieve", intermediate_reward=0.1)
                for i in range(5)
            ],
        )
        reward = await reward_fn.compute_episode_reward(traj, state)
        assert reward < 0.6

    @pytest.mark.asyncio
    async def test_anti_hacking_copy_paste(self, reward_fn: CompositeRewardFunction) -> None:
        query = "What is ion propulsion?"
        state = _make_state(answer=query)
        state.query_history = [query]
        traj = Trajectory(
            task_id="test",
            steps=[StepRecord(step_index=1, action_type="answer", intermediate_reward=0.3)],
        )
        reward = await reward_fn.compute_episode_reward(traj, state)
        assert reward < 0.5

    @pytest.mark.asyncio
    async def test_anti_hacking_degenerate_output(self, reward_fn: CompositeRewardFunction) -> None:
        state = _make_state(answer="the the the the the the the the the the the the")
        traj = Trajectory(
            task_id="test",
            steps=[StepRecord(step_index=1, action_type="answer", intermediate_reward=0.3)],
        )
        reward = await reward_fn.compute_episode_reward(traj, state)
        assert reward < 0.6

    def test_reward_bounds(self, reward_fn: CompositeRewardFunction) -> None:
        bounds = reward_fn.get_reward_bounds()
        assert bounds == (_SCORE_MIN, _SCORE_MAX)

    @pytest.mark.asyncio
    async def test_efficiency_penalizes_late_steps(self, reward_fn: CompositeRewardFunction) -> None:
        state = _make_state(docs=2)
        early_step = StepRecord(step_index=1, action_type="retrieve")
        late_step = StepRecord(step_index=9, action_type="retrieve")
        early_reward = await reward_fn.compute_step_reward(state, early_step)
        state2 = _make_state(docs=2)
        state2.current_step = 9
        late_reward = await reward_fn.compute_step_reward(state2, late_step)
        assert early_reward >= late_reward
