"""Tests for the orchestrator."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock

import pytest

from rag_master.adapters import BaseDomainConfig, BaseGrader, BaseLLMClient, BaseRewardFunction, BaseRetriever
from rag_master.models import (
    DifficultyLevel,
    Document,
    EpisodeState,
    RetrievalResult,
    StepRecord,
    TaskDefinition,
    Trajectory,
)
from rag_master.orchestrator import Orchestrator
from rag_master.rewards import _SCORE_MIN, clamp_score


class MockRetriever(BaseRetriever):
    """Mock retriever for testing."""

    def __init__(self) -> None:
        self._docs: List[Document] = []

    async def index_documents(self, documents: List[Document]) -> int:
        self._docs.extend(documents)
        return len(documents)

    async def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        return [
            RetrievalResult(
                document=Document(content="Test document about propulsion", source="test"),
                score=0.75,
                rank=0,
            )
        ]

    async def clear_index(self) -> None:
        self._docs.clear()


class MockLLM(BaseLLMClient):
    """Mock LLM client for testing."""

    async def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        return "Based on the analysis, ion propulsion offers higher specific impulse."

    async def generate_with_metadata(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        return {"content": "Test response", "total_tokens": 100, "model": "test"}


class MockRewardFn(BaseRewardFunction):
    """Mock reward function for testing."""

    async def compute_step_reward(self, state: EpisodeState, step: StepRecord) -> float:
        return clamp_score(0.5)

    async def compute_episode_reward(self, trajectory: Trajectory, state: EpisodeState) -> float:
        return clamp_score(0.7)

    def get_reward_bounds(self) -> Tuple[float, float]:
        return (0.01, 0.99)


class MockGrader(BaseGrader):
    """Mock grader for testing."""

    async def grade(self, state: EpisodeState, trajectory: Trajectory) -> float:
        return clamp_score(0.6)


class MockDomain(BaseDomainConfig):
    """Mock domain config for testing."""

    def get_tasks(self) -> List[TaskDefinition]:
        return [
            TaskDefinition(
                task_id="test_easy",
                name="Test Easy",
                description="An easy test task",
                difficulty=DifficultyLevel.EASY,
                max_steps=10,
            ),
            TaskDefinition(
                task_id="test_hard",
                name="Test Hard",
                description="A hard test task",
                difficulty=DifficultyLevel.HARD,
                max_steps=20,
            ),
        ]

    def get_documents(self) -> List[Document]:
        return [Document(content="Test doc", source="test")]

    def get_grader(self, task_id: str) -> BaseGrader:
        return MockGrader()

    def get_reward_function(self) -> BaseRewardFunction:
        return MockRewardFn()

    def get_system_prompt(self) -> str:
        return "You are a test assistant."


@pytest.fixture
def orchestrator() -> Orchestrator:
    return Orchestrator(
        domain_config=MockDomain(),
        retriever=MockRetriever(),
        llm_client=MockLLM(),
        reward_function=MockRewardFn(),
    )


class TestOrchestrator:
    """Test suite for the Orchestrator."""

    @pytest.mark.asyncio
    async def test_reset(self, orchestrator: Orchestrator) -> None:
        obs = await orchestrator.reset(task_id="test_easy")
        assert "task" in obs
        assert obs["task"]["id"] == "test_easy"
        assert obs["step"] == 0
        assert obs["done"] is False

    @pytest.mark.asyncio
    async def test_reset_specific_task(self, orchestrator: Orchestrator) -> None:
        obs = await orchestrator.reset(task_id="test_hard")
        assert obs["task"]["id"] == "test_hard"
        assert obs["task"]["difficulty"] == "hard"

    @pytest.mark.asyncio
    async def test_reset_invalid_task(self, orchestrator: Orchestrator) -> None:
        with pytest.raises(ValueError, match="not found"):
            await orchestrator.reset(task_id="nonexistent")

    @pytest.mark.asyncio
    async def test_step_without_reset(self, orchestrator: Orchestrator) -> None:
        with pytest.raises(RuntimeError, match="reset"):
            await orchestrator.step({"type": "retrieve"})

    @pytest.mark.asyncio
    async def test_step_retrieve(self, orchestrator: Orchestrator) -> None:
        await orchestrator.reset(task_id="test_easy")
        result = await orchestrator.step({"type": "retrieve", "query": "test query"})
        assert "observation" in result
        assert "reward" in result
        assert "done" in result
        assert 0.0 <= result["reward"] <= 1.0

    @pytest.mark.asyncio
    async def test_step_reason(self, orchestrator: Orchestrator) -> None:
        await orchestrator.reset(task_id="test_easy")
        await orchestrator.step({"type": "retrieve", "query": "test"})
        result = await orchestrator.step({"type": "reason"})
        assert result["observation"]["step"] == 2

    @pytest.mark.asyncio
    async def test_step_answer_terminates(self, orchestrator: Orchestrator) -> None:
        await orchestrator.reset(task_id="test_easy")
        await orchestrator.step({"type": "retrieve"})
        await orchestrator.step({"type": "reason"})
        # Answer on step 3 allows one more step (verify/critique)
        result = await orchestrator.step({"type": "answer", "answer": "Test answer"})
        assert result["done"] is False
        # A non-verify step after answer terminates
        result = await orchestrator.step({"type": "reason"})
        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_step_verify_after_answer(self, orchestrator: Orchestrator) -> None:
        """Verify step after answer should be allowed."""
        await orchestrator.reset(task_id="test_easy")
        await orchestrator.step({"type": "retrieve"})
        await orchestrator.step({"type": "reason"})
        await orchestrator.step({"type": "answer", "answer": "Test answer"})
        result = await orchestrator.step({"type": "verify"})
        assert result["done"] is False
        # Terminates 2 steps after answer
        result = await orchestrator.step({"type": "reason"})
        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_step_after_done(self, orchestrator: Orchestrator) -> None:
        await orchestrator.reset(task_id="test_easy")
        await orchestrator.step({"type": "retrieve"})
        await orchestrator.step({"type": "reason"})
        await orchestrator.step({"type": "answer", "answer": "Done"})
        # Step after answer (non-verify) triggers done
        await orchestrator.step({"type": "reason"})
        # Now episode is done, further steps should return done=True with 0 reward
        result = await orchestrator.step({"type": "retrieve"})
        assert result["done"] is True
        assert result["reward"] == 0.0

    @pytest.mark.asyncio
    async def test_state_before_reset(self, orchestrator: Orchestrator) -> None:
        state = orchestrator.state()
        assert state["initialized"] is False

    @pytest.mark.asyncio
    async def test_state_after_reset(self, orchestrator: Orchestrator) -> None:
        await orchestrator.reset(task_id="test_easy")
        state = orchestrator.state()
        assert state["initialized"] is True
        assert state["task_id"] == "test_easy"

    @pytest.mark.asyncio
    async def test_grade(self, orchestrator: Orchestrator) -> None:
        await orchestrator.reset(task_id="test_easy")
        await orchestrator.step({"type": "retrieve"})
        await orchestrator.step({"type": "reason"})
        await orchestrator.step({"type": "answer", "answer": "Test"})
        score = await orchestrator.grade()
        assert 0.01 <= score <= 0.99

    @pytest.mark.asyncio
    async def test_grade_before_reset(self, orchestrator: Orchestrator) -> None:
        score = await orchestrator.grade()
        assert score == 0.01

    @pytest.mark.asyncio
    async def test_max_steps_termination(self, orchestrator: Orchestrator) -> None:
        await orchestrator.reset(task_id="test_easy")
        done = False
        for i in range(15):
            if done:
                break
            result = await orchestrator.step({"type": "retrieve", "query": f"query {i}"})
            done = result["done"]
        assert done is True

    @pytest.mark.asyncio
    async def test_tasks_property(self, orchestrator: Orchestrator) -> None:
        tasks = orchestrator.tasks
        assert len(tasks) == 2
        assert tasks[0].task_id == "test_easy"

    @pytest.mark.asyncio
    async def test_full_episode_trajectory(self, orchestrator: Orchestrator) -> None:
        await orchestrator.reset(task_id="test_easy")
        await orchestrator.step({"type": "plan"})
        await orchestrator.step({"type": "retrieve", "query": "propulsion"})
        await orchestrator.step({"type": "reason"})
        await orchestrator.step({"type": "answer", "answer": "Final answer"})
        # One more step after answer to trigger done
        await orchestrator.step({"type": "reason"})

        traj = orchestrator.current_trajectory
        assert traj is not None
        assert traj.completed is True
        assert len(traj.steps) == 5
