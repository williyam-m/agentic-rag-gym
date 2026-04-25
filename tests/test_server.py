"""Tests for the FastAPI server endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest
from httpx import ASGITransport, AsyncClient

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
from rag_master.rewards import clamp_score

import server.app as server_app_module
from server.app import app


class _MockRetriever(BaseRetriever):
    async def index_documents(self, documents: List[Document]) -> int:
        return len(documents)

    async def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        return [
            RetrievalResult(
                document=Document(content="Test document about propulsion systems", source="test"),
                score=0.75, rank=0,
            )
        ]

    async def clear_index(self) -> None:
        pass


class _MockLLM(BaseLLMClient):
    async def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        return "Based on analysis, ion propulsion offers higher specific impulse."

    async def generate_with_metadata(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        return {"content": "Test response", "total_tokens": 50, "model": "test"}


class _MockRewardFn(BaseRewardFunction):
    async def compute_step_reward(self, state: EpisodeState, step: StepRecord) -> float:
        return clamp_score(0.5)

    async def compute_episode_reward(self, trajectory: Trajectory, state: EpisodeState) -> float:
        return clamp_score(0.7)

    def get_reward_bounds(self) -> Tuple[float, float]:
        return (0.01, 0.99)


class _MockGrader(BaseGrader):
    async def grade(self, state: EpisodeState, trajectory: Trajectory) -> float:
        return clamp_score(0.6)


class _MockDomain(BaseDomainConfig):
    def get_tasks(self) -> List[TaskDefinition]:
        return [
            TaskDefinition(task_id="test_easy", name="Test Easy", description="Easy test", difficulty=DifficultyLevel.EASY, max_steps=10),
            TaskDefinition(task_id="test_medium", name="Test Medium", description="Medium test", difficulty=DifficultyLevel.MEDIUM, max_steps=15),
            TaskDefinition(task_id="test_hard", name="Test Hard", description="Hard test", difficulty=DifficultyLevel.HARD, max_steps=20),
            TaskDefinition(task_id="test_expert1", name="Test Expert1", description="Expert test 1", difficulty=DifficultyLevel.EXPERT, max_steps=20),
            TaskDefinition(task_id="test_expert2", name="Test Expert2", description="Expert test 2", difficulty=DifficultyLevel.EXPERT, max_steps=20),
        ]

    def get_documents(self) -> List[Document]:
        return [Document(content="Test doc", source="test")]

    def get_grader(self, task_id: str) -> BaseGrader:
        return _MockGrader()

    def get_reward_function(self) -> BaseRewardFunction:
        return _MockRewardFn()

    def get_system_prompt(self) -> str:
        return "You are a test assistant."


@pytest.fixture(autouse=True)
def _inject_mock_orchestrator():
    """Inject a mock orchestrator into the server module for all tests."""
    orch = Orchestrator(
        domain_config=_MockDomain(),
        retriever=_MockRetriever(),
        llm_client=_MockLLM(),
        reward_function=_MockRewardFn(),
    )
    server_app_module._orchestrators = {"aerospace": orch}
    server_app_module._active_domain = "aerospace"
    yield
    server_app_module._orchestrators = {}
    server_app_module._active_domain = "aerospace"


@pytest.fixture
async def client():
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    @pytest.mark.asyncio
    async def test_health(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"


class TestTasksEndpoint:
    """Tests for the tasks listing endpoint."""

    @pytest.mark.asyncio
    async def test_list_tasks(self, client: AsyncClient) -> None:
        resp = await client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert "tasks" in data
        assert len(data["tasks"]) >= 5


class TestResetEndpoint:
    """Tests for the reset endpoint."""

    @pytest.mark.asyncio
    async def test_reset_default(self, client: AsyncClient) -> None:
        resp = await client.post("/reset", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data

    @pytest.mark.asyncio
    async def test_reset_specific_task(self, client: AsyncClient) -> None:
        resp = await client.post("/reset", json={"task_id": "test_easy"})
        assert resp.status_code == 200
        data = resp.json()
        obs = data["observation"]
        assert obs["task"]["id"] == "test_easy"

    @pytest.mark.asyncio
    async def test_reset_invalid_task(self, client: AsyncClient) -> None:
        resp = await client.post("/reset", json={"task_id": "nonexistent"})
        assert resp.status_code == 400


class TestStepEndpoint:
    """Tests for the step endpoint."""

    @pytest.mark.asyncio
    async def test_step_retrieve(self, client: AsyncClient) -> None:
        await client.post("/reset", json={})
        resp = await client.post("/step", json={"type": "retrieve", "query": "propulsion"})
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        assert "reward" in data

    @pytest.mark.asyncio
    async def test_step_after_reset(self, client: AsyncClient) -> None:
        resp = await client.post("/reset", json={})
        assert resp.status_code == 200


class TestStateEndpoint:
    """Tests for the state endpoint."""

    @pytest.mark.asyncio
    async def test_state(self, client: AsyncClient) -> None:
        await client.post("/reset", json={})
        resp = await client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data


class TestGradeEndpoint:
    """Tests for the grade endpoint."""

    @pytest.mark.asyncio
    async def test_grade(self, client: AsyncClient) -> None:
        await client.post("/reset", json={})
        await client.post("/step", json={"type": "retrieve", "query": "test"})
        resp = await client.post("/grade", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "score" in data
        assert 0.0 <= data["score"] <= 1.0


class TestServerModels:
    """Tests for server Pydantic models."""

    def test_action_model(self) -> None:
        from server.models import Action, ActionType
        action = Action(type=ActionType.RETRIEVE, query="test")
        assert action.type == ActionType.RETRIEVE

    def test_observation_model(self) -> None:
        from server.models import Observation, TaskObservation
        task = TaskObservation(id="t1", name="T1", description="D1", difficulty="easy", max_steps=10)
        obs = Observation(task=task, step=0)
        assert obs.done is False

    def test_health_status(self) -> None:
        from server.models import HealthStatus
        h = HealthStatus()
        assert h.status == "healthy"
