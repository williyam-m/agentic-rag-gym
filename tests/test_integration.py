"""Integration test: full episode flow through all endpoints."""

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
                document=Document(content="Ion propulsion achieves specific impulse of 4190 seconds.", source="test"),
                score=0.85, rank=0,
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
        ]

    def get_documents(self) -> List[Document]:
        return [Document(content="Test doc about aerospace", source="test")]

    def get_grader(self, task_id: str) -> BaseGrader:
        return _MockGrader()

    def get_reward_function(self) -> BaseRewardFunction:
        return _MockRewardFn()

    def get_system_prompt(self) -> str:
        return "You are a test assistant."


@pytest.fixture(autouse=True)
def _inject_mock():
    orch = Orchestrator(
        domain_config=_MockDomain(),
        retriever=_MockRetriever(),
        llm_client=_MockLLM(),
        reward_function=_MockRewardFn(),
    )
    server_app_module._orchestrator = orch
    yield
    server_app_module._orchestrator = None


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestFullEpisodeIntegration:
    """Full episode flow: reset → step(plan) → step(retrieve) → step(reason) → step(answer) → grade."""

    @pytest.mark.asyncio
    async def test_full_episode_flow(self, client: AsyncClient) -> None:
        # 1. Reset
        resp = await client.post("/reset", json={"task_id": "test_easy"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["observation"]["task"]["id"] == "test_easy"
        assert data["done"] is False

        # 2. Plan step
        resp = await client.post("/step", json={"type": "plan"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["observation"]["step"] == 1
        assert data["done"] is False

        # 3. Retrieve step
        resp = await client.post("/step", json={"type": "retrieve", "query": "ion propulsion"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["observation"]["step"] == 2

        # 4. Reason step
        resp = await client.post("/step", json={"type": "reason"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["observation"]["step"] == 3

        # 5. Answer step
        resp = await client.post("/step", json={"type": "answer", "answer": "Ion propulsion is better for deep space."})
        assert resp.status_code == 200
        data = resp.json()
        assert data["observation"]["step"] == 4

        # 6. Verify step (should be allowed after answer)
        resp = await client.post("/step", json={"type": "verify"})
        assert resp.status_code == 200

        # 7. State check
        resp = await client.get("/state")
        assert resp.status_code == 200
        state = resp.json()
        assert state["initialized"] is True
        assert state["task_id"] == "test_easy"
        assert state["answer_length"] > 0

        # 8. Grade
        resp = await client.post("/grade", json={})
        assert resp.status_code == 200
        grade = resp.json()
        assert 0.0 <= grade["score"] <= 1.0

    @pytest.mark.asyncio
    async def test_reset_clears_previous_episode(self, client: AsyncClient) -> None:
        # Run first episode
        await client.post("/reset", json={"task_id": "test_easy"})
        await client.post("/step", json={"type": "retrieve", "query": "test"})

        # Reset again
        resp = await client.post("/reset", json={"task_id": "test_easy"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["observation"]["step"] == 0

    @pytest.mark.asyncio
    async def test_health_always_works(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
