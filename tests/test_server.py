"""Tests for the FastAPI server endpoints."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from server.app import app


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
        resp = await client.post("/reset", json={"task_id": "aero_easy_propulsion_comparison"})
        assert resp.status_code == 200
        data = resp.json()
        obs = data["observation"]
        assert obs["task"]["id"] == "aero_easy_propulsion_comparison"

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
    async def test_step_without_reset(self, client: AsyncClient) -> None:
        # After server restart, state is None — reset first
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
