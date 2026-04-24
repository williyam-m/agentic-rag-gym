"""Tests for core domain models."""

from __future__ import annotations

import pytest

from rag_master.models import (
    AgentMessage,
    AgentRole,
    DifficultyLevel,
    Document,
    EpisodeState,
    RetrievalResult,
    StepRecord,
    TaskDefinition,
    Trajectory,
)


class TestDocument:
    """Test suite for Document model."""

    def test_create_document(self) -> None:
        doc = Document(content="Test content", source="test")
        assert doc.content == "Test content"
        assert doc.source == "test"
        assert doc.doc_id is not None
        assert doc.created_at is not None

    def test_document_with_metadata(self) -> None:
        doc = Document(content="x", metadata={"topic": "propulsion"})
        assert doc.metadata["topic"] == "propulsion"

    def test_document_default_embedding(self) -> None:
        doc = Document(content="x")
        assert doc.embedding is None


class TestRetrievalResult:
    """Test suite for RetrievalResult model."""

    def test_create_result(self) -> None:
        doc = Document(content="test")
        result = RetrievalResult(document=doc, score=0.85, rank=0)
        assert result.score == 0.85
        assert result.rank == 0

    def test_score_bounds(self) -> None:
        doc = Document(content="test")
        with pytest.raises(Exception):
            RetrievalResult(document=doc, score=1.5, rank=0)
        with pytest.raises(Exception):
            RetrievalResult(document=doc, score=-0.1, rank=0)


class TestAgentMessage:
    """Test suite for AgentMessage model."""

    def test_create_message(self) -> None:
        msg = AgentMessage(
            sender=AgentRole.RETRIEVER,
            receiver=AgentRole.REASONER,
            content="Found 3 documents",
        )
        assert msg.sender == AgentRole.RETRIEVER
        assert msg.receiver == AgentRole.REASONER

    def test_message_types(self) -> None:
        msg = AgentMessage(
            sender=AgentRole.CRITIC,
            content="Needs improvement",
            message_type="critique",
        )
        assert msg.message_type == "critique"


class TestStepRecord:
    """Test suite for StepRecord model."""

    def test_create_step(self) -> None:
        step = StepRecord(step_index=1, action_type="retrieve")
        assert step.step_index == 1
        assert step.intermediate_reward == 0.0

    def test_step_with_trace(self) -> None:
        step = StepRecord(
            step_index=2,
            action_type="reason",
            reasoning_trace="Based on the evidence...",
        )
        assert "evidence" in step.reasoning_trace


class TestTrajectory:
    """Test suite for Trajectory model."""

    def test_create_empty_trajectory(self) -> None:
        traj = Trajectory(task_id="test_task")
        assert len(traj.steps) == 0
        assert traj.completed is False

    def test_trajectory_with_steps(self) -> None:
        steps = [
            StepRecord(step_index=1, action_type="retrieve", intermediate_reward=0.3),
            StepRecord(step_index=2, action_type="reason", intermediate_reward=0.5),
        ]
        traj = Trajectory(task_id="test", steps=steps, total_reward=0.8)
        assert len(traj.steps) == 2
        assert traj.total_reward == 0.8


class TestTaskDefinition:
    """Test suite for TaskDefinition model."""

    def test_create_task(self) -> None:
        task = TaskDefinition(
            task_id="test_easy",
            name="Test Task",
            description="A test task",
            difficulty=DifficultyLevel.EASY,
        )
        assert task.difficulty == DifficultyLevel.EASY
        assert task.max_steps == 20

    def test_task_with_rubric(self) -> None:
        task = TaskDefinition(
            task_id="test",
            name="Test",
            description="Test",
            difficulty=DifficultyLevel.HARD,
            grading_rubric={"accuracy": 0.5, "completeness": 0.5},
        )
        assert sum(task.grading_rubric.values()) == 1.0


class TestEpisodeState:
    """Test suite for EpisodeState model."""

    def test_create_state(self) -> None:
        task = TaskDefinition(
            task_id="t1", name="T1", description="D1", difficulty=DifficultyLevel.MEDIUM
        )
        state = EpisodeState(task=task)
        assert state.current_step == 0
        assert state.done is False

    def test_state_history_tracking(self) -> None:
        task = TaskDefinition(
            task_id="t1", name="T1", description="D1", difficulty=DifficultyLevel.EASY
        )
        state = EpisodeState(task=task)
        state.query_history.append("ion propulsion")
        state.intermediate_rewards.append(0.4)
        assert len(state.query_history) == 1
        assert len(state.intermediate_rewards) == 1


class TestDifficultyLevel:
    """Test suite for DifficultyLevel enum."""

    def test_all_levels(self) -> None:
        assert DifficultyLevel.EASY == "easy"
        assert DifficultyLevel.MEDIUM == "medium"
        assert DifficultyLevel.HARD == "hard"
        assert DifficultyLevel.EXPERT == "expert"


class TestAgentRole:
    """Test suite for AgentRole enum."""

    def test_all_roles(self) -> None:
        assert AgentRole.RETRIEVER == "retriever"
        assert AgentRole.REASONER == "reasoner"
        assert AgentRole.CRITIC == "critic"
        assert AgentRole.PLANNER == "planner"
        assert AgentRole.EXECUTOR == "executor"
        assert AgentRole.VERIFIER == "verifier"
