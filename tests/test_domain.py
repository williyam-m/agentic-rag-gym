"""Tests for the aerospace domain configuration and knowledge base."""

from __future__ import annotations

import pytest

from domains.aerospace.config import AerospaceDomainConfig
from domains.aerospace.knowledge_base import get_aerospace_documents
from domains.aerospace.tasks import get_aerospace_tasks


class TestAerospaceDomainConfig:
    """Tests for the aerospace domain configuration."""

    @pytest.fixture
    def config(self) -> AerospaceDomainConfig:
        return AerospaceDomainConfig()

    def test_get_tasks(self, config: AerospaceDomainConfig) -> None:
        tasks = config.get_tasks()
        assert len(tasks) >= 5
        difficulties = {t.difficulty.value for t in tasks}
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties

    def test_get_documents(self, config: AerospaceDomainConfig) -> None:
        docs = config.get_documents()
        assert len(docs) >= 10
        for doc in docs:
            assert doc.content
            assert doc.source

    def test_get_grader(self, config: AerospaceDomainConfig) -> None:
        tasks = config.get_tasks()
        for task in tasks:
            grader = config.get_grader(task.task_id)
            assert grader is not None

    def test_get_reward_function(self, config: AerospaceDomainConfig) -> None:
        reward_fn = config.get_reward_function()
        assert reward_fn is not None
        bounds = reward_fn.get_reward_bounds()
        assert bounds[0] == 0.01
        assert bounds[1] == 0.99

    def test_get_system_prompt(self, config: AerospaceDomainConfig) -> None:
        prompt = config.get_system_prompt()
        assert "aerospace" in prompt.lower()
        assert len(prompt) > 50


class TestAerospaceKnowledgeBase:
    """Tests for the knowledge base."""

    def test_documents_have_metadata(self) -> None:
        docs = get_aerospace_documents()
        for doc in docs:
            assert "topic" in doc.metadata
            assert doc.metadata["topic"]

    def test_document_topics_diverse(self) -> None:
        docs = get_aerospace_documents()
        topics = {doc.metadata.get("topic") for doc in docs}
        assert len(topics) >= 6

    def test_document_content_not_empty(self) -> None:
        docs = get_aerospace_documents()
        for doc in docs:
            assert len(doc.content) > 100


class TestAerospaceTasks:
    """Tests for the tasks."""

    def test_task_difficulty_range(self) -> None:
        tasks = get_aerospace_tasks()
        difficulties = [t.difficulty.value for t in tasks]
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties

    def test_task_ids_unique(self) -> None:
        tasks = get_aerospace_tasks()
        ids = [t.task_id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_task_max_steps_vary(self) -> None:
        tasks = get_aerospace_tasks()
        max_steps = {t.max_steps for t in tasks}
        assert len(max_steps) >= 2

    def test_task_rubric_sums(self) -> None:
        tasks = get_aerospace_tasks()
        for task in tasks:
            if task.grading_rubric:
                total = sum(task.grading_rubric.values())
                assert 0.99 <= total <= 1.01, f"Task {task.task_id} rubric sums to {total}"
