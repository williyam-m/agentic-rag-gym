"""Aerospace domain configuration - plugs into the RAG Master orchestrator."""

from __future__ import annotations

from typing import List

from rag_master.adapters import BaseDomainConfig, BaseGrader, BaseRewardFunction
from rag_master.models import Document, TaskDefinition
from rag_master.rewards import CompositeRewardFunction

from domains.aerospace.graders import get_grader
from domains.aerospace.knowledge_base import get_aerospace_documents
from domains.aerospace.tasks import get_aerospace_tasks


class AerospaceDomainConfig(BaseDomainConfig):
    """Domain configuration for aerospace research tasks."""

    SYSTEM_PROMPT = (
        "You are an expert aerospace research analyst with deep knowledge of "
        "propulsion systems, orbital mechanics, materials science, thermal protection, "
        "life support systems, hypersonic flight, and autonomous spacecraft operations. "
        "Analyze the provided research documents carefully and synthesize comprehensive, "
        "technically accurate responses. Always cite specific data points from the source "
        "documents. Structure your analysis with clear sections and quantitative evidence."
    )

    def get_tasks(self) -> List[TaskDefinition]:
        """Return all aerospace research tasks."""
        return get_aerospace_tasks()

    def get_documents(self) -> List[Document]:
        """Return aerospace knowledge base documents."""
        return get_aerospace_documents()

    def get_grader(self, task_id: str) -> BaseGrader:
        """Return a grader for the specified task."""
        return get_grader(task_id)

    def get_reward_function(self) -> BaseRewardFunction:
        """Return the composite reward function."""
        return CompositeRewardFunction()

    def get_system_prompt(self) -> str:
        """Return the aerospace-specific system prompt."""
        return self.SYSTEM_PROMPT
