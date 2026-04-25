"""Legal Research domain configuration - plugs into the RAG Master orchestrator."""

from __future__ import annotations

from typing import List

from rag_master.adapters import BaseDomainConfig, BaseGrader, BaseRewardFunction
from rag_master.models import Document, TaskDefinition
from rag_master.rewards import CompositeRewardFunction

from domains.legal_research.graders import get_grader
from domains.legal_research.knowledge_base import get_legal_documents
from domains.legal_research.tasks import get_legal_tasks


class LegalResearchDomainConfig(BaseDomainConfig):
    """Domain configuration for legal research tasks."""

    SYSTEM_PROMPT = (
        "You are an expert legal research analyst with deep knowledge of "
        "contract law, intellectual property, regulatory compliance, corporate "
        "governance, employment law, litigation strategy, and international "
        "arbitration. Analyze the provided legal documents carefully and "
        "synthesize comprehensive, well-cited legal analyses. Always reference "
        "specific statutes, case law, and regulatory frameworks from the source "
        "documents. Structure your analysis with clear sections and quantitative "
        "risk assessments."
    )

    def get_tasks(self) -> List[TaskDefinition]:
        """Return all legal research tasks."""
        return get_legal_tasks()

    def get_documents(self) -> List[Document]:
        """Return legal knowledge base documents."""
        return get_legal_documents()

    def get_grader(self, task_id: str) -> BaseGrader:
        """Return a grader for the specified task."""
        return get_grader(task_id)

    def get_reward_function(self) -> BaseRewardFunction:
        """Return the composite reward function."""
        return CompositeRewardFunction()

    def get_system_prompt(self) -> str:
        """Return the legal-specific system prompt."""
        return self.SYSTEM_PROMPT
