"""Legal Research domain configuration stub.

This is a placeholder demonstrating how to add a new domain to the RAG Master framework.
Implement the methods with real legal research tasks, documents, and graders.
"""

from __future__ import annotations

from typing import List

from rag_master.adapters import BaseDomainConfig, BaseGrader, BaseRewardFunction
from rag_master.models import Document, DifficultyLevel, TaskDefinition
from rag_master.rewards import CompositeRewardFunction


class _StubGrader(BaseGrader):
    """Placeholder grader for legal tasks."""

    async def grade(self, state, trajectory) -> float:
        return 0.5


class LegalResearchDomainConfig(BaseDomainConfig):
    """Domain configuration for legal research tasks (stub)."""

    SYSTEM_PROMPT = (
        "You are an expert legal research analyst with deep knowledge of "
        "contract law, intellectual property, regulatory compliance, and "
        "case law analysis. Analyze the provided legal documents carefully "
        "and synthesize accurate, well-cited legal analyses."
    )

    def get_tasks(self) -> List[TaskDefinition]:
        """Return legal research tasks (stub)."""
        return [
            TaskDefinition(
                task_id="legal_easy_contract_review",
                name="Contract Clause Analysis",
                description=(
                    "Analyze the provided contract excerpts and identify key clauses "
                    "related to liability, indemnification, and termination. "
                    "Assess risks and suggest improvements."
                ),
                difficulty=DifficultyLevel.EASY,
                max_steps=12,
                success_criteria="Identify main clauses with risk assessment",
            ),
            TaskDefinition(
                task_id="legal_medium_ip_analysis",
                name="Intellectual Property Assessment",
                description=(
                    "Evaluate the intellectual property landscape for a technology "
                    "patent application. Analyze prior art, assess patentability, "
                    "and identify potential infringement risks."
                ),
                difficulty=DifficultyLevel.MEDIUM,
                max_steps=16,
                success_criteria="Comprehensive IP analysis with prior art review",
            ),
        ]

    def get_documents(self) -> List[Document]:
        """Return legal knowledge base documents (stub)."""
        return [
            Document(
                doc_id="legal_001",
                content=(
                    "A limitation of liability clause typically caps the total damages "
                    "recoverable by either party. Common formulations include caps equal "
                    "to the fees paid under the agreement in the preceding 12 months. "
                    "Courts generally enforce these clauses unless they are unconscionable "
                    "or involve gross negligence or willful misconduct."
                ),
                source="Legal Research Database",
                metadata={"topic": "contracts", "subtopic": "liability"},
            ),
            Document(
                doc_id="legal_002",
                content=(
                    "Prior art searches for patent applications should cover both patent "
                    "and non-patent literature. The USPTO uses classification systems "
                    "(CPC and USPC) to organize patents by technology area. A thorough "
                    "search includes keyword searches, citation analysis, and review "
                    "of related patent families across major jurisdictions."
                ),
                source="Patent Law Reference",
                metadata={"topic": "ip", "subtopic": "prior_art"},
            ),
        ]

    def get_grader(self, task_id: str) -> BaseGrader:
        """Return a grader for the specified task (stub)."""
        return _StubGrader()

    def get_reward_function(self) -> BaseRewardFunction:
        """Return the composite reward function."""
        return CompositeRewardFunction()

    def get_system_prompt(self) -> str:
        """Return the legal-specific system prompt."""
        return self.SYSTEM_PROMPT
