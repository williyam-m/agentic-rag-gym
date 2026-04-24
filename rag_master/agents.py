"""Multi-agent system for cooperative RAG processing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from rag_master.adapters import BaseAgent, BaseLLMClient, BaseRetriever
from rag_master.logging_config import get_logger
from rag_master.models import AgentMessage, AgentRole, EpisodeState, RetrievalResult

logger = get_logger(__name__)


class RetrieverAgent(BaseAgent):
    """Agent responsible for document retrieval and query reformulation."""

    role = AgentRole.RETRIEVER

    def __init__(self, retriever: BaseRetriever, llm: BaseLLMClient) -> None:
        self._retriever = retriever
        self._llm = llm

    async def act(
        self,
        state: EpisodeState,
        incoming_messages: List[AgentMessage],
    ) -> Tuple[Dict[str, Any], List[AgentMessage]]:
        """Retrieve documents based on current query or reformulated query."""
        query = self._extract_query(state, incoming_messages)
        results = await self._retriever.retrieve(query, top_k=5)

        outgoing: List[AgentMessage] = []
        if results:
            doc_summary = "\n".join(
                f"[{r.rank}] (score={r.score:.2f}) {r.document.content[:200]}"
                for r in results
            )
            outgoing.append(
                AgentMessage(
                    sender=self.role,
                    receiver=AgentRole.REASONER,
                    content=f"Retrieved {len(results)} documents for: {query}\n{doc_summary}",
                    message_type="retrieval_results",
                )
            )

        return {
            "action": "retrieve",
            "query": query,
            "results_count": len(results),
            "results": results,
        }, outgoing

    def _extract_query(
        self, state: EpisodeState, messages: List[AgentMessage]
    ) -> str:
        """Extract the best query from state or incoming messages."""
        for msg in reversed(messages):
            if msg.message_type == "query_refinement":
                return msg.content
        if state.query_history:
            return state.query_history[-1]
        return state.task.description


class ReasonerAgent(BaseAgent):
    """Agent responsible for reasoning over retrieved documents."""

    role = AgentRole.REASONER

    def __init__(self, llm: BaseLLMClient, system_prompt: str = "") -> None:
        self._llm = llm
        self._system_prompt = system_prompt

    async def act(
        self,
        state: EpisodeState,
        incoming_messages: List[AgentMessage],
    ) -> Tuple[Dict[str, Any], List[AgentMessage]]:
        """Reason over retrieved documents to build an answer."""
        context = self._build_context(state, incoming_messages)
        messages = [
            {"role": "system", "content": self._system_prompt or "You are a research analyst."},
            {"role": "user", "content": context},
        ]
        reasoning = await self._llm.generate(messages, temperature=0.3, max_tokens=1024)

        outgoing = [
            AgentMessage(
                sender=self.role,
                receiver=AgentRole.CRITIC,
                content=reasoning,
                message_type="reasoning_output",
            )
        ]
        return {"action": "reason", "reasoning": reasoning}, outgoing

    def _build_context(
        self, state: EpisodeState, messages: List[AgentMessage]
    ) -> str:
        """Build context string from state and messages."""
        parts = [f"Task: {state.task.description}"]
        for msg in messages:
            if msg.message_type == "retrieval_results":
                parts.append(f"Retrieved Documents:\n{msg.content}")
        if state.generated_answer:
            parts.append(f"Previous answer draft:\n{state.generated_answer}")
        parts.append("Please analyze the documents and provide a comprehensive answer.")
        return "\n\n".join(parts)


class CriticAgent(BaseAgent):
    """Agent that critiques and suggests improvements."""

    role = AgentRole.CRITIC

    def __init__(self, llm: BaseLLMClient) -> None:
        self._llm = llm

    async def act(
        self,
        state: EpisodeState,
        incoming_messages: List[AgentMessage],
    ) -> Tuple[Dict[str, Any], List[AgentMessage]]:
        """Critique the current reasoning and suggest improvements."""
        reasoning_content = ""
        for msg in incoming_messages:
            if msg.message_type == "reasoning_output":
                reasoning_content = msg.content
                break

        critique_prompt = (
            f"Task: {state.task.description}\n\n"
            f"Agent's reasoning:\n{reasoning_content}\n\n"
            f"Evaluate this reasoning. Identify:\n"
            f"1. Factual accuracy issues\n"
            f"2. Missing important points\n"
            f"3. Logical gaps\n"
            f"4. Suggestions for improvement\n"
            f"Rate quality 0-10 and explain."
        )
        critique = await self._llm.generate(
            [{"role": "user", "content": critique_prompt}],
            temperature=0.2,
            max_tokens=512,
        )

        needs_refinement = any(
            kw in critique.lower()
            for kw in ["missing", "incorrect", "improve", "lacks", "gap", "error"]
        )
        outgoing: List[AgentMessage] = []
        if needs_refinement:
            outgoing.append(
                AgentMessage(
                    sender=self.role,
                    receiver=AgentRole.RETRIEVER,
                    content=f"Refine search based on critique: {critique[:200]}",
                    message_type="query_refinement",
                )
            )
        return {
            "action": "critique",
            "critique": critique,
            "needs_refinement": needs_refinement,
        }, outgoing


class PlannerAgent(BaseAgent):
    """Agent that plans multi-step strategies for complex tasks."""

    role = AgentRole.PLANNER

    def __init__(self, llm: BaseLLMClient) -> None:
        self._llm = llm

    async def act(
        self,
        state: EpisodeState,
        incoming_messages: List[AgentMessage],
    ) -> Tuple[Dict[str, Any], List[AgentMessage]]:
        """Create or update the execution plan."""
        plan_prompt = (
            f"Task: {state.task.description}\n"
            f"Steps completed: {state.current_step}/{state.task.max_steps}\n"
            f"Documents retrieved: {len(state.retrieved_docs)}\n"
            f"Current answer length: {len(state.generated_answer)} chars\n\n"
            f"Create a concise plan for the remaining steps. "
            f"List 2-4 concrete actions."
        )
        plan = await self._llm.generate(
            [{"role": "user", "content": plan_prompt}],
            temperature=0.3,
            max_tokens=300,
        )

        outgoing = [
            AgentMessage(
                sender=self.role,
                receiver=AgentRole.REASONER,
                content=plan,
                message_type="execution_plan",
            )
        ]
        return {"action": "plan", "plan": plan}, outgoing


class VerifierAgent(BaseAgent):
    """Agent that verifies final answers against source documents."""

    role = AgentRole.VERIFIER

    def __init__(self, llm: BaseLLMClient) -> None:
        self._llm = llm

    async def act(
        self,
        state: EpisodeState,
        incoming_messages: List[AgentMessage],
    ) -> Tuple[Dict[str, Any], List[AgentMessage]]:
        """Verify the answer is grounded in retrieved documents."""
        docs_text = "\n".join(
            f"Doc {r.rank}: {r.document.content[:300]}"
            for r in state.retrieved_docs[:5]
        )
        verify_prompt = (
            f"Task: {state.task.description}\n\n"
            f"Source documents:\n{docs_text}\n\n"
            f"Answer to verify:\n{state.generated_answer[:500]}\n\n"
            f"Check if the answer is:\n"
            f"1. Factually supported by the documents\n"
            f"2. Complete (covers the task)\n"
            f"3. Free of hallucinations\n"
            f"Rate grounding score 0-10."
        )
        verification = await self._llm.generate(
            [{"role": "user", "content": verify_prompt}],
            temperature=0.1,
            max_tokens=300,
        )

        return {
            "action": "verify",
            "verification": verification,
            "grounded": "not grounded" not in verification.lower(),
        }, []
