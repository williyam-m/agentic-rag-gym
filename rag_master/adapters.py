"""Abstract base classes defining the RAG Master framework interfaces.

These adapters follow the Adapter design pattern, allowing any domain
to plug in its own implementations while the orchestrator remains generic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from rag_master.models import (
    AgentMessage,
    AgentRole,
    Document,
    EpisodeState,
    RetrievalResult,
    StepRecord,
    TaskDefinition,
    Trajectory,
)


class BaseRetriever(ABC):
    """Abstract retriever for document search."""

    @abstractmethod
    async def index_documents(self, documents: List[Document]) -> int:
        """Index documents into the retrieval store. Returns count indexed."""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query."""

    @abstractmethod
    async def clear_index(self) -> None:
        """Clear all indexed documents."""


class BaseLLMClient(ABC):
    """Abstract LLM client adapter (OpenAI-compatible)."""

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a response from the LLM."""

    @abstractmethod
    async def generate_with_metadata(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Generate a response with token usage and metadata."""


class BaseRewardFunction(ABC):
    """Abstract reward function for evaluating agent actions."""

    @abstractmethod
    async def compute_step_reward(
        self,
        state: EpisodeState,
        step: StepRecord,
    ) -> float:
        """Compute reward for a single step (process supervision)."""

    @abstractmethod
    async def compute_episode_reward(
        self,
        trajectory: Trajectory,
        state: EpisodeState,
    ) -> float:
        """Compute final reward for a complete episode."""

    @abstractmethod
    def get_reward_bounds(self) -> Tuple[float, float]:
        """Return (min_reward, max_reward) bounds. Must be within [0.01, 0.99]."""


class BaseGrader(ABC):
    """Abstract grader for evaluating task completion."""

    @abstractmethod
    async def grade(
        self,
        state: EpisodeState,
        trajectory: Trajectory,
    ) -> float:
        """Grade the agent's performance. Returns score in [0.01, 0.99]."""


class BaseAgent(ABC):
    """Abstract agent that can participate in the multi-agent system."""

    role: AgentRole

    @abstractmethod
    async def act(
        self,
        state: EpisodeState,
        incoming_messages: List[AgentMessage],
    ) -> Tuple[Dict[str, Any], List[AgentMessage]]:
        """
        Decide on an action given current state and incoming messages.
        Returns (action_payload, outgoing_messages).
        """


class BaseToolAdapter(ABC):
    """Abstract adapter for external tools/APIs."""

    @abstractmethod
    async def execute(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and return results."""

    @abstractmethod
    def list_tools(self) -> List[Dict[str, str]]:
        """List available tools with name and description."""


class BaseDomainConfig(ABC):
    """Abstract domain configuration that plugs into the orchestrator."""

    @abstractmethod
    def get_tasks(self) -> List[TaskDefinition]:
        """Return all tasks for this domain."""

    @abstractmethod
    def get_documents(self) -> List[Document]:
        """Return domain documents for indexing."""

    @abstractmethod
    def get_grader(self, task_id: str) -> BaseGrader:
        """Return a grader for the specified task."""

    @abstractmethod
    def get_reward_function(self) -> BaseRewardFunction:
        """Return the reward function for this domain."""

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for LLM interactions."""
