"""OpenEnv-compatible Pydantic models for the Agentic RAG Gym."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Valid action types in the environment."""

    RETRIEVE = "retrieve"
    REASON = "reason"
    ANSWER = "answer"
    CRITIQUE = "critique"
    PLAN = "plan"
    VERIFY = "verify"


class Action(BaseModel):
    """Action taken by the agent in the environment."""

    type: ActionType = Field(description="Type of action to perform.")
    query: Optional[str] = Field(default=None, description="Search query for retrieval actions.")
    answer: Optional[str] = Field(default=None, description="Answer text for answer actions.")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional action parameters."
    )


class DocumentObservation(BaseModel):
    """A retrieved document in the observation."""

    content: str = Field(description="Document content (truncated).")
    score: float = Field(ge=0.0, le=1.0, description="Relevance score.")
    source: str = Field(default="", description="Document source identifier.")


class TaskObservation(BaseModel):
    """Task information in the observation."""

    id: str = Field(description="Task identifier.")
    name: str = Field(description="Task name.")
    description: str = Field(description="Task description and objectives.")
    difficulty: str = Field(description="Difficulty level: easy, medium, hard, expert.")
    max_steps: int = Field(description="Maximum allowed steps.")


class Observation(BaseModel):
    """Observation returned by the environment after each step."""

    task: TaskObservation = Field(description="Current task information.")
    step: int = Field(ge=0, description="Current step number.")
    retrieved_docs: List[DocumentObservation] = Field(
        default_factory=list, description="Recently retrieved documents."
    )
    query_history: List[str] = Field(
        default_factory=list, description="Recent query history."
    )
    current_answer: str = Field(default="", description="Current answer draft.")
    last_reward: float = Field(default=0.0, description="Reward from last step.")
    done: bool = Field(default=False, description="Whether the episode has ended.")


class Reward(BaseModel):
    """Reward signal from the environment."""

    value: float = Field(ge=0.0, le=1.0, description="Reward value.")
    breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Reward component breakdown."
    )


class StepResult(BaseModel):
    """Result of a single environment step."""

    observation: Observation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool = Field(default=False)
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    """Result of environment reset."""

    observation: Observation


class StateResult(BaseModel):
    """Current environment state."""

    initialized: bool = Field(default=False)
    episode_id: str = Field(default="")
    task_id: str = Field(default="")
    task_name: str = Field(default="")
    task_description: str = Field(default="")
    difficulty: str = Field(default="")
    current_step: int = Field(default=0)
    max_steps: int = Field(default=0)
    done: bool = Field(default=False)
    query_history: List[str] = Field(default_factory=list)
    retrieved_doc_count: int = Field(default=0)
    answer_length: int = Field(default=0)
    intermediate_rewards: List[float] = Field(default_factory=list)
    generated_answer: str = Field(default="")


class TaskInfo(BaseModel):
    """Public task information."""

    task_id: str
    name: str
    description: str
    difficulty: str
    max_steps: int


class GradeResult(BaseModel):
    """Grading result for a task."""

    task_id: str
    score: float = Field(ge=0.0, le=1.0)
    episode_id: str = Field(default="")


class HealthStatus(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "1.0.0"
    environment: str = "agentic-rag-gym"
