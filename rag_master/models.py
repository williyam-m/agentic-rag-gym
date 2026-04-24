"""Core domain models for the RAG Master framework."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    """Roles an agent can assume in the multi-agent system."""

    RETRIEVER = "retriever"
    REASONER = "reasoner"
    CRITIC = "critic"
    PLANNER = "planner"
    VERIFIER = "verifier"


class DifficultyLevel(str, Enum):
    """Task difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class Document(BaseModel):
    """A document in the knowledge base."""

    doc_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RetrievalResult(BaseModel):
    """Result of a retrieval operation."""

    document: Document
    score: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=0)


class AgentMessage(BaseModel):
    """Message exchanged between agents."""

    sender: AgentRole
    receiver: Optional[AgentRole] = None
    content: str
    message_type: str = "text"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StepRecord(BaseModel):
    """Record of a single environment step for process supervision."""

    step_index: int
    action_type: str
    action_payload: Dict[str, Any] = Field(default_factory=dict)
    observation_summary: str = ""
    intermediate_reward: float = 0.0
    reasoning_trace: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Trajectory(BaseModel):
    """Full trajectory of an episode."""

    episode_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    task_id: str
    steps: List[StepRecord] = Field(default_factory=list)
    total_reward: float = 0.0
    final_score: float = 0.0
    completed: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskDefinition(BaseModel):
    """Definition of a task in the environment."""

    task_id: str
    name: str
    description: str
    difficulty: DifficultyLevel
    max_steps: int = 20
    success_criteria: str = ""
    grading_rubric: Dict[str, float] = Field(default_factory=dict)
    domain_data: Dict[str, Any] = Field(default_factory=dict)


class EpisodeState(BaseModel):
    """Current state of an episode."""

    episode_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    task: TaskDefinition
    current_step: int = 0
    query_history: List[str] = Field(default_factory=list)
    retrieved_docs: List[RetrievalResult] = Field(default_factory=list)
    agent_messages: List[AgentMessage] = Field(default_factory=list)
    generated_answer: str = ""
    intermediate_rewards: List[float] = Field(default_factory=list)
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)
