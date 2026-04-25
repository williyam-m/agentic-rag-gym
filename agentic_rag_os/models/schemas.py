"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --- Auth ---

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    password: str = Field(..., min_length=8)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserOut"


class UserOut(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    avatar_url: str = ""
    role: str = "user"
    tier: str = "free"
    created_at: str


# --- Domain ---

class DomainCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""


class DomainOut(BaseModel):
    id: str
    name: str
    description: str
    document_count: int = 0
    total_size_bytes: int = 0
    created_at: str


# --- Document ---

class DocumentOut(BaseModel):
    id: str
    filename: str
    size_bytes: int
    metadata: Dict[str, Any] = {}
    created_at: str


# --- RAG ---

class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class RAGResult(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any] = {}


class RAGQueryResponse(BaseModel):
    query: str
    results: List[RAGResult]
    domain: str


# --- Rewards-as-a-Service ---

class RewardConfigCreate(BaseModel):
    name: str
    algorithm: str = Field(default="grpo", pattern="^(grpo|ppo|dpo|reinforce|custom)$")
    domain_id: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=lambda: {
        "retrieval_relevance_weight": 0.25,
        "reasoning_quality_weight": 0.20,
        "answer_completeness_weight": 0.30,
        "efficiency_weight": 0.15,
        "anti_hack_weight": 0.10,
    })


class RewardConfigOut(BaseModel):
    id: str
    name: str
    algorithm: str
    domain_id: Optional[str] = None
    config: Dict[str, Any]
    created_at: str


class RewardJobCreate(BaseModel):
    reward_config_id: str
    input_data: Dict[str, Any] = {}


class RewardJobOut(BaseModel):
    id: str
    reward_config_id: str
    status: str
    result: Dict[str, Any] = {}
    created_at: str
    completed_at: Optional[str] = None


class RewardComputeRequest(BaseModel):
    """Compute reward on-the-fly without creating a job."""
    algorithm: str = "grpo"
    query: str = ""
    answer: str = ""
    retrieved_docs: List[str] = []
    config: Dict[str, Any] = {}


class RewardComputeResponse(BaseModel):
    total_reward: float
    breakdown: Dict[str, float] = {}
    algorithm: str
    metadata: Dict[str, Any] = {}


# --- API Key ---

class APIKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class APIKeyOut(BaseModel):
    id: str
    name: str
    key_prefix: str
    created_at: str
    last_used: Optional[str] = None
    is_active: bool = True


class APIKeyCreated(BaseModel):
    id: str
    name: str
    key: str  # Full key, shown only once
    created_at: str


# --- Dashboard Stats ---

class DashboardStats(BaseModel):
    total_domains: int = 0
    total_documents: int = 0
    total_queries: int = 0
    total_reward_configs: int = 0
    storage_used_bytes: int = 0
    storage_limit_bytes: int = 0


TokenResponse.model_rebuild()
