"""Rewards-as-a-Service API endpoints."""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag_os.api.deps import get_current_user
from agentic_rag_os.database import get_db
from agentic_rag_os.models.reward_config import (
    RewardAlgorithm,
    RewardComputation,
    RewardConfig,
    RewardStatus,
)
from agentic_rag_os.services.rewards_service import RULE_TYPES
from agentic_rag_os.models.user import User
from agentic_rag_os.services.rewards_service import get_rewards_service

router = APIRouter(prefix="/rewards", tags=["rewards"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RewardRule(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    type: str = Field(..., description=f"One of: {', '.join(RULE_TYPES.keys())}")
    weight: float = Field(default=1.0, ge=0.0, le=10.0)
    # Type-specific fields
    keywords: Optional[List[str]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    format: Optional[str] = None
    forbidden_phrases: Optional[List[str]] = None
    pattern: Optional[str] = None


class RewardConfigCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    description: Optional[str] = Field(default=None, max_length=1000)
    algorithm: RewardAlgorithm = RewardAlgorithm.GRPO
    rules: List[RewardRule] = Field(..., min_length=1)
    weights: Dict[str, float] = Field(default_factory=dict)


class RewardConfigResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    algorithm: str
    rules: List[Dict[str, Any]]
    weights: Dict[str, float]
    status: str
    compute_count: int
    created_at: str

    model_config = {"from_attributes": True}


class ComputeRequest(BaseModel):
    inputs: List[Dict[str, str]] = Field(
        ...,
        description="List of {prompt, response} pairs",
        min_length=1,
        max_length=500,
    )


class ComputationResponse(BaseModel):
    id: int
    config_id: int
    input_count: int
    avg_reward: float
    results: List[Dict[str, Any]]
    created_at: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/rule-types", summary="List available rule types")
async def list_rule_types():
    """Return all supported reward rule types with descriptions."""
    return {"rule_types": RULE_TYPES}


@router.post("/", response_model=RewardConfigResponse, status_code=status.HTTP_201_CREATED)
async def create_reward_config(
    body: RewardConfigCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new reward configuration."""
    # Validate rule types
    for rule in body.rules:
        if rule.type not in RULE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown rule type '{rule.type}'. Valid: {list(RULE_TYPES.keys())}",
            )

    rules_json = [r.model_dump(exclude_none=True) for r in body.rules]
    # Build weights map: use provided weights or rule-level weights
    weights = body.weights or {r.name: r.weight for r in body.rules}

    config = RewardConfig(
        owner_id=current_user.id,
        name=body.name,
        description=body.description,
        algorithm=body.algorithm,
        rules=rules_json,
        weights=weights,
    )
    db.add(config)
    await db.commit()
    await db.refresh(config)
    return _config_to_response(config)


@router.get("/", response_model=List[RewardConfigResponse])
async def list_reward_configs(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(RewardConfig)
        .where(RewardConfig.owner_id == current_user.id)
        .order_by(RewardConfig.created_at.desc())
    )
    return [_config_to_response(c) for c in result.scalars().all()]


@router.get("/{config_id}", response_model=RewardConfigResponse)
async def get_reward_config(
    config_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    config = await _get_owned_config(config_id, current_user.id, db)
    return _config_to_response(config)


@router.delete("/{config_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_reward_config(
    config_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    config = await _get_owned_config(config_id, current_user.id, db)
    await db.delete(config)
    await db.commit()


@router.post("/{config_id}/compute", response_model=ComputationResponse)
async def compute_rewards(
    config_id: int,
    body: ComputeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Compute rewards for a batch of prompt/response pairs."""
    config = await _get_owned_config(config_id, current_user.id, db)

    svc = get_rewards_service()
    try:
        results, avg = svc.compute_batch(
            rules=config.rules,
            weights=config.weights or {},
            algorithm=config.algorithm,
            inputs=body.inputs,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    computation = RewardComputation(
        config_id=config.id,
        input_count=len(body.inputs),
        results=results,
        avg_reward=avg,
    )
    db.add(computation)
    config.compute_count += 1
    config.status = RewardStatus.ACTIVE
    await db.commit()
    await db.refresh(computation)

    return ComputationResponse(
        id=computation.id,
        config_id=config.id,
        input_count=computation.input_count,
        avg_reward=avg,
        results=results,
        created_at=computation.created_at.isoformat(),
    )


@router.get("/{config_id}/computations")
async def list_computations(
    config_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all computation runs for a reward config."""
    config = await _get_owned_config(config_id, current_user.id, db)
    result = await db.execute(
        select(RewardComputation)
        .where(RewardComputation.config_id == config.id)
        .order_by(RewardComputation.created_at.desc())
        .limit(50)
    )
    comps = result.scalars().all()
    return [
        {
            "id": c.id,
            "input_count": c.input_count,
            "avg_reward": c.avg_reward,
            "created_at": c.created_at.isoformat(),
        }
        for c in comps
    ]


@router.get("/{config_id}/computations/{computation_id}/export")
async def export_computation(
    config_id: int,
    computation_id: int,
    format: str = "json",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Export computation results as JSON, CSV, or HuggingFace dataset format."""
    await _get_owned_config(config_id, current_user.id, db)
    result = await db.execute(
        select(RewardComputation).where(
            RewardComputation.id == computation_id,
            RewardComputation.config_id == config_id,
        )
    )
    comp = result.scalar_one_or_none()
    if comp is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Computation not found")

    rows = comp.results or []

    if format == "json":
        content = json.dumps(rows, indent=2)
        return StreamingResponse(
            io.StringIO(content),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=rewards_{computation_id}.json"},
        )
    elif format == "csv":
        output = io.StringIO()
        if rows:
            fieldnames = ["prompt", "response", "reward"] + [
                f"rule_{k}" for k in (rows[0].get("breakdown") or {}).keys()
            ]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                flat = {
                    "prompt": row.get("prompt", ""),
                    "response": row.get("response", ""),
                    "reward": row.get("reward", 0.0),
                }
                for k, v in (row.get("breakdown") or {}).items():
                    flat[f"rule_{k}"] = v
                writer.writerow(flat)
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=rewards_{computation_id}.csv"},
        )
    elif format == "hf_dataset":
        svc = get_rewards_service()
        hf_rows = svc.export_hf_dataset(rows)
        content = json.dumps(hf_rows, indent=2)
        return StreamingResponse(
            io.StringIO(content),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=hf_dataset_{computation_id}.json"
            },
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Format must be 'json', 'csv', or 'hf_dataset'",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_owned_config(config_id: int, owner_id: int, db: AsyncSession) -> RewardConfig:
    result = await db.execute(
        select(RewardConfig).where(
            RewardConfig.id == config_id, RewardConfig.owner_id == owner_id
        )
    )
    config = result.scalar_one_or_none()
    if config is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Reward config not found")
    return config


def _config_to_response(c: RewardConfig) -> RewardConfigResponse:
    return RewardConfigResponse(
        id=c.id,
        name=c.name,
        description=c.description,
        algorithm=c.algorithm.value if hasattr(c.algorithm, "value") else c.algorithm,
        rules=c.rules or [],
        weights=c.weights or {},
        status=c.status.value if hasattr(c.status, "value") else c.status,
        compute_count=c.compute_count,
        created_at=c.created_at.isoformat(),
    )
