"""Rewards-as-a-Service routes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from agentic_rag_os.api.deps import get_current_user
from agentic_rag_os.models.schemas import (
    RewardComputeRequest,
    RewardComputeResponse,
    RewardConfigCreate,
    RewardConfigOut,
    RewardJobCreate,
    RewardJobOut,
)
from agentic_rag_os.services.reward_service import (
    compute_reward,
    create_reward_config,
    create_reward_job,
    delete_reward_config,
    list_reward_configs,
    list_reward_jobs,
    get_reward_config,
)

router = APIRouter(prefix="/rewards", tags=["Rewards-as-a-Service"])


# --- Reward Configs ---

@router.post("/configs", response_model=RewardConfigOut)
async def create_config_route(body: RewardConfigCreate, user: Dict = Depends(get_current_user)):
    return await create_reward_config(user["user_id"], body.name, body.algorithm, body.domain_id, body.config)


@router.get("/configs", response_model=List[RewardConfigOut])
async def list_configs_route(user: Dict = Depends(get_current_user)):
    return await list_reward_configs(user["user_id"])


@router.get("/configs/{config_id}", response_model=RewardConfigOut)
async def get_config_route(config_id: str, user: Dict = Depends(get_current_user)):
    cfg = await get_reward_config(user["user_id"], config_id)
    if not cfg:
        raise HTTPException(status_code=404, detail="Config not found")
    return cfg


@router.delete("/configs/{config_id}")
async def delete_config_route(config_id: str, user: Dict = Depends(get_current_user)):
    ok = await delete_reward_config(user["user_id"], config_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Config not found")
    return {"ok": True}


# --- On-the-fly reward computation ---

@router.post("/compute", response_model=RewardComputeResponse)
async def compute_reward_route(body: RewardComputeRequest, user: Dict = Depends(get_current_user)):
    """Compute reward on-the-fly — no stored config needed."""
    result = compute_reward(
        algorithm=body.algorithm,
        query=body.query,
        answer=body.answer,
        retrieved_docs=body.retrieved_docs,
        config=body.config,
    )
    return result


# --- Reward Jobs ---

@router.post("/jobs", response_model=RewardJobOut)
async def create_job_route(body: RewardJobCreate, user: Dict = Depends(get_current_user)):
    try:
        return await create_reward_job(user["user_id"], body.reward_config_id, body.input_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/jobs", response_model=List[RewardJobOut])
async def list_jobs_route(config_id: Optional[str] = Query(None), user: Dict = Depends(get_current_user)):
    return await list_reward_jobs(user["user_id"], config_id)


# --- Supported Algorithms ---

@router.get("/algorithms")
async def list_algorithms():
    """List all supported reward algorithms."""
    return {
        "algorithms": [
            {"id": "grpo", "name": "GRPO", "description": "Group Relative Policy Optimization — rewards are relative within groups for stable training"},
            {"id": "ppo", "name": "PPO", "description": "Proximal Policy Optimization — clipped surrogate objective for monotonic improvement"},
            {"id": "dpo", "name": "DPO", "description": "Direct Preference Optimization — reward from preference pairs, no reward model needed"},
            {"id": "reinforce", "name": "REINFORCE", "description": "Vanilla policy gradient with baseline subtraction"},
            {"id": "custom", "name": "Custom", "description": "Define your own reward weights and scoring rules"},
        ]
    }
