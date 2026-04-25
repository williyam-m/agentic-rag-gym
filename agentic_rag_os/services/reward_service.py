"""Rewards-as-a-Service — dynamic reward computation for any algorithm."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional

from agentic_rag_os.models import execute, fetch_all, fetch_one, new_id, now_iso


# --- Reward Config CRUD ---

async def create_reward_config(user_id: str, name: str, algorithm: str, domain_id: Optional[str], config: Dict[str, Any]) -> Dict[str, Any]:
    rid = new_id()
    ts = now_iso()
    await execute(
        "INSERT INTO reward_configs (id, user_id, domain_id, name, algorithm, config, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
        (rid, user_id, domain_id, name, algorithm, json.dumps(config), ts, ts),
    )
    return {"id": rid, "name": name, "algorithm": algorithm, "domain_id": domain_id, "config": config, "created_at": ts}


async def list_reward_configs(user_id: str) -> List[Dict[str, Any]]:
    rows = await fetch_all("SELECT * FROM reward_configs WHERE user_id=?", (user_id,))
    return [
        {"id": r["id"], "name": r["name"], "algorithm": r["algorithm"],
         "domain_id": r.get("domain_id"), "config": json.loads(r["config"]), "created_at": r["created_at"]}
        for r in rows
    ]


async def get_reward_config(user_id: str, config_id: str) -> Optional[Dict[str, Any]]:
    r = await fetch_one("SELECT * FROM reward_configs WHERE id=? AND user_id=?", (config_id, user_id))
    if not r:
        return None
    return {"id": r["id"], "name": r["name"], "algorithm": r["algorithm"],
            "domain_id": r.get("domain_id"), "config": json.loads(r["config"]), "created_at": r["created_at"]}


async def delete_reward_config(user_id: str, config_id: str) -> bool:
    r = await fetch_one("SELECT id FROM reward_configs WHERE id=? AND user_id=?", (config_id, user_id))
    if not r:
        return False
    await execute("DELETE FROM reward_jobs WHERE reward_config_id=?", (config_id,))
    await execute("DELETE FROM reward_configs WHERE id=?", (config_id,))
    return True


# --- Reward Computation Engine ---

def compute_reward(
    algorithm: str,
    query: str = "",
    answer: str = "",
    retrieved_docs: List[str] = [],
    config: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """Compute reward dynamically based on algorithm and config."""

    weights = {
        "retrieval_relevance_weight": config.get("retrieval_relevance_weight", 0.25),
        "reasoning_quality_weight": config.get("reasoning_quality_weight", 0.20),
        "answer_completeness_weight": config.get("answer_completeness_weight", 0.30),
        "efficiency_weight": config.get("efficiency_weight", 0.15),
        "anti_hack_weight": config.get("anti_hack_weight", 0.10),
    }

    breakdown = {}

    # 1. Retrieval relevance — overlap between query and docs
    if retrieved_docs and query:
        query_tokens = set(query.lower().split())
        doc_scores = []
        for doc in retrieved_docs:
            doc_tokens = set(doc.lower().split())
            overlap = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
            doc_scores.append(min(overlap * 1.5, 1.0))
        breakdown["retrieval_relevance"] = sum(doc_scores) / len(doc_scores) if doc_scores else 0.0
    else:
        breakdown["retrieval_relevance"] = 0.0

    # 2. Reasoning quality — presence of reasoning markers in answer
    reasoning_markers = ["because", "therefore", "however", "analysis", "evidence", "conclusion", "furthermore"]
    if answer:
        answer_lower = answer.lower()
        marker_count = sum(1 for m in reasoning_markers if m in answer_lower)
        breakdown["reasoning_quality"] = min(marker_count / 4.0, 1.0)
    else:
        breakdown["reasoning_quality"] = 0.0

    # 3. Answer completeness — length and coverage
    if answer:
        word_count = len(answer.split())
        length_score = min(word_count / 150.0, 1.0)
        # Coverage of query terms in answer
        if query:
            q_tokens = set(query.lower().split())
            a_tokens = set(answer.lower().split())
            coverage = len(q_tokens & a_tokens) / max(len(q_tokens), 1)
        else:
            coverage = 0.5
        breakdown["answer_completeness"] = 0.6 * length_score + 0.4 * coverage
    else:
        breakdown["answer_completeness"] = 0.0

    # 4. Efficiency — penalize very long or very short answers
    if answer:
        wc = len(answer.split())
        if wc < 20:
            breakdown["efficiency"] = 0.3
        elif wc > 500:
            breakdown["efficiency"] = 0.6
        else:
            breakdown["efficiency"] = 0.9
    else:
        breakdown["efficiency"] = 0.0

    # 5. Anti-hack penalty
    penalty = 0.0
    if answer:
        words = answer.lower().split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        if unique_ratio < 0.3:
            penalty += 0.5
        # Repetition check
        if len(words) > 10:
            bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
            bigram_unique = len(set(bigrams)) / max(len(bigrams), 1)
            if bigram_unique < 0.4:
                penalty += 0.3
    breakdown["anti_hack_penalty"] = min(penalty, 1.0)

    # Compute total
    total = (
        weights["retrieval_relevance_weight"] * breakdown["retrieval_relevance"]
        + weights["reasoning_quality_weight"] * breakdown["reasoning_quality"]
        + weights["answer_completeness_weight"] * breakdown["answer_completeness"]
        + weights["efficiency_weight"] * breakdown["efficiency"]
        - weights["anti_hack_weight"] * breakdown["anti_hack_penalty"]
    )
    total = max(0.01, min(0.99, total))

    # Algorithm-specific adjustments
    algo_meta = {}
    if algorithm == "grpo":
        algo_meta["group_relative"] = True
        algo_meta["description"] = "Group Relative Policy Optimization — rewards are relative within groups"
    elif algorithm == "ppo":
        algo_meta["clipped"] = True
        algo_meta["clip_range"] = config.get("clip_range", 0.2)
        algo_meta["description"] = "Proximal Policy Optimization — clipped surrogate objective"
    elif algorithm == "dpo":
        algo_meta["preference_based"] = True
        algo_meta["beta"] = config.get("beta", 0.1)
        algo_meta["description"] = "Direct Preference Optimization — reward from preference pairs"
    elif algorithm == "reinforce":
        algo_meta["baseline"] = config.get("baseline", 0.5)
        algo_meta["description"] = "REINFORCE — vanilla policy gradient with baseline"
        total = total - algo_meta["baseline"]
        total = max(0.01, min(0.99, (total + 1) / 2))
    else:
        algo_meta["description"] = "Custom reward function"

    return {
        "total_reward": round(total, 6),
        "breakdown": {k: round(v, 4) for k, v in breakdown.items()},
        "algorithm": algorithm,
        "metadata": algo_meta,
    }


# --- Reward Jobs ---

async def create_reward_job(user_id: str, reward_config_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    config_row = await fetch_one("SELECT * FROM reward_configs WHERE id=? AND user_id=?", (reward_config_id, user_id))
    if not config_row:
        raise ValueError("Reward config not found")

    jid = new_id()
    ts = now_iso()

    # Compute immediately (synchronous for now)
    cfg = json.loads(config_row["config"])
    result = compute_reward(
        algorithm=config_row["algorithm"],
        query=input_data.get("query", ""),
        answer=input_data.get("answer", ""),
        retrieved_docs=input_data.get("retrieved_docs", []),
        config=cfg,
    )

    await execute(
        "INSERT INTO reward_jobs (id, user_id, reward_config_id, status, result, created_at, completed_at) VALUES (?,?,?,?,?,?,?)",
        (jid, user_id, reward_config_id, "completed", json.dumps(result), ts, ts),
    )
    return {"id": jid, "reward_config_id": reward_config_id, "status": "completed", "result": result, "created_at": ts, "completed_at": ts}


async def list_reward_jobs(user_id: str, config_id: Optional[str] = None) -> List[Dict[str, Any]]:
    if config_id:
        rows = await fetch_all(
            "SELECT * FROM reward_jobs WHERE user_id=? AND reward_config_id=? ORDER BY created_at DESC LIMIT 50",
            (user_id, config_id),
        )
    else:
        rows = await fetch_all(
            "SELECT * FROM reward_jobs WHERE user_id=? ORDER BY created_at DESC LIMIT 50",
            (user_id,),
        )
    return [
        {"id": r["id"], "reward_config_id": r["reward_config_id"], "status": r["status"],
         "result": json.loads(r.get("result", "{}")), "created_at": r["created_at"], "completed_at": r.get("completed_at")}
        for r in rows
    ]
