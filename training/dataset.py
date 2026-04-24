"""
Dataset builder for GRPO training.

Connects to the live Agentic RAG Gym environment to build training prompts.
Each prompt = system instruction + task description + retrieved documents.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List

import httpx
from datasets import Dataset

from training.config import ENV_URL

SYSTEM_PROMPT = (
    "You are an expert aerospace research analyst with deep knowledge of "
    "propulsion systems, orbital mechanics, materials science, thermal protection, "
    "life support systems, and space mission design. When analyzing aerospace topics:\n"
    "- Cite specific data points and numerical values from provided documents\n"
    "- Structure your analysis with clear sections\n"
    "- Compare alternatives with quantitative evidence\n"
    "- Provide actionable recommendations grounded in engineering constraints\n"
)


def _format_prompt(task: Dict[str, Any], docs: List[Dict[str, Any]]) -> str:
    """Build a user message from a task + retrieved docs."""
    doc_text = ""
    for i, doc in enumerate(docs, 1):
        src = doc.get("source", "unknown")
        doc_text += f"\n[Document {i} -- {src}]\n{doc['content']}\n"

    return (
        f"## Task: {task['name']}\n\n"
        f"{task['description']}\n\n"
        f"### Retrieved Reference Documents\n{doc_text}\n"
        "### Instructions\n"
        "Provide a comprehensive, well-structured answer to the task above. "
        "Cite specific data from the reference documents. "
        "Include quantitative evidence and clear recommendations."
    )


async def _fetch_tasks(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    resp = await client.get(f"{ENV_URL}/tasks")
    resp.raise_for_status()
    return resp.json()["tasks"]


def _query_variants(task: Dict[str, Any]) -> List[str]:
    """Generate diverse retrieval queries from a task description."""
    desc = task["description"]
    sentences = [s.strip() for s in re.split(r'[.!?]+', desc) if len(s.strip()) > 20]
    variants = [desc]
    variants.extend(sentences[:4])
    name_words = task["name"].lower()
    variants.append(name_words)
    return variants


async def _collect_one_prompt(
    client: httpx.AsyncClient,
    task: Dict[str, Any],
    query: str,
) -> Dict[str, Any] | None:
    """Reset env, retrieve docs, build a prompt."""
    resp = await client.post(f"{ENV_URL}/reset", json={"task_id": task["task_id"]})
    if resp.status_code != 200:
        return None

    resp = await client.post(
        f"{ENV_URL}/step", json={"type": "retrieve", "query": query}
    )
    if resp.status_code != 200:
        return None

    docs = resp.json()["observation"]["retrieved_docs"]
    user_msg = _format_prompt(task, docs)
    return {
        "task_id": task["task_id"],
        "task_name": task["name"],
        "difficulty": task.get("difficulty", "easy"),
        "prompt": user_msg,
    }


async def _build_dataset_async(num_per_task: int = 8) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = await _fetch_tasks(client)
        for task in tasks:
            variants = _query_variants(task)
            for i in range(num_per_task):
                query = variants[i % len(variants)]
                rec = await _collect_one_prompt(client, task, query)
                if rec:
                    records.append(rec)
    return records


def build_grpo_dataset(num_per_task: int = 8, seed: int = 42) -> Dataset:
    """
    Build a HuggingFace Dataset of prompts for GRPO training.

    Each row: prompt (str), task_id, task_name, difficulty.
    Prompts are collected from the LIVE environment (reset + retrieve).
    """
    records = asyncio.run(_build_dataset_async(num_per_task))
    if not records:
        raise RuntimeError(f"No prompts collected. Is the environment running at {ENV_URL}?")
    ds = Dataset.from_list(records)
    ds = ds.shuffle(seed=seed)
    print(f"Built GRPO dataset: {len(ds)} prompts across {len(ds.unique('task_id'))} tasks")
    return ds
