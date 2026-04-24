"""
Inference Script — Agentic RAG Gym
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# --- Configuration from environment ---
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:11434/v1")
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "ollama"
MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen2.5:7b")
ENV_URL: str = os.getenv("ENV_URL", "http://localhost:7860")
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS_ANSWER: int = 1024
MAX_TOKENS_STEP: int = 512

SYSTEM_PROMPT = (
    "You are an expert aerospace research analyst. You are interacting with a "
    "Retrieval-Augmented Generation environment. Your goal is to produce high-quality, "
    "well-sourced research analyses.\n\n"
    "Available actions:\n"
    "- retrieve: Search the knowledge base. Include a 'query' field.\n"
    "- reason: Analyze retrieved documents.\n"
    "- answer: Submit your final answer. Include an 'answer' field.\n"
    "- plan: Create a research plan.\n"
    "- critique: Evaluate your current reasoning.\n"
    "- verify: Check answer grounding against sources.\n\n"
    "Respond with a JSON object with fields: type, query (optional), answer (optional).\n"
    "Example: {\"type\": \"retrieve\", \"query\": \"ion propulsion specific impulse\"}\n"
    "Example: {\"type\": \"answer\", \"answer\": \"Based on the analysis...\"}"
)


def call_env(method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
    """Make an API call to the environment."""
    url = f"{ENV_URL}{endpoint}"
    with httpx.Client(timeout=120.0) as client:
        if method == "GET":
            resp = client.get(url)
        else:
            resp = client.post(url, json=data or {})
        resp.raise_for_status()
        return resp.json()


def parse_agent_action(response_text: str, observation: Dict) -> Dict[str, Any]:
    """Parse the LLM response into an action dict."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        text = text.strip()

    try:
        action = json.loads(text)
        if isinstance(action, dict) and "type" in action:
            return action
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: extract action type from text
    text_lower = text.lower()
    if "answer" in text_lower and len(text) > 100:
        return {"type": "answer", "answer": text}
    if "retrieve" in text_lower or "search" in text_lower:
        query = observation.get("task", {}).get("description", "aerospace research")
        return {"type": "retrieve", "query": query}
    if "plan" in text_lower:
        return {"type": "plan"}
    if "critiqu" in text_lower:
        return {"type": "critique"}
    if "verif" in text_lower:
        return {"type": "verify"}

    # Default fallback: retrieve with task description (more useful than reason)
    return {"type": "retrieve", "query": observation.get("task", {}).get("description", "aerospace research")}


def build_user_message(step: int, max_steps: int, observation: Dict, history: List[str]) -> str:
    """Build the user message for the LLM."""
    task = observation.get("task", {})
    parts = [
        f"Step: {step}/{max_steps}",
        f"Task: {task.get('description', 'N/A')}",
        f"Difficulty: {task.get('difficulty', 'N/A')}",
    ]

    docs = observation.get("retrieved_docs", [])
    if docs:
        parts.append("\nRetrieved Documents:")
        for i, doc in enumerate(docs):
            parts.append(f"  [{i+1}] (score={doc.get('score', 0):.2f}) {doc.get('content', '')[:300]}")

    current = observation.get("current_answer", "")
    if current:
        parts.append(f"\nCurrent Answer Draft:\n{current[:500]}")

    last_reward = observation.get("last_reward", 0.0)
    parts.append(f"\nLast reward: {last_reward:.4f}")

    if history:
        parts.append(f"\nPrevious actions: {', '.join(history[-5:])}")

    remaining = max_steps - step
    if remaining <= 2 and "answer" not in history:
        parts.append(
            "\n⚠️ IMPORTANT: You are running out of steps! You MUST submit an answer now. "
            "Respond with: {\"type\": \"answer\", \"answer\": \"Your comprehensive answer here...\"}"
        )
    elif remaining <= 4 and "answer" not in history:
        parts.append(
            "\nNote: Only a few steps remain. Start preparing your final answer soon."
        )

    parts.append("\nDecide your next action. Respond with a JSON object.")
    return "\n".join(parts)


def run_episode(client: OpenAI, task_id: str, max_steps: int = 20) -> Dict[str, Any]:
    """Run a single episode on a task."""
    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"{'='*60}")

    reset_result = call_env("POST", "/reset", {"task_id": task_id})
    observation = reset_result.get("observation", {})
    task_name = observation.get("task", {}).get("name", task_id)
    print(f"Task Name: {task_name}")
    print(f"Max Steps: {max_steps}")

    history: List[str] = []
    total_reward = 0.0
    answer_submitted = False

    for step in range(1, max_steps + 1):
        # Force answer on penultimate step if none submitted
        if step >= max_steps - 1 and not answer_submitted:
            action = {
                "type": "answer",
                "answer": _build_forced_answer(observation),
            }
        else:
            user_msg = build_user_message(step, max_steps, observation, history)
            try:
                max_tok = MAX_TOKENS_ANSWER if step >= max_steps - 2 else MAX_TOKENS_STEP
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=max_tok,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                print(f"  LLM Error: {exc}")
                response_text = '{"type": "retrieve", "query": "' + observation.get("task", {}).get("description", "") + '"}'

            action = parse_agent_action(response_text, observation)

        action_type = action.get("type", "reason")
        if action_type == "answer":
            answer_submitted = True
        print(f"  Step {step}: {action_type}", end="")

        result = call_env("POST", "/step", action)
        observation = result.get("observation", {})
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        total_reward += reward

        print(f" → reward={reward:.4f} done={done}")
        history.append(action_type)

        if done:
            break

    # Grade
    grade_result = call_env("POST", "/grade")
    final_score = grade_result.get("score", 0.0)

    print(f"\n  Final Score: {final_score:.4f}")
    print(f"  Total Reward: {total_reward:.4f}")
    print(f"  Steps Used: {len(history)}")

    return {
        "task_id": task_id,
        "task_name": task_name,
        "score": final_score,
        "total_reward": total_reward,
        "steps": len(history),
        "action_history": history,
        "answer_submitted": answer_submitted,
    }


def _build_forced_answer(observation: Dict) -> str:
    """Build a forced answer from available context when agent fails to answer."""
    task = observation.get("task", {})
    docs = observation.get("retrieved_docs", [])
    current = observation.get("current_answer", "")

    if current and len(current) > 50:
        return current

    parts = [f"Based on analysis of the available research documents for: {task.get('description', '')}"]
    for doc in docs[:3]:
        content = doc.get("content", "")
        if content:
            parts.append(f"\nKey finding: {content[:400]}")
    parts.append(
        "\nIn conclusion, the analysis of the retrieved aerospace research documents "
        "provides comprehensive technical evidence supporting the recommendations above."
    )
    return "\n".join(parts)


def main() -> None:
    """Run inference on all tasks."""
    print("=" * 60)
    print("Agentic RAG Gym — Inference Script")
    print("=" * 60)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Environment: {ENV_URL}")
    print(f"Temperature: {TEMPERATURE}")
    print()

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Get tasks
    tasks_result = call_env("GET", "/tasks")
    tasks = tasks_result.get("tasks", [])
    print(f"Found {len(tasks)} tasks\n")

    results: List[Dict[str, Any]] = []
    start_time = time.time()

    for task in tasks:
        task_id = task["task_id"]
        max_steps = int(task.get("max_steps", 20))
        try:
            result = run_episode(client, task_id, max_steps=max_steps)
            results.append(result)
        except Exception as exc:
            print(f"  ERROR on {task_id}: {exc}")
            results.append({"task_id": task_id, "score": 0.0, "error": str(exc)})

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Task ID':<45} {'Score':>8} {'Steps':>6}")
    print("-" * 60)
    for r in results:
        score = r.get("score", 0.0)
        steps = r.get("steps", 0)
        print(f"{r['task_id']:<45} {score:>8.4f} {steps:>6}")

    avg_score = sum(r.get("score", 0.0) for r in results) / max(len(results), 1)
    print("-" * 60)
    print(f"{'Average Score':<45} {avg_score:>8.4f}")
    print(f"Total Time: {elapsed:.1f}s")
    print("=" * 60)

    # Save results to JSON
    output = {
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "temperature": TEMPERATURE,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 1),
        "average_score": round(avg_score, 4),
        "results": results,
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to baseline_results.json")


if __name__ == "__main__":
    main()
