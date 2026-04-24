"""
Evaluation & plotting utilities for GRPO training.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training.config import PLOTS_DIR
from training.reward import grade_answer_sync


# ── Model evaluation ────────────────────────────────────────────────────

def evaluate_model_on_tasks(
    model,
    tokenizer,
    prompts: List[Dict[str, Any]],
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> List[Dict[str, Any]]:
    """Generate answers for each prompt and grade them with domain graders."""
    import torch
    results: List[Dict[str, Any]] = []
    device = next(model.parameters()).device

    for item in prompts:
        messages = [
            {"role": "system", "content": "You are an expert aerospace research analyst. Provide comprehensive answers citing specific data."},
            {"role": "user", "content": item["prompt"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
        score = grade_answer_sync(item["task_id"], answer)

        results.append({
            "task_id": item["task_id"],
            "task_name": item["task_name"],
            "difficulty": item["difficulty"],
            "answer": answer[:500],
            "score": score,
        })
        print(f"  [{item['task_id']}] score={score:.3f}  len={len(answer)}")
    return results


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_training_curves(log_history: List[Dict[str, Any]], out_dir: Path = PLOTS_DIR) -> Path:
    """Plot training loss and reward curves. Returns path to saved figure."""
    steps = [e["step"] for e in log_history if "loss" in e]
    losses = [e["loss"] for e in log_history if "loss" in e]
    reward_steps = [e["step"] for e in log_history if "reward" in e or "reward/mean" in e]
    rewards = [e.get("reward/mean", e.get("reward")) for e in log_history if "reward" in e or "reward/mean" in e]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    if steps and losses:
        ax.plot(steps, losses, color="#D4AF37", linewidth=2)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("GRPO Training Loss", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if reward_steps and rewards:
        ax.plot(reward_steps, rewards, color="#4CAF50", linewidth=2)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Mean Reward (grader score)", fontsize=12)
    ax.set_title("GRPO Mean Reward", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves -> {path}")
    return path


def plot_baseline_vs_trained(
    baseline_results: List[Dict[str, Any]],
    trained_results: List[Dict[str, Any]],
    out_dir: Path = PLOTS_DIR,
) -> Path:
    """Bar chart comparing baseline vs trained scores per task."""
    def _agg(results):
        sums: Dict[str, List[float]] = defaultdict(list)
        for r in results:
            sums[r["task_id"]].append(r["score"])
        return {k: float(np.mean(v)) for k, v in sums.items()}

    baseline_agg = _agg(baseline_results)
    trained_agg = _agg(trained_results)
    tasks = sorted(set(baseline_agg) | set(trained_agg))
    short_names = [t.replace("aero_", "").replace("_", " ").title() for t in tasks]

    x = np.arange(len(tasks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, [baseline_agg.get(t, 0) for t in tasks],
                   width, label="Baseline (untrained)", color="#8B0000", alpha=0.85, edgecolor="black")
    bars2 = ax.bar(x + width / 2, [trained_agg.get(t, 0) for t in tasks],
                   width, label="GRPO-trained", color="#D4AF37", alpha=0.85, edgecolor="black")

    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Grader Score (0-1)", fontsize=12)
    ax.set_title("Baseline vs. GRPO-Trained Model - Task Scores", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=20, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = out_dir / "baseline_vs_trained.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison chart -> {path}")
    return path


def plot_reward_distribution(
    baseline_results: List[Dict[str, Any]],
    trained_results: List[Dict[str, Any]],
    out_dir: Path = PLOTS_DIR,
) -> Path:
    """Histogram of score distributions."""
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist([r["score"] for r in baseline_results], bins=bins, alpha=0.6,
            label="Baseline", color="#8B0000", edgecolor="black")
    ax.hist([r["score"] for r in trained_results], bins=bins, alpha=0.6,
            label="GRPO-trained", color="#D4AF37", edgecolor="black")
    ax.set_xlabel("Grader Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Score Distribution - Baseline vs. GRPO-Trained", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "score_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved distribution plot -> {path}")
    return path


def save_eval_results(
    baseline_results: List[Dict[str, Any]],
    trained_results: List[Dict[str, Any]],
    out_dir: Path = PLOTS_DIR,
) -> Path:
    """Save evaluation results as JSON."""
    data = {
        "baseline": baseline_results,
        "trained": trained_results,
        "summary": {
            "baseline_mean": float(np.mean([r["score"] for r in baseline_results])) if baseline_results else 0,
            "trained_mean": float(np.mean([r["score"] for r in trained_results])) if trained_results else 0,
            "improvement": float(np.mean([r["score"] for r in trained_results]) - np.mean([r["score"] for r in baseline_results])) if baseline_results and trained_results else 0,
        },
    }
    path = out_dir / "eval_results.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"Saved eval results -> {path}")
    return path
