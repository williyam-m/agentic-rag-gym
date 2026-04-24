#!/usr/bin/env python3
"""
train.py — GRPO Fine-Tuning for Agentic RAG Gym (Aerospace Domain)
====================================================================

End-to-end training script that:
1. Connects to the live gym environment to collect prompts
2. Loads Qwen2.5-0.5B-Instruct with LoRA
3. Trains with GRPO (Group Relative Policy Optimization) via TRL
4. Evaluates baseline vs. trained model with domain graders
5. Generates publication-quality plots
6. Pushes the fine-tuned model to Hugging Face Hub

Usage:
    # Start the environment first:
    python main.py &

    # Then run training:
    python train.py

Environment variables (loaded from .env):
    HF_TOKEN          Hugging Face token (for model push)
    HF_USERNAME       Hugging Face username (default: williyam)
    ENV_URL           Gym environment URL (default: http://localhost:7860)
    BASE_MODEL_ID     Base model (default: Qwen/Qwen2.5-0.5B-Instruct)
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch
from dotenv import load_dotenv
from peft import LoraConfig, TaskType

load_dotenv()

from training.config import (
    BASE_MODEL_ID,
    CHECKPOINTS_DIR,
    FINETUNED_MODEL_ID,
    HF_TOKEN,
    PLOTS_DIR,
)
from training.dataset import SYSTEM_PROMPT, build_grpo_dataset
from training.evaluate import (
    evaluate_model_on_tasks,
    plot_baseline_vs_trained,
    plot_reward_distribution,
    plot_training_curves,
    save_eval_results,
)
from training.reward import grade_answer_sync

# ── Device ─────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"Device: {DEVICE}  |  PyTorch {torch.__version__}")

# ── Build dataset ──────────────────────────────────────────────────────
print("\n[1/6] Collecting prompts from environment...")
dataset = build_grpo_dataset(num_per_task=8, seed=42)

task_ids_map = {}
for row in dataset:
    task_ids_map[row["task_id"]] = row["task_name"]

# Format for TRL: prompt column must be list of message dicts
def format_for_trl(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ],
    }

train_dataset = dataset.map(format_for_trl, remove_columns=["task_name", "difficulty"])

# ── Load model ─────────────────────────────────────────────────────────
print(f"\n[2/6] Loading model: {BASE_MODEL_ID}")
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True,
)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)

# ── Baseline evaluation ───────────────────────────────────────────────
print("\n[3/6] Evaluating baseline (before training)...")
from peft import get_peft_model
eval_model = get_peft_model(model.to(DEVICE), peft_config)
eval_model.print_trainable_parameters()

eval_prompts = []
seen = set()
for row in dataset:
    if row["task_id"] not in seen:
        eval_prompts.append(row)
        seen.add(row["task_id"])

baseline_results = evaluate_model_on_tasks(
    eval_model, tokenizer, eval_prompts, max_new_tokens=512, temperature=0.1,
)
baseline_mean = np.mean([r["score"] for r in baseline_results])
print(f"  Baseline mean score: {baseline_mean:.4f}")
del eval_model
model = model.to("cpu")

# ── Reward function ────────────────────────────────────────────────────
def reward_fn(completions, **kwargs):
    """Score completions using domain graders."""
    rewards = []
    prompts = kwargs.get("prompts", [])
    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        text = text.strip()
        if len(text) < 10:
            rewards.append(0.01)
            continue
        task_id = None
        if i < len(prompts):
            p = prompts[i]
            if isinstance(p, list):
                p = " ".join(m.get("content", "") for m in p)
            for tid, name in task_ids_map.items():
                if name in str(p):
                    task_id = tid
                    break
        if task_id is None:
            task_id = list(task_ids_map.keys())[0]
        try:
            rewards.append(float(grade_answer_sync(task_id, text)))
        except Exception:
            rewards.append(0.01)
    return rewards

# ── GRPO Training ──────────────────────────────────────────────────────
print("\n[4/6] Starting GRPO training...")
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    output_dir=str(CHECKPOINTS_DIR / "grpo"),
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    logging_steps=1,
    save_steps=50,
    save_total_limit=2,
    bf16=False,
    fp16=False,
    seed=42,
    remove_unused_columns=False,
    num_generations=4,
    max_completion_length=512,
    temperature=0.7,
    use_vllm=False,
    report_to="none",
    optim="adamw_torch",
    gradient_checkpointing=True,
    log_completions=True,
    num_completions_to_print=1,
)

trainer = GRPOTrainer(
    model=BASE_MODEL_ID,
    reward_funcs=reward_fn,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)

t0 = time.time()
trainer.train()
elapsed = time.time() - t0
print(f"\nTraining completed in {elapsed/60:.1f} min")

# ── Post-training evaluation ──────────────────────────────────────────
print("\n[5/6] Evaluating trained model...")
trained_results = evaluate_model_on_tasks(
    trainer.model, tokenizer, eval_prompts, max_new_tokens=512, temperature=0.1,
)
trained_mean = np.mean([r["score"] for r in trained_results])
print(f"  Trained mean score: {trained_mean:.4f}")
print(f"  Improvement: {trained_mean - baseline_mean:+.4f}")

# ── Plots + save ──────────────────────────────────────────────────────
print("\n[6/6] Generating plots and saving...")
log_history = trainer.state.log_history if hasattr(trainer, "state") else []
plot_training_curves(log_history)
plot_baseline_vs_trained(baseline_results, trained_results)
plot_reward_distribution(baseline_results, trained_results)
save_eval_results(baseline_results, trained_results)

# Push
if HF_TOKEN:
    print(f"\nPushing to HF Hub: {FINETUNED_MODEL_ID}")
    trainer.model.push_to_hub(FINETUNED_MODEL_ID, token=HF_TOKEN, private=False)
    tokenizer.push_to_hub(FINETUNED_MODEL_ID, token=HF_TOKEN, private=False)
    print("Model pushed successfully")

print(f"\n{'='*50}")
print(f"  Base model:     {BASE_MODEL_ID}")
print(f"  Training time:  {elapsed/60:.1f} min")
print(f"  Baseline score: {baseline_mean:.4f}")
print(f"  Trained score:  {trained_mean:.4f}")
print(f"  Delta:          {trained_mean - baseline_mean:+.4f}")
print(f"{'='*50}")
