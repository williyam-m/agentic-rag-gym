"""Rewards-as-a-Service — dynamic reward computation engine."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from agentic_rag_os.models.reward_config import RewardAlgorithm


# ---------------------------------------------------------------------------
# Rule types
# ---------------------------------------------------------------------------

RULE_TYPES = {
    "keyword_match": "Reward when keywords appear in the response",
    "length_range": "Reward responses within a specific token/character range",
    "format_check": "Reward when response matches a format (JSON, markdown, etc.)",
    "no_hallucination": "Penalize when response contains forbidden phrases",
    "citation_present": "Reward when response includes citations/sources",
    "reasoning_steps": "Reward for step-by-step reasoning patterns",
    "custom_regex": "Reward when response matches a custom regex",
}


def _safe_regex(pattern: str, text: str) -> bool:
    """Match pattern against text safely."""
    try:
        return bool(re.search(pattern, text, re.IGNORECASE))
    except re.error:
        return False


def _apply_rule(rule: Dict[str, Any], prompt: str, response: str) -> float:
    """Compute a [0, 1] score for a single rule."""
    rule_type: str = rule.get("type", "keyword_match")
    weight: float = float(rule.get("weight", 1.0))
    score = 0.0

    if rule_type == "keyword_match":
        keywords: List[str] = rule.get("keywords", [])
        if not keywords:
            return 0.0
        matched = sum(1 for kw in keywords if kw.lower() in response.lower())
        score = matched / len(keywords)

    elif rule_type == "length_range":
        min_len: int = rule.get("min_length", 50)
        max_len: int = rule.get("max_length", 2000)
        length = len(response)
        if min_len <= length <= max_len:
            score = 1.0
        elif length < min_len:
            score = length / min_len
        else:
            score = max(0.0, 1.0 - (length - max_len) / max_len)

    elif rule_type == "format_check":
        fmt: str = rule.get("format", "text")
        if fmt == "json":
            try:
                json.loads(response)
                score = 1.0
            except json.JSONDecodeError:
                score = 0.0
        elif fmt == "markdown":
            score = 1.0 if re.search(r"[#*`\-\[\]]", response) else 0.3
        elif fmt == "numbered_list":
            score = 1.0 if re.search(r"^\d+[\.\)]\s", response, re.MULTILINE) else 0.0
        else:
            score = 0.5

    elif rule_type == "no_hallucination":
        forbidden: List[str] = rule.get("forbidden_phrases", [])
        found = any(phrase.lower() in response.lower() for phrase in forbidden)
        score = 0.0 if found else 1.0

    elif rule_type == "citation_present":
        patterns = [r"\[\d+\]", r"\(https?://\S+\)", r"\bsource:\b", r"\bref\b", r"\bcite\b"]
        score = 1.0 if any(_safe_regex(p, response) for p in patterns) else 0.0

    elif rule_type == "reasoning_steps":
        patterns = [r"\bstep \d+\b", r"^\d+[\.\)]\s", r"\bfirst\b.*\bthen\b", r"\btherefore\b"]
        matched = sum(1 for p in patterns if _safe_regex(p, response))
        score = min(1.0, matched / 2)

    elif rule_type == "custom_regex":
        pattern: str = rule.get("pattern", "")
        if pattern:
            score = 1.0 if _safe_regex(pattern, response) else 0.0
        else:
            score = 0.0

    return max(0.0, min(1.0, score * weight))


class RewardsService:
    """Compute rewards for prompt/response pairs given a reward config."""

    def compute_batch(
        self,
        rules: List[Dict[str, Any]],
        weights: Dict[str, float],
        algorithm: str,
        inputs: List[Dict[str, str]],
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Compute rewards for a batch of {prompt, response} pairs.

        Returns (results, avg_reward).
        """
        if not rules:
            raise ValueError("At least one reward rule is required.")
        if not inputs:
            raise ValueError("At least one input pair is required.")

        results: List[Dict[str, Any]] = []
        for item in inputs:
            prompt = item.get("prompt", "")
            response = item.get("response", "")
            breakdown: Dict[str, float] = {}
            total_weight = 0.0
            weighted_score = 0.0

            for rule in rules:
                rule_name = rule.get("name", rule.get("type", "rule"))
                w = float(weights.get(rule_name, rule.get("weight", 1.0)))
                raw_score = _apply_rule(rule, prompt, response)
                breakdown[rule_name] = round(raw_score, 4)
                weighted_score += raw_score * w
                total_weight += w

            base_reward = (weighted_score / total_weight) if total_weight > 0 else 0.0

            # Algorithm-specific normalization
            final_reward = self._normalize(base_reward, algorithm)

            results.append({
                "prompt": prompt[:200],  # truncate for storage
                "response": response[:500],
                "reward": round(final_reward, 4),
                "breakdown": breakdown,
            })

        avg = sum(r["reward"] for r in results) / len(results)
        return results, round(avg, 4)

    def _normalize(self, score: float, algorithm: str) -> float:
        """Apply algorithm-specific reward shaping."""
        # Clamp to (0.01, 0.99) — same convention as rag_master
        score = max(0.01, min(0.99, score))
        if algorithm == RewardAlgorithm.GRPO:
            # GRPO prefers normalized rewards in [0, 1]
            return score
        elif algorithm == RewardAlgorithm.PPO:
            # PPO typically uses advantage-shaped rewards; scale to [-1, 1]
            return (score - 0.5) * 2
        elif algorithm == RewardAlgorithm.DPO:
            # DPO uses log-ratio preference; keep as [0,1] probability-like
            return score
        return score

    def export_hf_dataset(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format results as HuggingFace dataset rows."""
        return [
            {
                "prompt": r["prompt"],
                "response": r["response"],
                "reward": r["reward"],
                "reward_breakdown": r["breakdown"],
            }
            for r in results
        ]


_rewards_service = RewardsService()


def get_rewards_service() -> RewardsService:
    return _rewards_service
