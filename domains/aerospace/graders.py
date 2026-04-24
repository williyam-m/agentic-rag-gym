"""Aerospace domain graders with deterministic scoring."""

from __future__ import annotations

import re
from typing import Dict, List, Set

from rag_master.adapters import BaseGrader
from rag_master.logging_config import get_logger
from rag_master.models import EpisodeState, Trajectory
from rag_master.rewards import _SCORE_MAX, _SCORE_MIN, clamp_score

logger = get_logger(__name__)


class KeywordCoverageGrader(BaseGrader):
    """Grades based on keyword coverage from expected topics."""

    def __init__(
        self,
        required_keywords: Dict[str, List[str]],
        rubric_weights: Dict[str, float],
    ) -> None:
        self._required_keywords = required_keywords
        self._rubric_weights = rubric_weights

    async def grade(self, state: EpisodeState, trajectory: Trajectory) -> float:
        """Grade based on keyword coverage in the answer."""
        answer = state.generated_answer.lower()
        if not answer or len(answer.strip()) < 20:
            return _SCORE_MIN

        total_score = 0.0
        for category, keywords in self._required_keywords.items():
            weight = self._rubric_weights.get(category, 0.0)
            if not keywords:
                continue
            hits = sum(1 for kw in keywords if kw.lower() in answer)
            coverage = hits / len(keywords)
            total_score += weight * coverage

        # Process quality bonus
        process_bonus = self._evaluate_process(trajectory)
        total_score = total_score * 0.8 + process_bonus * 0.2

        return clamp_score(total_score)

    def _evaluate_process(self, trajectory: Trajectory) -> float:
        """Evaluate quality of the reasoning process."""
        if not trajectory.steps:
            return 0.1

        action_types = [s.action_type for s in trajectory.steps]
        score = 0.2

        # Reward diverse action types
        unique_actions = len(set(action_types))
        if unique_actions >= 3:
            score += 0.3
        elif unique_actions >= 2:
            score += 0.15

        # Reward retrieval before answering
        if "retrieve" in action_types:
            retrieve_idx = action_types.index("retrieve")
            if "answer" in action_types:
                answer_idx = action_types.index("answer")
                if retrieve_idx < answer_idx:
                    score += 0.2

        # Reward reasoning steps
        reasoning_steps = sum(1 for a in action_types if a == "reason")
        if reasoning_steps >= 1:
            score += 0.15

        # Penalize excessive steps
        if len(trajectory.steps) > 15:
            score -= 0.1

        return min(max(score, 0.0), 0.95)


class PropulsionComparisonGrader(KeywordCoverageGrader):
    """Grader for the propulsion comparison task."""

    def __init__(self) -> None:
        super().__init__(
            required_keywords={
                "specific_impulse_comparison": [
                    "specific impulse", "4190", "4,190", "850", "1000", "1,000", "seconds"
                ],
                "thrust_comparison": [
                    "thrust", "0.5", "newton", "low thrust", "high thrust"
                ],
                "mission_duration": [
                    "mars", "transit", "month", "9 month", "4 month", "duration"
                ],
                "technology_readiness": [
                    "dawn", "draco", "darpa", "nasa", "technology", "demonstrated"
                ],
                "recommendation": [
                    "recommend", "better", "prefer", "advantage", "suitable", "crewed"
                ],
                "source_grounding": [
                    "xenon", "next", "nuclear thermal", "haleu", "hydrogen", "reactor"
                ],
            },
            rubric_weights={
                "specific_impulse_comparison": 0.20,
                "thrust_comparison": 0.15,
                "mission_duration": 0.20,
                "technology_readiness": 0.15,
                "recommendation": 0.15,
                "source_grounding": 0.15,
            },
        )


class DebrisMitigationGrader(KeywordCoverageGrader):
    """Grader for the space debris mitigation task."""

    def __init__(self) -> None:
        super().__init__(
            required_keywords={
                "kessler_syndrome": [
                    "kessler", "cascade", "collision", "density", "threshold"
                ],
                "tracking_data": [
                    "36,500", "36500", "track", "leo", "10 cm", "esa"
                ],
                "adr_technologies": [
                    "electrodynamic", "tether", "laser", "ablation", "robotic", "capture",
                    "clearspace"
                ],
                "cost_analysis": [
                    "cost", "million", "billion", "500", "economic"
                ],
                "recommendations": [
                    "recommend", "priority", "approach", "effective", "implement"
                ],
            },
            rubric_weights={
                "kessler_syndrome": 0.20,
                "tracking_data": 0.15,
                "adr_technologies": 0.25,
                "cost_analysis": 0.20,
                "recommendations": 0.20,
            },
        )


class MarsEDLGrader(KeywordCoverageGrader):
    """Grader for the Mars EDL architecture design task."""

    def __init__(self) -> None:
        super().__init__(
            required_keywords={
                "atmosphere_analysis": [
                    "atmosphere", "mars paradox", "thin", "heating", "deceleration"
                ],
                "edl_sequence": [
                    "entry", "descent", "landing", "aerocapture", "aerobraking",
                    "retropropulsion", "sequence"
                ],
                "thermal_protection": [
                    "heat shield", "pica", "ablative", "heat flux", "w/cm", "thermal"
                ],
                "propulsion_integration": [
                    "supersonic retro", "propulsion", "thrust", "falcon"
                ],
                "navigation": [
                    "terrain", "navigation", "lidar", "landing accuracy", "perseverance"
                ],
                "cross_domain_synthesis": [
                    "integrate", "combined", "system", "architecture", "design"
                ],
            },
            rubric_weights={
                "atmosphere_analysis": 0.15,
                "edl_sequence": 0.20,
                "thermal_protection": 0.20,
                "propulsion_integration": 0.15,
                "navigation": 0.15,
                "cross_domain_synthesis": 0.15,
            },
        )


class LifeSupportGrader(KeywordCoverageGrader):
    """Grader for the deep space life support design task."""

    def __init__(self) -> None:
        super().__init__(
            required_keywords={
                "water_recovery": [
                    "water", "recovery", "90%", "98%", "distillation", "vcd", "recycle"
                ],
                "oxygen_generation": [
                    "oxygen", "electrolysis", "ogs", "5.4", "o2"
                ],
                "co2_management": [
                    "co2", "carbon dioxide", "sabatier", "methane"
                ],
                "food_production": [
                    "food", "plant", "hydroponic", "bioregenerative", "melissa", "caloric"
                ],
                "radiation_protection": [
                    "radiation", "gcr", "shielding", "polyethylene", "msv", "dose"
                ],
                "mass_budget": [
                    "mass", "kg", "power", "budget", "requirement"
                ],
                "integration": [
                    "integrated", "combined", "system", "closure", "loop"
                ],
            },
            rubric_weights={
                "water_recovery": 0.18,
                "oxygen_generation": 0.15,
                "co2_management": 0.15,
                "food_production": 0.15,
                "radiation_protection": 0.15,
                "mass_budget": 0.12,
                "integration": 0.10,
            },
        )


class HypersonicVehicleGrader(KeywordCoverageGrader):
    """Grader for the reusable hypersonic vehicle design task."""

    def __init__(self) -> None:
        super().__init__(
            required_keywords={
                "propulsion_architecture": [
                    "combined cycle", "turbine", "scramjet", "rocket",
                    "mode transition", "mach", "x-51"
                ],
                "thermal_protection": [
                    "thermal", "heat", "temperature", "3000", "ablative",
                    "cooling", "leading edge"
                ],
                "materials_selection": [
                    "cmc", "silicon carbide", "uhtc", "hafnium", "ceramic",
                    "composite", "superalloy"
                ],
                "aerothermodynamics": [
                    "shock", "boundary layer", "hypersonic", "aerotherm",
                    "mach 5", "thermal choking"
                ],
                "orbital_mechanics": [
                    "orbit", "insertion", "delta-v", "trajectory", "access"
                ],
                "autonomy_integration": [
                    "autonomous", "flight management", "fdir", "fault",
                    "onboard", "control"
                ],
                "cross_domain_synthesis": [
                    "integrate", "system", "architecture", "design",
                    "combined", "comprehensive"
                ],
                "quantitative_analysis": [
                    "km/s", "m/s", "kg", "celsius", "efficiency",
                    "percent", "specific impulse"
                ],
            },
            rubric_weights={
                "propulsion_architecture": 0.18,
                "thermal_protection": 0.15,
                "materials_selection": 0.12,
                "aerothermodynamics": 0.12,
                "orbital_mechanics": 0.10,
                "autonomy_integration": 0.10,
                "cross_domain_synthesis": 0.13,
                "quantitative_analysis": 0.10,
            },
        )


# Registry of graders by task ID
GRADER_REGISTRY: Dict[str, type] = {
    "aero_easy_propulsion_comparison": PropulsionComparisonGrader,
    "aero_easy_debris_mitigation": DebrisMitigationGrader,
    "aero_medium_mars_edl": MarsEDLGrader,
    "aero_medium_life_support": LifeSupportGrader,
    "aero_hard_hypersonic_vehicle": HypersonicVehicleGrader,
}


def get_grader(task_id: str) -> BaseGrader:
    """Get a grader instance for the given task ID."""
    grader_cls = GRADER_REGISTRY.get(task_id)
    if grader_cls is None:
        raise ValueError(f"No grader registered for task: {task_id}")
    return grader_cls()
