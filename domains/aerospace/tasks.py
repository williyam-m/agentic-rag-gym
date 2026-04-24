"""Aerospace research tasks with graders."""

from __future__ import annotations

from typing import List

from rag_master.models import DifficultyLevel, TaskDefinition

AEROSPACE_TASKS: List[TaskDefinition] = [
    TaskDefinition(
        task_id="aero_easy_propulsion_comparison",
        name="Compare Propulsion Technologies",
        description=(
            "Compare ion propulsion and nuclear thermal propulsion for a Mars transit mission. "
            "Your analysis should cover: specific impulse, thrust levels, mission duration impact, "
            "technology readiness, and recommend which is better for a crewed Mars mission. "
            "Use the available research documents to support your analysis."
        ),
        difficulty=DifficultyLevel.EASY,
        max_steps=12,
        success_criteria="Accurate comparison with specific numbers from source documents",
        grading_rubric={
            "specific_impulse_comparison": 0.20,
            "thrust_comparison": 0.15,
            "mission_duration": 0.20,
            "technology_readiness": 0.15,
            "recommendation": 0.15,
            "source_grounding": 0.15,
        },
    ),
    TaskDefinition(
        task_id="aero_easy_debris_mitigation",
        name="Space Debris Mitigation Strategies",
        description=(
            "Analyze current space debris challenges and evaluate active debris removal (ADR) "
            "technologies. Cover the Kessler syndrome risk, current tracking capabilities, "
            "proposed removal methods, and provide cost-effectiveness assessment. "
            "Reference specific data from the knowledge base."
        ),
        difficulty=DifficultyLevel.EASY,
        max_steps=12,
        success_criteria="Comprehensive debris analysis with specific ADR technology comparison",
        grading_rubric={
            "kessler_syndrome": 0.20,
            "tracking_data": 0.15,
            "adr_technologies": 0.25,
            "cost_analysis": 0.20,
            "recommendations": 0.20,
        },
    ),
    TaskDefinition(
        task_id="aero_medium_mars_edl",
        name="Mars EDL Architecture Design",
        description=(
            "Design a complete Entry, Descent, and Landing (EDL) architecture for a 25-metric-ton "
            "payload delivery to Mars surface. Address the Mars atmosphere paradox, propose a "
            "multi-phase EDL sequence, specify thermal protection requirements, and integrate "
            "terrain-relative navigation. Cross-reference propulsion, materials, and thermal "
            "protection documents to create a coherent design."
        ),
        difficulty=DifficultyLevel.MEDIUM,
        max_steps=16,
        success_criteria="Coherent EDL architecture integrating multiple technology domains",
        grading_rubric={
            "atmosphere_analysis": 0.15,
            "edl_sequence": 0.20,
            "thermal_protection": 0.20,
            "propulsion_integration": 0.15,
            "navigation": 0.15,
            "cross_domain_synthesis": 0.15,
        },
    ),
    TaskDefinition(
        task_id="aero_medium_life_support",
        name="Deep Space Life Support Design",
        description=(
            "Design an integrated life support system for a 6-person crew on a 2.5-year Mars "
            "round-trip mission. Combine physicochemical and bioregenerative systems to maximize "
            "closure ratios. Address water recycling, oxygen generation, CO2 removal, food "
            "production, and radiation protection. Calculate mass budgets and power requirements "
            "using data from the knowledge base."
        ),
        difficulty=DifficultyLevel.MEDIUM,
        max_steps=16,
        success_criteria="Integrated system design with quantitative mass and power analysis",
        grading_rubric={
            "water_recovery": 0.18,
            "oxygen_generation": 0.15,
            "co2_management": 0.15,
            "food_production": 0.15,
            "radiation_protection": 0.15,
            "mass_budget": 0.12,
            "integration": 0.10,
        },
    ),
    TaskDefinition(
        task_id="aero_hard_hypersonic_vehicle",
        name="Reusable Hypersonic Space Access Vehicle",
        description=(
            "Design a conceptual reusable hypersonic space access vehicle using combined-cycle "
            "propulsion (turbine + scramjet + rocket). Address: the propulsion mode transitions, "
            "thermal protection system for sustained Mach 5-10 flight, structural materials "
            "selection (CMCs, UHTCs), aerothermodynamic challenges, orbital insertion strategy, "
            "and autonomous flight management. This requires synthesizing information from "
            "propulsion, materials, hypersonics, thermal protection, autonomy, and manufacturing "
            "documents. Provide specific performance estimates grounded in the knowledge base."
        ),
        difficulty=DifficultyLevel.HARD,
        max_steps=20,
        success_criteria="Comprehensive vehicle design synthesizing 5+ technical domains with quantitative analysis",
        grading_rubric={
            "propulsion_architecture": 0.18,
            "thermal_protection": 0.15,
            "materials_selection": 0.12,
            "aerothermodynamics": 0.12,
            "orbital_mechanics": 0.10,
            "autonomy_integration": 0.10,
            "cross_domain_synthesis": 0.13,
            "quantitative_analysis": 0.10,
        },
    ),
]


def get_aerospace_tasks() -> List[TaskDefinition]:
    """Return all aerospace research tasks."""
    return AEROSPACE_TASKS
