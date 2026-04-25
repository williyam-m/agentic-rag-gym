"""Legal research tasks with graders."""

from __future__ import annotations

from typing import List

from rag_master.models import DifficultyLevel, TaskDefinition

LEGAL_TASKS: List[TaskDefinition] = [
    TaskDefinition(
        task_id="legal_easy_contract_review",
        name="Contract Clause Analysis",
        description=(
            "Analyze the provided contract excerpts and identify key clauses related to "
            "liability limitation, indemnification, and termination. Assess risks for each "
            "clause type, compare common formulations, and suggest improvements. Reference "
            "relevant case law and best practices from the knowledge base."
        ),
        difficulty=DifficultyLevel.EASY,
        max_steps=12,
        success_criteria="Identify main clauses with risk assessment and improvement suggestions",
        grading_rubric={
            "liability_analysis": 0.20,
            "indemnification_analysis": 0.20,
            "termination_analysis": 0.20,
            "risk_assessment": 0.20,
            "recommendations": 0.20,
        },
    ),
    TaskDefinition(
        task_id="legal_easy_privacy_compliance",
        name="Data Privacy Compliance Review",
        description=(
            "Evaluate an organization's data handling practices against GDPR and CCPA/CPRA "
            "requirements. Identify compliance gaps, assess fine exposure, and recommend "
            "remediation steps. Cover key principles like data minimization, consent "
            "management, cross-border transfers, and data subject rights."
        ),
        difficulty=DifficultyLevel.EASY,
        max_steps=12,
        success_criteria="Comprehensive compliance gap analysis with remediation roadmap",
        grading_rubric={
            "gdpr_analysis": 0.25,
            "ccpa_analysis": 0.20,
            "gap_identification": 0.20,
            "fine_exposure": 0.15,
            "recommendations": 0.20,
        },
    ),
    TaskDefinition(
        task_id="legal_medium_ip_analysis",
        name="Intellectual Property Assessment",
        description=(
            "Evaluate the intellectual property landscape for a technology patent application. "
            "Analyze patent eligibility under §101 (Alice framework), conduct prior art "
            "analysis strategy, assess trade secret alternatives, and identify potential "
            "infringement risks. Cross-reference patent law, trade secret protection, and "
            "regulatory frameworks to provide a comprehensive IP strategy."
        ),
        difficulty=DifficultyLevel.MEDIUM,
        max_steps=16,
        success_criteria="Comprehensive IP analysis with prior art review and strategic recommendations",
        grading_rubric={
            "patent_eligibility": 0.20,
            "prior_art_analysis": 0.20,
            "trade_secret_options": 0.15,
            "infringement_risks": 0.20,
            "strategic_recommendation": 0.15,
            "cross_domain_synthesis": 0.10,
        },
    ),
    TaskDefinition(
        task_id="legal_medium_ma_due_diligence",
        name="M&A Due Diligence Assessment",
        description=(
            "Conduct a legal due diligence review for a technology company acquisition. "
            "Analyze corporate governance obligations, material contract risks (change-of-control "
            "provisions, IP ownership), regulatory compliance exposure (AI Act, data privacy), "
            "employment matters (non-compete enforceability), and litigation risks. Provide "
            "risk scoring and recommend deal structure protections."
        ),
        difficulty=DifficultyLevel.MEDIUM,
        max_steps=16,
        success_criteria="Systematic due diligence with risk scoring across all major areas",
        grading_rubric={
            "corporate_governance": 0.15,
            "contract_risks": 0.20,
            "regulatory_compliance": 0.20,
            "employment_matters": 0.15,
            "litigation_exposure": 0.15,
            "deal_recommendations": 0.15,
        },
    ),
    TaskDefinition(
        task_id="legal_hard_cross_border_dispute",
        name="Cross-Border Technology Dispute Resolution",
        description=(
            "Design a comprehensive legal strategy for a cross-border technology dispute "
            "involving patent infringement, trade secret misappropriation, antitrust claims, "
            "and data privacy violations across US, EU, and APAC jurisdictions. Address: "
            "choice of forum (litigation vs. arbitration), applicable law analysis, class "
            "action exposure, regulatory investigation risks, IP protection across jurisdictions, "
            "and settlement strategy. Synthesize contract law, IP law, regulatory compliance, "
            "corporate governance, and international arbitration frameworks."
        ),
        difficulty=DifficultyLevel.HARD,
        max_steps=20,
        success_criteria="Multi-jurisdictional strategy synthesizing 5+ legal domains with quantitative risk analysis",
        grading_rubric={
            "forum_selection": 0.15,
            "ip_strategy": 0.15,
            "antitrust_analysis": 0.15,
            "privacy_compliance": 0.12,
            "class_action_exposure": 0.12,
            "arbitration_strategy": 0.13,
            "cross_domain_synthesis": 0.10,
            "quantitative_analysis": 0.08,
        },
    ),
]


def get_legal_tasks() -> List[TaskDefinition]:
    """Return all legal research tasks."""
    return LEGAL_TASKS
