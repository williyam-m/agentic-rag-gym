"""Legal domain graders with deterministic scoring and anti-hack measures."""

from __future__ import annotations

from typing import Dict

from rag_master.adapters import BaseGrader
from rag_master.logging_config import get_logger

from domains.aerospace.graders import KeywordCoverageGrader

logger = get_logger(__name__)


class ContractReviewGrader(KeywordCoverageGrader):
    """Grader for the contract clause analysis task."""

    def __init__(self) -> None:
        super().__init__(
            required_keywords={
                "liability_analysis": [
                    "limitation of liability", "cap", "damages", "fees paid",
                    "12 months", "consequential", "ucc", "unconscionable",
                ],
                "indemnification_analysis": [
                    "indemnification", "indemnify", "broad-form", "intermediate",
                    "ip infringement", "data breach", "defense", "duty",
                ],
                "termination_analysis": [
                    "termination", "material breach", "cure period", "convenience",
                    "30 days", "90 days", "wind-down", "post-termination",
                ],
                "risk_assessment": [
                    "risk", "exposure", "enforce", "court", "delaware", "good faith",
                ],
                "recommendations": [
                    "recommend", "suggest", "improve", "best practice", "draft",
                ],
            },
            rubric_weights={
                "liability_analysis": 0.20,
                "indemnification_analysis": 0.20,
                "termination_analysis": 0.20,
                "risk_assessment": 0.20,
                "recommendations": 0.20,
            },
        )


class PrivacyComplianceGrader(KeywordCoverageGrader):
    """Grader for the data privacy compliance review task."""

    def __init__(self) -> None:
        super().__init__(
            required_keywords={
                "gdpr_analysis": [
                    "gdpr", "personal data", "data minimization", "lawfulness",
                    "consent", "dpia", "article", "€20 million", "4%",
                ],
                "ccpa_analysis": [
                    "ccpa", "cpra", "california", "opt-out", "right to know",
                    "right to delete", "$7,500", "sensitive personal",
                ],
                "gap_identification": [
                    "gap", "compliance", "deficiency", "non-compliant", "missing",
                ],
                "fine_exposure": [
                    "fine", "penalty", "million", "turnover", "violation", "enforcement",
                ],
                "recommendations": [
                    "recommend", "remediat", "implement", "policy", "framework",
                ],
            },
            rubric_weights={
                "gdpr_analysis": 0.25,
                "ccpa_analysis": 0.20,
                "gap_identification": 0.20,
                "fine_exposure": 0.15,
                "recommendations": 0.20,
            },
        )


class IPAnalysisGrader(KeywordCoverageGrader):
    """Grader for the intellectual property assessment task."""

    def __init__(self) -> None:
        super().__init__(
            required_keywords={
                "patent_eligibility": [
                    "patent", "§101", "alice", "abstract idea", "inventive concept",
                    "eligible", "two-step", "software",
                ],
                "prior_art_analysis": [
                    "prior art", "novelty", "§102", "obvious", "§103",
                    "phosita", "search", "cpc", "classification",
                ],
                "trade_secret_options": [
                    "trade secret", "dtsa", "reasonable measures", "nda",
                    "confidential", "economic value",
                ],
                "infringement_risks": [
                    "infringement", "freedom to operate", "risk", "claim",
                    "jurisdiction", "patent family",
                ],
                "strategic_recommendation": [
                    "recommend", "strategy", "protect", "portfolio", "filing",
                ],
                "cross_domain_synthesis": [
                    "combined", "comprehensive", "integrate", "regulatory", "ai act",
                ],
            },
            rubric_weights={
                "patent_eligibility": 0.20,
                "prior_art_analysis": 0.20,
                "trade_secret_options": 0.15,
                "infringement_risks": 0.20,
                "strategic_recommendation": 0.15,
                "cross_domain_synthesis": 0.10,
            },
        )


class MADueDiligenceGrader(KeywordCoverageGrader):
    """Grader for the M&A due diligence assessment task."""

    def __init__(self) -> None:
        super().__init__(
            required_keywords={
                "corporate_governance": [
                    "fiduciary", "duty of care", "duty of loyalty", "business judgment",
                    "caremark", "board", "director",
                ],
                "contract_risks": [
                    "change of control", "material contract", "assignment",
                    "mac", "material adverse", "representation", "warranty",
                ],
                "regulatory_compliance": [
                    "gdpr", "ai act", "regulatory", "compliance", "data privacy",
                    "high-risk", "fine",
                ],
                "employment_matters": [
                    "non-compete", "key employee", "garden leave", "enforceable",
                    "california", "consideration",
                ],
                "litigation_exposure": [
                    "litigation", "pending", "class action", "antitrust",
                    "standing", "damages",
                ],
                "deal_recommendations": [
                    "escrow", "holdback", "rwi", "insurance", "indemnification",
                    "recommend", "structure",
                ],
            },
            rubric_weights={
                "corporate_governance": 0.15,
                "contract_risks": 0.20,
                "regulatory_compliance": 0.20,
                "employment_matters": 0.15,
                "litigation_exposure": 0.15,
                "deal_recommendations": 0.15,
            },
        )


class CrossBorderDisputeGrader(KeywordCoverageGrader):
    """Grader for the cross-border technology dispute resolution task."""

    def __init__(self) -> None:
        super().__init__(
            required_keywords={
                "forum_selection": [
                    "forum", "jurisdiction", "venue", "litigation", "arbitration",
                    "icc", "new york convention",
                ],
                "ip_strategy": [
                    "patent", "trade secret", "infringement", "injunction",
                    "dtsa", "preliminary",
                ],
                "antitrust_analysis": [
                    "antitrust", "sherman", "monopoly", "anticompetitive",
                    "market", "dominant",
                ],
                "privacy_compliance": [
                    "gdpr", "ccpa", "data transfer", "cross-border",
                    "privacy", "personal data",
                ],
                "class_action_exposure": [
                    "class action", "rule 23", "numerosity", "commonality",
                    "certification", "standing",
                ],
                "arbitration_strategy": [
                    "arbitration", "arbitral", "enforcement", "award",
                    "confidential", "neutral forum",
                ],
                "cross_domain_synthesis": [
                    "integrated", "comprehensive", "multi-jurisdictional",
                    "combined", "strategy", "holistic",
                ],
                "quantitative_analysis": [
                    "million", "billion", "cost", "damages", "exposure",
                    "months", "percentage",
                ],
            },
            rubric_weights={
                "forum_selection": 0.15,
                "ip_strategy": 0.15,
                "antitrust_analysis": 0.15,
                "privacy_compliance": 0.12,
                "class_action_exposure": 0.12,
                "arbitration_strategy": 0.13,
                "cross_domain_synthesis": 0.10,
                "quantitative_analysis": 0.08,
            },
        )


# Registry of graders by task ID
GRADER_REGISTRY: Dict[str, type] = {
    "legal_easy_contract_review": ContractReviewGrader,
    "legal_easy_privacy_compliance": PrivacyComplianceGrader,
    "legal_medium_ip_analysis": IPAnalysisGrader,
    "legal_medium_ma_due_diligence": MADueDiligenceGrader,
    "legal_hard_cross_border_dispute": CrossBorderDisputeGrader,
}


def get_grader(task_id: str) -> BaseGrader:
    """Get a grader instance for the given task ID."""
    grader_cls = GRADER_REGISTRY.get(task_id)
    if grader_cls is None:
        raise ValueError(f"No grader registered for task: {task_id}")
    return grader_cls()
