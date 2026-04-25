"""Legal research knowledge base - curated documents for RAG."""

from __future__ import annotations

from typing import List

from rag_master.models import Document

LEGAL_DOCUMENTS: List[Document] = [
    # --- Contract Law ---
    Document(
        doc_id="contract_001",
        content=(
            "A limitation of liability clause typically caps the total damages recoverable "
            "by either party. Common formulations include caps equal to the fees paid under "
            "the agreement in the preceding 12 months. In Lucent Technologies v. AT&T, the court "
            "upheld a $500,000 liability cap in a $50 million contract. Courts generally enforce "
            "these clauses unless they are unconscionable, involve gross negligence, or willful "
            "misconduct. Under the UCC §2-719, consequential damage exclusions are presumptively "
            "valid in commercial transactions between sophisticated parties."
        ),
        source="Restatement (Second) of Contracts",
        metadata={"topic": "contracts", "subtopic": "liability_limitation", "difficulty": "medium"},
    ),
    Document(
        doc_id="contract_002",
        content=(
            "Indemnification clauses allocate risk between contracting parties by requiring one "
            "party to compensate the other for specified losses. A broad-form indemnity covers "
            "all claims including the indemnitee's own negligence, while an intermediate-form "
            "limits coverage to the indemnitor's proportionate fault. In Bridgestone v. IBM (2014), "
            "the court enforced a mutual indemnification clause awarding $600 million in damages. "
            "Key drafting considerations include carve-outs for IP infringement, data breaches, "
            "confidentiality violations, and willful misconduct. Defense obligations (duty to defend "
            "vs. duty to indemnify) should be explicitly addressed."
        ),
        source="ABA Model Contract Provisions",
        metadata={"topic": "contracts", "subtopic": "indemnification", "difficulty": "medium"},
    ),
    Document(
        doc_id="contract_003",
        content=(
            "Termination clauses define the conditions under which a contract may be ended "
            "before its natural expiration. Common termination triggers include: material breach "
            "(with 30-day cure period), insolvency, change of control, and convenience (typically "
            "requiring 60-90 days written notice). Under Delaware law, the implied covenant of good "
            "faith prevents termination for convenience in bad faith to avoid performance obligations. "
            "Post-termination obligations commonly include: data return/destruction within 30 days, "
            "wind-down services for 90 days, survival of confidentiality for 3-5 years, and "
            "settlement of outstanding payment obligations within 45 days."
        ),
        source="Practical Law Contract Drafting Guide",
        metadata={"topic": "contracts", "subtopic": "termination", "difficulty": "easy"},
    ),
    Document(
        doc_id="contract_004",
        content=(
            "Force majeure clauses excuse performance when extraordinary events beyond the parties' "
            "control prevent fulfillment. Post-COVID litigation expanded judicial interpretation: "
            "in Hess Corp v. Port Authority (2021), the court held that pandemic-related government "
            "shutdowns triggered force majeure only when the clause specifically listed 'epidemic' "
            "or 'government action.' Best practices include enumerated events (natural disasters, "
            "pandemics, government orders, cyberattacks, supply chain disruptions) plus a catch-all "
            "provision. The affected party must typically provide notice within 5-10 business days "
            "and demonstrate mitigation efforts. Prolonged force majeure (>180 days) often triggers "
            "termination rights for either party."
        ),
        source="Harvard Law Review - Contract Impossibility",
        metadata={"topic": "contracts", "subtopic": "force_majeure", "difficulty": "medium"},
    ),

    # --- Intellectual Property ---
    Document(
        doc_id="ip_001",
        content=(
            "Patent eligibility under 35 U.S.C. §101 requires that an invention be a process, "
            "machine, manufacture, or composition of matter. Following Alice Corp v. CLS Bank (2014), "
            "the Supreme Court established a two-step test: (1) determine if the claims are directed "
            "to an abstract idea, law of nature, or natural phenomenon; (2) if so, search for an "
            "'inventive concept' that transforms the claim into a patent-eligible application. "
            "For software patents, approximately 63% of §101 challenges succeed at the district "
            "court level post-Alice. Key strategies include claiming specific technical improvements, "
            "reciting particular hardware implementations, and demonstrating non-conventional "
            "combinations of known elements."
        ),
        source="USPTO Patent Eligibility Guidance",
        metadata={"topic": "ip", "subtopic": "patent_eligibility", "difficulty": "hard"},
    ),
    Document(
        doc_id="ip_002",
        content=(
            "Prior art searches for patent applications should cover both patent and non-patent "
            "literature. The USPTO uses the Cooperative Patent Classification (CPC) system to "
            "organize patents by technology area. A thorough prior art search includes: keyword "
            "searches across USPTO, EPO (Espacenet), WIPO (PatentScope), Google Patents; citation "
            "analysis of forward and backward references; review of related patent families across "
            "major jurisdictions (US, EU, CN, JP, KR). The novelty requirement under 35 U.S.C. §102 "
            "bars patents on inventions that were patented, described in a printed publication, or "
            "in public use before the effective filing date. Non-obvious under §103 requires that "
            "the invention would not have been obvious to a person of ordinary skill in the art (PHOSITA)."
        ),
        source="MPEP Chapter 2100 - Patentability",
        metadata={"topic": "ip", "subtopic": "prior_art", "difficulty": "medium"},
    ),
    Document(
        doc_id="ip_003",
        content=(
            "Trade secret protection under the Defend Trade Secrets Act (DTSA, 2016) and the "
            "Uniform Trade Secrets Act (UTSA) requires: (1) the information derives independent "
            "economic value from not being generally known; (2) the owner takes reasonable measures "
            "to maintain secrecy. In Waymo v. Uber (2018), the court issued a preliminary injunction "
            "after finding misappropriation of autonomous vehicle trade secrets valued at $1.86 billion. "
            "Reasonable protective measures include: NDAs with specific duration (typically 3-5 years), "
            "access controls (need-to-know basis), employee exit interviews, encryption of digital "
            "assets, and marking documents as 'CONFIDENTIAL.' The DTSA provides federal jurisdiction "
            "and allows ex parte seizure orders in extraordinary circumstances."
        ),
        source="Defend Trade Secrets Act Analysis",
        metadata={"topic": "ip", "subtopic": "trade_secrets", "difficulty": "medium"},
    ),

    # --- Regulatory Compliance ---
    Document(
        doc_id="reg_001",
        content=(
            "The General Data Protection Regulation (GDPR) imposes strict requirements on "
            "organizations processing personal data of EU residents. Key principles include: "
            "lawfulness, fairness, and transparency (Art. 5(1)(a)); purpose limitation (Art. 5(1)(b)); "
            "data minimization (Art. 5(1)(c)); and storage limitation (Art. 5(1)(e)). Maximum fines "
            "reach €20 million or 4% of annual global turnover, whichever is higher. In 2023, Meta "
            "was fined €1.2 billion for unlawful data transfers to the US. Data Protection Impact "
            "Assessments (DPIAs) are mandatory under Art. 35 for high-risk processing activities "
            "including systematic monitoring, large-scale processing of sensitive data, and automated "
            "decision-making with legal effects."
        ),
        source="EU GDPR Official Text and Guidance",
        metadata={"topic": "regulatory", "subtopic": "gdpr", "difficulty": "medium"},
    ),
    Document(
        doc_id="reg_002",
        content=(
            "The California Consumer Privacy Act (CCPA), as amended by the CPRA (effective 2023), "
            "grants California residents rights including: right to know what personal information "
            "is collected (§1798.100); right to delete (§1798.105); right to opt-out of sale/sharing "
            "(§1798.120); right to correct inaccurate data (§1798.106); and right to limit use of "
            "sensitive personal information (§1798.121). Businesses with >$25 million annual revenue, "
            "data on 100,000+ consumers, or deriving 50%+ revenue from selling data must comply. "
            "The California Privacy Protection Agency (CPPA) can impose fines of $2,500 per violation "
            "or $7,500 per intentional violation. Cross-border data transfer mechanisms must ensure "
            "substantially equivalent protection."
        ),
        source="California Privacy Rights Act Text",
        metadata={"topic": "regulatory", "subtopic": "ccpa_cpra", "difficulty": "medium"},
    ),
    Document(
        doc_id="reg_003",
        content=(
            "The EU AI Act (2024) establishes a risk-based regulatory framework for artificial "
            "intelligence. Risk categories include: Unacceptable (banned) — social scoring, real-time "
            "biometric identification in public spaces; High-risk — critical infrastructure, employment "
            "decisions, law enforcement, border control; Limited risk — chatbots, emotion recognition "
            "(transparency obligations); Minimal risk — spam filters, video games (no obligations). "
            "High-risk AI systems must implement: risk management systems, data governance, technical "
            "documentation, record-keeping, transparency to users, human oversight, accuracy/robustness "
            "requirements. Fines reach €35 million or 7% of global annual turnover for prohibited "
            "practices. General-purpose AI models must meet transparency and copyright compliance obligations."
        ),
        source="EU Artificial Intelligence Act",
        metadata={"topic": "regulatory", "subtopic": "ai_regulation", "difficulty": "hard"},
    ),

    # --- Case Law & Litigation ---
    Document(
        doc_id="case_001",
        content=(
            "In the landmark antitrust case United States v. Google LLC (2024), Judge Amit Mehta "
            "ruled that Google maintained an illegal monopoly in the search market by paying $26.3 "
            "billion annually to be the default search engine on browsers and devices (Apple, Samsung, "
            "Mozilla). The court applied the Sherman Act §2 framework: (1) possession of monopoly "
            "power in a relevant market (>89.2% of general search); (2) willful acquisition or "
            "maintenance of that power through anticompetitive conduct. The ruling could reshape "
            "Big Tech's distribution agreements. Remedies under consideration include structural "
            "separation, mandatory API access, and prohibition of exclusive default agreements."
        ),
        source="US District Court DC - Antitrust Division",
        metadata={"topic": "litigation", "subtopic": "antitrust", "difficulty": "hard"},
    ),
    Document(
        doc_id="case_002",
        content=(
            "Class action certification under Federal Rule of Civil Procedure 23 requires: "
            "numerosity (class so large that joinder is impracticable, typically 40+ members); "
            "commonality (common questions of law or fact, per Wal-Mart v. Dukes, 2011); typicality "
            "(named plaintiffs' claims are typical of the class); and adequacy (representative parties "
            "fairly and adequately protect class interests). Under Rule 23(b)(3), plaintiffs must "
            "additionally show that common questions predominate over individual ones and that class "
            "action is superior to other methods. In data breach class actions, courts increasingly "
            "require Article III standing showing concrete harm beyond mere data exposure, per "
            "TransUnion v. Ramirez (2021) requiring actual downstream consequences."
        ),
        source="Federal Rules of Civil Procedure - Advisory Notes",
        metadata={"topic": "litigation", "subtopic": "class_action", "difficulty": "hard"},
    ),

    # --- Corporate Governance ---
    Document(
        doc_id="corp_001",
        content=(
            "Fiduciary duties of corporate directors under Delaware law include the duty of care "
            "(informed business judgment) and the duty of loyalty (no self-dealing or conflicts of "
            "interest). The business judgment rule presumes that directors acted on an informed basis, "
            "in good faith, and in the honest belief that the action was in the company's best interest "
            "(Aronson v. Lewis, 1984). However, Caremark liability (In re Caremark International, 1996) "
            "holds directors liable for sustained failure to implement monitoring systems — requiring "
            "boards to ensure compliance programs, risk management frameworks, and reporting mechanisms. "
            "The Revlon duty applies during change-of-control transactions, requiring directors to "
            "maximize shareholder value by obtaining the highest reasonably available price."
        ),
        source="Delaware Chancery Court Decisions",
        metadata={"topic": "corporate", "subtopic": "fiduciary_duties", "difficulty": "medium"},
    ),

    # --- Employment Law ---
    Document(
        doc_id="employ_001",
        content=(
            "Non-compete agreements face increasing regulatory scrutiny. The FTC's 2024 rule banning "
            "most non-competes affects approximately 30 million workers, though enforcement is stayed "
            "pending litigation (Ryan LLC v. FTC). California Business and Professions Code §16600 "
            "voids non-competes entirely, with narrow exceptions for business sale contexts. "
            "Enforceable non-competes in other jurisdictions typically require: reasonable geographic "
            "scope (within the employer's market area), reasonable duration (6-24 months), protection "
            "of legitimate business interests (trade secrets, customer relationships), and adequate "
            "consideration (continued employment may suffice in at-will states). Garden leave clauses "
            "providing full salary during the restricted period are increasingly preferred as a "
            "more enforceable alternative."
        ),
        source="Employment Law Analysis - Non-Compete Trends",
        metadata={"topic": "employment", "subtopic": "non_compete", "difficulty": "medium"},
    ),

    # --- International Trade ---
    Document(
        doc_id="trade_001",
        content=(
            "International commercial arbitration under the New York Convention (1958) provides "
            "enforcement of arbitral awards in 172 signatory countries. Major arbitration institutions "
            "include ICC (Paris), LCIA (London), SIAC (Singapore), and HKIAC (Hong Kong). The average "
            "ICC arbitration takes 26 months with costs averaging $1.5 million for claims of $10-50 "
            "million. Key advantages over litigation: neutral forum, enforceability, confidentiality, "
            "party autonomy in selecting arbitrators. The UNCITRAL Model Law harmonizes procedural "
            "standards across jurisdictions. In investor-state disputes (ISDS), claims under bilateral "
            "investment treaties averaged $1.2 billion in 2023, with approximately 37% of cases "
            "resulting in awards favoring investors."
        ),
        source="ICC Dispute Resolution Statistics",
        metadata={"topic": "international", "subtopic": "arbitration", "difficulty": "hard"},
    ),

    # --- M&A ---
    Document(
        doc_id="ma_001",
        content=(
            "Merger and acquisition due diligence encompasses legal, financial, operational, and "
            "regulatory review. Legal due diligence focuses on: corporate structure and authority, "
            "material contracts (change-of-control provisions, assignment restrictions, key customer "
            "agreements), litigation exposure (pending and threatened claims), IP portfolio (ownership, "
            "encumbrances, freedom-to-operate), regulatory compliance, employment matters (key "
            "employee agreements, benefit liabilities), and real estate. Representation and warranty "
            "insurance (RWI) covers breaches discovered post-closing, with premiums typically 2-4% "
            "of coverage limits. Escrow holdbacks of 5-15% of purchase price for 12-18 months "
            "remain common even with RWI. Material adverse change (MAC) clauses, scrutinized in "
            "Akorn v. Fresenius (2018), allow buyers to terminate for significant undisclosed deterioration."
        ),
        source="M&A Transaction Handbook",
        metadata={"topic": "ma", "subtopic": "due_diligence", "difficulty": "hard"},
    ),
]


def get_legal_documents() -> List[Document]:
    """Return all legal research documents."""
    return LEGAL_DOCUMENTS
