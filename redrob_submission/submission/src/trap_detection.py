"""
trap_detection.py
Detects and penalises:
  1. Keyword stuffers (skills claimed without evidence in career history)
  2. Fake expertise (expert claimed but assessment scores are low)
  3. Honeypot candidates (impossible profiles, e.g. overlapping jobs)
  4. Pure consulting (entire career at services companies)
  5. Research-only backgrounds (no production deployments)
  6. Domain mismatch (CV / robotics / hardware without NLP crossover)
  7. Inflated signals (impossible engagement ratios)

Returns a TrapResult with a total penalty [0, 1] and a list of reasons.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from src.config import (
    CONSULTING_ONLY_COMPANIES,
    HONEYPOT_MAX_SIMULTANEOUS_JOBS,
    HONEYPOT_MIN_ASSESSMENT_VS_CLAIM,
    MUST_HAVE_SKILL_GROUPS,
    PENALTY_FAKE_EXPERTISE,
    PENALTY_HONEYPOT,
    PENALTY_INACTIVE,
    PENALTY_KEYWORD_STUFFING,
    PENALTY_NO_PRODUCTION,
    PENALTY_OUTSIDER_DOMAIN,
    PENALTY_PURE_CONSULTING,
    PENALTY_PURE_RESEARCH,
    SIGNAL_FRAUD_MIN_COMPLETENESS,
    SIGNAL_FRAUD_SAVED_VS_VIEWS_RATIO,
    STUFFING_MAX_SKILLS_PER_CATEGORY,
    STUFFING_MIN_DURATION_MONTHS,
    STUFFING_MIN_ENDORSEMENTS,
)
from src.utils import (
    check_date_overlap,
    clamp,
    contains_any,
    get_career_text,
    normalise,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# RESULT DATA CLASS
# ─────────────────────────────────────────────

@dataclass
class TrapResult:
    penalty: float = 0.0
    flags: List[str] = field(default_factory=list)
    is_honeypot: bool = False

    def add(self, penalty: float, reason: str) -> None:
        self.penalty = clamp(self.penalty + penalty)
        self.flags.append(reason)

    def mark_honeypot(self, reason: str) -> None:
        self.is_honeypot = True
        self.add(PENALTY_HONEYPOT, f"HONEYPOT: {reason}")


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def detect_traps(candidate: Dict) -> TrapResult:
    """
    Run all trap detectors. Returns TrapResult with cumulative penalty.
    """
    result = TrapResult()

    _check_honeypot(candidate, result)
    _check_keyword_stuffing(candidate, result)
    _check_fake_expertise(candidate, result)
    _check_pure_consulting(candidate, result)
    _check_research_only(candidate, result)
    _check_domain_mismatch(candidate, result)
    _check_signal_manipulation(candidate, result)
    _check_inactivity(candidate, result)

    result.penalty = clamp(result.penalty)
    return result


# ─────────────────────────────────────────────
# DETECTOR 1: HONEYPOT CANDIDATES
# ─────────────────────────────────────────────
# ~80 honeypots with "subtly impossible profiles" per the README.
# Red flags: overlapping job dates, impossible YoE, profile inconsistencies.

_IMPOSSIBLE_SKILL_COMBOS = [
    # Someone claiming expert-level in very different domains simultaneously
    ({"neurosurgery", "brain surgery"}, {"deep learning", "nlp"}),
]

def _check_honeypot(c: Dict, result: TrapResult) -> None:
    career = c.get("career_history", [])
    profile = c.get("profile", {})
    signals = c.get("redrob_signals", {})

    # 1. Overlapping job dates (impossible to work 2 full-time jobs simultaneously)
    overlaps = check_date_overlap(career)
    if overlaps >= HONEYPOT_MAX_SIMULTANEOUS_JOBS:
        result.mark_honeypot(f"{overlaps} simultaneous overlapping jobs")

    # 2. Declared YoE massively exceeds career history
    declared_yoe = float(profile.get("years_of_experience", 0))
    career_years = c.get("_career_years", 0)
    if declared_yoe > 0 and career_years > 0:
        ratio = declared_yoe / max(career_years, 0.5)
        if ratio > 2.5:  # Claims 2.5x more experience than career records show
            result.mark_honeypot(
                f"Declared YoE ({declared_yoe:.1f}y) vs career history ({career_years:.1f}y)"
            )

    # 3. Profile completeness very low but signals very high (fake engagement)
    completeness = signals.get("profile_completeness_score", 0)
    saved = signals.get("saved_by_recruiters_30d", 0)
    views = signals.get("profile_views_received_30d", 0)
    if completeness < 0.30 and saved > 20 and views < 5:
        result.mark_honeypot("Near-empty profile with inflated saved_by_recruiters signal")

    # 4. Career descriptions that don't match titles at all
    _check_description_title_mismatch(c, result)

    # 5. Future dates in history
    from datetime import date
    for job in career:
        start = job.get("_start")
        if start and start > date.today():
            result.mark_honeypot(f"Job start date in future: {start}")
            break


def _check_description_title_mismatch(c: Dict, result: TrapResult) -> None:
    """
    Flag candidates where all job descriptions are identical or nearly identical
    to each other (copy-pasted descriptions = synthetic/fake profile).
    """
    career = c.get("career_history", [])
    if len(career) < 2:
        return

    descs = [normalise(j.get("description", "")) for j in career]
    descs = [d for d in descs if len(d) > 50]

    if len(descs) < 2:
        return

    # Check if descriptions are suspiciously similar (share most words)
    identical_pairs = 0
    for i in range(len(descs)):
        for j in range(i + 1, len(descs)):
            words_i = set(descs[i].split())
            words_j = set(descs[j].split())
            if not words_i or not words_j:
                continue
            jaccard = len(words_i & words_j) / len(words_i | words_j)
            if jaccard > 0.85:
                identical_pairs += 1

    if identical_pairs >= 2:
        result.mark_honeypot("Multiple near-identical job descriptions (synthetic profile)")
    elif identical_pairs == 1:
        result.add(0.05, "One pair of nearly identical job descriptions")


# ─────────────────────────────────────────────
# DETECTOR 2: KEYWORD STUFFING
# ─────────────────────────────────────────────

# Skills that keyword stuffers tend to list without real evidence
HIGH_VALUE_AI_SKILLS = {
    "embeddings", "vector database", "rag", "retrieval", "semantic search",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "sentence-transformers",
    "sentence transformers", "fine-tuning", "lora", "qlora",
    "learning to rank", "ndcg", "reranking", "re-ranking",
    "llm", "large language model", "transformer",
}


def _check_keyword_stuffing(c: Dict, result: TrapResult) -> None:
    """
    Detect candidates who list high-value AI skills without career evidence.
    Logic: For each claimed AI skill at advanced/expert level, check:
    - endorsements >= threshold OR
    - duration_months >= threshold OR
    - career descriptions mention the skill
    Fail on 3+ unsupported claims.
    """
    career_text = get_career_text(c)
    skills = c.get("skills", [])

    unsupported_count = 0
    unsupported_skills = []

    for skill in skills:
        name_norm = skill.get("_name_norm", "")
        proficiency = skill.get("proficiency", "beginner")

        # Only flag high-value AI skills claimed at advanced/expert
        if name_norm not in HIGH_VALUE_AI_SKILLS:
            continue
        if proficiency not in ("advanced", "expert"):
            continue

        endorsements = skill.get("endorsements", 0)
        duration = skill.get("duration_months", 0)

        has_endorsements = endorsements >= STUFFING_MIN_ENDORSEMENTS
        has_duration = duration >= STUFFING_MIN_DURATION_MONTHS
        has_career_evidence = name_norm in normalise(career_text)

        if not (has_endorsements or has_duration or has_career_evidence):
            unsupported_count += 1
            unsupported_skills.append(name_norm)

    if unsupported_count >= 4:
        result.add(PENALTY_KEYWORD_STUFFING,
                   f"Keyword stuffing: {unsupported_count} unsupported high-value skills "
                   f"({', '.join(unsupported_skills[:3])}...)")
    elif unsupported_count >= 2:
        result.add(PENALTY_KEYWORD_STUFFING * 0.5,
                   f"Possible keyword stuffing: {unsupported_count} weakly supported skills")

    # Also flag: too many skills total with very low endorsements
    total_skills = len(skills)
    zero_endorsement_skills = sum(1 for s in skills if s.get("endorsements", 0) == 0)
    if total_skills > STUFFING_MAX_SKILLS_PER_CATEGORY and zero_endorsement_skills > total_skills * 0.6:
        result.add(PENALTY_KEYWORD_STUFFING * 0.4,
                   f"Skill inflation: {total_skills} skills, {zero_endorsement_skills} with 0 endorsements")


# ─────────────────────────────────────────────
# DETECTOR 3: FAKE EXPERTISE
# ─────────────────────────────────────────────

def _check_fake_expertise(c: Dict, result: TrapResult) -> None:
    """
    If candidate claims expert/advanced on a skill AND completed assessment
    on Redrob, but assessment score is very low — likely fake.
    """
    signals = c.get("redrob_signals", {})
    assessment_scores = signals.get("skill_assessment_scores", {}) or {}
    skills = c.get("skills", [])

    fake_count = 0
    for skill in skills:
        name = skill.get("name", "")
        name_norm = skill.get("_name_norm", "")
        proficiency = skill.get("proficiency", "beginner")

        if proficiency not in ("advanced", "expert"):
            continue

        # Check if there's an assessment score for this skill
        score = None
        for assess_name, assess_score in assessment_scores.items():
            if normalise(assess_name) == name_norm or name_norm in normalise(assess_name):
                score = assess_score
                break

        if score is not None and score < HONEYPOT_MIN_ASSESSMENT_VS_CLAIM:
            fake_count += 1

    if fake_count >= 2:
        result.add(PENALTY_FAKE_EXPERTISE,
                   f"Fake expertise: {fake_count} skills claimed advanced/expert "
                   f"but assessment scores < {HONEYPOT_MIN_ASSESSMENT_VS_CLAIM}")
    elif fake_count == 1:
        result.add(PENALTY_FAKE_EXPERTISE * 0.4,
                   "One skill claimed expert but assessment score very low")


# ─────────────────────────────────────────────
# DETECTOR 4: PURE CONSULTING
# ─────────────────────────────────────────────

def _check_pure_consulting(c: Dict, result: TrapResult) -> None:
    """
    JD explicitly flags candidates from pure consulting backgrounds.
    Penalise if ALL companies in career are services/consulting.
    """
    career = c.get("career_history", [])
    if not career:
        return

    product_count = c.get("_product_company_count", 0)
    services_count = c.get("_services_company_count", 0)
    total = len(career)

    if total >= 2 and product_count == 0 and services_count >= total:
        result.add(PENALTY_PURE_CONSULTING,
                   f"Pure consulting background: all {services_count} roles at services companies")
    elif total >= 3 and product_count == 0 and services_count >= total - 1:
        result.add(PENALTY_PURE_CONSULTING * 0.5,
                   "Near-pure consulting: almost no product company experience")


# ─────────────────────────────────────────────
# DETECTOR 5: RESEARCH-ONLY BACKGROUND
# ─────────────────────────────────────────────

RESEARCH_SIGNALS = [
    "researcher", "research scientist", "research intern", "phd student",
    "academic", "professor", "postdoc", "post-doctoral",
    "laboratory", "lab", "university", "institute",
]

PRODUCTION_SIGNALS = [
    "production", "deployed", "scaled", "serving", "inference",
    "api", "endpoint", "latency", "throughput", "a/b test",
    "monitoring", "logging", "rollout", "users", "customers",
    "ship", "shipped", "launch", "launched",
]


def _check_research_only(c: Dict, result: TrapResult) -> None:
    """
    JD explicitly rejects pure research backgrounds.
    Check: do career descriptions contain production signals?
    """
    career = c.get("career_history", [])
    if not career:
        return

    career_text = normalise(get_career_text(c))
    all_titles = " ".join(normalise(j.get("title", "")) for j in career)

    has_production_evidence = contains_any(career_text, PRODUCTION_SIGNALS)
    is_research_heavy = (
        contains_any(all_titles, RESEARCH_SIGNALS) or
        contains_any(career_text, ["research lab", "academic research", "published paper",
                                    "arxiv", "neurips", "icml", "acl", "emnlp"])
    )

    if is_research_heavy and not has_production_evidence:
        result.add(PENALTY_RESEARCH_ONLY if False else PENALTY_PURE_RESEARCH,
                   "Research-only background: no production deployment signals in descriptions")
    elif not has_production_evidence:
        result.add(PENALTY_NO_PRODUCTION * 0.6,
                   "Weak production evidence: descriptions don't mention shipping/deploying systems")


PENALTY_RESEARCH_ONLY = PENALTY_PURE_RESEARCH  # alias


# ─────────────────────────────────────────────
# DETECTOR 6: DOMAIN MISMATCH
# ─────────────────────────────────────────────

CV_ROBOTICS_SIGNALS = [
    "computer vision", "image classification", "object detection",
    "robotics", "autonomous", "slam", "lidar", "camera",
    "speech recognition", "speech synthesis", "text to speech", "tts",
    "audio processing", "signal processing",
]

NLP_IR_SIGNALS = [
    "nlp", "natural language", "text", "retrieval", "search",
    "ranking", "embeddings", "language model", "transformer",
    "question answering", "information retrieval",
]


def _check_domain_mismatch(c: Dict, result: TrapResult) -> None:
    """
    JD says: CV/robotics/speech without NLP crossover is a red flag.
    """
    career_text = normalise(get_career_text(c))
    profile_text = normalise(
        c.get("profile", {}).get("summary", "") + " " +
        c.get("profile", {}).get("headline", "")
    )
    all_text = career_text + " " + profile_text

    is_cv_robotics = contains_any(all_text, CV_ROBOTICS_SIGNALS)
    has_nlp_ir = contains_any(all_text, NLP_IR_SIGNALS)

    # Check skills too
    skill_names = " ".join(c.get("_skill_names", set()))
    is_cv_robotics = is_cv_robotics or contains_any(skill_names, CV_ROBOTICS_SIGNALS[:6])
    has_nlp_ir = has_nlp_ir or contains_any(skill_names, NLP_IR_SIGNALS[:6])

    if is_cv_robotics and not has_nlp_ir:
        result.add(PENALTY_OUTSIDER_DOMAIN,
                   "Domain mismatch: CV/robotics/speech without NLP/IR crossover")


# ─────────────────────────────────────────────
# DETECTOR 7: SIGNAL MANIPULATION
# ─────────────────────────────────────────────

def _check_signal_manipulation(c: Dict, result: TrapResult) -> None:
    """
    Detect implausible Redrob signal combinations that suggest synthetic inflation.
    """
    signals = c.get("redrob_signals", {})
    completeness = signals.get("profile_completeness_score", 0.5)  # already 0-1 from preprocess
    saved = signals.get("saved_by_recruiters_30d", 0)
    views = signals.get("profile_views_received_30d", 0)
    response_rate = signals.get("recruiter_response_rate", 0)
    interview_rate = signals.get("interview_completion_rate", 0)

    # Impossible: saved by more recruiters than viewed the profile
    if views > 0 and saved / views > SIGNAL_FRAUD_SAVED_VS_VIEWS_RATIO:
        result.add(0.08, f"Signal anomaly: saved ({saved}) > 90% of profile views ({views})")

    # Impossible: very low profile completeness but very high recruiter engagement
    if completeness < SIGNAL_FRAUD_MIN_COMPLETENESS / 100 and saved > 15:
        result.add(0.06, "Signal anomaly: low completeness but very high recruiter saves")

    # Impossible: 100% response rate AND 100% interview rate (too perfect)
    if response_rate >= 0.99 and interview_rate >= 0.99:
        result.add(0.05, "Signal anomaly: perfect response rate and interview rate (suspicious)")


# ─────────────────────────────────────────────
# DETECTOR 8: INACTIVITY
# ─────────────────────────────────────────────

def _check_inactivity(c: Dict, result: TrapResult) -> None:
    """
    A perfect-on-paper candidate inactive for 6+ months is effectively unavailable.
    """
    days_inactive = c.get("_days_inactive", 0)

    if days_inactive > 180:
        result.add(PENALTY_INACTIVE,
                   f"Inactive for {days_inactive} days (>6 months)")
    elif days_inactive > 90:
        result.add(PENALTY_INACTIVE * 0.5,
                   f"Inactive for {days_inactive} days (>3 months)")
