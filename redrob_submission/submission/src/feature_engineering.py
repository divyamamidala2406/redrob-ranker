"""
feature_engineering.py
Computes all scoring components for a candidate:
  - skill_score
  - experience_score
  - career_score
  - behavior_score
  - education_score
  - location_score
  - production_depth_score (unique differentiator)

Each function returns (score: float [0,1], debug: dict).
"""

import logging
import math
from datetime import datetime
from typing import Dict, List, Tuple

from src.config import (
    BEHAVIOR_WEIGHTS,
    DEGREE_SCORES,
    EDUCATION_TIER_SCORES,
    EXP_HARD_MAX,
    EXP_HARD_MIN,
    EXP_IDEAL_MAX,
    EXP_IDEAL_MIN,
    MUST_HAVE_SKILL_GROUPS,
    NICE_TO_HAVE_SKILLS,
    NOTICE_BUYOUT_MAX,
    NOTICE_HARD_MAX,
    NOTICE_IDEAL_MAX,
    PREFERRED_LOCATIONS,
    REFERENCE_DATE,
    RELEVANT_FIELDS,
    STRONG_AI_TITLES,
    WEAK_AI_TITLES,
)
from src.utils import (
    clamp,
    contains_any,
    count_matches,
    get_career_text,
    linear_interp,
    normalise,
    skill_trust_score,
)

logger = logging.getLogger(__name__)
REF_DATE = datetime.strptime(REFERENCE_DATE, "%Y-%m-%d").date()


# ─────────────────────────────────────────────
# SKILL SCORE
# ─────────────────────────────────────────────

def compute_skill_score(candidate: Dict) -> Tuple[float, Dict]:
    """
    Multi-factor skill score:
    1. Coverage of must-have skill groups (gate-level)
    2. Nice-to-have skill count
    3. Skill trust (endorsements x duration)
    4. Assessment scores from Redrob platform
    """
    skills = candidate.get("skills", [])
    skill_names = candidate.get("_skill_names", set())
    career_text = normalise(get_career_text(candidate))
    signals = candidate.get("redrob_signals", {})
    assessment_scores = signals.get("skill_assessment_scores", {}) or {}

    debug = {}

    # 1. Must-have group coverage
    group_scores = {}
    for group_name, keywords in MUST_HAVE_SKILL_GROUPS.items():
        skill_hit = any(
            any(kw in name for kw in keywords)
            for name in skill_names
        )
        career_hit = contains_any(career_text, keywords)

        if skill_hit and career_hit:
            group_scores[group_name] = 1.0
        elif career_hit:
            group_scores[group_name] = 0.85
        elif skill_hit:
            matching_skill = next(
                (s for s in skills if any(kw in s.get("_name_norm", "") for kw in keywords)),
                None
            )
            trust = skill_trust_score(matching_skill) if matching_skill else 0.35
            group_scores[group_name] = trust * 0.65
        else:
            group_scores[group_name] = 0.0

    must_have_score = (
        group_scores.get("embeddings", 0) * 0.35 +
        max(group_scores.get("vector_db", 0), group_scores.get("retrieval", 0)) * 0.35 +
        group_scores.get("python", 0) * 0.30
    )
    debug["must_have_groups"] = group_scores

    # 2. Nice-to-have skills
    combined_text = career_text + " " + " ".join(skill_names)
    nth_count = count_matches(combined_text, NICE_TO_HAVE_SKILLS)
    nth_score = clamp(math.log1p(nth_count) / math.log1p(8))
    debug["nice_to_have_count"] = nth_count

    # 3. Skill trust (quality-weighted)
    all_kws = [kw for group in MUST_HAVE_SKILL_GROUPS.values() for kw in group] + NICE_TO_HAVE_SKILLS
    relevant_skills = [
        s for s in skills
        if any(kw in s.get("_name_norm", "") or s.get("_name_norm", "") in kw for kw in all_kws)
    ]
    avg_trust = (
        sum(skill_trust_score(s) for s in relevant_skills) / len(relevant_skills)
        if relevant_skills else 0.0
    )
    debug["avg_skill_trust"] = avg_trust

    # 4. Redrob assessment scores for relevant skills
    relevant_kws = set()
    for group in MUST_HAVE_SKILL_GROUPS.values():
        relevant_kws.update(group)
    relevant_kws.update(NICE_TO_HAVE_SKILLS)

    ai_scores = [
        float(score) for name, score in assessment_scores.items()
        if any(kw in normalise(str(name)) for kw in relevant_kws)
    ]
    assessment_bonus = clamp(sum(ai_scores) / (len(ai_scores) * 100)) if ai_scores else 0.0
    debug["assessment_bonus"] = assessment_bonus

    final = clamp(
        0.50 * must_have_score +
        0.20 * nth_score +
        0.20 * avg_trust +
        0.10 * assessment_bonus
    )
    debug["final"] = final
    return final, debug


# ─────────────────────────────────────────────
# EXPERIENCE SCORE
# ─────────────────────────────────────────────

PRODUCTION_AI_KWS = [
    "embedding", "retrieval", "vector", "semantic search", "ranking",
    "recommendation", "deployed", "production", "real users", "scale",
    "llm", "language model", "nlp", "search", "reranking",
]


def compute_experience_score(candidate: Dict) -> Tuple[float, Dict]:
    """
    Score based on years of experience + production AI depth + recency.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    debug = {}

    declared_yoe = float(profile.get("years_of_experience", 0))
    career_years = candidate.get("_career_years", declared_yoe)
    effective_yoe = min(declared_yoe, career_years) if career_years > 0 else declared_yoe

    # Piecewise YoE score
    if effective_yoe < EXP_HARD_MIN:
        yoe_score = effective_yoe / max(EXP_HARD_MIN, 0.1) * 0.3
    elif effective_yoe < EXP_IDEAL_MIN:
        yoe_score = 0.3 + linear_interp(effective_yoe, EXP_HARD_MIN, EXP_IDEAL_MIN) * 0.5
    elif effective_yoe <= EXP_IDEAL_MAX:
        yoe_score = 0.8 + linear_interp(effective_yoe, EXP_IDEAL_MIN, EXP_IDEAL_MAX) * 0.2
    elif effective_yoe <= EXP_HARD_MAX:
        yoe_score = 1.0 - linear_interp(effective_yoe, EXP_IDEAL_MAX, EXP_HARD_MAX) * 0.15
    else:
        yoe_score = 0.70
    debug["yoe_score"] = yoe_score
    debug["effective_yoe"] = effective_yoe

    # Production AI evidence
    career_text = normalise(get_career_text(candidate))
    prod_ai_count = count_matches(career_text, PRODUCTION_AI_KWS)
    prod_ai_score = clamp(math.log1p(prod_ai_count) / math.log1p(12))
    debug["prod_ai_score"] = prod_ai_score

    # Recency of AI work
    recency_score = 0.0
    for i, job in enumerate(career[:3]):
        desc = normalise(job.get("description", ""))
        if contains_any(desc, PRODUCTION_AI_KWS):
            recency_score = max(recency_score, 1.0 - i * 0.2)
    debug["recency_score"] = recency_score

    final = clamp(
        0.40 * yoe_score +
        0.40 * prod_ai_score +
        0.20 * recency_score
    )
    debug["final"] = final
    return final, debug


# ─────────────────────────────────────────────
# CAREER SCORE
# ─────────────────────────────────────────────

STRONG_CAREER_PATTERNS = [
    "shipped", "deployed", "launched", "built", "designed", "architected",
    "improved", "reduced latency", "increased precision", "recall", "ndcg",
    "a/b test", "production", "serves", "billion", "million", "users",
]

FOUNDING_KWS = [
    "founding", "early-stage", "series a", "seed", "startup",
    "0 to 1", "from scratch", "built team", "first engineer",
    "founding engineer", "founding team",
]


def compute_career_score(candidate: Dict) -> Tuple[float, Dict]:
    """
    Score based on title quality, company types, tenure, impact, and founding experience.
    """
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    debug = {}

    if not career:
        return 0.0, {"final": 0.0}

    # 1. Title quality
    all_titles = " ".join(normalise(j.get("title", "")) for j in career)
    strong_hits = count_matches(all_titles, STRONG_AI_TITLES)
    weak_hits = count_matches(all_titles, WEAK_AI_TITLES)

    if strong_hits >= 2:
        title_score = 1.0
    elif strong_hits == 1:
        title_score = 0.75
    elif weak_hits == 0:
        title_score = 0.30
    else:
        title_score = 0.50
    debug["title_score"] = title_score

    # 2. Product company ratio
    product_count = candidate.get("_product_company_count", 0)
    product_frac = product_count / len(career) if career else 0.0
    company_score = clamp(product_frac * 1.2)
    debug["company_score"] = company_score

    # 3. Tenure stability
    durations = [j.get("duration_months", 0) for j in career]
    avg_tenure = sum(durations) / len(durations) if durations else 0
    if avg_tenure >= 36:
        tenure_score = 1.0
    elif avg_tenure >= 18:
        tenure_score = 0.8
    elif avg_tenure >= 12:
        tenure_score = 0.6
    else:
        tenure_score = 0.3
    debug["tenure_score"] = tenure_score

    # 4. Impact evidence
    career_text = normalise(get_career_text(candidate))
    impact_count = count_matches(career_text, STRONG_CAREER_PATTERNS)
    impact_score = clamp(math.log1p(impact_count) / math.log1p(10))
    debug["impact_score"] = impact_score

    # 5. Founding/early-stage bonus
    founding_bonus = 0.10 if contains_any(career_text, FOUNDING_KWS) else 0.0
    debug["founding_bonus"] = founding_bonus

    final = clamp(
        0.30 * title_score +
        0.25 * company_score +
        0.20 * tenure_score +
        0.20 * impact_score +
        founding_bonus
    )
    debug["final"] = final
    return final, debug


# ─────────────────────────────────────────────
# BEHAVIORAL SCORE
# ─────────────────────────────────────────────

def compute_behavior_score(candidate: Dict) -> Tuple[float, Dict]:
    """
    Score based on Redrob platform signals — availability + engagement.
    """
    sig = candidate.get("redrob_signals", {})
    debug = {}

    # 1. Open to work
    open_to_work = 1.0 if sig.get("open_to_work_flag", False) else 0.2
    debug["open_to_work"] = open_to_work

    # 2. Recency of activity
    days_inactive = candidate.get("_days_inactive", 999)
    if days_inactive <= 7:
        recency = 1.0
    elif days_inactive <= 30:
        recency = 0.85
    elif days_inactive <= 60:
        recency = 0.65
    elif days_inactive <= 90:
        recency = 0.45
    elif days_inactive <= 180:
        recency = 0.20
    else:
        recency = 0.05
    debug["recency"] = recency

    # 3. Recruiter response rate
    rrr = float(sig.get("recruiter_response_rate", 0.0))
    response_rate = clamp(rrr)
    debug["response_rate"] = response_rate

    # 4. Response speed
    avg_rt = float(sig.get("avg_response_time_hours", 999))
    if avg_rt <= 4:
        response_speed = 1.0
    elif avg_rt <= 24:
        response_speed = 0.85
    elif avg_rt <= 48:
        response_speed = 0.65
    elif avg_rt <= 72:
        response_speed = 0.45
    else:
        response_speed = clamp(1.0 - (avg_rt - 72) / 300)
    debug["response_speed"] = response_speed

    # 5. Profile completeness
    completeness = float(sig.get("profile_completeness_score", 0.0))
    debug["completeness"] = completeness

    # 6. GitHub activity
    gas = float(sig.get("github_activity_score", -1))
    github_score = 0.40 if gas == -1 else clamp(gas / 100.0)
    debug["github_score"] = github_score

    # 7. Recruiter market interest
    saved = int(sig.get("saved_by_recruiters_30d", 0))
    recruiter_interest = clamp(math.log1p(saved) / math.log1p(20))
    debug["recruiter_interest"] = recruiter_interest

    # 8. Interview completion rate
    icr = float(sig.get("interview_completion_rate", 0.0))
    interview_completion = clamp(icr)
    debug["interview_completion"] = interview_completion

    # Notice period modifier
    notice = int(sig.get("notice_period_days", 90))
    if notice <= NOTICE_IDEAL_MAX:
        notice_modifier = 1.0
    elif notice <= NOTICE_BUYOUT_MAX:
        notice_modifier = 0.85
    elif notice <= NOTICE_HARD_MAX:
        notice_modifier = 0.65
    else:
        notice_modifier = 0.45
    debug["notice_modifier"] = notice_modifier

    # Verification bonus
    verify_bonus = (
        0.04 * int(sig.get("verified_email", False)) +
        0.04 * int(sig.get("verified_phone", False)) +
        0.02 * int(sig.get("linkedin_connected", False))
    )

    raw = (
        BEHAVIOR_WEIGHTS["open_to_work"]        * open_to_work +
        BEHAVIOR_WEIGHTS["recency"]              * recency +
        BEHAVIOR_WEIGHTS["response_rate"]        * response_rate +
        BEHAVIOR_WEIGHTS["response_speed"]       * response_speed +
        BEHAVIOR_WEIGHTS["completeness"]         * completeness +
        BEHAVIOR_WEIGHTS["github_activity"]      * github_score +
        BEHAVIOR_WEIGHTS["recruiter_interest"]   * recruiter_interest +
        BEHAVIOR_WEIGHTS["interview_completion"] * interview_completion
    )

    final = clamp(raw * notice_modifier + verify_bonus)
    debug["final"] = final
    return final, debug


# ─────────────────────────────────────────────
# EDUCATION SCORE
# ─────────────────────────────────────────────

def compute_education_score(candidate: Dict) -> Tuple[float, Dict]:
    """
    Score based on institution tier, degree level, and field relevance.
    """
    education = candidate.get("education", [])
    debug = {}

    if not education:
        return 0.35, {"final": 0.35}

    best_score = 0.0
    for edu in education:
        tier = edu.get("tier", "unknown")
        tier_score = EDUCATION_TIER_SCORES.get(tier, 0.35)

        degree_norm = edu.get("_degree_norm", "")
        deg_score = 0.50
        for deg_key, score in DEGREE_SCORES.items():
            if deg_key in degree_norm:
                deg_score = max(deg_score, score)

        field_norm = edu.get("_field_norm", "")
        field_relevant = any(f in field_norm for f in RELEVANT_FIELDS)
        field_score = 0.90 if field_relevant else 0.40

        edu_score = 0.40 * tier_score + 0.35 * deg_score + 0.25 * field_score
        best_score = max(best_score, edu_score)

    debug["final"] = best_score
    return clamp(best_score), debug


# ─────────────────────────────────────────────
# LOCATION SCORE
# ─────────────────────────────────────────────

TOP_CITIES = {"pune", "noida", "delhi", "delhi ncr", "ncr", "gurgaon", "gurugram"}
INDIA_LOCS = {"india", "bangalore", "bengaluru", "hyderabad", "mumbai", "chennai", "kolkata"}


def compute_location_score(candidate: Dict) -> Tuple[float, Dict]:
    """
    JD: Pune/Noida preferred, Indian cities welcome, outside India = lower.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    debug = {}

    location_norm = profile.get("_location_norm", "")
    country_norm = profile.get("_country_norm", "")
    willing_to_relocate = signals.get("willing_to_relocate", False)
    work_mode = normalise(signals.get("preferred_work_mode", ""))

    in_top_city = any(city in location_norm for city in TOP_CITIES)
    in_india = any(
        loc in location_norm or loc in country_norm
        for loc in INDIA_LOCS | PREFERRED_LOCATIONS
    )

    if in_top_city:
        base_score = 1.0
    elif in_india:
        base_score = 0.80
    elif willing_to_relocate:
        base_score = 0.55
    else:
        base_score = 0.25

    mode_bonus = {"hybrid": 0.05, "flexible": 0.05, "onsite": 0.03}.get(work_mode, 0.0)

    final = clamp(base_score + mode_bonus)
    debug["base_score"] = base_score
    debug["final"] = final
    return final, debug


# ─────────────────────────────────────────────
# UNIQUE FEATURE: PRODUCTION DEPTH SCORE
# ─────────────────────────────────────────────

PRODUCTION_DEPTH_SIGNALS = {
    "scale": 3, "production": 3, "deployed": 3, "serving": 2,
    "latency": 2, "throughput": 2, "index refresh": 3,
    "embedding drift": 3, "retrieval quality": 3, "regression": 2,
    "a/b test": 3, "online evaluation": 3, "offline evaluation": 2,
    "ndcg": 2, "monitoring": 2, "billion": 2, "million users": 3,
    "real users": 3, "cold start": 2, "precision@": 2, "recall@": 2,
    "mrr": 2, "reranking pipeline": 3, "hybrid retrieval": 3,
    "vector index": 3, "ann index": 3, "embedding model": 2,
    "shipped": 3, "launched": 2, "end-to-end": 2,
}


def compute_production_depth_score(candidate: Dict) -> Tuple[float, Dict]:
    """
    UNIQUE DIFFERENTIATOR: Specifically measures evidence of operating
    ML/AI systems in production at scale — not just building prototypes.
    This punishes tutorial-followers and rewards real practitioners.
    """
    career_text = normalise(get_career_text(candidate))
    profile_text = normalise(
        candidate.get("profile", {}).get("summary", "") + " " +
        candidate.get("profile", {}).get("headline", "")
    )
    all_text = career_text + " " + profile_text

    total_weight = 0
    matched_signals = []
    for signal, weight in PRODUCTION_DEPTH_SIGNALS.items():
        if signal in all_text:
            total_weight += weight
            matched_signals.append(signal)

    score = clamp(total_weight / 25.0)
    return score, {"signals": matched_signals[:8], "weight": total_weight, "final": score}
