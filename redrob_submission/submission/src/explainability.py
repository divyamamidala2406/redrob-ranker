"""
explainability.py
Generates 1-2 sentence human-readable reasoning for each ranked candidate.
The reasoning must fit within the CSV "reasoning" column.
Format: positive signals first, then gaps, then flags if any.
"""

import logging
from typing import Dict, List, Optional

from src.config import MUST_HAVE_SKILL_GROUPS, NICE_TO_HAVE_SKILLS
from src.utils import contains_any, get_career_text, normalise, count_matches

logger = logging.getLogger(__name__)

# Max chars for reasoning column (keep short for CSV readability)
MAX_REASONING_LEN = 400


def generate_reasoning(scored: Dict) -> str:
    """
    Generate a 1-2 sentence reasoning string for the submission CSV.
    Pulls from score components and candidate data.
    """
    candidate = scored.get("_candidate", {})
    profile = candidate.get("profile", {})
    trap_flags = scored.get("trap_flags", [])
    is_honeypot = scored.get("is_honeypot", False)

    if is_honeypot:
        return _honeypot_reasoning(trap_flags)

    positives = _collect_positives(scored, candidate)
    gaps = _collect_gaps(scored, candidate)
    flags = _collect_flag_summary(trap_flags)

    # Build sentence 1: strengths
    if positives:
        s1 = "Strong fit: " + "; ".join(positives[:3]) + "."
    else:
        s1 = "Partial fit with limited evidence of key requirements."

    # Build sentence 2: gaps or flags
    parts2 = []
    if flags:
        parts2.append(flags)
    if gaps:
        parts2.append("Missing: " + ", ".join(gaps[:2]))

    s2 = " ".join(parts2) if parts2 else ""

    reasoning = (s1 + " " + s2).strip()
    return _truncate(reasoning)


def _honeypot_reasoning(flags: List[str]) -> str:
    summary = flags[0] if flags else "Profile inconsistency detected"
    return _truncate(f"Excluded: {summary}.")


def _collect_positives(scored: Dict, candidate: Dict) -> List[str]:
    positives = []

    career_text = normalise(get_career_text(candidate))
    profile = candidate.get("profile", {})
    skill_names = candidate.get("_skill_names", set())
    yoe = float(profile.get("years_of_experience", 0))
    current_title = profile.get("current_title", "")
    skills = candidate.get("skills", [])

    # Skill positives
    skill_debug = scored.get("skill_debug", {})
    groups = skill_debug.get("must_have_groups", {})

    if groups.get("embeddings", 0) >= 0.7:
        positives.append("embeddings expertise confirmed in career history")
    if max(groups.get("vector_db", 0), groups.get("retrieval", 0)) >= 0.7:
        positives.append("vector DB / retrieval experience evidenced")

    nth_count = skill_debug.get("nice_to_have_count", 0)
    if nth_count >= 5:
        positives.append(f"{nth_count} relevant AI skills (LLM, ranking, evaluation)")
    elif nth_count >= 2:
        positives.append(f"{nth_count} complementary AI skills")

    assess_bonus = skill_debug.get("assessment_bonus", 0)
    if assess_bonus >= 0.6:
        positives.append("high platform assessment scores")

    # Experience positives
    exp_debug = scored.get("exp_debug", {})
    eff_yoe = exp_debug.get("effective_yoe", yoe)
    if eff_yoe >= 7:
        positives.append(f"{eff_yoe:.0f} years experience")
    elif eff_yoe >= 5:
        positives.append(f"{eff_yoe:.0f} yrs relevant experience")

    if exp_debug.get("prod_ai_score", 0) >= 0.65:
        positives.append("strong production AI deployment background")

    # Career positives
    career_debug = scored.get("career_debug", {})
    if career_debug.get("title_score", 0) >= 0.75:
        positives.append(f"strong AI/ML title trajectory ({current_title})")
    if career_debug.get("company_score", 0) >= 0.6:
        positives.append("product company experience")
    if career_debug.get("founding_bonus", 0) > 0:
        positives.append("startup / founding team experience")

    # Behavior positives
    beh_debug = scored.get("beh_debug", {})
    if beh_debug.get("open_to_work", 0) == 1.0:
        positives.append("actively looking")
    if beh_debug.get("recency", 0) >= 0.8:
        positives.append("recently active on platform")
    if beh_debug.get("github_score", 0) >= 0.6:
        positives.append("active GitHub profile")
    if beh_debug.get("recruiter_interest", 0) >= 0.5:
        positives.append("high recruiter saves (market-validated)")

    # Production depth
    prod_debug = scored.get("prod_debug", {})
    prod_signals = prod_debug.get("signals", [])
    if len(prod_signals) >= 4:
        positives.append(f"production depth signals: {', '.join(prod_signals[:3])}")

    return positives


def _collect_gaps(scored: Dict, candidate: Dict) -> List[str]:
    gaps = []
    career_text = normalise(get_career_text(candidate))
    skill_names = candidate.get("_skill_names", set())

    skill_debug = scored.get("skill_debug", {})
    groups = skill_debug.get("must_have_groups", {})

    if groups.get("embeddings", 0) < 0.4:
        gaps.append("no clear embeddings experience")
    if max(groups.get("vector_db", 0), groups.get("retrieval", 0)) < 0.4:
        gaps.append("no vector DB / retrieval evidence")
    if groups.get("python", 0) < 0.5:
        gaps.append("Python not confirmed")

    exp_debug = scored.get("exp_debug", {})
    if exp_debug.get("effective_yoe", 0) < 3:
        gaps.append("limited years of experience")
    if exp_debug.get("prod_ai_score", 0) < 0.3:
        gaps.append("limited production AI evidence")

    beh_debug = scored.get("beh_debug", {})
    if beh_debug.get("recency", 0) < 0.3:
        gaps.append("inactive on platform")
    if beh_debug.get("notice_modifier", 1.0) < 0.7:
        notice = candidate.get("redrob_signals", {}).get("notice_period_days", 90)
        gaps.append(f"{notice}d notice period")

    return gaps


def _collect_flag_summary(trap_flags: List[str]) -> str:
    if not trap_flags:
        return ""
    # Show first flag briefly
    first = trap_flags[0]
    if len(trap_flags) > 1:
        return f"⚠ {first[:60]} (+{len(trap_flags)-1} other flags)."
    return f"⚠ {first[:80]}."


def _truncate(s: str, max_len: int = MAX_REASONING_LEN) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[:max_len - 3].rsplit(" ", 1)[0] + "..."


# ─────────────────────────────────────────────
# VERBOSE CONSOLE REPORT (for debugging top-10)
# ─────────────────────────────────────────────

def print_candidate_report(scored: Dict, rank: int) -> None:
    """Print detailed report for a candidate (debugging / demo)."""
    c = scored.get("_candidate", {})
    profile = c.get("profile", {})
    sig = c.get("redrob_signals", {})

    print(f"\n{'='*60}")
    print(f"RANK #{rank}  |  {scored['candidate_id']}")
    print(f"  Name:     {profile.get('anonymized_name', 'N/A')}")
    print(f"  Title:    {profile.get('current_title', 'N/A')} @ {profile.get('current_company', 'N/A')}")
    print(f"  Location: {profile.get('location', 'N/A')}, {profile.get('country', 'N/A')}")
    print(f"  YoE:      {profile.get('years_of_experience', 0):.1f} yrs")
    print(f"  FINAL SCORE: {scored['final_score']:.4f}")
    print(f"  Components:")
    print(f"    Skill:     {scored['skill_score']:.3f}  (w={0.32})")
    print(f"    Exp:       {scored['exp_score']:.3f}  (w={0.20})")
    print(f"    Career:    {scored['career_score']:.3f}  (w={0.18})")
    print(f"    Behavior:  {scored['behavior_score']:.3f}  (w={0.15})")
    print(f"    Education: {scored['edu_score']:.3f}  (w={0.08})")
    print(f"    Location:  {scored['loc_score']:.3f}  (w={0.07})")
    print(f"    ProdDepth: {scored['prod_depth_score']:.3f}  (bonus)")
    print(f"  Trap penalty: {scored['trap_penalty']:.3f}  | Honeypot: {scored['is_honeypot']}")
    if scored.get("trap_flags"):
        for f in scored["trap_flags"]:
            print(f"    ⚠ {f}")
    print(f"  Open to work: {sig.get('open_to_work_flag')}  | "
          f"Notice: {sig.get('notice_period_days')}d  | "
          f"Last active: {sig.get('last_active_date')}")
    print(f"  REASONING: {generate_reasoning(scored)}")
    print(f"{'='*60}")
