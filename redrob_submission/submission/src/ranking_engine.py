"""
ranking_engine.py
Combines component scores into a final weighted score.
Applies trap penalties multiplicatively to avoid gaming.
"""

import logging
from typing import Dict, List, Tuple

from src.config import WEIGHTS
from src.feature_engineering import (
    compute_behavior_score,
    compute_career_score,
    compute_education_score,
    compute_experience_score,
    compute_location_score,
    compute_production_depth_score,
    compute_skill_score,
)
from src.trap_detection import TrapResult, detect_traps
from src.utils import clamp

logger = logging.getLogger(__name__)


def score_candidate(candidate: Dict) -> Dict:
    """
    Full scoring pipeline for a single candidate.
    Returns a score dict with all components and the final score.
    """
    cid = candidate.get("candidate_id", "UNKNOWN")

    try:
        # Component scores
        skill_score, skill_debug       = compute_skill_score(candidate)
        exp_score,   exp_debug         = compute_experience_score(candidate)
        career_score, career_debug     = compute_career_score(candidate)
        behavior_score, beh_debug      = compute_behavior_score(candidate)
        edu_score,   edu_debug         = compute_education_score(candidate)
        loc_score,   loc_debug         = compute_location_score(candidate)
        prod_score,  prod_debug        = compute_production_depth_score(candidate)

        # Trap detection
        trap: TrapResult = detect_traps(candidate)

        # ── Weighted base score ──────────────────────
        base = (
            WEIGHTS["skill"]       * skill_score +
            WEIGHTS["experience"]  * exp_score +
            WEIGHTS["career"]      * career_score +
            WEIGHTS["behavior"]    * behavior_score +
            WEIGHTS["education"]   * edu_score +
            WEIGHTS["location"]    * loc_score
        )

        # Production depth is a BONUS on top (up to +0.08)
        # It rewards genuine practitioners without artificially inflating scores
        prod_bonus = prod_score * 0.08

        # ── Penalty application ──────────────────────
        # For honeypots, apply hard near-disqualifier
        if trap.is_honeypot:
            penalty_multiplier = 0.20   # Score collapses to 20% of raw
        elif trap.penalty > 0:
            # Penalty reduces score non-linearly
            # penalty=0.10 -> multiplier=0.92, penalty=0.40 -> multiplier=0.70
            penalty_multiplier = max(0.50, 1.0 - trap.penalty * 0.75)
        else:
            penalty_multiplier = 1.0

        final_score = clamp((base + prod_bonus) * penalty_multiplier)

        return {
            "candidate_id": cid,
            "final_score": final_score,
            "base_score": base,
            "prod_bonus": prod_bonus,
            "penalty_multiplier": penalty_multiplier,
            # Component scores
            "skill_score": skill_score,
            "exp_score": exp_score,
            "career_score": career_score,
            "behavior_score": behavior_score,
            "edu_score": edu_score,
            "loc_score": loc_score,
            "prod_depth_score": prod_score,
            # Trap info
            "trap_penalty": trap.penalty,
            "trap_flags": trap.flags,
            "is_honeypot": trap.is_honeypot,
            # Debug info
            "skill_debug": skill_debug,
            "exp_debug": exp_debug,
            "career_debug": career_debug,
            "beh_debug": beh_debug,
            "prod_debug": prod_debug,
            # Raw candidate ref for explainability
            "_candidate": candidate,
        }

    except Exception as e:
        logger.error(f"Scoring failed for {cid}: {e}", exc_info=True)
        return {
            "candidate_id": cid,
            "final_score": 0.0,
            "base_score": 0.0,
            "prod_bonus": 0.0,
            "penalty_multiplier": 0.0,
            "skill_score": 0.0,
            "exp_score": 0.0,
            "career_score": 0.0,
            "behavior_score": 0.0,
            "edu_score": 0.0,
            "loc_score": 0.0,
            "prod_depth_score": 0.0,
            "trap_penalty": 0.0,
            "trap_flags": [f"SCORING_ERROR: {str(e)}"],
            "is_honeypot": False,
            "_candidate": candidate,
        }


def rank_candidates(scored: List[Dict]) -> List[Dict]:
    """
    Sort candidates by final_score descending.
    Tie-break: candidate_id ascending (per spec).
    Assign ranks 1..N.
    """
    sorted_candidates = sorted(
        scored,
        key=lambda x: (-x["final_score"], x["candidate_id"])
    )
    for i, c in enumerate(sorted_candidates):
        c["rank"] = i + 1
    return sorted_candidates


def score_all_candidates(candidates_iter, max_workers: int = 1) -> List[Dict]:
    """
    Score all candidates from an iterator.
    Single-threaded for determinism (constraint: CPU only, 5-min limit).
    Streams candidates to keep memory < 2GB for 100K profiles.
    """
    results = []
    count = 0
    skipped = 0

    for candidate in candidates_iter:
        if candidate is None:
            skipped += 1
            continue

        scored = score_candidate(candidate)
        results.append(scored)
        count += 1

        if count % 10000 == 0:
            logger.info(f"Scored {count} candidates...")

    logger.info(f"Scored {count} candidates, skipped {skipped} malformed")
    return results
