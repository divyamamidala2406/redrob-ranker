"""
preprocess.py
Data loading, validation, cleaning, and normalization.
Outputs a cleaned candidate dict ready for feature engineering.
"""

import logging
from datetime import date, datetime
from typing import Dict, List, Optional

from src.utils import parse_date, normalise, clamp
from src.config import REFERENCE_DATE

logger = logging.getLogger(__name__)

REF_DATE = datetime.strptime(REFERENCE_DATE, "%Y-%m-%d").date()


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def preprocess_candidate(raw: Dict) -> Optional[Dict]:
    """
    Clean and validate a single candidate dict.
    Returns None if the candidate is so malformed it can't be scored.
    Mutates a copy of the dict with added/cleaned fields.
    """
    try:
        c = dict(raw)  # shallow copy — nested dicts are still shared
        _clean_profile(c)
        _clean_career(c)
        _clean_skills(c)
        _clean_education(c)
        _clean_signals(c)
        _add_derived_fields(c)
        return c
    except Exception as e:
        cid = raw.get("candidate_id", "UNKNOWN")
        logger.warning(f"Preprocessing failed for {cid}: {e}")
        return None


# ─────────────────────────────────────────────
# PROFILE CLEANING
# ─────────────────────────────────────────────

def _clean_profile(c: Dict) -> None:
    profile = c.get("profile", {})

    # Coerce years_of_experience to float
    yoe = profile.get("years_of_experience", 0)
    try:
        profile["years_of_experience"] = float(yoe)
    except (TypeError, ValueError):
        profile["years_of_experience"] = 0.0

    # Clamp to sane range
    profile["years_of_experience"] = clamp(profile["years_of_experience"], 0.0, 50.0)

    # Normalise location
    profile["_location_norm"] = normalise(profile.get("location", ""))
    profile["_country_norm"] = normalise(profile.get("country", ""))

    # Normalise current title
    profile["_title_norm"] = normalise(profile.get("current_title", ""))

    # Normalise company
    profile["_company_norm"] = normalise(profile.get("current_company", ""))

    c["profile"] = profile


# ─────────────────────────────────────────────
# CAREER CLEANING
# ─────────────────────────────────────────────

def _clean_career(c: Dict) -> None:
    history = c.get("career_history", [])
    cleaned = []
    for job in history:
        j = dict(job)

        # Parse dates
        j["_start"] = parse_date(j.get("start_date"))
        j["_end"] = parse_date(j.get("end_date")) if j.get("end_date") else REF_DATE

        # Coerce duration
        try:
            j["duration_months"] = int(j.get("duration_months", 0))
        except (TypeError, ValueError):
            j["duration_months"] = 0

        # Validate duration against dates if possible
        if j["_start"] and j["_end"]:
            calculated = max(0, (j["_end"].year - j["_start"].year) * 12
                            + (j["_end"].month - j["_start"].month))
            # If declared duration wildly off from calculated, use calculated
            if abs(j["duration_months"] - calculated) > 6:
                j["duration_months"] = calculated

        j["_title_norm"] = normalise(j.get("title", ""))
        j["_company_norm"] = normalise(j.get("company", ""))
        j["_industry_norm"] = normalise(j.get("industry", ""))
        j["_desc_norm"] = normalise(j.get("description", ""))

        cleaned.append(j)

    # Sort by start date descending (most recent first)
    cleaned.sort(key=lambda x: x["_start"] or date.min, reverse=True)
    c["career_history"] = cleaned


# ─────────────────────────────────────────────
# SKILLS CLEANING
# ─────────────────────────────────────────────

def _clean_skills(c: Dict) -> None:
    skills = c.get("skills", [])
    cleaned = []
    seen_names = set()

    for skill in skills:
        s = dict(skill)
        name_norm = normalise(s.get("name", ""))
        if not name_norm or name_norm in seen_names:
            continue  # Deduplicate
        seen_names.add(name_norm)

        s["_name_norm"] = name_norm

        # Coerce endorsements
        try:
            s["endorsements"] = max(0, int(s.get("endorsements", 0)))
        except (TypeError, ValueError):
            s["endorsements"] = 0

        # Coerce duration_months
        try:
            s["duration_months"] = max(0, int(s.get("duration_months", 0)))
        except (TypeError, ValueError):
            s["duration_months"] = 0

        cleaned.append(s)

    c["skills"] = cleaned


# ─────────────────────────────────────────────
# EDUCATION CLEANING
# ─────────────────────────────────────────────

def _clean_education(c: Dict) -> None:
    education = c.get("education", [])
    cleaned = []

    for edu in education:
        e = dict(edu)
        e["_institution_norm"] = normalise(e.get("institution", ""))
        e["_degree_norm"] = normalise(e.get("degree", ""))
        e["_field_norm"] = normalise(e.get("field_of_study", ""))
        e["tier"] = e.get("tier", "unknown")

        # Validate years
        try:
            e["start_year"] = int(e.get("start_year", 2000))
        except (TypeError, ValueError):
            e["start_year"] = 2000
        try:
            e["end_year"] = int(e.get("end_year", 2004))
        except (TypeError, ValueError):
            e["end_year"] = 2004

        cleaned.append(e)

    # Sort by end_year descending (most recent first)
    cleaned.sort(key=lambda x: x.get("end_year", 0), reverse=True)
    c["education"] = cleaned


# ─────────────────────────────────────────────
# REDROB SIGNALS CLEANING
# ─────────────────────────────────────────────

def _clean_signals(c: Dict) -> None:
    sig = c.get("redrob_signals", {})

    # Parse dates
    sig["_signup_date"] = parse_date(sig.get("signup_date"))
    sig["_last_active_date"] = parse_date(sig.get("last_active_date"))

    # Clamp rates
    for key in ["recruiter_response_rate", "interview_completion_rate"]:
        try:
            sig[key] = clamp(float(sig.get(key, 0.0)))
        except (TypeError, ValueError):
            sig[key] = 0.0

    # offer_acceptance_rate: -1 means no history
    oar = sig.get("offer_acceptance_rate", -1)
    try:
        sig["offer_acceptance_rate"] = float(oar)
    except (TypeError, ValueError):
        sig["offer_acceptance_rate"] = -1.0

    # github_activity_score: -1 means not linked
    gas = sig.get("github_activity_score", -1)
    try:
        sig["github_activity_score"] = float(gas)
    except (TypeError, ValueError):
        sig["github_activity_score"] = -1.0

    # Salary range
    sal = sig.get("expected_salary_range_inr_lpa", {})
    try:
        sig["_salary_min"] = float(sal.get("min", 0))
        sig["_salary_max"] = float(sal.get("max", 0))
    except (TypeError, ValueError):
        sig["_salary_min"] = 0.0
        sig["_salary_max"] = 0.0

    # Notice period
    try:
        sig["notice_period_days"] = int(sig.get("notice_period_days", 90))
    except (TypeError, ValueError):
        sig["notice_period_days"] = 90

    # Completeness
    try:
        sig["profile_completeness_score"] = clamp(
            float(sig.get("profile_completeness_score", 0)) / 100.0
        )
    except (TypeError, ValueError):
        sig["profile_completeness_score"] = 0.0

    # Boolean fields
    for key in ["open_to_work_flag", "willing_to_relocate", "verified_email",
                "verified_phone", "linkedin_connected"]:
        sig[key] = bool(sig.get(key, False))

    c["redrob_signals"] = sig


# ─────────────────────────────────────────────
# DERIVED FIELDS
# ─────────────────────────────────────────────

def _add_derived_fields(c: Dict) -> None:
    """Add pre-computed derived fields to reduce repeat computation."""
    sig = c["redrob_signals"]
    profile = c["profile"]

    # Days since last active
    last_active = sig.get("_last_active_date")
    if last_active:
        c["_days_inactive"] = max(0, (REF_DATE - last_active).days)
    else:
        c["_days_inactive"] = 999

    # Total months from career history
    total_career_months = sum(
        j.get("duration_months", 0) for j in c.get("career_history", [])
    )
    c["_total_career_months"] = total_career_months
    c["_career_years"] = total_career_months / 12.0

    # Number of distinct companies
    companies = set(
        j.get("_company_norm", "") for j in c.get("career_history", [])
        if j.get("_company_norm")
    )
    c["_distinct_companies"] = len(companies)

    # Number of product vs services companies
    from src.utils import classify_company
    classifications = [
        classify_company(j.get("company", ""), j.get("industry", ""))
        for j in c.get("career_history", [])
    ]
    c["_product_company_count"] = classifications.count("product")
    c["_services_company_count"] = classifications.count("services")

    # Is currently active (open to work + active recently)
    c["_is_available"] = (
        sig.get("open_to_work_flag", False) and c["_days_inactive"] < 90
    )

    # Skill name set for fast lookup
    c["_skill_names"] = {s["_name_norm"] for s in c.get("skills", [])}
