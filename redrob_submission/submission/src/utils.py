"""
utils.py
Shared utility functions used across the pipeline.
"""

import json
import re
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# TEXT NORMALISATION
# ─────────────────────────────────────────────

def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-/&]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_any(text: str, keywords: List[str]) -> bool:
    """Return True if normalised text contains any keyword."""
    norm = normalise(text)
    return any(kw in norm for kw in keywords)


def count_matches(text: str, keywords: List[str]) -> int:
    """Count how many distinct keywords appear in the text."""
    norm = normalise(text)
    return sum(1 for kw in keywords if kw in norm)


def all_text(candidate: Dict) -> str:
    """Concatenate all free-text fields for keyword scanning."""
    parts = []
    profile = candidate.get("profile", {})
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))
    parts.append(profile.get("current_title", ""))

    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
        parts.append(job.get("company", ""))

    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", ""))

    for cert in candidate.get("certifications", []):
        parts.append(cert.get("name", ""))
        parts.append(cert.get("issuer", ""))

    return " ".join(parts)


# ─────────────────────────────────────────────
# DATE UTILITIES
# ─────────────────────────────────────────────

def parse_date(s: Optional[str]) -> Optional[date]:
    """Parse ISO date string, return None on failure."""
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def days_since(d: Optional[date], ref: date) -> Optional[int]:
    """Days between d and ref. Positive if d is before ref."""
    if d is None:
        return None
    return (ref - d).days


def months_between(start: Optional[date], end: Optional[date]) -> int:
    """Calculate months between two dates."""
    if start is None or end is None:
        return 0
    delta = (end.year - start.year) * 12 + (end.month - start.month)
    return max(0, delta)


def check_date_overlap(jobs: List[Dict]) -> int:
    """
    Return count of overlapping job periods (honeypot signal).
    Allows 1-month grace for transitions.
    """
    periods = []
    for job in jobs:
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date")) if job.get("end_date") else date.today()
        if start and end:
            periods.append((start, end))

    periods.sort(key=lambda x: x[0])
    overlaps = 0
    for i in range(len(periods) - 1):
        _, end_i = periods[i]
        start_j, _ = periods[i + 1]
        # Allow 1 month transition gap
        if start_j < end_i and (end_i - start_j).days > 31:
            overlaps += 1
    return overlaps


# ─────────────────────────────────────────────
# SCORE NORMALISATION
# ─────────────────────────────────────────────

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def sigmoid(x: float, midpoint: float = 0.0, steepness: float = 1.0) -> float:
    """Soft sigmoid for smooth score transitions."""
    import math
    try:
        return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))
    except OverflowError:
        return 0.0 if x < midpoint else 1.0


def linear_interp(value: float, lo: float, hi: float) -> float:
    """Map value in [lo, hi] to [0, 1], clamped."""
    if hi <= lo:
        return 0.5
    return clamp((value - lo) / (hi - lo))


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_candidates_streaming(path: str):
    """
    Generator: yields one candidate dict at a time.
    Handles plain .jsonl or .jsonl.gz transparently.
    Memory-safe for 100K candidates.
    """
    path = Path(path)
    if not path.exists():
        # Try relative to project root
        path = Path(__file__).parent.parent / path
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    if path.suffix == ".gz":
        import gzip
        opener = lambda: gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = lambda: open(path, "r", encoding="utf-8")

    count = 0
    errors = 0
    with opener() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
                count += 1
            except json.JSONDecodeError as e:
                errors += 1
                if errors <= 5:
                    logger.warning(f"JSON parse error on line {count + errors}: {e}")

    logger.info(f"Loaded {count} candidates ({errors} parse errors)")


# ─────────────────────────────────────────────
# SKILL HELPERS
# ─────────────────────────────────────────────

def get_skills_dict(candidate: Dict) -> Dict[str, Dict]:
    """
    Return {normalised_skill_name: skill_obj} for fast lookup.
    """
    result = {}
    for skill in candidate.get("skills", []):
        name = normalise(skill.get("name", ""))
        if name:
            result[name] = skill
    return result


def skill_trust_score(skill: Dict) -> float:
    """
    Compute trust score [0,1] for a skill claim based on:
    - endorsements
    - duration_months
    - proficiency level
    Returns higher for well-endorsed, long-duration, verified skills.
    """
    endorsements = skill.get("endorsements", 0)
    duration = skill.get("duration_months", 0)
    proficiency = skill.get("proficiency", "beginner")

    # Proficiency baseline
    prof_base = {"beginner": 0.3, "intermediate": 0.55, "advanced": 0.75, "expert": 0.90}.get(proficiency, 0.3)

    # Endorsement boost (log-scaled, saturates at ~50 endorsements)
    import math
    end_score = min(1.0, math.log1p(endorsements) / math.log1p(50))

    # Duration (12+ months = full trust, < 3 months = suspicious)
    dur_score = clamp(duration / 12.0)

    # Weighted combination
    trust = 0.35 * prof_base + 0.40 * end_score + 0.25 * dur_score
    return clamp(trust)


def get_career_text(candidate: Dict) -> str:
    """All job descriptions concatenated."""
    return " ".join(
        job.get("description", "") for job in candidate.get("career_history", [])
    )


# ─────────────────────────────────────────────
# COMPANY TYPE DETECTION
# ─────────────────────────────────────────────

SERVICES_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree",
    "ltimindtree", "l&t infotech", "kpit", "cyient", "ibm", "deloitte",
    "pwc", "kpmg", "ey ", "ernst & young", "mckinsey", "bain", "bcg",
    "dunder mifflin", "globex", "initech", "umbrella corp", "acme",
}


def classify_company(company_name: str, industry: str) -> str:
    """
    Returns: 'product', 'services', 'startup', 'unknown'
    """
    cn = normalise(company_name)
    ind = normalise(industry)

    if any(s in cn for s in SERVICES_COMPANIES):
        return "services"

    # Industry hints
    if any(x in ind for x in ["it services", "outsourcing", "consulting", "bpo"]):
        return "services"

    if any(x in ind for x in ["saas", "software", "technology", "ai", "fintech",
                                "edtech", "healthtech", "startup"]):
        return "product"

    return "unknown"


def is_product_company(company: str, industry: str) -> bool:
    return classify_company(company, industry) == "product"


# ─────────────────────────────────────────────
# MISC
# ─────────────────────────────────────────────

def safe_get(d: Any, *keys, default=None):
    """Safe nested get."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
