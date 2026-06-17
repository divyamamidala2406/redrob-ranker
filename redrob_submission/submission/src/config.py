"""
config.py
All configuration constants derived from deep analysis of the JD and evaluation criteria.
Edit weights here to tune without touching logic files.
"""

# ─────────────────────────────────────────────
# SCORING WEIGHTS  (must sum to 1.0)
# Calibrated to NDCG@10 dominance (0.50 weight in evaluation)
# ─────────────────────────────────────────────
WEIGHTS = {
    "skill":       0.32,   # Highest — JD is very skill-specific
    "experience":  0.20,   # Years + production depth
    "career":      0.18,   # Title quality, company type, trajectory
    "behavior":    0.15,   # Redrob signals (availability, engagement)
    "education":   0.08,   # Tier + field relevance
    "location":    0.07,   # Pune/Noida/India preferred
}

# ─────────────────────────────────────────────
# JD REQUIREMENTS — Senior AI Engineer @ Redrob
# ─────────────────────────────────────────────

# Hard-required skills (MUST have ≥1 from each group to be competitive)
MUST_HAVE_SKILL_GROUPS = {
    "embeddings": [
        "sentence-transformers", "sentence transformers", "embeddings", "text embeddings",
        "openai embeddings", "bge", "e5", "embedding", "dense retrieval",
        "semantic embeddings", "vector embeddings",
    ],
    "vector_db": [
        "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
        "elasticsearch", "vector database", "vector db", "vector search",
        "hybrid search", "ann", "approximate nearest neighbor",
    ],
    "retrieval": [
        "retrieval", "information retrieval", "semantic search", "bm25",
        "hybrid retrieval", "dense retrieval", "sparse retrieval", "rag",
        "retrieval augmented generation", "reranking", "re-ranking",
    ],
    "python": [
        "python", "python3", "pyspark",
    ],
}

# Nice-to-have skills (boost score but not required)
NICE_TO_HAVE_SKILLS = [
    # LLM fine-tuning
    "lora", "qlora", "peft", "fine-tuning", "fine-tune", "fine tuning",
    "finetuning", "llm fine-tuning", "instruction tuning",
    # Learning-to-rank
    "learning to rank", "ltr", "lambdamart", "ranknet", "xgboost ranking",
    "neural ranking", "pointwise", "pairwise", "listwise",
    # Evaluation frameworks
    "ndcg", "mrr", "map", "a/b testing", "ab testing", "offline evaluation",
    "ranking evaluation", "precision@k", "recall@k",
    # LLMs
    "llm", "large language model", "gpt", "claude", "llama", "mistral",
    "transformer", "bert", "roberta",
    # MLOps / infra
    "mlflow", "weights & biases", "wandb", "ray", "triton", "onnx",
    "model serving", "inference optimization",
    # Adjacent ML
    "nlp", "natural language processing", "text classification", "ranking",
]

# Title keywords that signal AI Engineering background (career score boost)
STRONG_AI_TITLES = [
    "ai engineer", "ml engineer", "machine learning engineer",
    "applied scientist", "applied ml", "applied ai",
    "research engineer", "nlp engineer", "search engineer",
    "ranking engineer", "relevance engineer", "retrieval engineer",
    "data scientist", "senior data scientist", "staff data scientist",
    "senior ml", "staff ml", "principal ml",
    "senior ai", "staff ai", "principal ai",
    "founding engineer", "founding ai",
]

WEAK_AI_TITLES = [
    "data analyst", "business analyst", "product analyst",
    "software engineer", "backend engineer", "fullstack",
    "frontend engineer", "devops", "sre",
]

# Companies flagged as consulting-only (JD explicitly penalises pure consulting)
CONSULTING_ONLY_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree",
    "l&t infotech", "ltimindtree", "kpit", "cyient",
}

# Product-company signals (JD strongly prefers product experience)
PRODUCT_COMPANY_SIGNALS = [
    "startup", "series a", "series b", "seed", "saas", "product company",
    "product-led", "b2b saas", "platform", "ai-native",
]

# Indian locations preferred by JD
PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore",
    "bengaluru", "delhi ncr", "ncr", "gurgaon", "gurugram", "india",
}

# ─────────────────────────────────────────────
# EXPERIENCE THRESHOLDS
# ─────────────────────────────────────────────
EXP_IDEAL_MIN = 5.0
EXP_IDEAL_MAX = 9.0
EXP_HARD_MIN  = 2.0    # Below this = hard disqualifier territory
EXP_HARD_MAX  = 15.0   # Above this = senior/management drift risk

# Notice period (JD says sub-30 ideal, buyout up to 30, 30+ loses points)
NOTICE_IDEAL_MAX = 30
NOTICE_BUYOUT_MAX = 60  # Still acceptable
NOTICE_HARD_MAX = 90    # 90+ is a real hiring risk

# Salary range in INR LPA the role likely targets (adjust if you know better)
SALARY_TARGET_MIN = 25.0
SALARY_TARGET_MAX = 60.0

# ─────────────────────────────────────────────
# TRAP DETECTION THRESHOLDS
# ─────────────────────────────────────────────
# Keyword stuffing: skill claimed at expert/advanced but endorsements and
# duration are suspiciously low
STUFFING_MIN_ENDORSEMENTS = 3     # < 3 endorsements for "expert" = suspicious
STUFFING_MIN_DURATION_MONTHS = 6  # < 6 months for "advanced/expert" = suspicious
STUFFING_MAX_SKILLS_PER_CATEGORY = 15  # > 15 skills total = suspicious density

# Honeypot detection: impossible profiles
HONEYPOT_MAX_SIMULTANEOUS_JOBS = 2  # >2 overlapping roles = impossible
HONEYPOT_MAX_YOE_VS_CAREER = 3.0    # If declared yoe >> sum of career months/12 by this factor
HONEYPOT_MIN_ASSESSMENT_VS_CLAIM = 20.0  # Claimed expert but assessment < 20 = fake

# Signal manipulation: inflated signals with no real activity
SIGNAL_FRAUD_SAVED_VS_VIEWS_RATIO = 0.9  # saved > 90% of views = suspicious
SIGNAL_FRAUD_MIN_COMPLETENESS = 40.0     # Very low completeness + high signals = suspect

# Penalties applied to final score (0.0 – 1.0 reduction)
PENALTY_KEYWORD_STUFFING = 0.12
PENALTY_HONEYPOT = 0.40            # Near-disqualification
PENALTY_PURE_CONSULTING = 0.10
PENALTY_FAKE_EXPERTISE = 0.15
PENALTY_INACTIVE = 0.08            # Last active > 120 days ago
PENALTY_NO_PRODUCTION = 0.10       # No evidence of production deployment in descriptions
PENALTY_PURE_RESEARCH = 0.12       # Academic/research-only background
PENALTY_OUTSIDER_DOMAIN = 0.15     # CV/robotics/hardware without NLP

# ─────────────────────────────────────────────
# BEHAVIORAL SIGNAL WEIGHTS (within behavior score)
# ─────────────────────────────────────────────
BEHAVIOR_WEIGHTS = {
    "open_to_work":          0.15,
    "recency":               0.20,   # last_active_date recency
    "response_rate":         0.15,
    "response_speed":        0.08,
    "completeness":          0.10,
    "github_activity":       0.12,
    "recruiter_interest":    0.10,   # saved_by_recruiters_30d
    "interview_completion":  0.10,
}

# ─────────────────────────────────────────────
# EDUCATION SCORING
# ─────────────────────────────────────────────
EDUCATION_TIER_SCORES = {
    "tier_1": 1.0,
    "tier_2": 0.75,
    "tier_3": 0.50,
    "tier_4": 0.25,
    "unknown": 0.35,
}

RELEVANT_FIELDS = [
    "computer science", "cs", "information technology", "it",
    "electronics", "electrical", "data science", "statistics",
    "mathematics", "applied mathematics", "machine learning",
    "artificial intelligence", "ai", "ml", "engineering",
    "computational", "cognitive science",
]

DEGREE_SCORES = {
    "phd": 1.0, "ph.d": 1.0,
    "m.tech": 0.85, "m.e.": 0.85, "m.s.": 0.85, "ms": 0.85,
    "m.sc": 0.80, "msc": 0.80, "mca": 0.75,
    "b.tech": 0.70, "b.e.": 0.70, "be": 0.70,
    "b.sc": 0.60, "bsc": 0.60, "b.s.": 0.60,
    "mba": 0.50,
    "b.a.": 0.35, "ba": 0.35,
}

# ─────────────────────────────────────────────
# PIPELINE CONFIG
# ─────────────────────────────────────────────
TOP_K = 100                  # How many candidates to output
CANDIDATES_FILE = "data/candidates.jsonl"
OUTPUT_FILE = "outputs/submission.csv"
REFERENCE_DATE = "2026-06-16"  # Used for recency calculations

# How many candidates to process in streaming batches (memory safety)
BATCH_SIZE = 5000
