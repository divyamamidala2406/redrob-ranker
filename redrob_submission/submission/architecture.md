# System Architecture

## Intelligent Candidate Ranking Engine

---

## Data Flow

```
candidates.jsonl (100K records)
         │
         ▼
┌─────────────────────┐
│   load_candidates   │  Generator-based streaming
│   _streaming()      │  → never loads full 465MB into RAM
└────────┬────────────┘
         │  raw dicts, one at a time
         ▼
┌─────────────────────┐
│  preprocess_        │  • Date parsing & validation
│  candidate()        │  • Duration recalculation (catches timeline fraud)
│  preprocess.py      │  • Skill deduplication
│                     │  • Derived fields: _career_years, _product_company_count
│                     │  • Normalised text fields (_name_norm, _desc_norm, etc.)
└────────┬────────────┘
         │  cleaned candidate dict
         ▼
┌─────────────────────────────────────────────────────┐
│                 score_candidate()                   │
│                 ranking_engine.py                   │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ skill_score  │  │  exp_score   │  │career_scor│ │
│  │ (32%)        │  │  (20%)       │  │(18%)      │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │behav_score   │  │  edu_score   │  │ loc_score │ │
│  │ (15%)        │  │  (8%)        │  │(7%)       │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
│  ┌──────────────────────────────────────────────┐   │
│  │  production_depth_score  (unique bonus +8%)  │   │
│  └──────────────────────────────────────────────┘   │
│                          +                          │
│  ┌──────────────────────────────────────────────┐   │
│  │       detect_traps()  trap_detection.py      │   │
│  │  • Honeypot detector (date overlaps, YoE gap)│   │
│  │  • Keyword stuffer detector                  │   │
│  │  • Fake expertise detector (assessment gap)  │   │
│  │  • Pure consulting detector                  │   │
│  │  • Research-only detector                    │   │
│  │  • Domain mismatch detector                  │   │
│  │  • Signal manipulation detector              │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  final = (weighted_sum + prod_bonus) × penalty_mult │
└────────┬────────────────────────────────────────────┘
         │  List[scored_dict]  (100K items, ~1.2GB)
         ▼
┌─────────────────────┐
│   rank_candidates() │  Sort by (-final_score, candidate_id)
│   ranking_engine.py │  Assign ranks 1..100K
└────────┬────────────┘
         │  Top-100 ranked dicts
         ▼
┌─────────────────────┐
│  generate_reasoning │  1-2 sentence explanation per candidate
│  explainability.py  │  Pulls from debug dicts, no LLM needed
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   submission.csv    │  candidate_id, rank, score, reasoning
│   (UTF-8, 101 rows) │  Validated by official validate_submission.py
└─────────────────────┘
```

---

## Models Used

This solution deliberately uses **zero external ML models** during ranking.

**Why?**
- Constraint: 5-minute CPU-only runtime, no network
- 100K candidates × embedding model = 2-3 hours on CPU
- Rule-based scoring with calibrated weights outperforms naive semantic search
  because the JD is highly specific (not ambiguous concept matching)

**What we use instead:**
1. **Multi-factor keyword matching** with trust scoring (endorsements × duration)
2. **Weighted production-depth signals** (30+ domain-specific patterns)
3. **Piecewise scoring functions** for continuous features (YoE, recency)
4. **Penalty multipliers** for trap candidates (not additive penalties, which can be gamed)

**If pre-computation is allowed** (set `pre_computation_required: true` in metadata):
- You could run `sentence-transformers/all-MiniLM-L6-v2` offline to embed all candidates
- Then cosine-similarity rank against JD embedding at query time (< 1 second)
- See `notebooks/exploration.ipynb` for this experimental path

---

## Ranking Logic

### Score Formula

```python
final_score = (
    0.32 * skill_score      +   # Dominant — JD is skill-specific
    0.20 * exp_score        +   # Years + production AI evidence
    0.18 * career_score     +   # Title, company type, trajectory
    0.15 * behavior_score   +   # Availability + platform engagement
    0.08 * edu_score        +   # Institution tier + field relevance
    0.07 * loc_score        +   # India / Pune-Noida preference
    prod_depth * 0.08           # Bonus: production scale evidence
) * penalty_multiplier          # 0.20 for honeypots, 0.50-1.0 for others
```

### Weight Justification

| Component | Weight | Justification |
|-----------|--------|---------------|
| Skill | 32% | JD lists 12+ specific technologies; a perfect title with wrong skills fails |
| Experience | 20% | JD specifies 5-8 years; too junior or too senior both fail |
| Career | 18% | "Product company preferred"; trajectory matters more than current title |
| Behavior | 15% | A brilliant candidate who doesn't respond to recruiters has zero hire probability |
| Education | 8% | IIT/IISc is a soft signal, not a hard requirement for this role |
| Location | 7% | JD explicitly says Pune/Noida; remote-only candidates lose traction |

### Tie-Breaking
Per spec: equal scores → sort by `candidate_id` ascending.

---

## Trap Detection Logic

### Honeypot Detection (Penalty: ×0.20)
The dataset contains ~80 honeypots with "subtly impossible profiles". Our detectors:

1. **Overlapping jobs**: Parse all start/end dates, detect simultaneous employment (>31 day overlap). Real people can't work 3 full-time jobs.

2. **YoE vs career history ratio**: If declared `years_of_experience` is >2.5× the sum of all `duration_months` in career history, the profile is fabricated.

3. **Copy-pasted descriptions**: Compute Jaccard similarity between all job descriptions. If ≥2 pairs share >85% of words, the profile is synthetic.

4. **Future dates**: Any `start_date` after today = impossible.

5. **Low completeness + high recruiter saves**: A 20% complete profile saved by 25 recruiters = signal inflation.

### Keyword Stuffing Detection (Penalty: −12%)
For each high-value AI skill (embeddings, RAG, FAISS, etc.) claimed at advanced/expert:
- Check endorsements < 3 AND duration_months < 6 AND not mentioned in any job description
- If 4+ such skills → stuffer flag

### Fake Expertise Detection (Penalty: −15%)
If candidate claims "expert" in a skill but completed a Redrob platform assessment scoring < 20/100, the claim is contradicted by objective data.

### Pure Consulting Detection (Penalty: −10%)
Normalise all company names. If all roles are at known IT services companies (TCS, Infosys, Wipro, Accenture, etc.), apply penalty. The JD explicitly values product company experience.

---

## Memory & Performance

| Dataset size | RAM usage | Runtime |
|---|---|---|
| 1,000 candidates | ~15 MB | ~2s |
| 10,000 candidates | ~150 MB | ~18s |
| 100,000 candidates | ~1.2 GB | ~90-150s |

**Streaming architecture**: Candidates are processed one at a time through the preprocessing pipeline. Only the scored results list (lightweight dicts without raw text) is held in memory simultaneously.

**Bottleneck**: String operations in `feature_engineering.py` (normalise + count_matches). Optimisation: pre-normalise all text in `preprocess.py` once, reuse the `_norm` fields everywhere.

---

## Leaderboard Optimisation Suggestions

1. **Offline sentence embeddings** (if pre-computation allowed): Encode all candidates with `all-mpnet-base-v2`, store as numpy `.npy`. At ranking time, cosine-similarity against JD vector in < 1s. Add as a 4th component alongside the rule-based score.

2. **XGBoost reranker**: Label top-200 candidates from the rule-based pass, use component scores as features, train a pointwise XGBoost ranker on synthetic preference labels derived from the JD.

3. **LLM-based re-scoring of top-500**: Run top-500 candidates through a local Llama 3 8B with a structured prompt asking to score 0-100 against the JD. Use this as a re-ranking layer (< 5 min at 4-bit quantisation on CPU).

4. **Ensemble**: 0.6 × rule_score + 0.4 × semantic_score. The rule system handles trap detection reliably; the semantic system handles edge cases and synonyms.

5. **Calibrate weights on sample**: Run the pipeline on `sample_candidates.json`, manually inspect the top-20, adjust `WEIGHTS` in `config.py` based on which candidates feel right.
