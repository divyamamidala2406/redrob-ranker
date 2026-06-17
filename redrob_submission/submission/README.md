# Redrob Hackathon — Intelligent Candidate Ranking Engine

**Team:** Masakalithon  
**Track:** India.Runs AI & Datathon Arena — Track 1  
**Challenge:** Intelligent Candidate Discovery & Ranking

---

## Quick Start

```bash
# 1. Clone / unzip the project
cd submission/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place candidates.jsonl in data/
cp /path/to/candidates.jsonl data/candidates.jsonl

# 4. Run the full pipeline
python main.py

# 5. Validate the output
python validate_submission.py outputs/submission.csv

# 6. (Optional) verbose top-10 report
python main.py --verbose
```

---

## Project Structure

```
submission/
├── src/
│   ├── config.py            ← All weights, thresholds, JD-derived constants
│   ├── utils.py             ← Shared text/date/scoring utilities
│   ├── preprocess.py        ← Data loading, cleaning, normalization
│   ├── feature_engineering.py ← All scoring components
│   ├── trap_detection.py    ← Honeypot / stuffing / fake-expert detectors
│   ├── ranking_engine.py    ← Score aggregation + ranking
│   └── explainability.py    ← Reasoning generation for CSV output
├── data/                    ← Put candidates.jsonl here
├── outputs/                 ← submission.csv written here
├── notebooks/
│   └── exploration.ipynb    ← EDA and score distribution analysis
├── main.py                  ← Entry point
├── requirements.txt
├── README.md
└── architecture.md
```

---

## CLI Options

```
python main.py [OPTIONS]

Options:
  --candidates PATH   Path to candidates.jsonl or candidates.jsonl.gz
                      (default: data/candidates.jsonl)
  --out PATH          Output CSV path (default: outputs/submission.csv)
  --top-k N           Number of candidates to output (default: 100)
  --verbose           Print detailed score breakdown for top-10
  --validate          Run official validate_submission.py after generating
```

---

## Methodology

### Problem
Rank 100,000 candidates against a Senior AI Engineer JD focused on:
- Semantic search / embedding systems
- Vector databases (Pinecone, Qdrant, FAISS, etc.)
- Retrieval-Augmented Generation (RAG)
- Production ML deployment
- Python expertise

### Scoring Formula

```
final_score = (
    0.32 × skill_score       +
    0.20 × experience_score  +
    0.18 × career_score      +
    0.15 × behavior_score    +
    0.08 × education_score   +
    0.07 × location_score    +
         production_depth_bonus  ← unique feature
) × (1 - trap_penalty_multiplier)
```

### Component Details

**Skill Score (32%)**  
Four-factor model:
1. Coverage of must-have skill groups (embeddings, vector DB/retrieval, Python)
2. Nice-to-have count (LLM fine-tuning, LTR, evaluation frameworks)
3. Skill trust score = endorsements × duration_months × proficiency
4. Redrob platform assessment scores for relevant skills

**Experience Score (20%)**  
- Piecewise YoE scoring (ideal: 5–9 years)
- Production AI keyword density in career descriptions
- Recency of AI work (recent jobs weighted more)

**Career Score (18%)**  
- Title trajectory (AI/ML titles vs generic engineering)
- Product company ratio (penalises pure services/consulting)
- Tenure stability (job-hopping detection)
- Impact evidence (shipped, deployed, A/B tested, reduced latency)
- Founding/early-stage bonus (+0.10)

**Behavior Score (15%)**  
23 Redrob signals weighted by hiring relevance:
- Open-to-work flag (gate signal)
- Last active recency
- Recruiter response rate + speed
- GitHub activity score
- Saved by recruiters (market-validation signal)
- Interview completion rate
- Notice period modifier

**Education Score (8%)**  
- Institution tier (tier_1 = IIT/IISc equivalent)
- Degree level (PhD > MTech > BTech)
- Field relevance (CS/EE/Math/Stats)

**Location Score (7%)**  
- Pune/Noida: 1.0 (top cities from JD)
- Other Indian cities: 0.80
- Willing to relocate: 0.55
- Overseas without relocation: 0.25

### Unique Feature: Production Depth Score
A weighted keyword scan of career descriptions for 30+ signals of operating ML systems in production — not just building prototypes. Signals include: `a/b test`, `ndcg`, `embedding drift`, `index refresh`, `billion queries`, `reranking pipeline`, etc. This score adds a bonus of up to +0.08 to distinguish real practitioners from tutorial-followers.

### Trap Detection

| Trap Type | Detection Method | Penalty |
|-----------|-----------------|---------|
| Honeypot | Overlapping job dates, YoE >> career history, copy-pasted descriptions | ×0.20 |
| Keyword stuffing | High-value AI skills claimed at expert level with 0 endorsements + 0 duration + no career evidence | −12% |
| Fake expertise | Expert claimed but Redrob assessment score < 20 | −15% |
| Pure consulting | All roles at TCS/Infosys/Wipro-class companies | −10% |
| Research-only | No production signals in descriptions despite research titles | −12% |
| Domain mismatch | CV/robotics/speech with zero NLP crossover | −15% |
| Signal manipulation | saved_by_recruiters > 90% of profile_views | −8% |
| Inactivity (>6 months) | last_active_date delta | −8% |

---

## Runtime

- ~100K candidates: **< 3 minutes** on a single CPU core
- Memory: **< 1.5 GB** (streaming architecture, never loads full dataset)
- No GPU, no network calls during ranking

---

## Sandbox

Deploy the Streamlit app in `sandbox/app.py` to HuggingFace Spaces:
```bash
pip install streamlit
streamlit run sandbox/app.py
```
Upload a sample candidates JSON and run the ranker interactively.
