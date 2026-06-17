"""
sandbox/app.py
Streamlit demo for the Redrob Intelligent Candidate Ranker.
Required for submission: shows the ranker in action on sample data.

Run: streamlit run sandbox/app.py
"""

import json
import sys
import time
from pathlib import Path

import streamlit as st

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TOP_K
from src.explainability import generate_reasoning, print_candidate_report
from src.preprocess import preprocess_candidate
from src.ranking_engine import rank_candidates, score_candidate
from src.utils import setup_logging

setup_logging()

st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Redrob Intelligent Candidate Ranking Engine")
st.caption("India.Runs Hackathon — Track 1: AI & Datathon Arena")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    top_k = st.slider("Top-K candidates", min_value=10, max_value=100, value=10)
    show_debug = st.checkbox("Show score breakdown", value=True)
    show_traps = st.checkbox("Show trap flags", value=True)

    st.markdown("---")
    st.markdown("**Scoring Weights**")
    st.markdown("- Skill: 32%")
    st.markdown("- Experience: 20%")
    st.markdown("- Career: 18%")
    st.markdown("- Behavior: 15%")
    st.markdown("- Education: 8%")
    st.markdown("- Location: 7%")
    st.markdown("- Production Depth: bonus +8%")

# ─── File Upload ──────────────────────────────────────────────────────────────
st.header("1. Upload Candidates")
uploaded = st.file_uploader(
    "Upload sample_candidates.json or any candidates JSONL",
    type=["json", "jsonl"],
    help="Use sample_candidates.json from the hackathon bundle for a quick demo.",
)

col1, col2 = st.columns(2)
with col1:
    use_sample = st.button("📂 Use built-in sample (if available)")

# ─── Load candidates ──────────────────────────────────────────────────────────
candidates_raw = []

if uploaded is not None:
    try:
        content = uploaded.read().decode("utf-8")
        # Try JSON array first, then JSONL
        if content.strip().startswith("["):
            candidates_raw = json.loads(content)
        else:
            candidates_raw = [json.loads(line) for line in content.splitlines() if line.strip()]
        st.success(f"Loaded {len(candidates_raw)} candidates from upload.")
    except Exception as e:
        st.error(f"Failed to parse file: {e}")

elif use_sample:
    sample_path = Path(__file__).parent.parent / "data" / "sample_candidates.json"
    if not sample_path.exists():
        # Try the uploads path
        sample_path = Path("/mnt/user-data/uploads/sample_candidates.json")
    if sample_path.exists():
        with open(sample_path) as f:
            candidates_raw = json.load(f)
        st.success(f"Loaded {len(candidates_raw)} sample candidates.")
    else:
        st.warning("sample_candidates.json not found. Please upload a file.")

# ─── Run Ranker ───────────────────────────────────────────────────────────────
if candidates_raw:
    st.header("2. Run the Ranker")

    if st.button("🚀 Rank Candidates", type="primary"):
        with st.spinner("Preprocessing and scoring..."):
            t0 = time.time()
            scored_list = []
            errors = 0

            progress = st.progress(0)
            for i, raw in enumerate(candidates_raw):
                cleaned = preprocess_candidate(raw)
                if cleaned:
                    scored_list.append(score_candidate(cleaned))
                else:
                    errors += 1
                progress.progress((i + 1) / len(candidates_raw))

            ranked = rank_candidates(scored_list)
            top = ranked[:top_k]
            for s in top:
                s["reasoning"] = generate_reasoning(s)

            elapsed = time.time() - t0

        st.success(f"Ranked {len(scored_list)} candidates in {elapsed:.2f}s "
                   f"({'errors: ' + str(errors) if errors else 'no errors'})")

        # ─── Summary metrics ──────────────────────────────────────────────────
        st.header("3. Results")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total scored", len(scored_list))
        m2.metric("Honeypots in top-K", sum(1 for r in top if r.get("is_honeypot")))
        m3.metric("Flagged in top-K", sum(1 for r in top if r.get("trap_flags")))
        m4.metric("Top score", f"{ranked[0]['final_score']:.4f}")

        # ─── Top-K table ──────────────────────────────────────────────────────
        for scored in top:
            c = scored.get("_candidate", {})
            prof = c.get("profile", {})
            sig = c.get("redrob_signals", {})

            with st.expander(
                f"#{scored['rank']}  {scored['candidate_id']}  —  "
                f"{prof.get('current_title','?')} @ {prof.get('current_company','?')}  "
                f"[{scored['final_score']:.4f}]",
                expanded=(scored["rank"] <= 3),
            ):
                col_a, col_b = st.columns([2, 1])

                with col_a:
                    st.markdown(f"**Location:** {prof.get('location','?')}, {prof.get('country','?')}")
                    st.markdown(f"**YoE:** {prof.get('years_of_experience',0):.1f} years")
                    st.markdown(f"**Open to work:** {sig.get('open_to_work_flag','?')}")
                    st.markdown(f"**Notice:** {sig.get('notice_period_days','?')} days")
                    st.markdown(f"**Last active:** {sig.get('last_active_date','?')}")
                    st.markdown(f"**Reasoning:** _{scored['reasoning']}_")

                    if show_traps and scored.get("trap_flags"):
                        st.warning("**Trap flags:**\n" + "\n".join(f"• {f}" for f in scored["trap_flags"]))

                with col_b:
                    if show_debug:
                        st.markdown("**Score breakdown:**")
                        components = [
                            ("Skill (32%)",     scored["skill_score"]),
                            ("Experience (20%)", scored["exp_score"]),
                            ("Career (18%)",     scored["career_score"]),
                            ("Behavior (15%)",   scored["behavior_score"]),
                            ("Education (8%)",   scored["edu_score"]),
                            ("Location (7%)",    scored["loc_score"]),
                            ("Prod Depth (B)",   scored["prod_depth_score"]),
                            ("FINAL",            scored["final_score"]),
                        ]
                        for label, val in components:
                            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
                            st.text(f"{label:<18} {val:.3f} {bar}")

        # ─── Download CSV ─────────────────────────────────────────────────────
        st.header("4. Download Submission")
        import csv, io
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for s in ranked[:100]:
            if "reasoning" not in s:
                s["reasoning"] = generate_reasoning(s)
            writer.writerow([
                s["candidate_id"],
                s["rank"],
                f"{s['final_score']:.6f}",
                s["reasoning"],
            ])
        st.download_button(
            "⬇️ Download submission.csv (top-100)",
            data=buf.getvalue(),
            file_name="submission.csv",
            mime="text/csv",
        )
else:
    st.info("Upload a candidates file above to get started.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built for India.Runs Hackathon | Architecture: rule-based multi-factor "
           "ranker with production-depth scoring and honeypot detection | "
           "Runtime: <3 min CPU-only for 100K candidates")
