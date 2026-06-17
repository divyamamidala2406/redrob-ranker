#!/usr/bin/env python3
"""
main.py
Redrob Hackathon — Intelligent Candidate Ranking Engine
Entry point. Run: python main.py [--candidates PATH] [--out PATH] [--top-k N] [--verbose]
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

# Ensure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.config import CANDIDATES_FILE, OUTPUT_FILE, TOP_K
from src.explainability import generate_reasoning, print_candidate_report
from src.preprocess import preprocess_candidate
from src.ranking_engine import rank_candidates, score_all_candidates
from src.utils import load_candidates_streaming, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Redrob Intelligent Candidate Ranker"
    )
    parser.add_argument(
        "--candidates",
        default=CANDIDATES_FILE,
        help=f"Path to candidates.jsonl or .jsonl.gz (default: {CANDIDATES_FILE})",
    )
    parser.add_argument(
        "--out",
        default=OUTPUT_FILE,
        help=f"Output CSV path (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of top candidates to output (default: {TOP_K})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed report for top-10 candidates",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run the official validator after generating submission",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(logging.INFO)
    logger = logging.getLogger("main")

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Redrob Intelligent Candidate Ranking Engine")
    logger.info("=" * 60)

    # ── Step 1: Load & preprocess ────────────────────────────────
    logger.info(f"Step 1/5: Loading candidates from {args.candidates}")
    raw_stream = load_candidates_streaming(args.candidates)

    def preprocess_stream():
        for raw in raw_stream:
            cleaned = preprocess_candidate(raw)
            if cleaned is not None:
                yield cleaned

    # ── Step 2: Score all candidates ────────────────────────────
    logger.info("Step 2/5: Scoring candidates (this is the heavy step)...")
    scored_list = score_all_candidates(preprocess_stream())
    logger.info(f"  → Scored {len(scored_list)} candidates in {time.time()-t0:.1f}s")

    # ── Step 3: Rank ─────────────────────────────────────────────
    logger.info("Step 3/5: Ranking...")
    ranked = rank_candidates(scored_list)

    # Quick stats
    honeypots_in_top100 = sum(1 for r in ranked[:100] if r.get("is_honeypot"))
    flagged_in_top100 = sum(1 for r in ranked[:100] if r.get("trap_flags"))
    logger.info(f"  Top-100 stats: honeypots={honeypots_in_top100}, flagged={flagged_in_top100}")
    logger.info(f"  Score range (top-100): "
                f"{ranked[99]['final_score']:.4f} — {ranked[0]['final_score']:.4f}")

    # ── Step 4: Generate reasoning ───────────────────────────────
    logger.info("Step 4/5: Generating explanations...")
    top_k = ranked[:args.top_k]
    for scored in top_k:
        scored["reasoning"] = generate_reasoning(scored)

    # Optional verbose report for top-10
    if args.verbose:
        for scored in top_k[:10]:
            print_candidate_report(scored, scored["rank"])

    # ── Step 5: Write submission CSV ─────────────────────────────
    logger.info(f"Step 5/5: Writing submission to {args.out}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for scored in top_k:
            writer.writerow([
                scored["candidate_id"],
                scored["rank"],
                f"{scored['final_score']:.6f}",
                scored["reasoning"],
            ])

    total_time = time.time() - t0
    logger.info(f"Done in {total_time:.1f}s  →  {out_path}")

    # ── Optional: validate ───────────────────────────────────────
    if args.validate:
        logger.info("Running official validator...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "validate_submission.py", str(out_path)],
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            sys.exit(1)

    # ── Summary ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"SUBMISSION READY: {out_path}")
    logger.info(f"Total time: {total_time:.1f}s | Candidates scored: {len(scored_list)}")
    logger.info(f"Top candidate: {ranked[0]['candidate_id']} "
                f"(score={ranked[0]['final_score']:.4f})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
