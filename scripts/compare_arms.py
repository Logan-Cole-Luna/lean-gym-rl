#!/usr/bin/env python3
"""Paired arm-vs-arm comparison at matched training steps.

The placebo control (compute_score_placebo) only means something relative to
the real arm at the SAME step on the SAME validation slice: both arms start
from sft-step390, see the same prompt order, and differ only in what the
reward pays. This script pairs their per-example eval records step by step and
runs McNemar's exact test between the ARMS (select_checkpoint.py tests each
arm against SFT, which is a different question).

Usage:
    python scripts/compare_arms.py rl_gated_clean rl_placebo_clean
    python scripts/compare_arms.py rl_gated_clean rl_gated_bigbatch --metric typecheck
"""
from __future__ import annotations

import argparse
import json
import re
from math import comb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"


def mcnemar_exact(a, b, key):
    n = min(len(a), len(b))
    lost = sum(1 for i in range(n) if a[i][key] and not b[i][key])
    gained = sum(1 for i in range(n) if not a[i][key] and b[i][key])
    disc = lost + gained
    if disc == 0:
        return lost, gained, 1.0
    k = min(lost, gained)
    return lost, gained, min(2.0 * sum(comb(disc, i) for i in range(k + 1)) / (2 ** disc), 1.0)


def load_arm(arm: str, n_eval: int) -> dict[int, dict]:
    """step -> eval record for every cached eval of this arm."""
    out = {}
    for p in RESULTS.glob(f"eval_{arm}-step*_n{n_eval}.json"):
        m = re.search(r"-step(\d+)_n\d+\.json$", p.name)
        if not m:
            continue
        data = json.loads(p.read_text())
        rec = next(iter(data.values()))
        if "per_example" in rec:
            out[int(m.group(1))] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_a")
    ap.add_argument("arm_b")
    ap.add_argument("--metric", default="beq_plus", choices=["beq_plus", "typecheck"])
    ap.add_argument("--n-eval", type=int, default=400)
    args = ap.parse_args()

    rate_key = "beq_plus_rate" if args.metric == "beq_plus" else "typecheck_rate"
    a_recs, b_recs = load_arm(args.arm_a, args.n_eval), load_arm(args.arm_b, args.n_eval)
    common = sorted(set(a_recs) & set(b_recs))
    if not common:
        raise SystemExit(f"No common steps: {args.arm_a} has {sorted(a_recs)}, "
                         f"{args.arm_b} has {sorted(b_recs)}")

    print(f"metric={args.metric}  A={args.arm_a}  B={args.arm_b}  (paired, n={args.n_eval})")
    hdr = f"{'step':>5} {'A%':>6} {'B%':>6} {'Δ(B-A) pp':>10} {'A>B':>5} {'B>A':>5} {'p':>8}"
    print(hdr)
    print("-" * len(hdr))
    for s in common:
        ra, rb = a_recs[s], b_recs[s]
        lost, gained, p = mcnemar_exact(ra["per_example"], rb["per_example"], args.metric)
        mark = "*" if p < 0.05 else " "
        print(f"{s:>5} {100*ra[rate_key]:>6.1f} {100*rb[rate_key]:>6.1f} "
              f"{100*(rb[rate_key]-ra[rate_key]):>+10.1f} {lost:>5} {gained:>5} {p:>8.4f}{mark}")
    print("\n* = arms differ significantly at that step (McNemar exact, two-sided)")


if __name__ == "__main__":
    main()
