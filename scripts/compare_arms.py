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


def load_arm(arm: str, n_eval: int | None) -> dict[int, dict]:
    """step -> eval record for every cached eval of this arm.

    n_eval=None matches ANY n. Arms are routinely evaluated at different n --
    hpc/grpo_eval.slurm defaults to N_EVAL=400 and the 3B arms were run with
    N_EVAL=1000 explicitly -- and requiring an exact filename match made
    perfectly comparable arms look like they shared no steps at all. Pairing is
    then done on the common PREFIX of per_example (see main), which is valid
    because evaluate_checkpoints.py always scores the first n rows of the same
    val parquet in order.
    """
    pattern = f"eval_{arm}-step*_n{n_eval}.json" if n_eval else f"eval_{arm}-step*_n*.json"
    out = {}
    for p in RESULTS.glob(pattern):
        m = re.search(r"-step(\d+)_n(\d+)\.json$", p.name)
        if not m:
            continue
        data = json.loads(p.read_text())
        rec = next(iter(data.values()))
        if "per_example" not in rec:
            continue
        step = int(m.group(1))
        # Same step evaluated at two different n: keep the larger.
        if step not in out or len(rec["per_example"]) > len(out[step]["per_example"]):
            out[step] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_a")
    ap.add_argument("arm_b")
    ap.add_argument("--metric", default="beq_plus", choices=["beq_plus", "typecheck"])
    ap.add_argument("--n-eval", type=int, default=0,
                    help="0 (default) = match any n and pair on the common prefix")
    args = ap.parse_args()


    a_recs, b_recs = (load_arm(args.arm_a, args.n_eval or None),
                      load_arm(args.arm_b, args.n_eval or None))
    common = sorted(set(a_recs) & set(b_recs))
    if not common:
        raise SystemExit(f"No common steps: {args.arm_a} has {sorted(a_recs)}, "
                         f"{args.arm_b} has {sorted(b_recs)}")

    # Truncate to the shortest record so both arms are scored on identical
    # examples. Rates are RECOMPUTED over that prefix rather than read from
    # *_rate, which describes whatever n the file was written at.
    n = min(min(len(a_recs[s]["per_example"]) for s in common),
            min(len(b_recs[s]["per_example"]) for s in common))
    print(f"metric={args.metric}  A={args.arm_a}  B={args.arm_b}  (paired on the first n={n})")
    hdr = f"{'step':>5} {'A%':>6} {'B%':>6} {'Δ(B-A) pp':>10} {'A>B':>5} {'B>A':>5} {'p':>8}"
    print(hdr)
    print("-" * len(hdr))
    for s in common:
        pa = a_recs[s]["per_example"][:n]
        pb = b_recs[s]["per_example"][:n]
        lost, gained, p = mcnemar_exact(pa, pb, args.metric)
        ra = 100 * sum(bool(x[args.metric]) for x in pa) / n
        rb = 100 * sum(bool(x[args.metric]) for x in pb) / n
        mark = "*" if p < 0.05 else " "
        print(f"{s:>5} {ra:>6.1f} {rb:>6.1f} "
              f"{rb - ra:>+10.1f} {lost:>5} {gained:>5} {p:>8.4f}{mark}")
    print("\n* = arms differ significantly at that step (McNemar exact, two-sided)")


if __name__ == "__main__":
    main()
