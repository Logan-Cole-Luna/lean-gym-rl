#!/usr/bin/env python3
"""Extract the DEAD BAND from an already-scored rollout pool.

The dead band is the rollouts that **type-check but earn nothing**: no direction
of BEq+ proves, so `compute_score_gated` scores them 0.0 -- identical to
unparseable garbage. At the 3B SFT baseline that is ~32% of all rollouts, and it
is the structural reason the gain channel has been pinned (`results/FINDINGS.md`:
"for a third of prompts the reward is constant, so GRPO's advantage is zero
there no matter how the group is composed").

WHY SUBSET RATHER THAN RE-SCORE EVERYTHING. The question Phase 0 asks is about
the dead band only: how often does BEq+'s cascade already reach rung 3, where it
computes -- and discards -- whether the PREDICTION proves on its own? Scoring a
BEq+ hit or a non-elaborating rollout tells us nothing about that, and the dead
band is also the *expensive* class (it runs the full cascade to exhaustion, ~175
Lean-seconds). Selecting on the previous scoring makes every example we pay for
answer the question.

    python scripts/make_deadband_subset.py \
        --scored data_3b/rollouts/passk_sft3b-step93_k32.scored.jsonl \
        --rollouts data_3b/rollouts/passk_sft3b-step93_k32.jsonl \
        --out data_3b/rollouts/deadband_sft3b-step93.jsonl --n 600

Output is in `scripts/score_rollouts.py`'s input format, so it re-scores with no
special casing.
"""
import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="*.scored.jsonl from a previous pass")
    ap.add_argument("--rollouts", required=True, help="the matching raw *.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600,
                    help="sample size; 600 gives ~+-4pp at 95%% on a proportion")
    ap.add_argument("--seed", type=int, default=0, help="pinned, so the subset is reproducible")
    ap.add_argument("--band", choices=["dead", "no_typecheck", "one_direction", "beq_plus"],
                    default="dead")
    args = ap.parse_args()

    def band_of(r: dict) -> str:
        if not r.get("typecheck"):
            return "no_typecheck"
        if r.get("beq_plus"):
            return "beq_plus"
        if int(r.get("semantic_signal", 0)) >= 1:
            return "one_direction"
        return "dead"

    scored = [json.loads(l) for l in open(args.scored) if l.strip()]
    wanted = {(r["prompt_index"], r["sample_index"]) for r in scored if band_of(r) == args.band}
    counts: dict[str, int] = {}
    for r in scored:
        counts[band_of(r)] = counts.get(band_of(r), 0) + 1
    total = len(scored)
    print(f"[deadband] {args.scored}: {total} rollouts")
    for k in sorted(counts, key=lambda k: -counts[k]):
        mark = "  <-- selected" if k == args.band else ""
        print(f"[deadband]   {k:16} {counts[k]:6}  {counts[k]/total*100:5.1f}%{mark}")

    raw = [json.loads(l) for l in open(args.rollouts) if l.strip()]
    pool = [r for r in raw if (r["prompt_index"], r["sample_index"]) in wanted]
    if len(pool) != len(wanted):
        # Not fatal, but it means the two files disagree -- say so rather than
        # silently sampling from whatever matched.
        print(f"[deadband] WARNING: {len(wanted)} selected but only {len(pool)} found in "
              f"{args.rollouts}; the two files may not be from the same run.")

    random.Random(args.seed).shuffle(pool)
    pool = pool[:args.n]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for r in pool:
            fh.write(json.dumps(r) + "\n")
    print(f"[deadband] wrote {len(pool)} rollouts -> {args.out}")


if __name__ == "__main__":
    main()
