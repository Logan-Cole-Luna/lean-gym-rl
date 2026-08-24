#!/usr/bin/env python3
"""Extract the prompts a policy NEVER solves, so k can be raised on them alone.

Why this exists: 47% of the RL pool produced 0/8 BEq+ at k=8. Whether those are
beyond the model's reach (a CAPABILITY limit -> scale up) or merely rarely
reached (an EXPLORATION limit -> raise rollout_n / temperature) is the single
question separating "buy a bigger model" from "tune the sampler", and it has
never been measured.

Extrapolating from the k=8 data cannot answer it: a binomial fit assigns p=0 to
every 0/8 prompt, which forces the pessimistic conclusion rather than testing it.
The only honest way is to actually sample more on exactly those prompts.

IMPORTANT -- how prompts are identified. generate_rollouts.py DEDUPLICATES and
drops prompts over --max-prompt-length, then re-indexes what survives from 0. So
`prompt_index` is a position in that FILTERED list, not a row number in
data/train.parquet (1,280 rows -> 1,191 rollout groups). Joining positionally
silently selects the wrong prompts: a first attempt did exactly that and produced
"starved" prompts scoring 32/32, which is impossible for a prompt that went 0/8.
This script therefore joins on PROMPT TEXT, carried by the raw rollouts file, and
verifies the gold matches before writing anything.

Usage:
    python scripts/make_starved_subset.py \
        --scored data/rollouts/sft390_k8.scored.jsonl \
        --rollouts data/rollouts/sft390_k8.jsonl \
        --parquet data/train.parquet --out data/starved.parquet
then generate at k=32 on --parquet data/starved.parquet and score as usual.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="data/rollouts/sft390_k8.scored.jsonl")
    ap.add_argument("--rollouts", default="data/rollouts/sft390_k8.jsonl",
                    help="raw rollouts, which carry the prompt/gold text used to join")
    ap.add_argument("--parquet", default="data/train.parquet",
                    help="the parquet the rollouts were generated from")
    ap.add_argument("--out", default="data/starved.parquet")
    ap.add_argument("--metric", default="beq_plus", choices=["beq_plus", "typecheck"])
    # Scoring dominates: 560 prompts x k=32 is ~18k BEq+ calls (~8h even at 8
    # workers). A random subsample answers the same question -- the statistic is
    # a single proportion, and 250 prompts bounds it to about +/-3% at 95%.
    ap.add_argument("--max-prompts", type=int, default=0, help="0 = keep all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd

    scored = [json.loads(l) for l in Path(args.scored).read_text().splitlines() if l.strip()]
    by = collections.defaultdict(list)
    for s in scored:
        by[s["prompt_index"]].append(s)

    # A prompt is starved iff NO sample in its group ever satisfied the metric.
    starved = sorted(pi for pi, g in by.items() if not any(x[args.metric] for x in g))
    n_all = len(starved)

    # prompt_index -> (prompt text, gold), taken from the rollouts themselves.
    seen: dict[int, tuple[str, str]] = {}
    for line in Path(args.rollouts).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        seen.setdefault(d["prompt_index"], (d["prompt"], d["gold"]))

    df = pd.read_parquet(args.parquet)
    by_text = {row[0]["content"]: i for i, row in enumerate(df["prompt"])}

    rows, missing, mismatched = [], 0, 0
    for pi in starved:
        if pi not in seen:
            missing += 1
            continue
        ptext, gold = seen[pi]
        j = by_text.get(ptext)
        if j is None:
            missing += 1
            continue
        # Positional joins silently mis-select here; verify the gold agrees.
        if df["reward_model"].iloc[j]["ground_truth"] != gold:
            mismatched += 1
            continue
        rows.append(j)
    if missing or mismatched:
        print(f"[starved] WARNING: {missing} prompts not found by text, "
              f"{mismatched} whose gold disagreed -- excluded")
    if not rows:
        raise SystemExit("no starved prompts could be matched back to the parquet")

    if args.max_prompts and len(rows) > args.max_prompts:
        import random
        rows = sorted(random.Random(args.seed).sample(rows, args.max_prompts))
    out = df.iloc[rows].reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)

    k = len(next(iter(by.values())))
    print(f"[starved] {len(by)} prompts scored at k={k}")
    print(f"[starved] {n_all} never satisfied {args.metric} "
          f"({100*n_all/len(by):.1f}%)")
    if len(out) != n_all:
        print(f"[starved] using {len(out)} of them (seed {args.seed}) to bound scoring cost")
    print(f"[starved] wrote {args.out} ({len(out)} rows), joined by prompt text with gold verified")
    print(f"[starved] next: generate at higher k on this subset, score, then the")
    print(f"[starved] fraction with >=1 success is the answer -- near 0% means")
    print(f"[starved] capability, clearly above 0% means exploration.")


if __name__ == "__main__":
    main()
