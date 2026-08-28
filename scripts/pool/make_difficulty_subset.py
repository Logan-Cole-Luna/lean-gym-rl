#!/usr/bin/env python3
"""Restrict the RL pool to prompts at the policy's EDGE OF COMPETENCE.

Why. GRPO's advantage is A_i = r_i - mean(r_group). With binary reward and k=8,
a group that scores 0/8 and a group that scores 8/8 both give every rollout an
advantage of exactly zero. Only 1-7/8 teaches anything. On the 3B pool that is
46.0% of prompts (starved 45.3%, saturated 8.7%), so more than half of every
batch is dead weight that still pays full Lean cost.

Selecting on the measured k=8 success count takes informative to ~100% by
construction. This is the cheap, static form of what FILTER_GROUPS=1 (DAPO) does
online -- online refill was measured at ~8x cost here, not the estimated 3x,
because verl refills one prompt at a time.

WHAT THIS BUYS AND WHAT IT COSTS. It is a STATIC filter on a MOVING target:
prompts migrate starved -> informative -> saturated as the policy improves, so
a pool curated against the SFT checkpoint decays in usefulness over the run.
Paired against `gated` on the full pool, that decay IS the measurement.

Also: the surviving pool is smaller, so a fixed step budget means more epochs.
The script prints steps-per-epoch and warns past 3, because at that point
memorisation confounds any gain.

IDENTIFYING PROMPTS -- read before changing the join. generate_rollouts.py
DEDUPLICATES and drops prompts over --max-prompt-length, then re-indexes what
survives from 0, so `prompt_index` is a position in that FILTERED list, NOT a
row number in the parquet (600 sampled -> 585 groups here; 1,280 -> 1,191 at
0.5B). Joining positionally silently selects the wrong prompts -- an earlier
attempt did exactly that and produced "starved" prompts scoring 32/32. This
script joins on PROMPT TEXT and verifies the gold agrees before writing.

Usage (accepts several scored/rollout pairs, one per scoring job):
    python scripts/pool/make_difficulty_subset.py \
        --pair data_3b/rollouts/sft3b_k8.scored.jsonl:data_3b/rollouts/sft3b_k8.jsonl \
        --pair data_3b/rollouts/pool_s1.scored.jsonl:data_3b/rollouts/pool_s1.jsonl \
        --parquet data_3b/train.parquet --out data_3b/train_edge.parquet
"""
from __future__ import annotations

import argparse
import os

DATA_DIR = os.environ.get("DATA_DIR", "data_3b")   # corpus root
import collections
import json
from pathlib import Path


def _user_turn(row) -> str:
    """The LAST user turn of a chat-format prompt row.

    NOT row[0]: a parquet that carries a system turn puts the system text at
    index 0, and every prompt silently collapses to the same string. That
    produced a LoCoLib eval where one identical theorem was scored for all 999
    rows at 100% type-check and 0% BEq+.
    """
    turns = [m for m in row if m.get("role") == "user"]
    if not turns:
        raise ValueError(f"no user turn in prompt row: {row!r}")
    return turns[-1]["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True,
                    metavar="SCORED:ROLLOUTS",
                    help="scored jsonl and the raw rollouts it came from, colon-separated. "
                         "Repeat for each scoring job; prompt_index namespaces are per-file "
                         "and are NOT merged numerically -- the join is on prompt text.")
    ap.add_argument("--parquet", default=f"{DATA_DIR}/train.parquet")
    ap.add_argument("--out", default=f"{DATA_DIR}/train_edge.parquet")
    ap.add_argument("--metric", default="beq_plus", choices=["beq_plus", "typecheck"])
    ap.add_argument("--min-wins", type=int, default=1, help="inclusive lower bound on k=8 successes")
    ap.add_argument("--max-wins", type=int, default=7, help="inclusive upper bound")
    ap.add_argument("--batch-size", type=int, default=16, help="only used to report steps/epoch")
    ap.add_argument("--total-steps", type=int, default=150)
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_parquet(args.parquet)
    by_text = {_user_turn(row): i for i, row in enumerate(df["prompt"])}

    wins_by_text: dict[str, tuple[int, int, str]] = {}   # text -> (wins, k, gold)
    for spec in args.pair:
        scored_p, _, roll_p = spec.partition(":")
        if not roll_p:
            raise SystemExit(f"--pair needs SCORED:ROLLOUTS, got {spec!r}")
        groups = collections.defaultdict(list)
        for line in Path(scored_p).read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                groups[d["prompt_index"]].append(d)
        text_of: dict[int, tuple[str, str]] = {}
        for line in Path(roll_p).read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                text_of.setdefault(d["prompt_index"], (d["prompt"], d["gold"]))
        # RAGGED GROUPS. A run killed mid-write leaves a group with fewer than k
        # samples, and its win count is then compared against --max-wins on a
        # different denominator: a 7/7 prompt is fully SATURATED (zero advantage,
        # the exact thing this filter exists to remove) but would pass a
        # `1 <= wins <= 7` test and be labelled informative. Drop them.
        modal = collections.Counter(len(g) for g in groups.values()).most_common(1)[0][0]
        ragged = dupes = 0
        for pi, g in groups.items():
            if pi not in text_of:
                continue
            if len(g) != modal:
                ragged += 1
                continue
            ptext, gold = text_of[pi]
            if ptext in wins_by_text:
                dupes += 1
                continue
            wins_by_text[ptext] = (sum(1 for x in g if x[args.metric]), len(g), gold)
        print(f"[edge] {scored_p}: {len(groups)} groups at k={modal}"
              + (f", dropped {ragged} ragged" if ragged else "")
              + (f", {dupes} already seen in an earlier file" if dupes else ""))

    hist = collections.Counter(w for w, _k, _g in wins_by_text.values())
    total = len(wins_by_text)
    ks = {k for _w, k, _g in wins_by_text.values()}
    print(f"[edge] {total} distinct prompts scored, k in {sorted(ks)}")
    print(f"[edge] success histogram: {dict(sorted(hist.items()))}")

    rows, missing, mismatched = [], 0, 0
    for ptext, (wins, _k, gold) in wins_by_text.items():
        if not (args.min_wins <= wins <= args.max_wins):
            continue
        j = by_text.get(ptext)
        if j is None:
            missing += 1
            continue
        if df["reward_model"].iloc[j]["ground_truth"] != gold:
            mismatched += 1
            continue
        rows.append(j)
    if missing or mismatched:
        print(f"[edge] WARNING: {missing} not found by text, {mismatched} gold mismatch -- excluded")
    if not rows:
        raise SystemExit("no prompts survived the difficulty filter")

    rows = sorted(set(rows))
    out = df.iloc[rows].reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)

    kept = len(out)
    spe = kept / args.batch_size
    epochs = args.total_steps / spe if spe else float("inf")
    print(f"[edge] kept {kept}/{total} prompts with {args.min_wins}-{args.max_wins} "
          f"successes ({100*kept/total:.1f}%) -> {args.out}")
    k = max(w for w, _k, _g in wins_by_text.values()) if wins_by_text else 8
    if args.min_wins >= 1 and args.max_wins < k:
        print(f"[edge] informative fraction of this pool is ~100% by construction "
              f"(was {100*sum(v for w, v in hist.items() if 1 <= w < k)/total:.1f}%)")
    else:
        # e.g. --min-wins 0 --max-wins 0 extracts STARVED prompts, which are the
        # zero-advantage ones. Saying "informative ~100%" there would be exactly
        # backwards.
        print(f"[edge] NOTE: this selection is not the informative band; "
              f"{args.min_wins}-{args.max_wins} of k={k} produces "
              f"{'zero' if args.max_wins == 0 else 'partial'} GRPO advantage.")
    print(f"[edge] {spe:.1f} steps/epoch at batch {args.batch_size}; "
          f"{args.total_steps} steps = {epochs:.1f} epochs")
    if epochs > 3 and args.min_wins >= 1:
        print(f"[edge] WARNING: {epochs:.1f} epochs over a curated pool -- memorisation will")
        print(f"[edge] confound any gain. Score more prompts or cut --total-steps.")


if __name__ == "__main__":
    main()
