#!/usr/bin/env python3
"""Turn scored rollouts into a rejection-sampling fine-tuning set, and report
the group-level signal statistics of the same rollouts.

Two outputs, both from ONE scoring pass:

1. **An RFT dataset** in verl's SFT `messages` schema, filtered by a semantic
   criterion. `--filter beq_plus` keeps only rollouts BEq+ certifies as
   equivalent to the gold; `--filter typecheck` keeps everything that merely
   elaborates. Training one arm on each is the paper's exploitability claim
   tested where it can actually be measured -- rejection sampling has no
   advantage-estimation noise, unlike GRPO at this batch size.

2. **Clean-data group statistics** (`--stats-out`). The existing gradient probe
   covered 48 prompts of the OLD, SFT-contaminated RL pool. The same rollouts
   scored here cover the entire disjoint pool, so the starved/saturated split
   and within-group reward variance can finally be reported on the distribution
   training actually sees.

Usage:
    python scripts/build_rft_dataset.py \
        --scored data/rollouts/sft390_k8.scored.jsonl \
        --rollouts data/rollouts/sft390_k8.jsonl \
        --filter beq_plus --out-dir data/rft_beq --stats-out results/rollout_stats.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--rollouts", required=True, help="the unscored file, for prompt/gold text")
    ap.add_argument("--filter", default="beq_plus", choices=["beq_plus", "typecheck"])
    ap.add_argument("--max-per-prompt", type=int, default=2,
                    help="cap accepted rollouts per prompt so easy prompts do not "
                         "dominate the fine-tuning set")
    ap.add_argument("--val-frac", type=float, default=0.05)
    # The type-check filter accepts about twice as many rollouts as BEq+ does,
    # so an unmatched comparison would confound "which reward selects better
    # data" with "which arm got more data". Build the BEq+ arm first, then pass
    # its reported pair count here when building the type-check arm.
    ap.add_argument("--match-size", type=int, default=0,
                    help="truncate to exactly N accepted pairs (0 = keep all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stats-out", default="")
    args = ap.parse_args()

    rolls = {(r["prompt_index"], r["sample_index"]): r for r in load_jsonl(args.rollouts)}
    scored = load_jsonl(args.scored)
    print(f"[rft] {len(scored)} scored rollouts, {len(rolls)} generated")

    by_prompt: dict[int, list[dict]] = defaultdict(list)
    for s in scored:
        by_prompt[s["prompt_index"]].append(s)

    # ── group statistics on the clean, disjoint pool ──────────────────────────
    # GRPO's advantage is r_i - mean(r_group); a group whose rollouts all score
    # alike yields exactly zero gradient regardless of the level. Reported for
    # BOTH candidate rewards so the "which reward teaches more" question is
    # answered on the real training distribution, not the memorised one.
    stats: dict[str, dict] = {}
    for name, key in (("gated_beq_plus", "beq_plus"), ("typecheck_only", "typecheck")):
        deg = starved = saturated = 0
        stds, means = [], []
        for pi, group in by_prompt.items():
            vals = [1.0 if g[key] else 0.0 for g in group]
            if len(vals) < 2:
                continue
            m = st.mean(vals)
            means.append(m)
            stds.append(st.pstdev(vals))
            if len(set(vals)) == 1:
                deg += 1
                if vals[0] == 0.0:
                    starved += 1
                else:
                    saturated += 1
        n = len(means)
        stats[name] = {
            "n_groups": n, "mean_reward": st.mean(means) if means else 0.0,
            "degenerate_frac": deg / n if n else 0.0,
            "starved": starved, "saturated": saturated,
            "informative_groups": n - deg,
            "informative_frac": (n - deg) / n if n else 0.0,
            "mean_within_group_std": st.mean(stds) if stds else 0.0,
        }

    n_err = sum(1 for s in scored if s.get("error_kind"))
    beq_not_tc = sum(1 for s in scored if s["beq_plus"] and not s["typecheck"])
    stats["_meta"] = {
        "n_scored": len(scored), "n_prompts": len(by_prompt),
        "scorer_error_rate": n_err / len(scored) if scored else 0.0,
        "beq_plus_without_typecheck": beq_not_tc,
        "typecheck_rate": sum(s["typecheck"] for s in scored) / len(scored),
        "beq_plus_rate": sum(s["beq_plus"] for s in scored) / len(scored),
    }

    print("\n=== group signal on the clean disjoint pool ===")
    hdr = f"{'reward':<18} {'mean':>7} {'informative':>12} {'starved':>8} {'saturated':>10} {'wg_std':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name in ("typecheck_only", "gated_beq_plus"):
        s = stats[name]
        print(f"{name:<18} {s['mean_reward']:>7.3f} {100*s['informative_frac']:>11.1f}% "
              f"{s['starved']:>8} {s['saturated']:>10} {s['mean_within_group_std']:>8.4f}")
    print(f"\nBEq+ without type-check: {beq_not_tc} (expected 0 -- BEq+ implies type-check)")
    print(f"scorer errors: {100*stats['_meta']['scorer_error_rate']:.2f}%")

    if args.stats_out:
        Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_out).write_text(json.dumps(stats, indent=2))
        print(f"[rft] stats -> {args.stats_out}")

    # ── build the fine-tuning set ─────────────────────────────────────────────
    rng = random.Random(args.seed)
    rows, n_prompts_covered = [], 0
    for pi, group in sorted(by_prompt.items()):
        # A scorer failure is recorded as False; it is not evidence the rollout
        # is wrong, but it is also not evidence it is right -- exclude either way.
        good = [g for g in group if g[args.filter] and not g.get("error_kind")]
        if not good:
            continue
        n_prompts_covered += 1
        # Deduplicate identical statements before capping, so the cap buys
        # DIVERSITY rather than N copies of the same string.
        uniq: dict[str, dict] = {}
        for g in good:
            uniq.setdefault(g["pred"].strip(), g)
        picks = list(uniq.values())
        rng.shuffle(picks)
        for g in picks[: args.max_per_prompt]:
            src = rolls.get((g["prompt_index"], g["sample_index"]))
            if not src:
                continue
            rows.append({"messages": [
                {"role": "user", "content": src["prompt"]},
                {"role": "assistant", "content": g["pred"].strip()},
            ]})

    rng.shuffle(rows)
    if args.match_size and len(rows) > args.match_size:
        print(f"[rft] size-matching: {len(rows)} -> {args.match_size} pairs")
        rows = rows[: args.match_size]
    elif args.match_size:
        print(f"[rft] WARNING: only {len(rows)} pairs available, cannot match "
              f"{args.match_size} -- the arms are NOT size-matched.")
    n_val = max(1, int(len(rows) * args.val_frac)) if rows else 0
    val, train = rows[:n_val], rows[n_val:]
    print(f"\n[rft] filter={args.filter}: {len(rows)} accepted pairs from "
          f"{n_prompts_covered}/{len(by_prompt)} prompts "
          f"({100*n_prompts_covered/len(by_prompt):.1f}% coverage)")
    print(f"[rft] split: {len(train)} train / {len(val)} val")

    import pandas as pd

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train).to_parquet(out_dir / "train.parquet")
    pd.DataFrame(val).to_parquet(out_dir / "val.parquet")
    print(f"[rft] wrote {out_dir}/train.parquet and val.parquet")
    if train:
        print("\nexample accepted target:\n  " + train[0]["messages"][1]["content"][:180])


if __name__ == "__main__":
    main()
