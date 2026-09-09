#!/usr/bin/env python3
"""Do the policy's proofs walk the same road as the gold proofs? OFFLINE ONLY.

Adapted from `faithful_score.py`. Three things changed, and each is the point:

NAME. `faithful_score.py` was slotted against `reward.reward.proof_follows_argument`,
which reward.py specifies as NL -> FL ("the Lean PROOF against the English proof
the agent was given ... the only place the English enters the score"). This
metric never reads English. It compares two LEAN proofs' intermediate goal
states, so it is FL <-> FL proof-STRUCTURE agreement against a reference proof.
Calling it faithfulness would silently redefine a documented column, so it is
`proof_follows_reference` here and does NOT feed `proof_follows_argument`.

NORMALISATION. The original `calculate_score` returns an unnormalised total that
is MONOTONE IN MATCHED-CHECKPOINT COUNT. As a reward that is backwards: the
candidate picks its own checkpoints, the DP maximises over alignments, so
padding a proof with intermediate `have`s buys more chances to hit a gold
checkpoint. That is precisely the erosion channel reward.py warns about ("a
`have` chain restating each informal step, each closed by `grind`"), except
mechanically incentivised. F1 over the two trajectories removes the incentive:
precision divides by the CANDIDATE's checkpoint count, so padding costs exactly
what it buys.

DEPENDENCY. The original needed Pantograph (a second Lean driver, async, and the
proof had to be written into a built Lean project to get `tactic_invocations`).
`lean_interact` -- already this repo's dependency -- returns the same
goal-before-tactic states from `Command(..., all_tactics=True)`.

WHY THIS IS NOT A REWARD. The comparison is an n_pred x n_gold matrix of FULL
BEq+ cascades, plus self-validation. Measured on this repo's own hardware the
cascade is seconds per pair, so a 5x7 matrix is minutes -- per rollout, against
a GRPO step of 64 rollouts. It belongs at checkpoint boundaries on the val
split, where it answers "are the policy's proofs converging on the golds'
structure?", and nowhere near the inner loop.

    python scripts/eval/proof_alignment.py --pairs pairs.jsonl --out results/alignment
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lean_interact import Command

from reward.beq_plus import (BEqPlusScorer, check_theorem_equivalence,
                             split_header_and_theorem)


@dataclass(frozen=True)
class Checkpoint:
    step: int
    tactic: str
    premises: tuple[str, ...]
    goal: str


def parse_goal_state(state: str) -> tuple[tuple[str, ...], str] | None:
    """Split one visible Lean tactic state into (premises, goal).

    None -- rather than the original's ValueError -- when the state has zero or
    several visible goals. A multi-goal state has no reliable machine-readable
    boundary in its pretty-printed form, and on a real rollout that is common
    enough that raising would abort the whole evaluation over one branchy proof.
    Skipping the checkpoint costs recall on that proof and nothing else.
    """
    state = state.strip()
    turnstiles = list(re.finditer(r"(?m)^\s*⊢\s*", state))
    if len(turnstiles) != 1:
        return None
    t = turnstiles[0]
    context = state[: t.start()].strip()
    premises = tuple(ln.strip() for ln in context.splitlines()
                     if ln.strip() and not ln.strip().startswith("case "))
    return premises, state[t.end():].strip()


def checkpoint_theorem(cp: Checkpoint, name: str = "checkpoint") -> str:
    """`Γ ⊢ goal` -> `theorem name (Γ) : goal := by sorry`, so BEq+ (a
    STATEMENT metric) can compare two mid-proof states at all."""
    binders = "".join(f"  ({p})\n" for p in cp.premises)
    return f"theorem {name}\n{binders}  : {cp.goal} := by\n  sorry"


def extract_checkpoints(scorer: BEqPlusScorer, decl: str, context: str,
                        max_checkpoints: int = 24) -> list[Checkpoint]:
    """The goal state before each tactic of `decl`, via the REPL's `allTactics`.

    Capped: the matrix is quadratic in this count, and a 100-step proof would
    cost more than the rest of the evaluation combined.
    """
    env = scorer.get_env(context)
    if env is None:
        return []
    # NOT `scorer._run`: that builds its own `Command` and cannot pass
    # `all_tactics`. Same contract though -- any Lean failure yields no
    # checkpoints, never a wrong verdict.
    try:
        out = scorer.server.run(Command(cmd=decl, env=env, all_tactics=True),
                                timeout=scorer.timeout_per_proof)
    except Exception:
        return []
    tactics = getattr(out, "tactics", None) or []
    cps = []
    for step, t in enumerate(tactics, 1):
        parsed = parse_goal_state(t.goals)
        if parsed is None:
            continue
        cps.append(Checkpoint(step, t.tactic.splitlines()[0].strip(), *parsed))
        if len(cps) >= max_checkpoints:
            break
    return cps


def compare_matrix(scorer: BEqPlusScorer, a: list[Checkpoint], b: list[Checkpoint],
                   env: int) -> list[list[float]]:
    """BEq+ verdict for every checkpoint pair. Binary, per the published metric.

    Cached on the (theorem, theorem) pair: proofs revisit equivalent states, and
    each cache hit is a whole cascade not paid for.
    """
    cache: dict[tuple[str, str], float] = {}
    rows = []
    for ca in a:
        row = []
        for cb in b:
            t1, t2 = checkpoint_theorem(ca), checkpoint_theorem(cb)
            key = tuple(sorted((t1, t2)))          # BEq+ is symmetric
            if key not in cache:
                try:
                    res = check_theorem_equivalence(t1, t2, scorer.server, env,
                                                    scorer.timeout_per_proof)
                    cache[key] = float(res.beq_plus())
                except Exception:
                    cache[key] = 0.0
            row.append(cache[key])
        rows.append(row)
    return rows


def alignment_f1(table: list[list[float]], *, threshold: float = 1.0) -> dict:
    """Order-preserving checkpoint alignment -> precision / recall / F1.

    Plain LCS. The original `calculate_score` carries a third axis to report the
    best total AT each match count k; a single score needs only the last layer,
    so the 3-D array collapses to this 2-D recurrence and the O(min(n,m)*n*m)
    cost drops to O(n*m).

        precision = matched / |candidate checkpoints|   penalises padding
        recall    = matched / |reference checkpoints|   penalises skipping

    F1 rather than recall alone is what makes the metric length-neutral, and so
    safe to read next to a proof-length term without the two fighting.
    """
    n, m = len(table), len(table[0]) if table else 0
    if not n or not m:
        return {"matched": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_pred": n, "n_gold": m}
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(1, n + 1):
        for k in range(1, m + 1):
            dp[j][k] = max(dp[j - 1][k], dp[j][k - 1],
                           dp[j - 1][k - 1] + (table[j - 1][k - 1] >= threshold))
    matched = dp[n][m]
    p, r = matched / n, matched / m
    return {"matched": matched, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if matched else 0.0,
            "n_pred": n, "n_gold": m}


def score_pair(scorer: BEqPlusScorer, pred_full: str, gold_full: str,
               max_cells: int = 64) -> dict:
    """One (candidate proof, gold proof) pair -> alignment metrics + cost."""
    t0 = time.perf_counter()
    gold_ctx, gold_decl = split_header_and_theorem(gold_full)
    _pred_ctx, pred_decl = split_header_and_theorem(pred_full)
    env = scorer.get_env(gold_ctx)
    if env is None:
        return {"error": "no_env", "seconds": time.perf_counter() - t0}
    cps_pred = extract_checkpoints(scorer, pred_decl, gold_ctx)
    cps_gold = extract_checkpoints(scorer, gold_decl, gold_ctx)
    if not cps_pred or not cps_gold:
        return {"error": "no_checkpoints", "n_pred": len(cps_pred),
                "n_gold": len(cps_gold), "seconds": time.perf_counter() - t0}
    cells = len(cps_pred) * len(cps_gold)
    if cells > max_cells:
        # Refuse rather than silently truncate: a truncated matrix would report a
        # low F1 that means "we did not look", indistinguishable from a real miss.
        return {"error": "too_large", "cells": cells, "n_pred": len(cps_pred),
                "n_gold": len(cps_gold), "seconds": time.perf_counter() - t0}
    table = compare_matrix(scorer, cps_pred, cps_gold, env)
    out = alignment_f1(table)
    out.update({"cells": cells, "seconds": time.perf_counter() - t0, "error": None,
                "table": table})
    return out


# A gold that stops at `:= by` with nothing after it has no proof to align
# against. miniF2F is entirely like this -- it ships STATEMENTS, `formal_proof`
# is literally "sorry" -- so FL <-> FL is undefined there and the pair must be
# dropped rather than scored 0.0, which would read as a policy failure.
_PROOFLESS = re.compile(r":=\s*(by)?\s*$")


def has_gold_proof(gold: str) -> bool:
    return not _PROOFLESS.search(gold.rstrip())


def load_pairs(args) -> list[dict]:
    """{pred, gold} pairs from either a hand-built --pairs file or a gen_*.jsonl.

    The gen file already carries both snippets under exactly these two keys, so
    reading it directly removes a conversion step that could silently pair the
    wrong rows.
    """
    src = args.pairs or args.gen
    rows = [json.loads(l) for l in src.open() if l.strip()]
    if args.gen:
        kept = [r for r in rows if has_gold_proof(r.get("gold", ""))]
        print(f"[align] {len(kept)}/{len(rows)} rows have a gold PROOF to align "
              f"against ({len(rows) - len(kept)} statement-only golds dropped)")
        rows = kept
        if not rows:
            print("[align] FATAL: no gold proofs in this corpus. FL <-> FL "
                  "alignment is undefined on a statement-only benchmark such as "
                  "miniF2F; run it on a corpus whose golds carry proofs.")
            sys.exit(2)
    for i, r in enumerate(rows):
        r.setdefault("i", i)
    if args.n and args.n < len(rows):
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.n]
        rows.sort(key=lambda r: r["i"])
    return rows


def summarize(recs: list[dict]) -> dict:
    ok = [r for r in recs if not r.get("error")]
    return {
        "n": len(recs), "n_scored": len(ok),
        "errors": {k: sum(r.get("error") == k for r in recs)
                   for k in ("no_env", "no_checkpoints", "too_large")},
        "mean_f1": sum(r["f1"] for r in ok) / max(len(ok), 1),
        "mean_precision": sum(r["precision"] for r in ok) / max(len(ok), 1),
        "mean_recall": sum(r["recall"] for r in ok) / max(len(ok), 1),
        "mean_seconds": sum(r["seconds"] for r in recs) / max(len(recs), 1),
        "total_seconds": sum(r["seconds"] for r in recs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path,
                    help="jsonl with {pred, gold} full Lean snippets per line")
    ap.add_argument("--gen", type=Path,
                    help="a gen_*.jsonl from evaluate_checkpoints.py; its `pred` "
                         "and `gold` are the pair, and statement-only golds are "
                         "dropped with a count")
    ap.add_argument("--out", type=Path, default=Path("results/alignment"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n", type=int, default=0,
                    help="score a random subsample of this size, seeded, so two "
                         "checkpoints are compared on the SAME rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cells", type=int, default=64)
    # One pair is an n_pred x n_gold matrix of full BEq+ cascades, i.e. seconds
    # to minutes each, and this loop is single-process by construction (one Lean
    # REPL, BEQ_MAX_CONCURRENT=1). Shards are separate PROCESSES with their own
    # REPL, which is the only parallelism available here.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true",
                    help="combine records.shard*.jsonl in --out into summary.json")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.merge:
        recs = [json.loads(l) for f in sorted(args.out.glob("records.shard*.jsonl"))
                for l in f.open() if l.strip()]
        summary = summarize(recs)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

    if not (args.pairs or args.gen):
        ap.error("pass --pairs or --gen")
    pairs = load_pairs(args)
    if args.limit:
        pairs = pairs[: args.limit]
    mine = [p for k, p in enumerate(pairs) if k % args.num_shards == args.shard]
    print(f"[align] shard {args.shard}/{args.num_shards}: {len(mine)} of "
          f"{len(pairs)} pairs", flush=True)

    scorer = BEqPlusScorer()
    fh = (args.out / f"records.shard{args.shard}.jsonl").open("w")
    recs = []
    for i, p in enumerate(mine, 1):
        r = score_pair(scorer, p["pred"], p["gold"], args.max_cells)
        r["index"] = p["i"]
        recs.append(r)
        fh.write(json.dumps({k: v for k, v in r.items() if k != "table"}) + "\n")
        fh.flush()
        print(f"[align] {i}/{len(mine)} f1={r.get('f1', 0):.2f} "
              f"({r.get('n_pred', 0)}x{r.get('n_gold', 0)}) {r['seconds']:.1f}s "
              f"{r.get('error') or ''}", flush=True)
    print(json.dumps(summarize(recs), indent=2))


if __name__ == "__main__":
    main()
