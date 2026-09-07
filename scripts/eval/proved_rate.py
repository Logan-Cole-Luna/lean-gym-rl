#!/usr/bin/env python3
"""Measure the `proved` column that nothing else in this repo reports.

`scripts/eval/evaluate_checkpoints.py` reports `typecheck_rate` (the STATEMENT
elaborates, proof sorry'd away) and `beq_plus_rate`. Neither says whether the
candidate's OWN proof closes -- and that is the gate between
`incomplete_faithful` (0.50) and `solved` (0.85) in reward/reward.py's table,
i.e. the entire top of the ladder. Measured over the finished HPC arms, the
consequence was that the arms looked flat on both reported metrics while
`proved` differed by 5-8pp between them.

Reads the raw `gen_*.jsonl` an eval already wrote, so it costs one elaboration
per example (~0.01s here) and no regeneration.

    python scripts/eval/proved_rate.py results/eval/<arm>/gen_<arm>-step90_n760.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reward.beq_plus import BEqPlusScorer, split_header_and_theorem


def run(path: Path, n: int, seed: int) -> list[dict]:
    rows = [json.loads(l) for l in path.open() if l.strip()]
    # Sample the WHOLE slice, not just the rows that already type-check: the
    # rate wanted is over all rollouts, not conditional on elaborating.
    random.Random(seed).shuffle(rows)
    rows = rows[:n] if n else rows
    scorer = BEqPlusScorer()
    out, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        context, _ = split_header_and_theorem(r["gold"])
        try:
            p = scorer.check_own_proof(r["pred"], context)
        except Exception as e:
            p = {"type_correct": False, "proved": False, "sorry_used": False,
                 "error_kind": type(e).__name__}
        out.append({"i": r["i"], "beq_plus": bool(r["beq_plus"]),
                    "statement_typecheck": bool(r["typecheck"]),
                    "own_proof_elaborates": p["type_correct"],
                    "proved": p["proved"], "sorry_used": p["sorry_used"],
                    "error_kind": p["error_kind"]})
        if i % 50 == 0:
            print(f"[proved] {i}/{len(rows)} {(time.time()-t0)/i:.3f}s/ex", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gen", type=Path, help="a gen_*.jsonl written by evaluate_checkpoints.py")
    ap.add_argument("--n", type=int, default=250, help="0 for the whole file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    recs = run(args.gen, args.n, args.seed)
    n = len(recs)
    rate = lambda k: sum(r[k] for r in recs) / max(n, 1)
    solved = sum(r["proved"] and r["beq_plus"] for r in recs) / max(n, 1)
    rep = {"source": str(args.gen), "n": n,
           "statement_typecheck_rate": rate("statement_typecheck"),
           "beq_plus_rate": rate("beq_plus"),
           "own_proof_elaborates_rate": rate("own_proof_elaborates"),
           "sorry_rate": rate("sorry_used"),
           "proved_rate": rate("proved"),
           "solved_rate": solved}
    (args.out or args.gen.with_suffix(".proved.json")).write_text(
        json.dumps({"summary": rep, "per_example": recs}, indent=2))
    print(f"\n=== {args.gen.name}  n={n} ===")
    for k in ("statement_typecheck_rate", "beq_plus_rate", "own_proof_elaborates_rate",
              "sorry_rate", "proved_rate"):
        print(f"  {k:28s} {rep[k]:.3f}")
    print(f"  {'solved_rate (proved & BEq+)':28s} {solved:.3f}   <- the top of the ladder")


if __name__ == "__main__":
    main()
