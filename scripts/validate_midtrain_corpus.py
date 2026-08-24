#!/usr/bin/env python3
"""Type-check a SAMPLE of the mid-training corpus and report how much of it is
self-contained under `import Mathlib`.

WHY THIS GATE EXISTS. Mathlib declarations routinely depend on `variable` lines
and open namespaces, so a statement lifted out of its file may leave identifiers
free -- `theorem geom_mean_le_arith_mean_weighted (w z : ι -> R) ...` never binds
`i` or `s`. Our task's outputs are self-contained, so a corpus that is mostly
NOT would be teaching a different shape from the one we want.

Elaborating all 108k declarations is ~30h of Lean and is not worth it. A sample
answers the question: this is a single proportion, and 500 statements bounds it
to about +/-4.4% at 95%. Run this BEFORE spending training compute, and read the
number honestly -- if standalone-valid comes back low, the corpus still teaches
vocabulary but the claim about shape has to be dropped.

Usage (needs Mathlib staged; run it from a compute node, see hpc/midtrain_3b.slurm):
    python scripts/validate_midtrain_corpus.py --sample data_3b/midtrain/sample.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data_3b/midtrain/sample.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/midtrain_corpus_validation.json")
    args = ap.parse_args()

    from reward.beq_plus import BEqPlusScorer

    rows = [json.loads(l) for l in Path(args.sample).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[validate] {len(rows)} statements, {args.workers} workers")

    scorer = BEqPlusScorer()

    def check(r):
        # `:= by sorry` is what the eval harness appends too, so this measures
        # the same notion of "elaborates" the BEq+ pipeline uses.
        ok, msg = scorer.typecheck_message(r["statement"] + " := by sorry")
        return ok, (msg or "")[:200]

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(check, rows))

    ok = sum(1 for o, _ in results if o)
    n = len(results)
    print(f"\n=== standalone-valid: {ok}/{n} = {100*ok/n:.1f}% ===")

    # The failure MODES matter more than the rate: "unknown identifier" means the
    # statement leaned on a `variable` line and is a shape mismatch, whereas a
    # timeout is just cost.
    kinds: dict[str, int] = {}
    for o, msg in results:
        if o:
            continue
        low = msg.lower()
        k = ("unknown identifier" if "unknown identifier" in low
             else "unknown constant" if "unknown constant" in low
             else "timeout" if "timeout" in low
             else "other")
        kinds[k] = kinds.get(k, 0) + 1
    for k, v in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v:4d}  ({100*v/n:4.1f}% of all)")

    payload = {"n": n, "valid": ok, "rate": ok / n, "failure_kinds": kinds,
               "examples": [{"statement": r["statement"], "error": m}
                            for r, (o, m) in zip(rows, results) if not o][:20]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    if ok / n < 0.40:
        print("\n[validate] Under 40% standalone-valid. The corpus still teaches Mathlib")
        print("[validate] vocabulary, but do NOT claim it teaches the task's statement shape.")


if __name__ == "__main__":
    main()
