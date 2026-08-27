#!/usr/bin/env python3
"""Run the FULL self-prove ladder over a rollout pool, and over the golds.

WHY. Phase 0 found that BEq+'s cascade reaches rung 3 on 99.7% of dead-band
rollouts -- so its `provable_without_have` flag is essentially free -- but that
**0 of 598** predictions proved. That kills the proposed 0.15 "true and
non-trivial" rung *if* the statements are genuinely unprovable, and merely
mismeasures it if rung 3's tactic list is too weak. Rung 3 uses
tauto + simp_all_arith! (its `exact? using this` has no `this` in scope);
`self_prove` also tries `noncomm_ring` and a bare `exact?`, which searches
Mathlib and is far stronger.

    python scripts/probe_selfprove.py --rollouts <pool>.jsonl --out <pool>.selfprove.jsonl

`--golds` scores the REFERENCE statement instead of the prediction. That is the
control that settles it: Lean-Workbook ships problems as `:= by sorry`, i.e.
unproved competition statements. If the golds are not self-provable either, the
rung is structurally empty for this dataset and no tactic list rescues it --
which is a fact about the data, not about our predictions.
"""
import argparse
import collections
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_scorer = None


def _init_worker(memory_limit_mb: int, probe_timeout: int) -> None:
    global _scorer
    os.environ["BEQ_MEMORY_LIMIT_MB"] = str(memory_limit_mb)
    os.environ["BEQ_PROBE_TIMEOUT"] = str(probe_timeout)
    from reward.beq_plus import BEqPlusScorer

    _scorer = BEqPlusScorer(memory_hard_limit_mb=memory_limit_mb)
    print(f"[selfprove] worker {os.getpid()} ready", flush=True)


def _one(rec: dict) -> dict:
    from reward.beq_plus import split_header_and_theorem
    from reward.reward_fn import _clean_solution

    if rec.get("_use_gold"):
        _ctx, target = split_header_and_theorem(rec["gold"])
    else:
        target = _clean_solution(rec["completion"])
    base = {"prompt_index": rec["prompt_index"], "sample_index": rec["sample_index"]}
    try:
        r = _scorer.self_prove(rec["gold"], target)
    except Exception as exc:
        return {**base, "typecheck": False, "provable": False, "trivial": False,
                "self_prove": False, "error_kind": f"worker:{type(exc).__name__}"}
    return {**base, "typecheck": bool(r["typecheck"]), "provable": bool(r["provable"]),
            "trivial": bool(r["trivial"]), "self_prove": bool(r["self_prove"]),
            "error_kind": r.get("error_kind")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--golds", action="store_true",
                    help="score the reference statement instead of the prediction")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--memory-limit-mb", type=int, default=8000)
    ap.add_argument("--probe-timeout", type=int, default=10)
    ap.add_argument("--result-timeout", type=float, default=600,
                    help="a Pool hangs forever if a worker dies holding a task")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.rollouts) if l.strip()]
    if args.golds:
        # One probe per PROMPT -- the gold does not vary across samples.
        seen, uniq = set(), []
        for r in recs:
            if r["prompt_index"] not in seen:
                seen.add(r["prompt_index"])
                uniq.append({**r, "_use_gold": True})
        recs = uniq
    if args.limit:
        recs = recs[:args.limit]
    print(f"[selfprove] {len(recs)} targets ({'GOLD' if args.golds else 'prediction'})", flush=True)

    out = []
    with Pool(args.workers, initializer=_init_worker,
              initargs=(args.memory_limit_mb, args.probe_timeout)) as pool:
        it = pool.imap_unordered(_one, recs)
        for n in range(len(recs)):
            try:
                out.append(it.next(timeout=args.result_timeout))
            except StopIteration:
                break
            except Exception as exc:
                print(f"[selfprove] STALLED after {n}: {type(exc).__name__}", flush=True)
                break
            if (n + 1) % 50 == 0:
                sp = sum(r["self_prove"] for r in out)
                print(f"[selfprove] {n+1}/{len(recs)}  self_prove {sp} ({sp/len(out)*100:.1f}%)", flush=True)

    with open(args.out, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")

    n = len(out) or 1
    c = collections.Counter()
    for r in out:
        c["typecheck"] += r["typecheck"]
        c["provable"] += r["provable"]
        c["trivial"] += r["trivial"]
        c["self_prove"] += r["self_prove"]
        c["error"] += bool(r["error_kind"])
    print(f"\n=== FULL self-prove ladder, {'GOLD' if args.golds else 'prediction'}, n={len(out)} ===")
    for k in ("typecheck", "provable", "trivial", "self_prove", "error"):
        print(f"  {k:12} {c[k]:5}  {c[k]/n*100:5.1f}%")
    print("\n  self_prove = provable AND NOT trivial -- the 0.15 rung of the")
    print("  proposed `verified` ladder. If this is ~0 on the golds too, the rung")
    print("  is empty for this DATASET and no tactic list will fill it.")
    print(f"\n[selfprove] wrote {args.out}")


if __name__ == "__main__":
    main()
