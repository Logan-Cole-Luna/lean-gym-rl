#!/usr/bin/env python3
"""Do LoCoLib golds elaborate under OUR Mathlib?

THE GATE. LoCoLib targets Lean 4.23.0; we are pinned to v4.8.0-rc1 to match
Lean-Workbook. If the gold statements do not elaborate here, BEq+ cannot score
anything against them and **RL on LoCoLib is impossible** regardless of how good
the corpus looks. SFT is unaffected: it never opens Lean.

Two rates, and the gap between them is the actionable number:

  context_ok    the preamble (`open` / `variable` / `namespace`) elaborates
  gold_ok       the preamble AND the gold theorem signature elaborate

A low `context_ok` means API drift in the opens/namespaces, which is often
fixable by dropping unresolvable lines. A high `context_ok` with low `gold_ok`
means the statements themselves use post-4.8 Mathlib, which is not fixable
without a toolchain move.

    python scripts/probe_locolib_elab.py --parquet data_locolib/rl.parquet --n 400
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


def _init_worker(memory_limit_mb: int) -> None:
    global _scorer
    os.environ["BEQ_MEMORY_LIMIT_MB"] = str(memory_limit_mb)
    from reward.beq_plus import BEqPlusScorer
    _scorer = BEqPlusScorer(memory_hard_limit_mb=memory_limit_mb)
    print(f"[elab] worker {os.getpid()} ready", flush=True)


def _one(rec: dict) -> dict:
    from reward.beq_plus import split_header_and_theorem
    out = {"i": rec["i"], "domain": rec["domain"],
           "context_ok": False, "gold_ok": False, "err": None}
    try:
        context, theorem = split_header_and_theorem(rec["gold"])
        # get_env returns None when the header itself will not elaborate.
        out["context_ok"] = _scorer.get_env(context) is not None
        if out["context_ok"]:
            ok, kind = _scorer.typecheck_ex(theorem, context)
            out["gold_ok"] = bool(ok)
            out["err"] = kind
    except Exception as exc:
        out["err"] = f"worker:{type(exc).__name__}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data_locolib/rl.parquet")
    ap.add_argument("--out", default="results/locolib_elab.jsonl")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--memory-limit-mb", type=int, default=8000)
    ap.add_argument("--result-timeout", type=float, default=600)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(args.parquet).head(args.n)
    recs = [{"i": i, "gold": r["ground_truth"], "domain": e.get("domain", "?")}
            for i, (r, e) in enumerate(zip(df["reward_model"], df["extra_info"]))]
    print(f"[elab] {len(recs)} golds from {args.parquet}", flush=True)

    out = []
    with Pool(args.workers, initializer=_init_worker, initargs=(args.memory_limit_mb,)) as pool:
        it = pool.imap_unordered(_one, recs)
        for n in range(len(recs)):
            try:
                out.append(it.next(timeout=args.result_timeout))
            except StopIteration:
                break
            except Exception as exc:
                print(f"[elab] STALLED after {n}: {type(exc).__name__}", flush=True)
                break
            if (n + 1) % 50 == 0:
                g = sum(r["gold_ok"] for r in out)
                print(f"[elab] {n+1}/{len(recs)}  gold_ok {g} ({g/len(out)*100:.1f}%)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")

    n = len(out) or 1
    ctx = sum(r["context_ok"] for r in out)
    gold = sum(r["gold_ok"] for r in out)
    print(f"\n=== LoCoLib under Mathlib v4.8.0-rc1, n={len(out)} ===")
    print(f"  context elaborates : {ctx:5}  {ctx/n*100:5.1f}%")
    print(f"  gold elaborates    : {gold:5}  {gold/n*100:5.1f}%")
    print("\n  by domain:")
    per = collections.defaultdict(lambda: [0, 0])
    for r in out:
        per[r["domain"]][0] += 1
        per[r["domain"]][1] += r["gold_ok"]
    for d, (tot, g) in sorted(per.items()):
        print(f"    {d:24} {g:5}/{tot:<5} {g/max(1,tot)*100:5.1f}%")
    kinds = collections.Counter(r["err"] for r in out if r["err"])
    if kinds:
        print(f"\n  error kinds: {dict(kinds)}")
    print("\n  gold_ok is the ceiling on any BEq+ reward over this corpus: a gold")
    print("  that will not elaborate can never be proved equivalent to anything.")
    print("  Below roughly 50% this corpus is not viable for RL as-is.\n")


if __name__ == "__main__":
    main()
