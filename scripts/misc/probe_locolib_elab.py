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

    python scripts/misc/probe_locolib_elab.py --parquet data_locolib/rl.parquet --n 400

For the theorem+proof-pair task, add --check-proof: instead of forcing `sorry`
onto the gold SIGNATURE (what `typecheck_ex` does), this elaborates the gold's
OWN real Mathlib PROOF as written (`check_own_proof`) -- the actual question
RL on the proof-pair task needs answered before committing, since it is the
proof body, not just the statement, that must survive the toolchain gap
(LoCoLib targets Lean 4.23.0; we are pinned to v4.8.0-rc1). `reward_model.
ground_truth` already carries the gold's full untouched proof in BOTH
data_locolib/rl.parquet and rl_proof.parquet (same field, same golds), so
either file works with --check-proof.

    python scripts/misc/probe_locolib_elab.py --parquet data_locolib/rl.parquet --n 400 --check-proof
"""
import argparse
import collections
import functools
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: F401  -- repo root + every stage folder

import leanpool


def _one(rec: dict, check_proof: bool = False) -> dict:
    from reward.beq_plus import split_header_and_theorem
    out = {"i": rec["i"], "domain": rec["domain"],
           "context_ok": False, "gold_ok": False, "proved": False, "err": None}
    scorer = leanpool.scorer()
    try:
        context, theorem = split_header_and_theorem(rec["gold"])
        # get_env returns None when the header itself will not elaborate.
        out["context_ok"] = scorer.get_env(context) is not None
        if out["context_ok"]:
            if check_proof:
                # gold_ok here means the gold's OWN proof elaborates (sorry
                # allowed); proved additionally requires sorry-free and
                # axiom-clean -- the ceiling that actually matters for RL,
                # since a gold that elaborates only via `sorry` cannot back a
                # `proved` verdict for anything compared against it.
                p = scorer.check_own_proof(theorem, context)
                out["gold_ok"] = bool(p["type_correct"])
                out["proved"] = bool(p["proved"])
                out["err"] = p["error_kind"]
            else:
                ok, kind = scorer.typecheck_ex(theorem, context)
                out["gold_ok"] = bool(ok)
                out["err"] = kind
    except Exception as exc:
        out["err"] = f"worker:{type(exc).__name__}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data_locolib/rl.parquet")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--memory-limit-mb", type=int, default=8000)
    ap.add_argument("--result-timeout", type=float, default=600)
    ap.add_argument("--check-proof", action="store_true",
                    help="elaborate the gold's own real proof (check_own_proof) "
                         "instead of forcing sorry onto the signature")
    args = ap.parse_args()
    if args.out is None:
        # Not model-specific -- this probes the CORPUS (does the gold survive our
        # Mathlib pin), independent of any checkpoint, so it lives alongside
        # other training-time diagnostics rather than under results/eval/<model>/.
        args.out = ("results/train/_probes/locolib_elab_proof.jsonl" if args.check_proof
                    else "results/train/_probes/locolib_elab.jsonl")

    import pandas as pd
    # The `prompt` column is the bulk of the file and is never read here.
    df = pd.read_parquet(args.parquet, columns=["reward_model", "extra_info"]).head(args.n)
    recs = [{"i": i, "gold": r["ground_truth"], "domain": e.get("domain", "?")}
            for i, (r, e) in enumerate(zip(df["reward_model"], df["extra_info"]))]
    print(f"[elab] {len(recs)} golds from {args.parquet}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).unlink(missing_ok=True)   # leanpool.run appends
    n_ok = 0

    def progress(i, res):
        nonlocal n_ok
        n_ok += bool(res["gold_ok"])
        return f"gold_ok {n_ok} ({100 * n_ok / i:.1f}%)"

    out = leanpool.run(functools.partial(_one, check_proof=args.check_proof), recs,
                       workers=args.workers,
                       memory_limit_mb=args.memory_limit_mb,
                       result_timeout=args.result_timeout,
                       out_path=args.out, on_result=progress,
                       tag="elab", report_every=50)

    n = len(out) or 1
    ctx = sum(r["context_ok"] for r in out)
    gold = sum(r["gold_ok"] for r in out)
    proved = sum(r["proved"] for r in out)
    import os as _os
    _toolchain_path = _os.path.join(_os.environ.get("MATHLIB_ROOT", ""), "lean-toolchain")
    try:
        _toolchain = open(_toolchain_path).read().strip()
    except OSError:
        _toolchain = "unknown"
    print(f"\n=== LoCoLib under {_toolchain}, n={len(out)} ===")
    print(f"  context elaborates : {ctx:5}  {ctx/n*100:5.1f}%")
    if args.check_proof:
        print(f"  gold's own proof elaborates (sorry allowed) : {gold:5}  {gold/n*100:5.1f}%")
        print(f"  gold's own proof PROVED (sorry-free, axiom-clean) : {proved:5}  {proved/n*100:5.1f}%")
    else:
        print(f"  gold elaborates    : {gold:5}  {gold/n*100:5.1f}%")
    print("\n  by domain:")
    per = collections.defaultdict(lambda: [0, 0])
    for r in out:
        per[r["domain"]][0] += 1
        per[r["domain"]][1] += (r["proved"] if args.check_proof else r["gold_ok"])
    for d, (tot, g) in sorted(per.items()):
        print(f"    {d:24} {g:5}/{tot:<5} {g/max(1,tot)*100:5.1f}%")
    kinds = collections.Counter(r["err"] for r in out if r["err"])
    if kinds:
        print(f"\n  error kinds: {dict(kinds)}")
    if args.check_proof:
        print("\n  PROVED is the ceiling for the proof-pair `outcome` arm's SOLVED row: a")
        print("  gold whose own proof will not survive our Mathlib pin can never anchor")
        print("  a `proved`+BEq+-matched verdict for anything compared against it.")
    else:
        print("\n  gold_ok is the ceiling on any BEq+ reward over this corpus: a gold")
        print("  that will not elaborate can never be proved equivalent to anything.")
    print("  Below roughly 50% this corpus is not viable for RL as-is.\n")


if __name__ == "__main__":
    main()
