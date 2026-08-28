#!/usr/bin/env python3
"""One-off diagnostic: what does Lean actually SAY when a LoCoLib gold proof
fails to elaborate? Not part of the permanent pipeline -- answers a single
question (is this a renamed-API drift or a missing-local-context problem)
before deciding how to handle the low elaboration rate.

Unlike typecheck_message, this checks the PROOF, not a sorry'd signature.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: F401

import pandas as pd

from lean_interact.interface import LeanError
from reward.beq_plus import BEqPlusScorer, _rename_theorem_keep_proof, split_header_and_theorem

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15

df = pd.read_parquet("data_locolib/rl_proof.parquet", columns=["reward_model"]).head(N)
scorer = BEqPlusScorer()

shown = 0
for i, r in enumerate(df["reward_model"]):
    gold = r["ground_truth"]
    context, theorem = split_header_and_theorem(gold)
    env = scorer.get_env(context)
    if env is None:
        print(f"row {i}: context itself failed to elaborate")
        continue
    try:
        code = _rename_theorem_keep_proof(theorem, "candidate_theorem")
    except ValueError as e:
        print(f"row {i}: unparseable ({e})")
        continue
    out = scorer._run(code, env, scorer.timeout_per_proof)
    if out is None:
        print(f"row {i}: infra failure (timeout/dead REPL)")
        continue
    if isinstance(out, LeanError):
        shown += 1
        print(f"\n===== row {i} (LeanError) =====")
        print("--- theorem (first 300 chars) ---")
        print(theorem[:300])
        print("--- error ---")
        print(str(out)[:600])
    elif not out.lean_code_is_valid(allow_sorry=True):
        shown += 1
        errs = [f"{m.severity}: {m.data}" for m in out.messages if m.severity == "error"]
        print(f"\n===== row {i} (elaboration errors) =====")
        print("--- theorem (first 300 chars) ---")
        print(theorem[:300])
        print("--- errors ---")
        print("\n".join(errs)[:600])
    if shown >= 8:
        break

print(f"\nshown {shown} failures out of {len(df)} checked")
