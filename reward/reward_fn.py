#!/usr/bin/env python3
"""verl `custom_reward_function` entry points.

THE FILE VERL LOADS. `configs/run_grpo.sh` sets
`reward.custom_reward_function.path` to this path and `.name` to one of the
functions below. Each is built from its row in `reward/arms.py`, which is the
single place an arm's makeup is written down -- what it pays for, which terms
are live, and which Lean evidence it gathers. Nothing about a reward is decided
in this file; it only binds a name to a row. Read `ARMS` for what each arm is
and what it pays for.

ONE MODULE ATTRIBUTE PER ARM, not one entry point reading the arm from the
environment. verl records `custom_reward_function.name` in its config dump and
that dump is how a finished run is identified afterwards, so two arms that
differ only by an env var would be indistinguishable there. Adding an arm is a
row in `ARMS` and one line here.

Each returns a dict, not a float. verl forwards every key other than `score`
into `reward_extra_info` and aggregates it into a train/val metric. Emitting
`acc` = BEq+ makes `val-core/<data_source>/acc/mean@1` the true BEq+ rate rather
than the mean of whatever reward the run happens to use.

All entry points share one lazily built `BEqPlusScorer` per process, so repeated
calls reuse a single Lean REPL instead of re-importing Mathlib.
"""
from __future__ import annotations

import os

from reward.arms import ARMS, build
from reward.reward import OUTCOMES

# Re-exported because callers outside this package import them from here:
# `scripts/eval/evaluate_checkpoints.py` and `scripts/pool/score_rollouts.py`
# strip fences with `_clean_solution` so an offline score matches a training one
# exactly.
from reward.evidence import _clean_solution, _proof_body_len  # noqa: F401

compute_score_outcome         = build("outcome")
compute_score_outcome_brevity = build("outcome_brevity")
compute_score_gated           = build("gated")
compute_score_typecheck       = build("typecheck")

# Used when custom_reward_function.name is left unset. Every job here sets the
# name explicitly, so this alias is a fallback rather than what the arms run.
compute_score = compute_score_outcome

__all__ = ["compute_score"] + [f"compute_score_{n}" for n in ARMS]


if __name__ == "__main__":
    # A LoCoLib-shaped gold: full record text, real proof body, no `sorry`.
    gold = ("theorem gold_add_comm (a b : Nat) : a + b = b + a := by\n"
            "  induction a with\n"
            "  | zero => simp\n"
            "  | succ n ih => omega")
    gold_body = _proof_body_len(gold)

    # Each case is a full theorem AND proof, because compute_score_outcome runs
    # check_own_proof, which elaborates the submission AS WRITTEN.
    cases = [
        # Right answer, right size: brevity must cost nothing.
        ("concise",   "theorem r (a b : Nat) : a + b = b + a := by omega"),
        # Right answer, padded with vacuous `have`s. Same verdict from BEq+ and
        # the same `proved`, so brevity is the ONLY term that separates it from
        # `concise` -- which is exactly the tiebreak this term exists to make.
        ("bloated",   "theorem r (a b : Nat) : a + b = b + a := by\n" +
                      "".join(f"  have h{i} : {i} + 0 = {i} := by simp\n" for i in range(24)) +
                      "  omega"),
        # Degenerately short AND wrong: lands on a lower rung, never reaches the
        # band. This is the case that shows the gate, not the ratio, is the guard.
        ("tiny_wrong", "theorem r (a b : Nat) : a + b = b + a := by rfl"),
        # Unfinished. MATCHED_NOT_PROVED at most; brevity must not touch it.
        ("sorry",     "theorem r (a b : Nat) : a + b = b + a := by sorry"),
    ]

    # ---- the pre-existing arm probe, unchanged in substance -----------------
    # `check_own_proof` elaborates the candidate AS WRITTEN, so every case needs
    # a real `:=`, not a bare signature. `:= by sorry` still counts as
    # type-correct (a warning, not an error); "hack" gets a real closing tactic
    # to show a genuine 1.0 on the exploitable arm.
    sig_gold = ("theorem lean_workbook_plus_2 (x : ℝ) : "
                "x^2 - 2*x - 24 < 0 ↔ x ∈ Set.Ioo (-4) 6 := by sorry")
    sig_cases = [
        ("good", "theorem restated (x : ℝ) : x^2 - 2*x - 24 < 0 ↔ x ∈ Set.Ioo (-4) 6 := by sorry"),
        ("wrong", "theorem restated (x : ℝ) : x^2 - 2*x - 24 > 0 ↔ x ∈ Set.Ioo (-4) 6 := by sorry"),
        ("fenced", "Here is the formalization:\n```lean\ntheorem restated (x : ℝ) : "
                   "x^2 - 2*x - 24 < 0 ↔ x ∈ Set.Ioo (-4) 6 := by sorry\n```"),
        # Strictly stronger: implies the gold but is not implied by it. Lands on
        # 0 directions unless BEQ_PROBE_STRONGER=1; see beq_plus.py.
        ("one_way", "theorem restated (x : ℝ) : x^2 - 2*x - 24 < 0 ∧ x > -4 ↔ "
                    "x ∈ Set.Ioo (-4) 6 := by sorry"),
        ("near_miss", "theorem restated (x : ℝ) : x^2 - 2*x - 25 < 0 ↔ x ∈ Set.Ioo (-4) 6 := by sorry"),
        # Trivially true AND sorry-free. 1.0 on typecheck-only, 0 on gated.
        ("hack", "theorem hack (x : ℝ) : x - 1 = -1 + x := by ring"),
    ]
    print(f"probe_stronger={os.environ.get('BEQ_PROBE_STRONGER', '0')}")
    print(f"{'case':12s} {'typecheck':>9} {'gated':>7}")
    for name, pred in sig_cases:
        tc = compute_score_typecheck("lean_workbook", pred, sig_gold)["score"]
        gated = compute_score_gated("lean_workbook", pred, sig_gold)["score"]
        print(f"{name:12s} {tc:>9.2f} {gated:>7.2f}")

    # ---- the brevity probe --------------------------------------------------
    print(f"\ngold proof body = {gold_body} chars\n")
    print(f"{'case':12s} {'score':>6} {'outcome':>22} {'proved':>7} "
          f"{'pred_ch':>8} {'brevity':>8}")
    for name, pred in cases:
        r = compute_score_outcome("locolib", pred, gold)
        outcome = OUTCOMES[int(r["outcome_code"])]
        print(f"{name:12s} {r['score']:>6.3f} {outcome:>22} "
              f"{r['proved']:>7.0f} {r['pred_proof_chars']:>8.0f} {r['brevity']:>8.2f}")
