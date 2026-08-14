#!/usr/bin/env python3
"""verl `custom_reward_function` entry points for the BEq+ RL PoC.

Two reward functions, selected per training run via `custom_reward_function.name`
(see repos/verl/docs/preparation/reward_function.rst):

- `compute_score_typecheck_only`: the ablation baseline from the source paper --
  reward = 1 iff the generated statement type-checks (with `sorry`), regardless
  of whether it means the same thing as the gold statement. Expected to be
  exploitable (syntactically valid, semantically vacuous outputs).
- `compute_score_composite`: reward = w_typecheck * typecheck_pass +
  w_beq_plus * beq_plus_match, matching the paper's composite formulation
  (syntactic signal + BEq+ semantic-equivalence signal against ground truth).

Both share one process-wide, lazily-initialized `BEqPlusScorer` (one persistent
Lean REPL server per worker process) so repeated calls don't re-import Mathlib.
"""
from __future__ import annotations

import re

from reward.beq_plus import BEqPlusScorer, split_header_and_theorem

W_TYPECHECK = 0.1
W_BEQ_PLUS = 0.9

_scorer: BEqPlusScorer | None = None


def _get_scorer() -> BEqPlusScorer:
    global _scorer
    if _scorer is None:
        _scorer = BEqPlusScorer()
    return _scorer


_CODE_FENCE_RE = re.compile(r"```(?:lean4?|Lean4?)?\s*(.*?)```", re.DOTALL)


def _clean_solution(solution_str: str) -> str:
    """Strip a ```lean ... ``` fence if the model wrapped its answer in one;
    otherwise pass the raw completion through (BEqPlusScorer's own
    `extract_last_theorem`/`clean_last_theorem_string` already locate and
    isolate the theorem declaration within surrounding text)."""
    m = _CODE_FENCE_RE.search(solution_str)
    return m.group(1).strip() if m else solution_str.strip()


def compute_score_typecheck_only(data_source, solution_str, ground_truth, extra_info=None) -> float:
    scorer = _get_scorer()
    pred = _clean_solution(solution_str)
    context, _gold_theorem = split_header_and_theorem(ground_truth)
    return 1.0 if scorer.typecheck(pred, context) else 0.0


def compute_score_composite(data_source, solution_str, ground_truth, extra_info=None) -> float:
    scorer = _get_scorer()
    pred = _clean_solution(solution_str)
    result = scorer.score(ground_truth, pred)
    reward = W_TYPECHECK * float(result["typecheck"]) + W_BEQ_PLUS * float(result["beq_plus"])
    return reward


# alias expected when custom_reward_function.name is left unset
compute_score = compute_score_composite


if __name__ == "__main__":
    gold = "theorem lean_workbook_plus_2 (x : ℝ) : x^2 - 2*x - 24 < 0 ↔ x ∈ Set.Ioo (-4) 6 := by sorry"
    pred_good = "theorem restated (x : ℝ) : x^2 - 2*x - 24 < 0 ↔ x ∈ Set.Ioo (-4) 6"
    pred_wrong = "theorem restated (x : ℝ) : x^2 - 2*x - 24 > 0 ↔ x ∈ Set.Ioo (-4) 6"
    pred_fenced = f"Here is the formalization:\n```lean\n{pred_good}\n```"

    for name, pred in [("good", pred_good), ("wrong", pred_wrong), ("fenced", pred_fenced)]:
        tc = compute_score_typecheck_only("lean_workbook", pred, gold)
        comp = compute_score_composite("lean_workbook", pred, gold)
        print(f"{name:8s} typecheck_only={tc:.2f}  composite={comp:.2f}")
