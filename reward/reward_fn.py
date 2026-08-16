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

import os
import re
import threading

from reward.beq_plus import BEqPlusScorer, split_header_and_theorem

W_TYPECHECK = 0.1
W_BEQ_PLUS = 0.9

_scorer: BEqPlusScorer | None = None
_scorer_failures = 0

# Serialises access to the shared scorer. DEFAULT 1 -- THIS IS A CORRECTNESS
# REQUIREMENT, not just a memory knob.
#
# verl dispatches `compute_score` through a thread pool (reward_manager's
# `run_in_executor`), so every thread in this process shares the one module-global
# BEqPlusScorer and its AutoLeanServer. `lean_interact`'s session cache is NOT
# thread-safe: two threads materialising the same base env race and raise
#     RuntimeError: Session state -1 is already being materialized.
# which fails the rollout. Observed with a value of 4: `failure: 14` and a wedged
# run. Keep this at 1 unless lean_interact gains thread-safe session handling.
#
# It also happens to cap memory, which matters because AutoLeanServer keeps pooled
# REPL processes with Mathlib resident (~4.3GB each), and neither
# `reward.num_workers` nor `rollout.agent.num_workers` bounds that count
# (measured: 30-46 REPLs, ~50GB host RAM).
#
# To get parallel Lean scoring, add PROCESSES (raise rollout.agent.num_workers),
# each of which gets its own serialised scorer -- and budget ~4.3GB of host RAM
# per process for Mathlib.
BEQ_MAX_CONCURRENT = int(os.environ.get("BEQ_MAX_CONCURRENT", "1"))
_lean_slot = threading.Semaphore(BEQ_MAX_CONCURRENT)

# Constructing a BEqPlusScorer spawns a Lean REPL and imports Mathlib (~4.3GB,
# tens of seconds). If that construction FAILS, the naive `if _scorer is None`
# cache never gets populated, so every subsequent reward call retries it -- and
# each retry spawns another Lean process. Observed in practice: a too-small
# memory cap killed the REPL at startup and the retry storm left **38 concurrent
# Lean REPLs** alive, which exhausted host RAM and got the Ray workers
# OOM-killed. The memory symptom looked like the cause; it was the consequence.
#
# So: bound the retries. After this many consecutive construction failures, give
# up and score 0 rather than keep forking Lean processes. A run whose Lean
# environment is broken should degrade to "no reward signal", not take the
# machine down with it.
_MAX_SCORER_FAILURES = 3


class ScorerUnavailable(RuntimeError):
    """Raised when the Lean scorer could not be constructed after repeated tries."""


# Construction must be serialised too: verl calls the reward from a thread pool,
# so without this two threads can both observe `_scorer is None` and each build a
# Lean server (doubling Mathlib's ~4.3GB, and racing inside lean_interact).
_scorer_lock = threading.Lock()


def _get_scorer() -> BEqPlusScorer:
    global _scorer, _scorer_failures
    if _scorer is not None:
        return _scorer
    with _scorer_lock:
        # Re-check inside the lock: another thread may have built it while we
        # waited.
        if _scorer is not None:
            return _scorer
        if _scorer_failures >= _MAX_SCORER_FAILURES:
            raise ScorerUnavailable(
                f"BEqPlusScorer failed to start {_scorer_failures}x; refusing to spawn more "
                "Lean processes. Check BEQ_MEMORY_LIMIT_MB (must exceed Mathlib's ~4.5GB "
                "baseline) and available host RAM."
            )
        try:
            _scorer = BEqPlusScorer()
        except Exception:
            _scorer_failures += 1
            raise
        _scorer_failures = 0
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
    with _lean_slot:
        return 1.0 if scorer.typecheck(pred, context) else 0.0


def compute_score_composite(data_source, solution_str, ground_truth, extra_info=None) -> float:
    scorer = _get_scorer()
    pred = _clean_solution(solution_str)
    with _lean_slot:
        result = scorer.score(ground_truth, pred)
    reward = W_TYPECHECK * float(result["typecheck"]) + W_BEQ_PLUS * float(result["beq_plus"])
    return reward


# ── Shaped / "process-level" reward ───────────────────────────────────────────
# Motivation (see README's Results): the strict composite reward puts 90% of its
# mass on BEq+, a hard binary that a weak policy achieves ~0-4% of the time. GRPO
# computes advantages WITHIN each rollout group, so with rollout_n=4 nearly every
# group is all-zero on that term -> zero advantage -> no semantic gradient at all.
#
# BEq+ internally proves equivalence in TWO directions (gold=>pred and pred=>gold)
# and only calls it equivalent when both succeed. Those per-direction results are
# real intermediate progress that the binary metric discards: one direction proving
# means the prediction is a strictly stronger or strictly weaker version of the
# gold -- much closer than an unrelated statement. BEqL (the `exact?`-only tier) is
# a further intermediate signal. This ladder turns them into graded credit so that
# rollout groups differ in reward and GRPO has something to learn from.
#
# Deliberately monotone: every rung is a strict superset of the one below, so the
# argmax is still exact BEq+ equivalence -- this shapes the path, it does not
# change the target.
W_SHAPED_TYPECHECK = 0.15   # produced Lean that elaborates at all
W_SHAPED_ONE_DIR = 0.35     # + implies (or is implied by) the gold
W_SHAPED_BOTH_DIR = 0.50    # + full BEq+ equivalence   (total 1.00)


def compute_score_shaped(data_source, solution_str, ground_truth, extra_info=None) -> float:
    scorer = _get_scorer()
    pred = _clean_solution(solution_str)
    with _lean_slot:
        r = scorer.score(ground_truth, pred)

    if not r["typecheck"]:
        return 0.0
    reward = W_SHAPED_TYPECHECK
    n_dir = r.get("n_directions", 0)
    if n_dir >= 1:
        reward += W_SHAPED_ONE_DIR
    if n_dir >= 2:  # == r["beq_plus"]
        reward += W_SHAPED_BOTH_DIR
    return reward


# ── Guided reward: continuous similarity + BEq+ certification ────────────────
# Motivation, measured on our SFT policy over 80 validation examples:
#     26%  does not type-check      -> shaped reward 0.00  (no information)
#     41%  type-checks, no BEq+     -> shaped reward 0.15  (ALL identical)
#     32%  full BEq+                -> shaped reward 1.00
# i.e. two thirds of the data gave the policy no directional gradient, and a
# near-miss scored the same as `theorem x : 1 = 1`.
#
# This reward adds a CONTINUOUS term (reward/similarity.py: GTED-style operator
# -tree edit distance + library-constant overlap, optionally embeddings) that is
# defined even when the statement fails to elaborate, so it grades both dead
# buckets.
#
# CRITICAL ORDERING CONSTRAINT: structural similarity cannot detect semantic
# inversion -- flipping `<` to `>` is one token and scores ~0.98, the same as a
# harmless typo. So the similarity weight is deliberately kept small enough that
# NO amount of structural resemblance can outscore actually proving equivalence.
# Similarity guides the search; BEq+ certifies the answer.
#
#   typecheck * (0.15 + 0.20 * similarity)   <- similarity is GATED, see below
# + 0.20 * one BEq+ direction
# + 0.45 * both directions (full BEq+)
#   -------------------------------------
#   = 1.00 for an exactly-equivalent statement
W_G_SIM = float(os.environ.get("BEQ_W_G_SIM", "0.20"))
W_G_TYPECHECK = float(os.environ.get("BEQ_W_G_TYPECHECK", "0.15"))
W_G_ONE_DIR = float(os.environ.get("BEQ_W_G_ONE_DIR", "0.20"))
W_G_BOTH_DIR = float(os.environ.get("BEQ_W_G_BOTH_DIR", "0.45"))


# GATING (learned the hard way -- do not remove). The first version of this
# reward paid the similarity term UNCONDITIONALLY, so it was available to
# outputs that never elaborate. Trained from the base model -- where BEq+ is
# effectively unreachable -- similarity became the ONLY climbable gradient, and
# the policy promptly learned to farm it: it copied the gold's tokens into
# invalid Lean. Actual step-30 generation, against a gold of
# `(f : ℝ → ℝ) (C : ℝ) (h₁ : f = fun x => -x^4 / 2 - C * x + x / 2)`:
#     lemma (x: ℝ): f(x) = -x^4/2 - C*x + x/2
# -- no theorem name, `f(x)` is not Lean application, `f`/`C` unbound. Entropy
# collapsed to 0.127, response length fell 107 -> 54, and mean reward sat at
# 0.186, just under the 0.20 similarity cap: almost pure similarity credit with
# essentially no type-checking. Shaping had created a NEW, easier hack than the
# one it was designed to prevent.
#
# So similarity is now multiplied by the type-check indicator. Structural
# resemblance is only worth anything once the statement is actually valid Lean,
# which removes the degenerate optimum while keeping the dense signal exactly
# where it was wanted: among outputs that elaborate but fail BEq+ (41% of the
# SFT policy's outputs, previously all scored identically).
def compute_score_guided(data_source, solution_str, ground_truth, extra_info=None) -> float:
    from reward.similarity import similarity

    scorer = _get_scorer()
    pred = _clean_solution(solution_str)
    _gold_context, gold_theorem = split_header_and_theorem(ground_truth)

    with _lean_slot:
        r = scorer.score(ground_truth, pred)

    if not r["typecheck"]:
        # No credit for resembling the target in invalid Lean.
        return 0.0

    reward = W_G_TYPECHECK + W_G_SIM * similarity(pred, gold_theorem)
    n_dir = r.get("n_directions", 0)
    if n_dir >= 1:
        reward += W_G_ONE_DIR
    if n_dir >= 2:  # == r["beq_plus"]
        reward += W_G_BOTH_DIR
    return reward


# alias expected when custom_reward_function.name is left unset
compute_score = compute_score_composite


if __name__ == "__main__":
    gold = "theorem lean_workbook_plus_2 (x : ℝ) : x^2 - 2*x - 24 < 0 ↔ x ∈ Set.Ioo (-4) 6 := by sorry"
    pred_good = "theorem restated (x : ℝ) : x^2 - 2*x - 24 < 0 ↔ x ∈ Set.Ioo (-4) 6"
    pred_wrong = "theorem restated (x : ℝ) : x^2 - 2*x - 24 > 0 ↔ x ∈ Set.Ioo (-4) 6"
    pred_fenced = f"Here is the formalization:\n```lean\n{pred_good}\n```"

    # A strictly stronger statement: implies the gold but is not implied by it,
    # so it should land on the shaped reward's middle rung (typecheck + one
    # direction) while strict BEq+ still scores it 0.
    pred_one_way = "theorem restated (x : ℝ) : x^2 - 2*x - 24 < 0 ∧ x > -4 ↔ x ∈ Set.Ioo (-4) 6"
    # Trivially-true boilerplate -- the exact reward hack the typecheck-only arm
    # learned. Should score 1.0 on typecheck-only but bottom out elsewhere.
    pred_hack = "theorem hack (x : ℝ) : x - 1 = -1 + x"

    # A near-miss that still elaborates: the shaped reward cannot tell it apart
    # from the degenerate `hack` below (both 0.15); the guided reward should.
    pred_near = "theorem restated (x : \u211d) : x^2 - 2*x - 25 < 0 \u2194 x \u2208 Set.Ioo (-4) 6"

    print(f"{'case':12s} {'typecheck':>9} {'composite':>9} {'shaped':>7} {'guided':>7}")
    for name, pred in [("good", pred_good), ("wrong", pred_wrong),
                       ("fenced", pred_fenced), ("one_way", pred_one_way),
                       ("near_miss", pred_near), ("hack", pred_hack)]:
        tc = compute_score_typecheck_only("lean_workbook", pred, gold)
        comp = compute_score_composite("lean_workbook", pred, gold)
        shaped = compute_score_shaped("lean_workbook", pred, gold)
        guided = compute_score_guided("lean_workbook", pred, gold)
        print(f"{name:12s} {tc:>9.2f} {comp:>9.2f} {shaped:>7.2f} {guided:>7.2f}")
