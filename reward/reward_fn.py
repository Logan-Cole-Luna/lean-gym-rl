#!/usr/bin/env python3
"""verl `custom_reward_function` entry points for the BEq+ RL PoC.

Selected per training run via `custom_reward_function.name`
(see repos/verl/docs/preparation/reward_function.rst):

- `compute_score_typecheck_only`: the ablation baseline from the source paper --
  reward = 1 iff the generated statement type-checks. Expected to be exploitable.
- `compute_score_composite`: 0.1*typecheck + 0.9*BEq+, the paper's composite.
- `compute_score_shaped`: graded ladder over BEq+'s per-direction results.
- `compute_score_guided`: shaped + a continuous structural-similarity term.
- `compute_score_gated`: **the current default.** Semantic-signal-only ladder;
  see "WHY GATED IS THE DEFAULT" below.

All of them return a DICT, not a float. verl's reward managers accept
`{"score": float, ...}` and forward every other key into `reward_extra_info`,
which buys three things this project needs:

  1. `val-core/<data_source>/acc/mean@1` becomes the **BEq+ rate itself**,
     because we emit `acc` = BEq+. Previously that metric was the mean of
     whatever reward the run used, on a per-run scale, and the README had to
     warn people not to compare arms with it. Now it is the real objective, so
     checkpoint selection and early stopping can key off it directly.
  2. DAPO-style group filtering can key off `semantic_signal`
     (`algorithm.filter_groups.metric`), which is how a run drops rollout
     groups that carry no semantic gradient.
  3. Lean scorer failures become a visible, logged rate instead of silently
     poisoning the reward.

Both share one process-wide, lazily-initialized `BEqPlusScorer` (one persistent
Lean REPL server per worker process) so repeated calls don't re-import Mathlib.


WHY GATED IS THE DEFAULT (measured, results/compare.txt)
-------------------------------------------------------
SFT reaches 38.8% BEq+ / 76.2% type-check. 200 steps of GRPO on top with
`compute_score_guided` moved that to 29.0% BEq+ / 84.2% type-check -- type-check
UP, semantics DOWN, p < 1e-4 by McNemar at n=400. Of the 56 examples the RL
policy lost, 48 still type-check: valid Lean that no longer means the right
thing.

The mechanism is GRPO's within-group advantage. In a rollout group where NO
sample achieves BEq+ -- the majority of prompts for a policy at ~35% -- the only
terms that can differentiate the samples are `typecheck` and `similarity`. The
group's "winner" is therefore whichever rollout elaborates and looks most like
the gold, and the gradient teaches exactly that. Summed over a run, the
non-semantic terms are the dominant signal, and they point away from the target.

The gated reward removes the differentiator: every rollout with no semantic
signal scores the SAME flat floor, so those groups produce an advantage of
exactly zero and contribute no gradient at all. That is the intended
"train only on the learnable window" behaviour, obtained inside the reward
function at zero extra compute -- as opposed to `algorithm.filter_groups`, which
achieves the same thing by DISCARDING those groups and generating replacements
(correct, but it multiplies generation cost by 1/keep_rate, ~3x here). Both are
wired up; the reward-side gate is on by default and the sampler-side filter is
opt-in via FILTER_GROUPS=1 in configs/run_grpo.sh.
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


# ── Scorer-failure instrumentation ────────────────────────────────────────────
# A Lean timeout or a dead REPL scores a rollout 0, exactly like a wrong answer.
# For a CORRECT rollout that is a gradient pointing away from correctness, and it
# is invisible in every metric the run currently logs. verl gives a reward
# function no way to say "skip this sample", so we cannot fix it here -- but an
# unmeasured version of this bug is much worse than a measured one. Every reward
# emits `scorer_error` per sample (so it shows up as a val/train mean), and the
# counters below print a summary line every _ERROR_LOG_EVERY calls.
_ERROR_LOG_EVERY = int(os.environ.get("BEQ_ERROR_LOG_EVERY", "200"))
_calls = 0
_errors = 0
_counter_lock = threading.Lock()


def _note_call(error_kind: str | None) -> None:
    global _calls, _errors
    with _counter_lock:
        _calls += 1
        if error_kind:
            _errors += 1
        due = _calls % _ERROR_LOG_EVERY == 0
        calls, errors = _calls, _errors
    if due:
        rate = 100.0 * errors / max(calls, 1)
        stats = getattr(_scorer, "stats", {})
        print(
            f"[reward] {calls} scored, {errors} Lean-scorer failures ({rate:.1f}%) "
            f"timeout={stats.get('timeout', 0)} infra={stats.get('infra', 0)} "
            f"other={stats.get('other_error', 0)}",
            flush=True,
        )
        if rate > 5.0:
            print(
                "[reward] WARNING: >5% of rewards come from a FAILED Lean call, not a "
                "real verdict. Those rollouts are scored 0 regardless of correctness, "
                "which biases the gradient. Raise BEQ_TIMEOUT_PER_PROOF or reduce "
                "concurrency.",
                flush=True,
            )


def _diagnostics(r: dict, similarity_value: float = 0.0) -> dict[str, float]:
    """The per-sample fields every reward forwards into `reward_extra_info`.

    Keys must be numeric: verl aggregates each one into a val/train metric, and
    a string would break that. `acc` is emitted deliberately -- verl treats an
    `acc` key as the run's CORE validation variable, so this makes
    `val-core/<data_source>/acc/mean@1` the true BEq+ rate.
    """
    return {
        "acc": float(r["beq_plus"]),
        "beq_plus": float(r["beq_plus"]),
        "typecheck": float(r["typecheck"]),
        "semantic_signal": float(r.get("semantic_signal", r.get("n_directions", 0))),
        "n_directions": float(r.get("n_directions", 0)),
        "gold_implies_pred": float(r.get("gold_implies_pred", False)),
        "pred_implies_gold": float(r.get("pred_implies_gold", False)),
        "similarity": float(similarity_value),
        "scorer_error": float(bool(r.get("error_kind"))),
    }


def _score_pair(solution_str: str, ground_truth: str) -> tuple[dict, str, str]:
    """Run the full BEq+ scorer once. Returns (result, pred, gold_theorem)."""
    scorer = _get_scorer()
    pred = _clean_solution(solution_str)
    _gold_context, gold_theorem = split_header_and_theorem(ground_truth)
    with _lean_slot:
        r = scorer.score(ground_truth, pred)
    _note_call(r.get("error_kind"))
    return r, pred, gold_theorem


def compute_score_typecheck_only(data_source, solution_str, ground_truth, extra_info=None) -> dict:
    scorer = _get_scorer()
    pred = _clean_solution(solution_str)
    context, _gold_theorem = split_header_and_theorem(ground_truth)
    with _lean_slot:
        ok, err = scorer.typecheck_ex(pred, context)
    _note_call(err)
    # No `acc` key: this arm never computes BEq+, so there is no honest value to
    # report and verl correctly falls back to the reward mean for val-core.
    return {"score": 1.0 if ok else 0.0, "typecheck": float(ok), "scorer_error": float(bool(err))}


def compute_score_composite(data_source, solution_str, ground_truth, extra_info=None) -> dict:
    r, _pred, _gold = _score_pair(solution_str, ground_truth)
    score = W_TYPECHECK * float(r["typecheck"]) + W_BEQ_PLUS * float(r["beq_plus"])
    return {"score": score, **_diagnostics(r)}


# ── Shaped / "process-level" reward ───────────────────────────────────────────
# BEq+ internally proves equivalence in TWO directions and only calls it
# equivalent when both succeed. Those per-direction results are real intermediate
# progress that the binary metric discards, so this ladder turns them into graded
# credit. Deliberately monotone: every rung is a strict superset of the one
# below, so the argmax is still exact BEq+ equivalence.
#
# READ reward/beq_plus.py's DIRECTION SEMANTICS note before touching the
# one-direction rung. Because the vendored cascade short-circuits, the ONLY
# observable one-direction state is "prediction is strictly WEAKER than the
# gold" -- a strictly stronger prediction scores 0 directions, same as garbage.
# So this rung pays exclusively for weakening, which is also the direction that
# trends toward vacuity. Its weight is deliberately smaller than it used to be
# (0.35 -> 0.20, with the difference moved onto full equivalence). Set
# BEQ_PROBE_STRONGER=1 to make the signal direction-symmetric at the cost of an
# extra Lean cascade per non-equivalent rollout.
W_SHAPED_TYPECHECK = float(os.environ.get("BEQ_W_SHAPED_TYPECHECK", "0.10"))
W_SHAPED_ONE_DIR = float(os.environ.get("BEQ_W_SHAPED_ONE_DIR", "0.20"))
W_SHAPED_BOTH_DIR = float(os.environ.get("BEQ_W_SHAPED_BOTH_DIR", "0.70"))


def compute_score_shaped(data_source, solution_str, ground_truth, extra_info=None) -> dict:
    r, _pred, _gold = _score_pair(solution_str, ground_truth)
    if not r["typecheck"]:
        return {"score": 0.0, **_diagnostics(r)}
    signal = r.get("semantic_signal", r.get("n_directions", 0))
    score = W_SHAPED_TYPECHECK
    if signal >= 1:
        score += W_SHAPED_ONE_DIR
    if signal >= 2:
        score += W_SHAPED_BOTH_DIR
    return {"score": score, **_diagnostics(r)}


# ── Gated reward: semantic signal only (DEFAULT) ──────────────────────────────
# See "WHY GATED IS THE DEFAULT" in the module docstring. The rule is one line:
#
#     no semantic signal  ->  a FLAT floor, identical for every such rollout
#
# so a rollout group in which nothing proves anything has zero reward variance,
# hence zero GRPO advantage, hence no gradient. Type-check and similarity are
# deliberately NOT paid here. They are the terms that let a hard prompt's group
# rank its rollouts by "valid and superficially similar", which is the measured
# failure mode this reward exists to remove. Their pedagogical value (teaching
# Lean syntax from a cold start) is real but belongs to the SFT stage, which
# already delivers 76% type-check before RL begins.
#
# The floor is 0.0 rather than a positive constant only for readability -- any
# constant gives the same advantage, since GRPO subtracts the group mean.
W_GATED_ONE_DIR = float(os.environ.get("BEQ_W_GATED_ONE_DIR", "0.25"))


def compute_score_gated(data_source, solution_str, ground_truth, extra_info=None) -> dict:
    r, _pred, _gold = _score_pair(solution_str, ground_truth)
    signal = r.get("semantic_signal", r.get("n_directions", 0))
    if signal >= 2:
        score = 1.0
    elif signal >= 1:
        score = W_GATED_ONE_DIR
    else:
        score = 0.0
    return {"score": score, **_diagnostics(r)}


# ── Guided reward: continuous similarity + BEq+ certification ────────────────
# Kept for ablation against `compute_score_gated`. This is the reward the
# 200-step SFT->RL run used, and the one whose measured result motivated the
# gate: it is the version WITH a similarity term, re-weighted so the non-semantic
# terms it pays are a smaller share of the total.
#
# Old weights (the ones that produced the 38.8% -> 29.0% BEq+ regression):
#   sim 0.20, typecheck 0.15, one-dir 0.20, both 0.45
#     -> a rollout that proves NOTHING could still bank up to 0.35.
# New weights: sim 0.10, typecheck 0.10, one-dir 0.15, both 0.65
#     -> the same rollout banks at most 0.20, and full equivalence is worth 5x
#        the best non-semantic score instead of under 3x.
# To reproduce the old behaviour exactly, set the four BEQ_W_G_* env vars back.
#
# This re-weighting only shrinks the biased gradient; it does not remove it, and
# on its own it is NOT expected to fix the regression. It matters mainly in
# combination with `algorithm.norm_adv_by_std_in_grpo=False`, which is what makes
# a small reward spread produce a correspondingly small gradient instead of being
# renormalised back to full scale.
#
# CRITICAL ORDERING CONSTRAINT: structural similarity cannot detect semantic
# inversion -- flipping `<` to `>` is one token and scores ~0.98, the same as a
# harmless typo. So the similarity weight is deliberately kept small enough that
# NO amount of structural resemblance can outscore actually proving equivalence.
# Similarity guides the search; BEq+ certifies the answer.
W_G_SIM = float(os.environ.get("BEQ_W_G_SIM", "0.10"))
W_G_TYPECHECK = float(os.environ.get("BEQ_W_G_TYPECHECK", "0.10"))
W_G_ONE_DIR = float(os.environ.get("BEQ_W_G_ONE_DIR", "0.15"))
W_G_BOTH_DIR = float(os.environ.get("BEQ_W_G_BOTH_DIR", "0.65"))


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
# So similarity is multiplied by the type-check indicator. Structural
# resemblance is only worth anything once the statement is actually valid Lean.
def compute_score_guided(data_source, solution_str, ground_truth, extra_info=None) -> dict:
    from reward.similarity import similarity

    r, pred, gold_theorem = _score_pair(solution_str, ground_truth)

    if not r["typecheck"]:
        # No credit for resembling the target in invalid Lean.
        return {"score": 0.0, **_diagnostics(r)}

    sim = similarity(pred, gold_theorem)
    score = W_G_TYPECHECK + W_G_SIM * sim
    signal = r.get("semantic_signal", r.get("n_directions", 0))
    if signal >= 1:
        score += W_G_ONE_DIR
    if signal >= 2:
        score += W_G_BOTH_DIR
    return {"score": score, **_diagnostics(r, sim)}


# alias expected when custom_reward_function.name is left unset
compute_score = compute_score_gated


if __name__ == "__main__":
    gold = "theorem lean_workbook_plus_2 (x : ℝ) : x^2 - 2*x - 24 < 0 ↔ x ∈ Set.Ioo (-4) 6 := by sorry"
    pred_good = "theorem restated (x : ℝ) : x^2 - 2*x - 24 < 0 ↔ x ∈ Set.Ioo (-4) 6"
    pred_wrong = "theorem restated (x : ℝ) : x^2 - 2*x - 24 > 0 ↔ x ∈ Set.Ioo (-4) 6"
    pred_fenced = f"Here is the formalization:\n```lean\n{pred_good}\n```"

    # A strictly stronger statement: implies the gold but is not implied by it.
    # NOTE it lands on 0 directions unless BEQ_PROBE_STRONGER=1 -- see the
    # DIRECTION SEMANTICS note in reward/beq_plus.py. Running this self-test both
    # ways is the quickest demonstration of that asymmetry.
    pred_one_way = "theorem restated (x : ℝ) : x^2 - 2*x - 24 < 0 ∧ x > -4 ↔ x ∈ Set.Ioo (-4) 6"
    # Trivially-true boilerplate -- the exact reward hack the typecheck-only arm
    # learned. Should score 1.0 on typecheck-only but bottom out elsewhere.
    pred_hack = "theorem hack (x : ℝ) : x - 1 = -1 + x"

    # A near-miss that still elaborates.
    pred_near = "theorem restated (x : ℝ) : x^2 - 2*x - 25 < 0 ↔ x ∈ Set.Ioo (-4) 6"

    print(f"probe_stronger={os.environ.get('BEQ_PROBE_STRONGER', '0')}")
    print(f"{'case':12s} {'typecheck':>9} {'composite':>9} {'shaped':>7} {'guided':>7} {'gated':>7}")
    for name, pred in [("good", pred_good), ("wrong", pred_wrong),
                       ("fenced", pred_fenced), ("one_way", pred_one_way),
                       ("near_miss", pred_near), ("hack", pred_hack)]:
        tc = compute_score_typecheck_only("lean_workbook", pred, gold)["score"]
        comp = compute_score_composite("lean_workbook", pred, gold)["score"]
        shaped = compute_score_shaped("lean_workbook", pred, gold)["score"]
        guided = compute_score_guided("lean_workbook", pred, gold)["score"]
        gated = compute_score_gated("lean_workbook", pred, gold)["score"]
        print(f"{name:12s} {tc:>9.2f} {comp:>9.2f} {shaped:>7.2f} {guided:>7.2f} {gated:>7.2f}")
