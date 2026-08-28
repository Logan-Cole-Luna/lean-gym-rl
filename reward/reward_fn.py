#!/usr/bin/env python3
"""verl `custom_reward_function` entry points.

Selected per run via `custom_reward_function.name`:

- `compute_score_outcome` (default): the six-outcome ladder from reward/reward.py.
- `compute_score_gated`: pays only for BEq+ equivalence, flat floor below it.
- `compute_score_typecheck_only`: pays for elaboration alone. Exploitable, and
  kept as the ablation baseline.

Each returns a dict, not a float. verl forwards every key other than `score`
into `reward_extra_info` and aggregates it into a train/val metric. Emitting
`acc` = BEq+ makes `val-core/<data_source>/acc/mean@1` the true BEq+ rate rather
than the mean of whatever reward the run happens to use.

All entry points share one lazily built `BEqPlusScorer` per process, so repeated
calls reuse a single Lean REPL instead of re-importing Mathlib.
"""
from __future__ import annotations

import functools
import os
import re
import threading

from reward.beq_plus import BEqPlusScorer, split_header_and_theorem
from reward.reward import (OUTCOMES, gated_reward, outcome_for, reward_for,
                           signals_from_score, typecheck_reward)

_scorer: BEqPlusScorer | None = None
_scorer_failures = 0
_scorer_lock = threading.Lock()

# Serialises Lean access. Must stay at 1: verl calls the reward from a thread
# pool, and lean_interact's session cache is not thread-safe -- two threads
# materialising the same base env raise "Session state -1 is already being
# materialized". For parallel scoring add processes (rollout.agent.num_workers),
# each of which gets its own scorer and its own ~4.3GB resident Mathlib.
BEQ_MAX_CONCURRENT = int(os.environ.get("BEQ_MAX_CONCURRENT", "1"))
_lean_slot = threading.Semaphore(BEQ_MAX_CONCURRENT)

# Building a scorer spawns a Lean REPL and imports Mathlib. A plain
# `if _scorer is None` cache retries that on every call when construction fails,
# and each retry leaks another REPL until the host runs out of RAM. Bound the
# retries and degrade to "no reward signal" instead.
_MAX_SCORER_FAILURES = 3


class ScorerUnavailable(RuntimeError):
    """The Lean scorer could not be constructed after repeated attempts."""


def _get_scorer() -> BEqPlusScorer:
    global _scorer, _scorer_failures
    if _scorer is not None:
        return _scorer
    with _scorer_lock:
        if _scorer is not None:      # another thread built it while we waited
            return _scorer
        if _scorer_failures >= _MAX_SCORER_FAILURES:
            raise ScorerUnavailable(
                f"BEqPlusScorer failed to start {_scorer_failures}x. Check "
                "BEQ_MEMORY_LIMIT_MB (must exceed Mathlib's ~4.5GB baseline) "
                "and available host RAM."
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
    """Strip a ```lean fence if the model wrapped its answer in one.

    Otherwise pass the completion through; the scorer's own
    `extract_last_theorem` isolates the declaration from surrounding text.
    """
    m = _CODE_FENCE_RE.search(solution_str)
    return m.group(1).strip() if m else solution_str.strip()


_ERROR_LOG_EVERY = int(os.environ.get("BEQ_ERROR_LOG_EVERY", "200"))
_calls = 0
_errors = 0
_counter_lock = threading.Lock()


def _note_call(error_kind: str | None) -> None:
    """Count Lean-scorer failures and log the rate periodically.

    A timeout or a dead REPL scores a rollout 0, indistinguishably from a wrong
    answer, so for a correct rollout it is a gradient pointing the wrong way.
    verl offers no way to skip a sample, so the rate is measured instead: every
    reward emits `scorer_error`, and a high rate means the run is not measuring
    what it claims to.
    """
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
        print(f"[reward] {calls} scored, {errors} Lean-scorer failures ({rate:.1f}%) "
              f"timeout={stats.get('timeout', 0)} infra={stats.get('infra', 0)} "
              f"other={stats.get('other_error', 0)}", flush=True)
        if rate > 5.0:
            print("[reward] WARNING: >5% of rewards come from a failed Lean call "
                  "rather than a real verdict. Raise BEQ_TIMEOUT_PER_PROOF or "
                  "reduce concurrency.", flush=True)


# Every reward must emit the same key set: verl aggregates each key into a
# metric, and an omitted key gives a ragged schema. These are the "not
# applicable" values, not measurements.
_SCHEMA = {
    "score": 0.0, "acc": 0.0, "beq_plus": 0.0, "typecheck": 0.0,
    "semantic_signal": 0.0, "n_directions": 0.0,
    "gold_implies_pred": 0.0, "pred_implies_gold": 0.0,
    "similarity": 0.0, "scorer_error": 0.0,
    "beql": 0.0, "rung": 0.0, "convert_level": 0.0,
    "provable_alone": 0.0, "provable_alone_known": 0.0, "stop_reason": 0.0,
    # Index into reward.reward.OUTCOMES. Only the outcome arm varies it; the
    # others emit it so every arm has one key set.
    "outcome_code": 0.0,
    # Only the outcome arm on the proof-pair task sets these for real (from
    # BEqPlusScorer.check_own_proof); every other arm emits the "not
    # applicable" default so the schema stays uniform.
    "proved": 0.0, "sorry_used": 0.0,
}

_ZERO = {**_SCHEMA, "scorer_error": 1.0}

_STOP_REASON_CODE = {"unparseable": 1, "sorry_gate": 2, "no_rung": 3}


def _never_raises(fn):
    """Score 0 rather than propagating an exception.

    A reward that raises does not just lose its rollout: verl marks the agent
    task failed, the GRPO step never closes, and the job holds a GPU until
    walltime. Scoring 0 is not neutral -- a correct rollout lost to an
    infrastructure fault pushes the gradient the wrong way -- so it is reported
    as `scorer_error=1` rather than hidden.
    """
    @functools.wraps(fn)
    def wrapped(data_source, solution_str, ground_truth, extra_info=None):
        try:
            return fn(data_source, solution_str, ground_truth, extra_info)
        except Exception as e:
            print(f"[reward] {fn.__name__} FAILED, scoring 0: "
                  f"{type(e).__name__}: {e}"[:300], flush=True)
            return dict(_ZERO)
    return wrapped


def _diagnostics(r: dict, proof_check: dict | None = None) -> dict[str, float]:
    """Per-sample fields forwarded into `reward_extra_info`. All numeric.

    `proof_check`, when given, is a `check_own_proof()` result for the same
    rollout -- only `compute_score_outcome` on the proof-pair task passes one.
    """
    provable_alone = r.get("provable_alone")
    return {
        **_SCHEMA,
        "acc": float(r["beq_plus"]),
        "beq_plus": float(r["beq_plus"]),
        "typecheck": float(r["typecheck"]),
        "semantic_signal": float(r.get("semantic_signal", r.get("n_directions", 0))),
        "n_directions": float(r.get("n_directions", 0)),
        "gold_implies_pred": float(r.get("gold_implies_pred", False)),
        "pred_implies_gold": float(r.get("pred_implies_gold", False)),
        "scorer_error": float(bool(r.get("error_kind"))),
        # BEqL is strictly tighter than BEq+; only cascade rung 1 sets it.
        "beql": float(r.get("beql", False)),
        # Which cascade rung closed direction 0, 0 if none. Lower is a tighter
        # match. Read the per-example histogram, not the mean, which mixes
        # "did not close" with "closed at rung k".
        "rung": float(r.get("rung") or 0),
        # `convert ... using k`; only meaningful when rung == 4.
        "convert_level": float(r.get("convert_level") or 0),
        # Two keys because None (rung 3 never ran) and False (it ran and the
        # statement did not prove) are different facts. The conditional rate is
        # mean(provable_alone) / mean(provable_alone_known).
        "provable_alone": float(provable_alone is True),
        "provable_alone_known": float(provable_alone is not None),
        # Why direction 0 stopped, which n_directions=0 conflates.
        # 0 = ran to completion, 1 = unparseable, 2 = sorry gate, 3 = no rung.
        "stop_reason": float(_STOP_REASON_CODE.get(r.get("stop_reason"), 0)),
        "outcome_code": float(OUTCOMES.index(outcome_for(signals_from_score(r, proof_check)))),
        "proved": float(proof_check.get("proved", False)) if proof_check else 0.0,
        "sorry_used": float(proof_check.get("sorry_used", False)) if proof_check else 0.0,
    }


def _score_pair(solution_str: str, ground_truth: str) -> tuple[dict, str, str]:
    """Run the BEq+ scorer once. Returns (result, prediction, gold theorem)."""
    scorer = _get_scorer()
    pred = _clean_solution(solution_str)
    _gold_context, gold_theorem = split_header_and_theorem(ground_truth)
    with _lean_slot:
        r = scorer.score(ground_truth, pred)
    _note_call(r.get("error_kind"))
    return r, pred, gold_theorem


@_never_raises
def compute_score_typecheck_only(data_source, solution_str, ground_truth, extra_info=None) -> dict:
    """1.0 if the candidate's OWN submission elaborates, else 0.0.

    Uses `check_own_proof`, not `typecheck_ex`: on the proof-pair task the
    candidate writes a real proof body, and `typecheck_ex` would discard it
    and re-inject `sorry` before elaborating, silently checking only the
    signature. `sorry` inside the candidate's own proof still counts as
    type-correct here (a warning, not an error), so this stays the same
    exploitable, cheap ablation baseline it always was -- just checking the
    real submission instead of a forced stand-in.
    """
    scorer = _get_scorer()
    pred = _clean_solution(solution_str)
    context, _gold_theorem = split_header_and_theorem(ground_truth)
    with _lean_slot:
        p = scorer.check_own_proof(pred, context)
    _note_call(p["error_kind"])
    # Deliberately no `acc`: this arm never computes BEq+, so there is no honest
    # value to report and verl falls back to the reward mean for val-core.
    return {"score": typecheck_reward(p["type_correct"], p["error_kind"]),
            "typecheck": float(p["type_correct"]),
            "scorer_error": float(bool(p["error_kind"]))}


W_GATED_ONE_DIR = float(os.environ.get("BEQ_W_GATED_ONE_DIR", "0.25"))


@_never_raises
def compute_score_gated(data_source, solution_str, ground_truth, extra_info=None) -> dict:
    """BEq+ ladder: 1.0 for both directions, W_GATED_ONE_DIR for one, else 0.

    Type-check is not paid. The gate is implicit, since nothing that fails to
    elaborate can prove in either direction, and paying for elaboration is the
    measured failure mode this reward exists to avoid.

    The flat floor is the point. Under GRPO the advantage is the reward minus
    the group mean, so a group with no semantic signal produces an advantage of
    exactly zero and contributes no gradient. Any term that varies within such a
    group -- similarity, type-check -- becomes the only climbable signal there,
    and the policy learns to farm it.
    """
    r, _pred, _gold = _score_pair(solution_str, ground_truth)
    return {"score": gated_reward(r, W_GATED_ONE_DIR), **_diagnostics(r)}


@_never_raises
def compute_score_outcome(data_source, solution_str, ground_truth, extra_info=None) -> dict:
    """The six-outcome ladder from reward/reward.py, now fully reachable on the
    proof-pair task (the candidate writes a real proof, not just a signature):

        0.00  no answer, or the scorer failed
        0.05  Lean rejected it
        0.15  elaborates, does not match the gold, proof unfinished/absent
        0.30  a real finished proof, but of a theorem BEq+ could not match
        0.50  BEq+ matches the gold, proof still unfinished
        0.85-1.00  BEq+ matches AND the candidate's own proof is sorry-free
                   and axiom-clean (`solved`)

    `_score_pair` runs BEq+'s statement cascade (ignores proof bodies on both
    sides, unchanged); `check_own_proof` separately elaborates the candidate's
    OWN proof, sorry included, to determine `proved`. Unlike `gated`, this
    pays a graded reward below BEq+, so groups with no semantic signal still
    carry a within-group gradient. That is the property `gated` deliberately
    removes, so the two are an A/B and this is not yet the default. Emits
    `outcome_code` for the per-example histogram.
    """
    scorer = _get_scorer()
    r, pred, _gold = _score_pair(solution_str, ground_truth)
    context, _gold_theorem = split_header_and_theorem(ground_truth)
    with _lean_slot:
        proof_check = scorer.check_own_proof(pred, context)
    s = signals_from_score(r, proof_check)
    return {"score": reward_for(s), **_diagnostics(r, proof_check=proof_check)}


# Used when custom_reward_function.name is left unset. Every job here sets the
# name explicitly, so this alias is a fallback rather than what the arms run.
compute_score = compute_score_outcome


if __name__ == "__main__":
    gold = ("theorem lean_workbook_plus_2 (x : ℝ) : "
            "x^2 - 2*x - 24 < 0 ↔ x ∈ Set.Ioo (-4) 6 := by sorry")
    cases = [
        ("good", "theorem restated (x : ℝ) : x^2 - 2*x - 24 < 0 ↔ x ∈ Set.Ioo (-4) 6"),
        ("wrong", "theorem restated (x : ℝ) : x^2 - 2*x - 24 > 0 ↔ x ∈ Set.Ioo (-4) 6"),
        ("fenced", "Here is the formalization:\n```lean\ntheorem restated (x : ℝ) : "
                   "x^2 - 2*x - 24 < 0 ↔ x ∈ Set.Ioo (-4) 6\n```"),
        # Strictly stronger: implies the gold but is not implied by it. Lands on
        # 0 directions unless BEQ_PROBE_STRONGER=1; see beq_plus.py.
        ("one_way", "theorem restated (x : ℝ) : x^2 - 2*x - 24 < 0 ∧ x > -4 ↔ "
                    "x ∈ Set.Ioo (-4) 6"),
        ("near_miss", "theorem restated (x : ℝ) : x^2 - 2*x - 25 < 0 ↔ x ∈ Set.Ioo (-4) 6"),
        # Trivially true. Should score 1.0 on typecheck-only and 0 on gated.
        ("hack", "theorem hack (x : ℝ) : x - 1 = -1 + x"),
    ]

    print(f"probe_stronger={os.environ.get('BEQ_PROBE_STRONGER', '0')}")
    print(f"{'case':12s} {'typecheck':>9} {'gated':>7}")
    for name, pred in cases:
        tc = compute_score_typecheck_only("lean_workbook", pred, gold)["score"]
        gated = compute_score_gated("lean_workbook", pred, gold)["score"]
        print(f"{name:12s} {tc:>9.2f} {gated:>7.2f}")
