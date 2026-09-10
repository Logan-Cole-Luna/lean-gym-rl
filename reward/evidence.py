#!/usr/bin/env python3
"""The Lean evidence an arm scores, and the diagnostic schema verl aggregates.

GATHERING, NOT SCORING. Nothing here decides what a rollout is worth: it runs
the BEq+ cascade and the candidate's own proof through one persistent Lean REPL,
measures the two proof bodies, and shapes the result into the flat numeric dict
verl expects. `reward/terms/` turns those numbers into a reward and
`reward/arms.py` says which terms apply.

ONE SCORER PER PROCESS. Building a `BEqPlusScorer` spawns a Lean REPL and
imports Mathlib (~4.3GB resident, minutes to start), so it is built lazily once
and reused, with `_lean_slot` serialising access.
"""
from __future__ import annotations

import functools
import os
import re
import threading

from lean_interact.utils import split_implementation

from reward.beq_plus import BEqPlusScorer, split_header_and_theorem
from reward.reward import OUTCOMES, outcome_for, signals_from_score

_scorer: BEqPlusScorer | None = None
_scorer_failures = 0
_scorer_lock = threading.Lock()

# Serialises Lean access.
BEQ_MAX_CONCURRENT = int(os.environ.get("BEQ_MAX_CONCURRENT", "1"))
_lean_slot = threading.Semaphore(BEQ_MAX_CONCURRENT)

# Building a scorer spawns a Lean REPL and imports Mathlib. 
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


def _proof_body_len(text: str) -> int | None:
    """Characters in the declaration's PROOF BODY, or None if it has none.

    `split_implementation` finds the first BRACKET-BALANCED `:=` that is not a
    `let`/`haveI` binder. Do not be tempted by `rfind(":=")` -- MEASURED over
    the 18,757 LoCoLib golds, 22.4% carry a `:=` inside the proof (`have h : P
    := ...`) and for 16.8% `rfind` undercounts the body by more than 40 chars,
    worst case 5986 -> 259. That would invent a one-line phantom reference and
    mark every honest proof as bloated. (`strip_proof` in
    scripts/data/prepare_locolib.py still has this bug; it only reaches the
    dedup key on the proof-pair arm, so it is not fixed here.)

    `split_header_and_theorem` runs FIRST and is not optional: a self-contained
    prediction brings its own `variable`/`def` preamble, and a `:=` in there
    would otherwise be mistaken for the start of the proof.
    """
    try:
        _ctx, decl = split_header_and_theorem(text)
        i = split_implementation(decl)
    except Exception:
        return None
    if i is None:
        return None
    body = decl[i + 2:].strip()
    return len(body) or None


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


def _never_raises(zero: dict):
    """Decorator factory: score `zero` rather than propagating an exception.

    A reward that raises does not just lose its rollout: verl marks the agent
    task failed, the GRPO step never closes, and the job holds a GPU until
    walltime. Scoring 0 is not neutral -- a correct rollout lost to an
    infrastructure fault pushes the gradient the wrong way -- so it is reported
    as `scorer_error=1` rather than hidden. `zero` is per-function (see the
    _*_ZERO dicts above) so the fallback never hands verl a different key set
    than the function's own normal-path return.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapped(data_source, solution_str, ground_truth, extra_info=None):
            try:
                return fn(data_source, solution_str, ground_truth, extra_info)
            except Exception as e:
                print(f"[reward] {fn.__name__} FAILED, scoring 0: "
                      f"{type(e).__name__}: {e}"[:300], flush=True)
                return dict(zero)
        return wrapped
    return decorator


def _score_pair(solution_str: str, ground_truth: str) -> tuple[dict, str, str]:
    """Run the BEq+ scorer once. Returns (result, prediction, gold theorem)."""
    scorer = _get_scorer()
    pred = _clean_solution(solution_str)
    _gold_context, gold_theorem = split_header_and_theorem(ground_truth)
    with _lean_slot:
        r = scorer.score(ground_truth, pred)
    _note_call(r.get("error_kind"))
    return r, pred, gold_theorem


# Column 2 of reward.py's table ("Lean compiles it") now reads the SUBMISSION AS
# WRITTEN via check_own_proof, not BEq+'s statement-only check which sorries the
# proof away. Set to 0 to restore the old wiring when comparing against numbers
# recorded before the fix -- it changes what a run scores, so the two are not
# directly comparable. See `signals_from_score`.
TYPECHECK_FROM_OWN_PROOF = os.environ.get("BEQ_TYPECHECK_FROM_OWN_PROOF", "1") == "1"


# The rich diagnostic set, used only by `compute_score_outcome` -- the arm
# actually under iteration, where the cascade internals (rung, convert_level,
# proved, ...) are worth reading. `gated` and `typecheck` return small,
# hand-built dicts instead of this (see their functions below): fewer moving
# parts, and each one's own zero-fallback can match its own shape exactly.
#
# Every reward must emit the same key set ACROSS ITS OWN CALLS: verl
# aggregates each key into a metric, and an omitted key gives a ragged
# schema. These are the "not applicable" values, not measurements.
_SCHEMA = {
    "acc": 0.0, "beq_plus": 0.0, "typecheck": 0.0,
    "semantic_signal": 0.0, "n_directions": 0.0,
    "gold_implies_pred": 0.0, "pred_implies_gold": 0.0,
    "similarity": 0.0, "scorer_error": 0.0,
    "beql": 0.0, "rung": 0.0, "convert_level": 0.0,
    "provable_alone": 0.0, "provable_alone_known": 0.0, "stop_reason": 0.0,
    # Index into reward.reward.OUTCOMES.
    "outcome_code": 0.0,
    # Set for real from BEqPlusScorer.check_own_proof; "not applicable"
    # otherwise (no other current caller reaches this dict).
    "proved": 0.0, "sorry_used": 0.0,
    # Length agreement with the gold proof. The "not applicable" value is 1.0,
    # NOT 0.0 -- brevity is subtracted, so an unknown must leave the score
    # alone. `brevity_known` separates "in tolerance" from "never measured",
    # which a mean over `brevity` alone would conflate.
    "brevity": 1.0, "brevity_known": 0.0,
    "pred_proof_chars": 0.0, "gold_proof_chars": 0.0,
}

# `outcome`'s zero-fallback: the full schema above.
_OUTCOME_ZERO = {"score": 0.0, **_SCHEMA, "scorer_error": 1.0}
# `gated` and `typecheck` each emit a small dict matching their own function
# below -- their zero-fallback must match THAT shape, not the outcome one, or
# a mid-run scorer failure would change the key set verl sees mid-stream.
_GATED_ZERO = {"score": 0.0, "acc": 0.0, "beq_plus": 0.0, "typecheck": 0.0, "scorer_error": 1.0}
_TYPECHECK_ZERO = {"score": 0.0, "typecheck": 0.0, "scorer_error": 1.0}

_STOP_REASON_CODE = {"unparseable": 1, "sorry_gate": 2, "no_rung": 3}


def _diagnostics(r: dict, proof_check: dict | None = None,
                 brevity: float | None = None,
                 pred_chars: int | None = None,
                 gold_chars: int | None = None,
                 wrote_something: bool = True) -> dict[str, float]:
    """Per-sample fields forwarded into `reward_extra_info`. All numeric.
    Used only by `compute_score_outcome` -- see the module note above `_SCHEMA`.

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
        # Same wiring as the score above, or the per-example histogram would
        # disagree with the reward it is meant to explain.
        "outcome_code": float(OUTCOMES.index(outcome_for(signals_from_score(
            r, proof_check, typecheck_from_own_proof=TYPECHECK_FROM_OWN_PROOF,
            wrote_something=wrote_something)))),
        "proved": float(proof_check.get("proved", False)) if proof_check else 0.0,
        "sorry_used": float(proof_check.get("sorry_used", False)) if proof_check else 0.0,
        # 1.0 when unmeasurable, so mean(brevity) reads as "how much of the band
        # is being paid" and never dips for a missing measurement.
        "brevity": 1.0 if brevity is None else float(brevity),
        "brevity_known": float(brevity is not None),
        # Raw lengths, so proof INFLATION is visible directly. Watch
        # mean(pred_proof_chars) against mean(gold_proof_chars): the failure
        # this term exists to prevent is the first climbing away from the second.
        "pred_proof_chars": float(pred_chars or 0),
        "gold_proof_chars": float(gold_chars or 0),
    }
