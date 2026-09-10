#!/usr/bin/env python3
"""THE ARM TABLE. What each reward arm is made of, in one place.

An ARM is not a function here, it is a ROW: a pipeline that gathers Lean
evidence, a reward table over the six outcomes, and which optional terms are
live. `ARMS` below is the only place that composition is written down. Adding an
arm is a row here plus one binding in `reward/reward_fn.py`; changing what an
existing arm pays for is editing its row and nothing else.

    reward/reward.py    the six outcomes and the ladder. How an outcome is DECIDED.
    reward/terms/       one module per term. What each component is WORTH.
    reward/evidence.py  the Lean calls and the diagnostic schema.
    this file           which of those a given run combines.

SEPARATE NAMES ARE DELIBERATE, and that is why `reward_fn.py` binds one module
attribute per arm rather than reading the arm from an environment variable. verl
records `custom_reward_function.name` in its config dump, and that dump is how a
finished run is identified months later. Two arms distinguishable only by an env
var are indistinguishable there. The MAKEUP is configurable; the NAME is not.

THREE PIPELINES, because the arms do not all ask Lean the same question. The
ladder runs the BEq+ cascade AND elaborates the candidate's own proof; `gated`
runs the cascade only; `typecheck` elaborates only. They emit different key
sets, so each carries its own zero-fallback -- see `reward/evidence.py`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from reward.beq_plus import split_header_and_theorem
from reward.evidence import (_GATED_ZERO, _OUTCOME_ZERO, _TYPECHECK_ZERO,
                             _clean_solution, _diagnostics, _get_scorer,
                             _lean_slot, _never_raises, _note_call,
                             _proof_body_len, _score_pair,
                             TYPECHECK_FROM_OWN_PROOF)
from reward.reward import reward_for, signals_from_score
from reward.terms.brevity import BREVITY_BAND, brevity_for
from reward.terms.gated import GATED_REWARDS, gated_reward
from reward.terms.typecheck import TYPECHECK_REWARDS, typecheck_reward

W_GATED_ONE_DIR = float(os.environ.get("BEQ_W_GATED_ONE_DIR", "0.25"))


# How much of a SOLVED rollout's reward the proof-length term can take, for the
# arms that turn it on. `compute_score_outcome` passes 0.0 and is unaffected.
#
# THE TERM NEEDS A RESPONSE CAP THAT DOES NOT BOUND THE LENGTH DISTRIBUTION.
# Under the old 128-token `max_response_length` it was unreachable: 0 of 182
# SOLVED rollouts across gated/outcome/typecheck at step 90 were long enough to
# charge, because the cap already truncated the top of the distribution. Gold
# ANSWERS (theorem+proof; the context is in the prompt) are p50 54 / p90 115 /
# max 275 Qwen tokens on the 760-row proof val slice, so 128 truncates 6.1% of
# reference-length answers. `configs/run_grpo.sh` defaults to 512.
#
# MEASURED at 512 over the certified population: mean b 0.834, mean charge
# 0.017 of the band, against 0.001 under the earlier dead-band formulation.
W_BREVITY_BAND_ARM = float(os.environ.get("BEQ_BREVITY_BAND", str(BREVITY_BAND)))



@dataclass(frozen=True)
class ArmSpec:
    """One arm, as a composition rather than a function.

    `rewards=None` means `reward.reward.DEFAULT_REWARDS`, the graded ladder.

    THE FAITHFULNESS BAND IS NOT A FIELD, and that is not an oversight. The band
    is carved out of the top row whether or not an arm measures f, so an arm
    that does not measure it still tops out at 0.85 (see
    `reward/terms/faithfulness.py`). `faith` therefore says whether f is
    MEASURED, not how much it is worth -- turning it on cannot move any
    already-recorded number, only make the band reachable.
    """
    pipeline: str
    rewards: dict | None = None
    brevity: bool = False
    faith: bool = False
    doc: str = ""

    @property
    def brevity_band(self) -> float:
        """What the length term may take from a SOLVED rollout. 0.0 when off."""
        return W_BREVITY_BAND_ARM if self.brevity else 0.0


def _ladder(spec: ArmSpec, solution_str: str, ground_truth: str, extra_info) -> dict:
    """BEq+ cascade AND the candidate's own proof, scored on the six-outcome ladder.

    The shared body of every ladder arm. They differ only in `spec`, which is
    what makes `outcome` a true control for the rest: one field changes, nothing
    else does.
    """
    scorer = _get_scorer()
    r, pred, _gold = _score_pair(solution_str, ground_truth)
    context, _gold_theorem = split_header_and_theorem(ground_truth)
    with _lean_slot:
        proof_check = scorer.check_own_proof(pred, context)
    # Both sides measured the same way, from the same splitter. No Lean call:
    # this is pure string work and adds nothing to the reward's latency.
    # Computed for EVERY arm even when the band is 0, because it is a
    # diagnostic first: `pred_proof_chars` drifting away from
    # `gold_proof_chars` is the failure the band exists to catch, and an arm
    # that does not charge for it still needs to report it.
    pred_chars = _proof_body_len(pred)
    gold_chars = _proof_body_len(ground_truth)
    brevity = brevity_for(pred_chars, gold_chars)
    wrote_something = bool(pred.strip())
    s = signals_from_score(r, proof_check, brevity=brevity,
                           typecheck_from_own_proof=TYPECHECK_FROM_OWN_PROOF,
                           wrote_something=wrote_something)
    return {"score": reward_for(s, spec.rewards, brevity_band=spec.brevity_band),
            **_diagnostics(r, proof_check=proof_check, brevity=brevity,
                           pred_chars=pred_chars, gold_chars=gold_chars,
                           wrote_something=wrote_something)}


def _gated(spec: ArmSpec, solution_str: str, ground_truth: str, extra_info) -> dict:
    """BEq+ cascade only. No own-proof elaboration, and a small key set.
    BEq+ ladder: 1.0 for both directions, W_GATED_ONE_DIR for one, else 0.

        Type-check is not paid. The gate is implicit, since nothing that fails to
        elaborate can prove in either direction, and paying for elaboration is the
        measured failure mode this reward exists to avoid.

        The flat floor is the point. Under GRPO the advantage is the reward minus
        the group mean, so a group with no semantic signal produces an advantage of
        exactly zero and contributes no gradient. Any term that varies within such a
        group -- similarity, type-check -- becomes the only climbable signal there,
        and the policy learns to farm it.

        Same shape as `compute_score_typecheck`: a small hand-built dict, not
        `_diagnostics()` -- this arm only needs the two signals in its name.    """
    r, _pred, _gold = _score_pair(solution_str, ground_truth)
    return {"score": gated_reward(r, W_GATED_ONE_DIR, spec.rewards),
            "acc": float(r["beq_plus"]),
            "beq_plus": float(r["beq_plus"]),
            "typecheck": float(r["typecheck"]),
            "scorer_error": float(bool(r.get("error_kind")))}


def _typecheck(spec: ArmSpec, solution_str: str, ground_truth: str, extra_info) -> dict:
    """One elaboration of the candidate's own submission. No BEq+ cascade.
    1.0 if the candidate's OWN submission elaborates, else 0.0.

        Uses `check_own_proof`, not `typecheck_ex`: on the proof-pair task the
        candidate writes a real proof body, and `typecheck_ex` would discard it
        and re-inject `sorry` before elaborating, silently checking only the
        signature. `sorry` inside the candidate's own proof still counts as
        type-correct here (a warning, not an error), so this stays the same
        exploitable, cheap ablation baseline it always was -- just checking the
        real submission instead of a forced stand-in.    """
    scorer = _get_scorer()
    pred = _clean_solution(solution_str)
    context, _gold_theorem = split_header_and_theorem(ground_truth)
    with _lean_slot:
        p = scorer.check_own_proof(pred, context)
    _note_call(p["error_kind"])
    # Deliberately no `acc`: this arm never computes BEq+, so there is no honest
    # value to report and verl falls back to the reward mean for val-core.
    return {"score": typecheck_reward(p["type_correct"], p["error_kind"], spec.rewards),
            "typecheck": float(p["type_correct"]),
            "scorer_error": float(bool(p["error_kind"]))}


# Pipeline -> (implementation, the zero-fallback matching ITS key set). The zero
# is per-pipeline rather than per-arm because the key set is a property of what
# Lean was asked, not of what the arm pays.
PIPELINES = {
    "ladder":    (_ladder, _OUTCOME_ZERO),
    "gated":     (_gated, _GATED_ZERO),
    "typecheck": (_typecheck, _TYPECHECK_ZERO),
}

_DOC_OUTCOME = """The six-outcome ladder from reward/reward.py, now fully reachable on the
    proof-pair task (the candidate writes a real proof, not just a signature):

        0.00  no answer, or the scorer failed
        0.05  Lean rejected it
        0.15  elaborates, does not match the gold, proof unfinished/absent
        0.30  a real finished proof, but of a theorem BEq+ could not match
        0.50  BEq+ matches the gold, proof still unfinished
        0.85  BEq+ matches AND the candidate's own proof is sorry-free and
              axiom-clean (`solved`). Flat at 0.85, not 0.85-1.00: this arm
              measures no faithfulness, and an unmeasured f is not paid.

    `_score_pair` runs BEq+'s statement cascade (ignores proof bodies on both
    sides, unchanged); `check_own_proof` separately elaborates the candidate's
    OWN proof, sorry included, to determine `proved`. Unlike `gated`, this
    pays a graded reward below BEq+, so groups with no semantic signal still
    carry a within-group gradient. That is the property `gated` deliberately
    removes, so the two are an A/B and this is not yet the default. Emits
    `outcome_code` for the per-example histogram.

    This is also the CONTROL for the arm below: that arm is this function plus
    one term, so a difference between the two is attributable to that term and
    to nothing else -- provided both run at the same `max_response_length`,
    which is why the 128-token cap had to be lifted before either could be
    interpreted.
"""

_DOC_BREVITY = """`compute_score_outcome` plus the proof-LENGTH deduction. Nothing else.

    A SOLVED rollout is charged up to `W_BREVITY_BAND_ARM` for a proof body far
    from the gold's length, as `exp(-|ln((L+s)/(L*+s))|)` with s=60, symmetric
    in the softened log ratio -- see `reward.reward.brevity_for`. Rows 1-5 are
    untouched, which is the anti-exploit: a degenerately short non-proof lands
    on `incomplete` (0.15) and never reaches the term.

    THIS ARM IS A NO-OP BELOW A ~384-TOKEN RESPONSE CAP. At the old
    `max_response_length=128` the term fired on 0 of 182 SOLVED rollouts,
    because the cap already bounded the length distribution below anything the
    band charges for. Running it there produces a bit-identical copy of
    `compute_score_outcome`, not a null result.
"""

_DOC_GATED = """BEq+ ladder: 1.0 for both directions, W_GATED_ONE_DIR for one, else 0.

    Type-check is not paid. The gate is implicit, since nothing that fails to
    elaborate can prove in either direction, and paying for elaboration is the
    measured failure mode this reward exists to avoid.

    The flat floor is the point. Under GRPO the advantage is the reward minus
    the group mean, so a group with no semantic signal produces an advantage of
    exactly zero and contributes no gradient. Any term that varies within such a
    group -- similarity, type-check -- becomes the only climbable signal there,
    and the policy learns to farm it.

    Same shape as `compute_score_typecheck`: a small hand-built dict, not
    `_diagnostics()` -- this arm only needs the two signals in its name.
"""

_DOC_TYPECHECK = """1.0 if the candidate's OWN submission elaborates, else 0.0.

    Uses `check_own_proof`, not `typecheck_ex`: on the proof-pair task the
    candidate writes a real proof body, and `typecheck_ex` would discard it
    and re-inject `sorry` before elaborating, silently checking only the
    signature. `sorry` inside the candidate's own proof still counts as
    type-correct here (a warning, not an error), so this stays the same
    exploitable, cheap ablation baseline it always was -- just checking the
    real submission instead of a forced stand-in.
"""

# ---------------------------------------------------------------------------
# THE TABLE. Edit a row to change what an arm is made of.
# ---------------------------------------------------------------------------
ARMS = {
    "outcome":         ArmSpec(pipeline="ladder", doc=_DOC_OUTCOME),
    "outcome_brevity": ArmSpec(pipeline="ladder", brevity=True, doc=_DOC_BREVITY),
    "gated":           ArmSpec(pipeline="gated", rewards=GATED_REWARDS, doc=_DOC_GATED),
    "typecheck":       ArmSpec(pipeline="typecheck", rewards=TYPECHECK_REWARDS,
                               doc=_DOC_TYPECHECK),
}


def build(name: str):
    """Turn one row of `ARMS` into a verl `custom_reward_function`.

    VALIDATION HAPPENS HERE, AT IMPORT, on purpose. A misconfigured arm must
    fail the job at startup, not per-rollout: every entry point carries
    `@_never_raises`, so a term this branch cannot compute would otherwise
    surface as a silent stream of zeros that trains the policy on noise for a
    full walltime.
    """
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; ARMS has {sorted(ARMS)}")
    spec = ARMS[name]
    if spec.pipeline not in PIPELINES:
        raise ValueError(f"arm {name!r} names no known pipeline: {spec.pipeline!r}")
    if spec.faith:
        raise ValueError(
            f"arm {name!r} sets faith=True, but no faithfulness judge is wired "
            "on this branch. `reward/terms/faithfulness.py` defines the band; "
            "the metric that fills it does not exist here yet."
        )
    if spec.brevity and spec.pipeline != "ladder":
        raise ValueError(
            f"arm {name!r} turns brevity on, but the {spec.pipeline!r} pipeline "
            "never reaches the SOLVED row the term applies to."
        )
    run, zero = PIPELINES[spec.pipeline]

    def compute(data_source, solution_str, ground_truth, extra_info=None) -> dict:
        return run(spec, solution_str, ground_truth, extra_info)

    # Set before decorating: `_never_raises` copies these onto its wrapper, and
    # its failure line prints `__name__`.
    compute.__name__ = compute.__qualname__ = f"compute_score_{name}"
    compute.__doc__ = spec.doc
    return _never_raises(zero)(compute)
