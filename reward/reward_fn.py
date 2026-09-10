#!/usr/bin/env python3
"""verl `custom_reward_function` entry points.

Selected per run via `custom_reward_function.name`:

- `compute_score_outcome` (default): the six-outcome ladder from reward/reward.py.
- `compute_score_outcome_brevity`: the ladder, plus a proof-LENGTH deduction on
  the top row.
- `compute_score_outcome_faith`: the ladder, plus the NL -> FL faithfulness band
  on the top row. Needs a judge server and an informal-carrying corpus.
- `compute_score_outcome_v2`: the ladder with BOTH of the above live.
- `compute_score_gated`: pays only for BEq+ equivalence, flat floor below it.
- `compute_score_typecheck`: pays for elaboration alone. Exploitable, and
  kept as the ablation baseline.

The four `outcome*` arms share one body, `_outcome_core`, and differ only in two
keyword flags, so `outcome` is a true control for the other three. They are
separate NAMES rather than one name plus env vars on purpose: verl writes
`custom_reward_function.name` into its config dump, and that dump is how a run
is identified after the fact.

Each returns a dict, not a float. verl forwards every key other than `score`
into `reward_extra_info` and aggregates it into a train/val metric. Emitting
`acc` = BEq+ makes `val-core/<data_source>/acc/mean@1` the true BEq+ rate rather
than the mean of whatever reward the run happens to use.

All entry points share one lazily built `BEqPlusScorer` per process, so repeated
calls reuse a single Lean REPL instead of re-importing Mathlib.
"""
from __future__ import annotations

import dataclasses
import functools
import os
import re
import threading

from lean_interact.utils import split_implementation

from reward.beq_plus import BEqPlusScorer, split_header_and_theorem
from reward.reward import (BREVITY_BAND, FAITHFULNESS_BAND, OUTCOMES, SOLVED,
                           brevity_for, gated_reward, outcome_for, reward_for,
                           signals_from_score, typecheck_reward)

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
    # NL -> FL faithfulness. The "not applicable" value is 0.0, the OPPOSITE of
    # brevity's, and the asymmetry is the point: faithfulness is PAID and
    # brevity is DEDUCTED, so an unknown must never award the one nor charge the
    # other. `faith_known` is the number to actually watch. Coverage is
    # structurally bounded here -- a term-mode proof emits no tactics and so has
    # no FL blocks to judge, which was 43.8% of the policy's own proofs -- so
    # mean(faith) alone cannot distinguish "unfaithful" from "unjudgeable", and
    # a faith arm that raises mean(faith) purely by raising `faith_known` has
    # learned to write tactic-mode proofs, not faithful ones. Watch both.
    "faith": 0.0, "faith_known": 0.0, "faith_seconds": 0.0,
    # Index into _FAITH_ERROR_CODE; 0 means "scored, no error".
    "faith_error": 0.0,
}

# Why a faithfulness verdict was unavailable. Numeric because verl aggregates
# reward_extra_info by mean, so the useful reading is the SHIFT in the mix, not
# any one value: `fl_blocks_empty` (term-mode) is a property of the policy's
# output and is expected to move during training, while `no_judge` is the judge
# server falling over and must not be mistaken for it.
_FAITH_ERROR_CODE = {
    None: 0, "no_informal": 1, "no_autofaith": 2, "no_judge": 3, "bad_json": 4,
    "no_declaration": 5, "fl_extract_failed": 6, "fl_blocks_empty": 7,
    "nl_blocks_failed": 8,
}

# `outcome`'s zero-fallback: the full schema above.
_OUTCOME_ZERO = {"score": 0.0, **_SCHEMA, "scorer_error": 1.0}
# `gated` and `typecheck` each emit a small dict matching their own function
# below -- their zero-fallback must match THAT shape, not the outcome one, or
# a mid-run scorer failure would change the key set verl sees mid-stream.
_GATED_ZERO = {"score": 0.0, "acc": 0.0, "beq_plus": 0.0, "typecheck": 0.0, "scorer_error": 1.0}
_TYPECHECK_ZERO = {"score": 0.0, "typecheck": 0.0, "scorer_error": 1.0}

_STOP_REASON_CODE = {"unparseable": 1, "sorry_gate": 2, "no_rung": 3}


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


def _diagnostics(r: dict, proof_check: dict | None = None,
                 brevity: float | None = None,
                 pred_chars: int | None = None,
                 gold_chars: int | None = None,
                 wrote_something: bool = True,
                 faith=None) -> dict[str, float]:
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
        # 0.0 when unmeasured, matching the band's own convention that an
        # unknown is not paid. Read alongside `faith_known`, never alone.
        "faith": 0.0 if (faith is None or faith.score is None) else float(faith.score),
        "faith_known": float(faith is not None and faith.score is not None),
        "faith_seconds": float(faith.seconds) if faith is not None else 0.0,
        "faith_error": float(_FAITH_ERROR_CODE.get(
            faith.error if faith is not None else None, 0)),
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


@_never_raises(_TYPECHECK_ZERO)
def compute_score_typecheck(data_source, solution_str, ground_truth, extra_info=None) -> dict:
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

# Column 2 of reward.py's table ("Lean compiles it") now reads the SUBMISSION AS
# WRITTEN via check_own_proof, not BEq+'s statement-only check which sorries the
# proof away. Set to 0 to restore the old wiring when comparing against numbers
# recorded before the fix -- it changes what a run scores, so the two are not
# directly comparable. See `signals_from_score`.
TYPECHECK_FROM_OWN_PROOF = os.environ.get("BEQ_TYPECHECK_FROM_OWN_PROOF", "1") == "1"


@_never_raises(_GATED_ZERO)
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

    Same shape as `compute_score_typecheck`: a small hand-built dict, not
    `_diagnostics()` -- this arm only needs the two signals in its name.
    """
    r, _pred, _gold = _score_pair(solution_str, ground_truth)
    return {"score": gated_reward(r, W_GATED_ONE_DIR),
            "acc": float(r["beq_plus"]),
            "beq_plus": float(r["beq_plus"]),
            "typecheck": float(r["typecheck"]),
            "scorer_error": float(bool(r.get("error_kind")))}


# How much of a SOLVED rollout the faithfulness band is worth, for the arms that
# turn it on. This is reward/reward.py's FAITHFULNESS_BAND, restated here only
# so a reader of the arm sees the number without a second file.
#
# THE BAND IS ALREADY CARVED OUT WHETHER OR NOT AN ARM MEASURES f. With f=None,
# `reward_for_outcome` scores SOLVED at 0.85, and that is what every arm before
# this one recorded. Turning faithfulness on therefore does not move the floor;
# it makes the top 0.15 REACHABLE. So `outcome` and `outcome_faith` differ only
# in whether a solved rollout can earn above 0.85 -- which is exactly the
# contrast the arm is supposed to isolate.
W_FAITH_BAND = float(os.environ.get("BEQ_FAITH_BAND", str(FAITHFULNESS_BAND)))

_faith_calls = 0
_faith_unknown = 0
_FAITH_LOG_EVERY = int(os.environ.get("BEQ_FAITH_LOG_EVERY", "100"))


def _note_faith(res) -> None:
    """Periodic coverage line for the faithfulness judge.

    A judge server that dies scores every rollout `no_judge` -> f unknown -> a
    flat 0.85 on every SOLVED row, which is silently identical to running the
    plain `outcome` arm. Nothing else in the loop would say so: the reward keeps
    returning plausible numbers and the run completes. Hence this.
    """
    global _faith_calls, _faith_unknown
    _faith_calls += 1
    _faith_unknown += res.score is None
    if _FAITH_LOG_EVERY and _faith_calls % _FAITH_LOG_EVERY == 0:
        rate = _faith_unknown / _faith_calls
        print(f"[reward] faith: {_faith_calls} judged, "
              f"{100 * rate:.1f}% unknown (last cause: {res.error})", flush=True)
        if rate > 0.9:
            print("[reward] WARNING: >90% of faithfulness verdicts are unknown. "
                  "If the cause is `no_judge` the judge server is down and this "
                  "arm has silently degenerated into plain `outcome`; check "
                  f"FAITH_JUDGE_URL ({os.environ.get('FAITH_JUDGE_URL', 'default')}).",
                  flush=True)


def _faithfulness(pred: str, context: str, extra_info):
    """f in [0,1] for one rollout, as a FaithfulnessResult. Never raises.

    THE INFORMAL TEXT COMES FROM `extra_info`, not from the prompt: a verl
    custom reward is handed (data_source, solution_str, ground_truth,
    extra_info) and never sees the prompt the rollout answered. The stock
    LoCoLib parquets carry only split/index/id/domain, so a faith arm must
    train on a corpus built by `scripts/data/add_informal_to_extra_info.py`,
    which copies the two fields out of each row's own prompt. Missing fields
    are `no_informal`, i.e. UNKNOWN -- never a zero the rollout earned.
    """
    from reward.faithfulness import FaithfulnessResult, score_faithfulness
    info = extra_info or {}
    statement = info.get("informal") or ""
    proof_nl = info.get("informal_proof") or ""
    if not proof_nl:
        return FaithfulnessResult(None, "no_informal")
    # `_lean_slot` is handed in, not held around this call: score_faithfulness
    # takes it for its Lean step and drops it before the judge, per that
    # module's "never holds the Lean semaphore across a network call".
    res = score_faithfulness(statement, proof_nl, pred, scorer=_get_scorer(),
                             context=context, lean_lock=_lean_slot)
    _note_faith(res)
    return res


def _outcome_core(solution_str: str, ground_truth: str, extra_info,
                  *, brevity_band: float, faith: bool) -> dict:
    """The shared body of every `outcome`-family arm.

    ONE BODY, THREE ARMS, differing only in the two keyword flags. They are
    separate named entry points rather than one function reading env vars
    because verl records `custom_reward_function.name` in its config dump, and
    that dump is how a finished run is identified months later -- two arms that
    differ only by an environment variable are indistinguishable there.

    FAITHFULNESS IS COMPUTED ON THE `SOLVED` ROW ALONE, and that is a cost
    decision as much as a semantic one: `reward_for_outcome` reads f nowhere
    else, and ~24% of rollouts reach that row, so gating here removes ~76% of
    the judge traffic without changing a single score.
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
    f_res = None
    if faith and outcome_for(s) == SOLVED:
        f_res = _faithfulness(pred, context, extra_info)
        s = dataclasses.replace(s, proof_follows_argument=f_res.score)
    return {"score": reward_for(s, band=W_FAITH_BAND, brevity_band=brevity_band),
            **_diagnostics(r, proof_check=proof_check, brevity=brevity,
                           pred_chars=pred_chars, gold_chars=gold_chars,
                           wrote_something=wrote_something, faith=f_res)}


@_never_raises(_OUTCOME_ZERO)
def compute_score_outcome(data_source, solution_str, ground_truth, extra_info=None) -> dict:
    """The six-outcome ladder from reward/reward.py, now fully reachable on the
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

    This is also the CONTROL for the three arms below: they are this function
    plus one term each, so a difference between them and this is attributable
    to that term and to nothing else -- provided all four run at the same
    `max_response_length`, which is why the 128-token cap had to be lifted
    before any of them could be interpreted.
    """
    return _outcome_core(solution_str, ground_truth, extra_info,
                         brevity_band=0.0, faith=False)


@_never_raises(_OUTCOME_ZERO)
def compute_score_outcome_brevity(data_source, solution_str, ground_truth,
                                  extra_info=None) -> dict:
    """`compute_score_outcome` plus the proof-LENGTH deduction. Nothing else.

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
    return _outcome_core(solution_str, ground_truth, extra_info,
                         brevity_band=W_BREVITY_BAND_ARM, faith=False)


@_never_raises(_OUTCOME_ZERO)
def compute_score_outcome_faith(data_source, solution_str, ground_truth,
                                extra_info=None) -> dict:
    """`compute_score_outcome` plus the NL -> FL faithfulness band. Nothing else.

    A SOLVED rollout scores `0.85 + 0.15 * f`, where f asks whether the Lean
    proof carries out the ENGLISH argument the prompt supplied. This is the only
    place the informal text enters the score.

    REQUIRES A JUDGE SERVER on `FAITH_JUDGE_URL` and a corpus whose `extra_info`
    carries `informal` / `informal_proof`. Both missing pieces degrade to
    unknown, i.e. to a flat 0.85, i.e. to plain `outcome` -- silently, except
    for `_note_faith`'s coverage line and the `faith_known` metric. Check both
    before trusting a finished run.

    KNOWN CONFOUND, and it is not fixable inside the reward. Coverage is
    structural: a term-mode proof (`:= rfl`) emits no tactics, so AutoFaith has
    no FL blocks to judge and f is unknown, which pays nothing. Term-mode was
    43.8% of the policy's own proofs. So this band pays tactic-mode proofs and
    not term-mode ones, independently of whether either is faithful, and part of
    any gain it shows will be the policy learning to write `by ...` instead of a
    proof term. `faith_known` measures exactly that share and must be reported
    beside any headline this arm produces.
    """
    return _outcome_core(solution_str, ground_truth, extra_info,
                         brevity_band=0.0, faith=True)


@_never_raises(_OUTCOME_ZERO)
def compute_score_outcome_v2(data_source, solution_str, ground_truth,
                             extra_info=None) -> dict:
    """OUTCOME V2: the six-outcome ladder with BOTH new terms live.

    A SOLVED rollout scores `0.85 + 0.15*f - 0.10*(1-b)`, so the two terms pull
    in opposite directions by construction -- faithfulness pays for a proof that
    follows the argument, brevity charges for one that rambles to get there --
    and this arm is where they are allowed to interact. Rows 1-5 are the
    untouched constants in every case.

    Run it against `outcome_brevity` and `outcome_faith`, not only against
    `outcome`: the three-way contrast is what separates "the two terms add" from
    "one of them is carrying the whole difference".
    """
    return _outcome_core(solution_str, ground_truth, extra_info,
                         brevity_band=W_BREVITY_BAND_ARM, faith=True)



# Used when custom_reward_function.name is left unset. Every job here sets the
# name explicitly, so this alias is a fallback rather than what the arms run.
compute_score = compute_score_outcome


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
