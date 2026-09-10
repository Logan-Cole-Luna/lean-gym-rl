#!/usr/bin/env python3
"""How one autoformalization rollout is scored. The shared definition.

Used in two places and deliberately owned by neither:

    harness evolution   the mean of this over a frozen task set IS a harness's fitness
    RL fine-tuning      this is the per-rollout reward

Adopted from the harness-evolution project. The table below is unchanged; what
this repo adds is `signals_from_score` and the `compute_score_outcome` verl entry
point at the bottom, which build `Signals` from a BEqPlusScorer verdict.

Stdlib only, no config, no record shapes, no Lean. Fill in `Signals` from whatever a rollout
looks like in your pipeline and call `reward_for`. That is the whole interface.

THE TABLE. This is the specification; everything else in this file only implements it.

 wrote       type-correct        proof finished  FL statement matches  FL proof follows the
 something?  (Lean compiles it)  (no `sorry`)?   FL reference (BEq+)?  NL proof (faithfulness)?  reward
 ----------  ------------------  --------------  --------------------  ------------------------  -------------
 no          --                  --              --                    --                        0.00
 yes         no                  --              --                    --                        0.05
 yes         yes                 no              not verified          --                        0.15
 yes         yes                 YES             not verified          --                        0.30
 yes         yes                 no              VERIFIED              --                        0.50
 yes         yes                 YES             VERIFIED              f in [0,1]                0.85 + 0.15*f

 ROW 6 ONLY carries one further term, a DEDUCTION rather than a payment:

 brevity     how close the candidate's proof body is in LENGTH to the reference proof's,
             as b in (0,1] from `brevity_for`   ->  reward -= 0.10 * (1 - b)

 b == 1 (in tolerance, or unmeasurable) subtracts nothing, so the five rows above and a
 concise ROW 6 are all EXACTLY the numbers they were before brevity existed. Only ROW 6 is
 reachable, which is what stops the degenerate exploit: a two-token non-proof scores 0.15 on
 ROW 3 and never reaches the term.

A crash, a timeout or a missing verdict is also 0.00 AND stays in the denominator, so a
policy can never gain by failing to produce a result.

WHY THE STEPS ARE UNEVEN. Each is strictly more of the task done than the last, and the gaps
say how hard each property is to get right. Getting the statement verified when the proof is
already finished is worth 0.55 (0.30 -> 0.85); finishing the proof once the statement is
verified is worth 0.35 (0.50 -> 0.85); making the proof follow the given argument is worth
0.15. The statement stays the dominant term, which is correct -- a beautifully structured
proof of the wrong theorem is worth nothing extra.

TWO NAMES THAT HAVE CAUSED REAL CONFUSION, because the underlying grader's field names are
misleading and are not ours to rename:

    type_correct    `lean file.lean` exited 0. NOTE `sorry` is a WARNING in Lean, not an
                    error, so a file whose only proof is `sorry` IS type-correct. The grader
                    calls this `exit_ok`.
    proved          type_correct AND sorry-free AND declares a real theorem (not `True`, not
                    a `False` hypothesis, not empty) AND not cheating with a false axiom AND
                    Lean itself healthy. The grader calls this `compiles`, which reads like
                    "Lean accepted it" but is strictly stronger.

So the two are nested, never independent: proved implies type_correct. There is no case
where `compiles` holds and `exit_ok` does not.

TWO DIFFERENT COMPARISONS, AND THEY ARE EASY TO CONFLATE. NL = natural language (the English
the agent was handed), FL = formal language (Lean).

    BEq+           FL <-> FL.  The generated Lean STATEMENT against the gold Lean STATEMENT,
                   both `sorry`d and proofs ignored, asking whether each is provable from the
                   other. It never reads the English -- so it measures agreement with the
                   reference formalization, which is a PROXY for faithfulness and is only as
                   good as the reference. A failing forward direction means the answer is too
                   strong (over-generalised); backward, too weak (under-specified).

    faithfulness   NL -> FL.  The Lean PROOF against the English proof the agent was given,
                   asking whether it formalizes that argument or merely closes the goal some
                   other way. This is the only place the English enters the score.

So the statement is judged against Lean, and the proof against English. Both are needed: a
proof that follows the argument perfectly still proves whatever the statement says.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# The six outcomes.
#
# The STRINGS are a wire format: they are stored in every evolution node's fitness.json and
# are the keys of the fingerprinted reward table in harness_evolution/config.yaml. Changing a
# string makes every recorded number incomparable. The NAMES are free -- use these.
# Rewards shown are the defaults below.
# ---------------------------------------------------------------------------
NOTHING_WRITTEN    = "no_answer"            # -> 0.00  no file, or the same boilerplate that
                                            #          every problem got back
DOES_NOT_COMPILE   = "no_elaborate"         # -> 0.05  there is Lean, and Lean rejected it
UNFINISHED         = "incomplete"           # -> 0.15  Lean took it, but it is not an answer:
                                            #          a `sorry`, or a vacuous file
PROVED_NOT_MATCHED = "compiles"             # -> 0.30  a real, finished proof of a non-trivial
                                            #          theorem we could not match to the gold
MATCHED_NOT_PROVED = "incomplete_faithful"  # -> 0.50  statement verified, proof still missing
SOLVED             = "solved"               # -> 0.85 + 0.15*f   the deliverable

# Worst to best. Every consumer that reports a histogram should use this order.
OUTCOMES = (NOTHING_WRITTEN, DOES_NOT_COMPILE, UNFINISHED,
            PROVED_NOT_MATCHED, MATCHED_NOT_PROVED, SOLVED)

OUTCOME_LABEL = {
    NOTHING_WRITTEN:    "nothing written",
    DOES_NOT_COMPILE:   "written, does not compile",
    UNFINISHED:         "compiles, proof unfinished",
    PROVED_NOT_MATCHED: "proved, statement not matched to the reference",
    MATCHED_NOT_PROVED: "statement matches the reference, proof unfinished",
    SOLVED:             "solved",
}

# The defaults. harness_evolution reads its own table from config.yaml and passes it in
# explicitly, because that table is fingerprinted and must be able to change under a rescore
# without editing code. An RL caller that has no opinion should use these.
DEFAULT_REWARDS = {
    NOTHING_WRITTEN:    0.00,
    DOES_NOT_COMPILE:   0.05,
    UNFINISHED:         0.15,
    PROVED_NOT_MATCHED: 0.30,
    MATCHED_NOT_PROVED: 0.50,
    SOLVED:             1.00,   # the CEILING of the top band, not a flat value; see below
}

# Of the top reward, how much is paid for the proof following the informal argument rather
# than merely being a proof Lean accepted. Kept small on purpose: a large bonus rewards proofs
# that MIMIC the argument's shape -- a `have` chain restating each informal step, each closed
# by `grind` -- without doing its work, which is an erosion channel the statement-level
# monitors cannot see. f == 1 gives 0.85 + 0.15 == 1.00, i.e. exactly the pre-faithfulness
# score, so turning this on moves no existing number.
FAITHFULNESS_BAND = 0.15

# ---------------------------------------------------------------------------
# BREVITY. How close the candidate's proof is in LENGTH to the reference proof.
#
# A DEDUCTION FROM THE TOP ROW, NOT A PAYMENT. Every other signal here is paid
# for, so an unknown must never RAISE a score. This one is subtracted, so an
# unknown must never LOWER one: `brevity=None` means b == 1.0 means no change,
# and every already-recorded SOLVED number survives turning this on.
#
# IT APPLIES TO `SOLVED` ONLY, and that gate -- not the ratio -- is what stops
# the obvious exploit. A three-token proof that does not prove lands on
# UNFINISHED (0.15) and never reaches this term at all. Tying the target to the
# gold length is what stops the OTHER direction: a policy that has learned to
# solve cannot then farm the band by emitting something degenerately short,
# because short is scored against the gold's own length, not against zero.
#
# MEASURED, over the 18,550 LoCoLib golds carrying a real proof body: median 67
# chars, p75 170, p90 405, p99 1452, max ~7.3k. That range is why the ratio is
# SOFTENED. `python -m reward.reward` prints b per gold percentile.
#
# SYMMETRIC in the softened log ratio: exp(-lam * |ln((L+s)/(L*+s))|). There is
# no dead band, so the term carries gradient everywhere except at L == L*.
#
# WHAT THAT COSTS, measured over the certified population rather than assumed:
# 55% of proofs are shorter than their reference and 26% are under half its
# length, so most of the charge falls on the short side (mean 0.025 of the band
# against 0.013 long). A shorter proof of a matched statement is usually the
# better artifact, the reference being one author's draft rather than an
# optimum, so the term should be read as "distance from reference length", not
# as "bloat". Watch `pred_proof_chars` against `gold_proof_chars` directly
# rather than inferring the direction from the reward.
BREVITY_BAND = 0.10           # most this can ever cost a SOLVED rollout
BREVITY_SOFTEN = 60.0         # chars added to BOTH sides before the ratio
BREVITY_LAMBDA = 1.0          # decay per unit of |log ratio|; 1.0 == min(r, 1/r)


def brevity_for(pred_len: int | None, gold_len: int | None,
                *, soften: float = BREVITY_SOFTEN,
                lam: float = BREVITY_LAMBDA) -> float | None:
    """Length agreement with the reference proof, in (0,1]. 1.0 == no penalty.

        b = exp(-lam * |ln((pred + s) / (gold + s))|)

    Exponential decay in the SOFTENED log ratio, so tolerance is multiplicative
    and the term is smooth everywhere, with no interval of zero gradient. At
    lam == 1 this is exactly min(r, 1/r) for r the softened ratio.

    SOFTENED BY `s`, and that is what the constant is for. A raw pred/gold is
    degenerate at the short end: MEASURED over this corpus, the reference proof
    body is p10 3 / p25 16 / p50 40 chars, and 30% are under 20, so against a
    5-char `rfl` a perfectly good `by simp [foo]` reads as a fourfold overrun.
    Adding a constant to both sides makes the tolerance absolute where the gold
    is tiny and relative where it is large.

    SYMMETRIC in the log ratio: a proof at 2x the reference length and one at
    half of it are charged equally. Note this is a choice about what the term
    measures, not a claim that the two failure modes are equally bad. Over the
    certified population, 55% of proofs are SHORTER than their reference and
    26% are under half its length, so the short side carries most of the charge:
    mean 0.025 of the band against 0.013 on the long side.

    None means there is no reference length (the gold carries no proof body --
    1.1% of LoCoLib -- or the candidate could not be parsed), and MUST be
    treated as 1.0 by the caller.
    """
    if not gold_len or not pred_len:
        return None
    return math.exp(-lam * abs(math.log((pred_len + soften) / (gold_len + soften))))


@dataclass(frozen=True)
class Signals:
    """What is known about one rollout. Plain values; no Lean, no config, no record shapes.

    Tri-states are `True | False | None`, and None ALWAYS means "we could not find out".
    None is never good news -- see `outcome_for`.
    """
    graded: bool = False                             # a trustworthy verdict exists at all
    wrote_something: bool = False                    # a non-empty answer file
    problem_specific: bool = True                    # not the boilerplate every problem got
    type_correct: bool = False                       # Lean accepted it (`sorry` still counts)
    proved: bool = False                             # ...and sorry-free, real, and honest
    declares_theorem: bool = True                    # a theorem can actually be extracted
    statement_is_trivial: bool | None = None         # closed by the automation on its own
    statement_matches_reference: bool | None = None  # BEq+, FL vs FL: against the gold
                                                     #   Lean statement, not the English
    proof_follows_argument: float | None = None      # NL vs FL: does the Lean proof
                                                     #   formalize the English proof?
    brevity: float | None = None                     # FL vs FL: is the proof the same
                                                     #   LENGTH as the reference proof?
                                                     #   None == unmeasurable == no penalty


def outcome_for(s: Signals) -> str:
    """Which ROW of the table is this rollout? -> one of OUTCOMES.

    The branches below follow the table's COLUMNS, left to right: wrote something, then
    type-correct, then proof finished, then statement matches. Each leaf is one row.

    This returns the NAME only. `reward_for` is the entry point that returns the number, and
    faithfulness (f) deliberately does not appear here: it scales the top row alone and is
    applied in `reward_for_outcome`. The split exists because the name is what histograms and
    round-over-round deltas are counted in, and those need six buckets, not a continuum.
    """
    # COLUMN 1 -- wrote something?
    # No usable result, an empty file, or one answer shared across many problems.
    if not s.graded or not s.wrote_something or s.problem_specific is False:
        return NOTHING_WRITTEN                  # ROW 1  ->  0.00

    # COLUMN 2 -- type-correct? There is Lean, and Lean rejected it.
    if not s.type_correct:
        return DOES_NOT_COMPILE                 # ROW 2  ->  0.05

    # COLUMN 4 -- does the FL statement match the FL reference (BEq+)?
    # `is True`, not truthiness: an unknown is NOT a match, and must never score as one.
    matches = s.statement_matches_reference is True

    # COLUMN 3 -- proof finished?
    if not s.proved:
        if matches:
            return MATCHED_NOT_PROVED           # ROW 5  ->  0.50
        return UNFINISHED                       # ROW 3  ->  0.15

    if matches:
        # The NAME of row 6. Its number is not a constant -- `reward_for_outcome` computes
        # 0.85 + 0.15*f for it, and this is the only row f applies to.
        return SOLVED                           # ROW 6  ->  0.85 + 0.15*f

    # ROW 4, but it has TWO admission tests, because this is the row a policy would farm:
    #   declares_theorem      a file no statement can be extracted from has formalized
    #                         nothing (`instance : Inhabited Nat := <0>` compiles clean).
    #   statement_is_trivial  nor has one the solver tactics close unaided
    #                         (`theorem t (n : Nat) : n = n := rfl` compiles, is not vacuous).
    # `is not True` keeps an unmeasurable statement admissible, so a flaky checker cannot
    # demote honest work; a high unknown rate should trip a validity gate instead.
    if s.declares_theorem and s.statement_is_trivial is not True:
        return PROVED_NOT_MATCHED               # ROW 4  ->  0.30

    # Failed an admission test: a finished proof, but of nothing. Demoted to row 3 -- this is
    # the one path the table above does not show as its own line.
    return UNFINISHED


def reward_for_outcome(outcome: str, rewards: dict | None = None,
                       faithfulness: float | None = None,
                       *, band: float = FAITHFULNESS_BAND,
                       brevity: float | None = None,
                       brevity_band: float = BREVITY_BAND) -> float:
    """Outcome -> scalar. Only the top outcome reads `faithfulness`.

    An unmeasured f scores as 0.0, the same convention every other unknown here follows: an
    unknown must never RAISE a score. That is exactly why the placeholder in
    `proof_follows_argument` returns 1.0 rather than None -- "not implemented yet" must not
    silently dock every solved rollout by the whole band.
    """
    rewards = rewards or DEFAULT_REWARDS
    value = rewards[outcome]

    # ROWS 1-5 are constants, straight from the table: 0.00 / 0.05 / 0.15 / 0.30 / 0.50.
    # Brevity does not reach them either, and that gate is the whole anti-exploit
    # story: a degenerately short answer that does not prove never gets here.
    if outcome != SOLVED:
        return value
    if not band:
        return _apply_brevity(value, brevity, brevity_band)

    # ROW 6 is the only one that is not a constant. `value` is the CEILING of the band, so
    # with the defaults (value = 1.00, band = 0.15) the line below IS the table's entry:
    #
    #       (1.00 - 0.15) + 0.15 * f   ==   0.85 + 0.15 * f
    #
    #       f == 0    -> 0.85   a proof Lean accepted that ignores the given argument
    #       f == 1    -> 1.00   and this is exactly the pre-faithfulness score, which is why
    #                           turning faithfulness on moves no already-recorded number
    #       f is None -> 0.85   unmeasured scores as 0, like every other unknown here
    f = 0.0 if faithfulness is None else min(1.0, max(0.0, float(faithfulness)))
    return _apply_brevity((value - band) + band * f, brevity, brevity_band)


def _apply_brevity(top: float, brevity: float | None, brevity_band: float) -> float:
    """Subtract the length penalty from an already-computed ROW 6 reward.

    Written as `top - band*(1-b)` rather than as a second band carved off the
    ceiling so that b == 1 -- in tolerance, or unmeasurable -- returns `top`
    EXACTLY. That keeps every already-recorded SOLVED number intact and leaves
    the table's documented step sizes (0.50 -> 0.85, and the 0.35 that finishing
    a proof is worth) unchanged for a concise proof.
    """
    b = 1.0 if brevity is None else min(1.0, max(0.0, float(brevity)))
    return top - brevity_band * (1.0 - b)


def reward_for(s: Signals, rewards: dict | None = None,
               *, band: float = FAITHFULNESS_BAND,
               brevity_band: float = BREVITY_BAND) -> float:
    """THE ENTRY POINT. Signals -> the scalar reward for one rollout.

    Two steps: `outcome_for` picks the row, `reward_for_outcome` turns it into a number and
    is where faithfulness enters. Call this unless you specifically need the row name too --
    in which case call both, which is what the harness-evolution side does so it can report
    a histogram alongside the mean.
    """
    return reward_for_outcome(outcome_for(s), rewards, s.proof_follows_argument,
                              band=band, brevity=s.brevity, brevity_band=brevity_band)


def proof_follows_argument(informal_proof: str | None, lean_proof: str | None) -> float | None:
    """PLACEHOLDER. Does the Lean proof formalize the informal proof it was handed?

    Returns f in [0,1]: 0.0 for a proof that ignores the argument (a `grind`, or an `exact?`
    that just retrieves the Mathlib lemma), 1.0 for one that follows it.

    RETURNS 1.0 UNCONDITIONALLY FOR NOW, which makes the top outcome score exactly
    `rewards[SOLVED]` and leaves every already-recorded number unchanged. Swapping in a real
    metric therefore CHANGES PAST NUMBERS, so for harness evolution it must go through
    `evolve.py rescore --into <dir>` rather than being edited in place.

    When the real one lands, three things come with it and none are optional:

      * an UNKNOWN return of None, so a judge that fails does not silently award the band;
      * a validity gate on the unknown rate, so a broken judge invalidates the batch instead
        of quietly capping everything at the band's floor;
      * a NEGATIVE CONTROL -- score a proof against a DIFFERENT problem's informal proof and
        confirm the metric says no. Unlike BEq+ this will be judged rather than derived, and
        is therefore gameable by exactly the policy being trained.
    """
    return 1.0


# ---------------------------------------------------------------------------
# This repo's adaptation. Everything above is the shared definition; everything
# below maps a BEqPlusScorer verdict onto it.
#
# LoCoLib proof-pair task (`typecheck`/`outcome` arms, `data_locolib/*_proof.parquet`):
# the model emits a full theorem AND its own proof, so every row is reachable.
# `proof_check` is a `BEqPlusScorer.check_own_proof()` result -- `proved` and
# `statement_is_trivial` (via BEq+'s own `provable_alone`) are real signals,
# not placeholders. `proof_follows_argument` has no metric yet (see the
# PLACEHOLDER note above) and stays None, which scores as 0 -- it never lifts
# a score, only would cap it lower once implemented.
#
# For the older signature-only tasks (Lean-Workbook, LoCoLib statement-only),
# call this with `proof_check=None`: the ladder tops out at MATCHED_NOT_PROVED
# (0.50) because there is no proof body to check.
# ---------------------------------------------------------------------------


def signals_from_score(r: dict, proof_check: dict | None = None,
                       brevity: float | None = None,
                       *, typecheck_from_own_proof: bool = True,
                       wrote_something: bool = True) -> Signals:
    """Build `Signals` from a `BEqPlusScorer.score()` result.

    `proof_check`, when given, is a `BEqPlusScorer.check_own_proof()` result
    for the SAME rollout -- pass it whenever the candidate was asked for a
    full proof, not just a signature. Omitted (or None), `proved` stays False
    and `statement_is_trivial` stays unknown, matching the old signature-only
    behaviour exactly.

    WHICH ELABORATION FEEDS COLUMN 2. `r["typecheck"]` is BEq+'s own check, and
    it SORRIES THE PROOF AWAY before elaborating -- it answers "does the
    statement elaborate", not the table's "`lean file.lean` exited 0". On the
    proof-pair task those differ, and the difference was a live mis-calibration:
    a submission whose proof body Lean rejects outright
    (`:= by exact?_no_such_tactic`) still passed column 2, and with a matching
    statement scored `incomplete_faithful` (0.50) -- the same as an honest
    `sorry`, and on the `gated` arm a full 1.0. Measured over 120 golds it hit
    100% of that perturbation.

    So when a `proof_check` exists it wins: it elaborates the submission AS
    WRITTEN. `typecheck_from_own_proof=False` restores the old wiring for
    comparison against numbers recorded before this was fixed -- it is a real
    change to what a run scores, not a no-op refactor.
    """
    failed = bool(r.get("error_kind"))
    proved = bool(proof_check and proof_check.get("proved"))
    if proof_check and typecheck_from_own_proof:
        type_correct = bool(proof_check.get("type_correct"))
    else:
        type_correct = bool(r.get("typecheck"))
    return Signals(
        # A Lean or infrastructure failure is not a verdict about the model, so
        # it is ungraded rather than a zero earned by the rollout.
        graded=not failed,
        # Was hardcoded True, which made ROW 1 unreachable: an empty completion
        # scored `no_elaborate` (0.05) rather than `no_answer` (0.00), so "wrote
        # nothing" and "wrote broken Lean" were the same reward. Small in
        # magnitude and it creates no within-group variance on its own, but the
        # table's floor is 0.00 and it should be reachable.
        wrote_something=wrote_something,
        type_correct=type_correct,
        proved=proved,
        # BEq+'s cascade already computes this as a side effect (rung 3's
        # `provable_without_have`): whether the SECOND theorem of the pair --
        # here, the prediction -- proves on its own, with no reference to the
        # gold. That is exactly what "closed by the automation on its own"
        # means. Only meaningful once there is a proof to judge triviality
        # of; left unknown otherwise.
        statement_is_trivial=(r.get("provable_alone") if proof_check else None),
        statement_matches_reference=bool(r.get("beq_plus")) or None,
        proof_follows_argument=None,
        # Only ever set by a caller that measured BOTH proof bodies; see
        # `brevity_for`. It reaches the score on the SOLVED row alone, so it is
        # inert for `gated` and `typecheck`, which never reach that row.
        brevity=brevity,
    )


def signals_from_typecheck(ok: bool, error_kind: str | None = None) -> Signals:
    """Build `Signals` when only elaboration was checked.

    The type-check arm never runs the BEq+ cascade: one elaboration is ~0.36s
    against ~175 Lean-seconds for a full cascade, and routing it through
    `score()` to reach the same verdict would make the cheap arm 500x dearer.
    `statement_matches_reference` stays None, i.e. unknown, which the table
    already treats as "not a match".
    """
    return Signals(
        graded=not error_kind,
        wrote_something=True,
        type_correct=bool(ok),
        proved=False,
        statement_matches_reference=None,
        proof_follows_argument=None,
    )


# The arms, as reward tables over the same six outcomes. Each is the `rewards`
# argument `reward_for_outcome` already takes, so the arms differ only in what
# they pay, never in how an outcome is decided.
TYPECHECK_REWARDS = {
    NOTHING_WRITTEN: 0.0, DOES_NOT_COMPILE: 0.0,
    UNFINISHED: 1.0, PROVED_NOT_MATCHED: 1.0,
    MATCHED_NOT_PROVED: 1.0, SOLVED: 1.0,
}

# Pays only for semantic equivalence; everything below it shares one flat floor.
# Under GRPO the advantage is the reward minus the group mean, so a group with no
# semantic signal contributes no gradient at all. Any term that varies within
# such a group becomes the only climbable signal there.
GATED_REWARDS = {
    NOTHING_WRITTEN: 0.0, DOES_NOT_COMPILE: 0.0,
    UNFINISHED: 0.0, PROVED_NOT_MATCHED: 0.0,
    MATCHED_NOT_PROVED: 1.0, SOLVED: 1.0,
}

# BEq+ proves in one direction but not both. The table has no row for it: its
# `statement_matches_reference` is a boolean, so a partial match is simply not a
# match. The gated arm pays a step for it, applied as an override on the
# UNFINISHED row rather than by inventing a seventh outcome.
GATED_ONE_DIRECTION = 0.25


def gated_reward(r: dict, one_direction: float = GATED_ONE_DIRECTION) -> float:
    """The gated arm: 1.0 for BEq+, `one_direction` for a single direction, else 0."""
    outcome = outcome_for(signals_from_score(r))
    if outcome == UNFINISHED and int(r.get("semantic_signal", 0) or 0) >= 1:
        return one_direction
    return reward_for_outcome(outcome, GATED_REWARDS)


def typecheck_reward(ok: bool, error_kind: str | None = None) -> float:
    """The type-check arm: 1.0 if the statement elaborates, else 0."""
    return reward_for_outcome(outcome_for(signals_from_typecheck(ok, error_kind)),
                              TYPECHECK_REWARDS)


if __name__ == "__main__":
    # Stdlib-only invariant check, matching this file's contract: no Lean, no
    # config, no record shapes. `python -m reward.reward`.
    def _s(**kw) -> Signals:
        base = dict(graded=True, wrote_something=True, type_correct=True)
        return Signals(**{**base, **kw})

    ladder = [
        ("no_answer",            _s(graded=False),                                    0.00),
        ("no_elaborate",         _s(type_correct=False),                              0.05),
        ("incomplete",           _s(),                                                0.15),
        ("compiles",             _s(proved=True),                                     0.30),
        ("incomplete_faithful",  _s(statement_matches_reference=True),                0.50),
        ("solved",               _s(proved=True, statement_matches_reference=True),   0.85),
    ]
    print("THE LADDER, with brevity unmeasured -- must equal the pre-brevity numbers")
    for name, sig, expect in ladder:
        got = reward_for(sig)
        assert abs(got - expect) < 1e-9, f"{name}: {got} != {expect}"
        assert outcome_for(sig) == name, f"{name}: {outcome_for(sig)}"
        print(f"  {name:22s} {got:.2f}  ok")

    print("\nBREVITY REACHES ROW 6 ONLY (b=0.0 is the worst possible length)")
    for name, sig, expect in ladder:
        worst = reward_for(Signals(**{**sig.__dict__, "brevity": 0.0}))
        delta = worst - expect
        assert abs(delta) < 1e-9 or name == "solved", \
            f"{name} moved by {delta}, but only `solved` may"
        print(f"  {name:22s} {expect:.2f} -> {worst:.2f}   delta {delta:+.2f}")
    assert abs(reward_for(Signals(**{**ladder[-1][1].__dict__, "brevity": 0.0}))
               - (0.85 - BREVITY_BAND)) < 1e-9

    print("\nb == 1 AND b is None BOTH leave every number untouched")
    for b in (1.0, None):
        for name, sig, expect in ladder:
            got = reward_for(Signals(**{**sig.__dict__, "brevity": b}))
            assert abs(got - expect) < 1e-9, f"b={b} {name}: {got} != {expect}"
    print("  ok")

    print("\nbrevity_for, against the measured LoCoLib gold percentiles")
    print(f"  {'gold':>6}  b(1.5x)  b(2x)   b(4x)  b(gold/2)  b(gold/4)")
    for lg in (5, 28, 67, 170, 405, 1452):
        print(f"  {lg:>6}   {brevity_for(round(1.5 * lg), lg):.3f}   "
              f"{brevity_for(2 * lg, lg):.3f}   {brevity_for(4 * lg, lg):.3f}   "
              f"{brevity_for(max(1, lg // 2), lg):.3f}      "
              f"{brevity_for(max(1, lg // 4), lg):.3f}")
    # Symmetry, stated as the identity it actually is: |ln(u/v)| == |ln(v/u)|,
    # so swapping candidate and reference cannot change the value.
    for a, b in ((10, 67), (67, 10), (5, 1452), (170, 170), (1, 7300)):
        assert abs(brevity_for(a, b) - brevity_for(b, a)) < 1e-12, (a, b)
    assert brevity_for(67, 67) == 1.0
    assert brevity_for(0, 67) is None and brevity_for(67, 0) is None
    assert 0.0 < brevity_for(7300, 1) < 1e-2
    print("  symmetric, bounded in (0,1], and unmeasurable stays None: ok")

    print("\nall invariants hold")
