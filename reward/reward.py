#!/usr/bin/env python3
"""How one autoformalization rollout is scored. The shared definition.

Used in two places and deliberately owned by neither:

    harness evolution   the mean of this over a frozen task set IS a harness's fitness
    RL fine-tuning      this is the per-rollout reward

Adopted from the harness-evolution project. The table below is unchanged; what
this repo adds is `signals_from_score`, which builds `Signals` from a
BEqPlusScorer verdict.

WHERE THE REST LIVES. This file owns the six outcomes and the ladder over them,
and nothing else. To change what an arm is made of, edit the table in
`reward/arms.py`, not this file; its header maps the rest of the package.

Stdlib only, no config, no record shapes, no Lean. Fill in `Signals` from whatever a rollout
looks like in your pipeline and call `reward_for`. That is the whole interface.

THE TABLE. This is the specification; everything else in this file only implements it.

                                                 FL statement       FL proof follows  FL proof LENGTH
 wrote       type-correct        proof finished  matches the FL     the NL proof      near the reference
 something?  (Lean compiles it)  (no `sorry`)?   reference (BEq+)?  (faithfulness)?   (brevity)?          reward
 ----------  ------------------  --------------  -----------------  ----------------  ------------------  --------------------------
 no          --                  --              --                 --                --                  0.00
 yes         no                  --              --                 --                --                  0.05
 yes         yes                 no              not verified       --                --                  0.15
 yes         yes                 YES             not verified       --                --                  0.30
 yes         yes                 no              VERIFIED           --                --                  0.50
 yes         yes                 YES             VERIFIED           f in [0,1]        b in (0,1]          0.85 + 0.15*f - 0.10*(1-b)

 f AND b ARE THE ONLY NON-CONSTANT ENTRIES, and both live on ROW 6 alone. An unmeasured f
 scores 0, so the faithfulness band goes unpaid; an unmeasured or in-tolerance b is 1, so
 the length term deducts nothing. ROW 6 with neither measured is therefore a flat 0.85, and
 every row above it is EXACTLY the number it was before either term existed.

 THAT ROW 6 GATE, not the shape of either term, is what stops the degenerate exploit: a
 two-token non-proof lands on ROW 3 (0.15) and never reaches them. `reward/terms/` holds
 both terms; this file only decides which row a rollout is on.

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

from dataclasses import dataclass

from reward.terms.brevity import BREVITY_BAND, apply_brevity, brevity_for
from reward.terms.faithfulness import FAITHFULNESS_BAND, apply_faithfulness

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
    """Outcome -> scalar. Only the top outcome reads either term.

    ROWS 1-5 ARE CONSTANTS, straight from the table: 0.00 / 0.05 / 0.15 / 0.30 /
    0.50. Neither term reaches them, and that gate -- not the shape of either
    term -- is the whole anti-exploit story: a degenerately short answer that
    does not prove lands on ROW 3 and never gets here.

    ROW 6 is the only one that is not a constant. `rewards[SOLVED]` is the
    CEILING of the faithfulness band, so with the defaults (1.00, band 0.15) the
    row pays `0.85 + 0.15 * f`, less up to `brevity_band * (1 - b)`. Both terms
    are no-ops when unmeasured, so an arm that measures neither scores a flat
    0.85 here and every already-recorded number survives.
    """
    rewards = rewards or DEFAULT_REWARDS
    value = rewards[outcome]
    if outcome != SOLVED:
        return value
    return apply_brevity(apply_faithfulness(value, faithfulness, band),
                         brevity, brevity_band)


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
