#!/usr/bin/env python3
"""FAITHFULNESS: does the Lean PROOF carry out the English argument it was given?

NL -> FL, and the only place the English enters the score. Distinct from BEq+,
which is FL <-> FL over STATEMENTS and never reads the English at all.

THE BAND IS CARVED OUT WHETHER OR NOT AN ARM MEASURES f. An unmeasured f scores
0.0, so the top row pays `ceiling - band` either way. Measuring f does not move
the floor; it makes the band reachable. That is why an arm measuring no
faithfulness still tops out at 0.85 rather than 1.00.

Stdlib only, and there is no metric here yet -- see `proof_follows_argument`.
"""
from __future__ import annotations

# Of the top reward, how much is paid for the proof following the informal argument rather
# than merely being a proof Lean accepted. Kept small on purpose: a large bonus rewards proofs
# that MIMIC the argument's shape -- a `have` chain restating each informal step, each closed
# by `grind` -- without doing its work, which is an erosion channel the statement-level
# monitors cannot see. f == 1 gives 0.85 + 0.15 == 1.00, i.e. exactly the pre-faithfulness
# score, so turning this on moves no existing number.
FAITHFULNESS_BAND = 0.15


def apply_faithfulness(ceiling: float, f: float | None,
                       band: float = FAITHFULNESS_BAND) -> float:
    """ROW 6's ceiling, less the unearned part of the band.

    `(ceiling - band) + band * f`, which with the defaults (1.00, 0.15) is the
    table's `0.85 + 0.15 * f`:

        f == 0     -> 0.85   a proof Lean accepted that ignores the argument
        f == 1     -> 1.00   exactly the pre-faithfulness score, which is why
                             turning f on moves no already-recorded number
        f is None  -> 0.85   unmeasured scores as 0, like every other unknown

    A band of 0 returns the ceiling untouched.
    """
    if not band:
        return ceiling
    f = 0.0 if f is None else min(1.0, max(0.0, float(f)))
    return (ceiling - band) + band * f


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
