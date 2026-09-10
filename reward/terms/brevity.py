#!/usr/bin/env python3
"""BREVITY: how close the candidate's proof is in LENGTH to the reference proof.

A DEDUCTION FROM THE TOP ROW, NOT A PAYMENT, and it reaches no other row. See
`reward/reward.py` for the table it is applied on top of and `reward/arms.py`
for which arms turn it on.

Stdlib only. Measuring the two proof bodies is `reward/evidence.py`'s job; this
module only turns two lengths into a number.
"""
from __future__ import annotations

import math

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


def apply_brevity(top: float, brevity: float | None, brevity_band: float) -> float:
    """Subtract the length penalty from an already-computed ROW 6 reward.

    Written as `top - band*(1-b)` rather than as a second band carved off the
    ceiling so that b == 1 -- in tolerance, or unmeasurable -- returns `top`
    EXACTLY. That keeps every already-recorded SOLVED number intact and leaves
    the table's documented step sizes (0.50 -> 0.85, and the 0.35 that finishing
    a proof is worth) unchanged for a concise proof.
    """
    b = 1.0 if brevity is None else min(1.0, max(0.0, float(brevity)))
    return top - brevity_band * (1.0 - b)
