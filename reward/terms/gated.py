#!/usr/bin/env python3
"""GATED: the reward table that pays for BEq+ equivalence and nothing below it.

An alternative `rewards` table over the six outcomes, plus one override for the
half-match BEq+ can report. Arms differ only in what they PAY, never in how an
outcome is decided, which is why this is a table and not a second ladder.
"""
from __future__ import annotations

from reward.reward import (DOES_NOT_COMPILE, MATCHED_NOT_PROVED, NOTHING_WRITTEN,
                           PROVED_NOT_MATCHED, SOLVED, UNFINISHED, outcome_for,
                           reward_for_outcome, signals_from_score)

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


def gated_reward(r: dict, one_direction: float = GATED_ONE_DIRECTION,
                 rewards: dict | None = None) -> float:
    """The gated arm: 1.0 for BEq+, `one_direction` for a single direction, else 0."""
    outcome = outcome_for(signals_from_score(r))
    if outcome == UNFINISHED and int(r.get("semantic_signal", 0) or 0) >= 1:
        return one_direction
    return reward_for_outcome(outcome, rewards or GATED_REWARDS)
