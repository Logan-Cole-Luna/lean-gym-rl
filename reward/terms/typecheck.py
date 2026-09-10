#!/usr/bin/env python3
"""TYPECHECK: the reward table that pays for elaboration alone.

KNOWN EXPLOITABLE and kept anyway as the ablation baseline: a trivially true
theorem satisfies it, so it measures loop health rather than autoformalization.
"""
from __future__ import annotations

from reward.reward import (DOES_NOT_COMPILE, MATCHED_NOT_PROVED, NOTHING_WRITTEN,
                           PROVED_NOT_MATCHED, SOLVED, UNFINISHED, outcome_for,
                           reward_for_outcome, signals_from_typecheck)

# The arms, as reward tables over the same six outcomes. Each is the `rewards`
# argument `reward_for_outcome` already takes, so the arms differ only in what
# they pay, never in how an outcome is decided.
TYPECHECK_REWARDS = {
    NOTHING_WRITTEN: 0.0, DOES_NOT_COMPILE: 0.0,
    UNFINISHED: 1.0, PROVED_NOT_MATCHED: 1.0,
    MATCHED_NOT_PROVED: 1.0, SOLVED: 1.0,
}


def typecheck_reward(ok: bool, error_kind: str | None = None,
                     rewards: dict | None = None) -> float:
    """The type-check arm: 1.0 if the statement elaborates, else 0."""
    return reward_for_outcome(outcome_for(signals_from_typecheck(ok, error_kind)),
                              rewards or TYPECHECK_REWARDS)
