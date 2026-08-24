#!/usr/bin/env python3
"""A Lean type-check tool the policy can call mid-rollout, so it sees the
compiler's error and gets to revise before the reward is assigned.

Ported in spirit from project-numina/kimina-prover-rl, whose reward returns
`(rewards, tool_feedbacks)` and feeds `create_tool_message(formal_code,
lean_feedback)` back into the conversation.

Why it should matter MORE here than it does for them: 23.4% of our rollouts
fail to elaborate at all, and a rollout that does not type-check can never earn
BEq+ (verified: BEq+ implies type-check, 0 exceptions in 9,528 rollouts). Those
are dead weight in their group -- they push prompts toward the "starved" bucket
that produces no gradient. Converting even a fraction of them into live rollouts
widens the informative-group share, which is the measured bottleneck.

What it will NOT fix: semantically-wrong-but-well-formed statements. On the
starved prompts we sampled at k=32, 84.3% type-checked and were still wrong.
The compiler cannot see that; only BEq+ can.

Registered with verl via
  actor_rollout_ref.rollout.multi_turn.function_tool_path=reward/lean_tool.py
"""
from __future__ import annotations

import os

_MAX_FEEDBACK_CHARS = int(os.environ.get("LEAN_TOOL_MAX_FEEDBACK", "600"))

try:
    from verl.tools.function_tool import function_tool
except Exception:  # importable standalone for testing
    def function_tool(fn=None, **_kw):
        return fn if callable(fn) else (lambda f: f)


@function_tool
def check_lean_statement(statement: str) -> str:
    """Type-check a Lean 4 theorem statement and report any elaboration errors.

    Args:
        statement: A single Lean 4 theorem declaration, signature only, ending in
            `:= by sorry`. Example:
            `theorem foo (x : ℝ) (hx : 0 < x) : Real.log x < x := by sorry`

    Returns:
        "OK" if the statement elaborates, otherwise the Lean error message.
    """
    # Imported lazily: this module is loaded by verl at config time, long before
    # a Lean REPL should be spawned, and building a scorer costs ~4.3GB.
    from reward.reward_fn import _clean_solution, _get_scorer

    stmt = _clean_solution(statement)
    if not stmt.strip():
        return "ERROR: empty statement."
    try:
        scorer = _get_scorer()
    except Exception as e:  # a broken Lean env must not kill the rollout
        return f"ERROR: Lean unavailable ({type(e).__name__}). Proceed without checking."

    ctx = os.environ.get("LEAN_TOOL_CONTEXT", "import Mathlib\nset_option maxRecDepth 10000")
    try:
        # typecheck_message, not typecheck_ex: the latter returns an error
        # CATEGORY ("infra"/"timeout"/"other_error"), which tells the policy
        # nothing it can act on. This returns Lean's actual diagnostics.
        ok, detail = scorer.typecheck_message(stmt, ctx)
    except Exception as e:
        return f"ERROR: check failed ({type(e).__name__}). Proceed without checking."
    if ok:
        return "OK — the statement elaborates."
    return (f"FAILED to elaborate:\n{detail[:_MAX_FEEDBACK_CHARS]}\n"
            f"Revise the statement and try again.")
