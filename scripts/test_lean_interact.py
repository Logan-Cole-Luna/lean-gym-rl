#!/usr/bin/env python3
"""Smoke test: does lean-interact typecheck against this project's own
Mathlib4 checkout (repos/mathlib4), reusing the existing REPL cache?"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lean_interact import AutoLeanServer, Command, LeanREPLConfig
from lean_interact.interface import LeanError
from lean_interact.project import LocalProject

from reward.beq_plus import LEAN_INTERACT_CACHE_DIR as CACHE_DIR
from reward.beq_plus import MATHLIB_ROOT

BASE_IMPORT = "import Mathlib\nset_option maxRecDepth 10000"


def main():
    config = LeanREPLConfig(
        project=LocalProject(directory=MATHLIB_ROOT, auto_build=False),
        cache_dir=CACHE_DIR,
        verbose=True,
    )
    server = AutoLeanServer(config=config, max_total_memory=1.5, max_process_memory=None)

    t0 = time.time()
    print("importing Mathlib base env ...", flush=True)
    base_out = server.run(Command(cmd=BASE_IMPORT), timeout=600, add_to_session_cache=True)
    base_env = getattr(base_out, "env", None)
    print(f"base env = {base_env} (took {time.time() - t0:.1f}s)", flush=True)
    if base_env is None:
        raise RuntimeError(f"failed to import Mathlib base env: {base_out}")

    good_stmt = "theorem test_add_zero (n : ℕ) : n + 0 = n := by rfl"
    print("--- typechecking a known-good statement ---")
    out = server.run(Command(cmd=good_stmt, env=base_env), timeout=60)
    if isinstance(out, LeanError):
        print("LEAN ERROR:", out)
    else:
        print("valid:", out.lean_code_is_valid(allow_sorry=False))

    bad_stmt = "theorem test_bad (n : ℕ) : n + 0 = n + 1 := by rfl"
    print("--- typechecking a known-bad statement (should fail) ---")
    out2 = server.run(Command(cmd=bad_stmt, env=base_env), timeout=60)
    if isinstance(out2, LeanError):
        print("LEAN ERROR (expected):", out2)
    else:
        print("valid (should be False):", out2.lean_code_is_valid(allow_sorry=False))


if __name__ == "__main__":
    main()
