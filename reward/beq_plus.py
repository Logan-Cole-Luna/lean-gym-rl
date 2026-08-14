#!/usr/bin/env python3
"""Online, persistent-server wrapper around the BEq+ symbolic equivalence metric
(Poiroux et al., EMNLP 2025), for use as a per-example RL reward function.

`check_theorem_equivalence` below is vendored verbatim from
`augustepoiroux/RLMEval` (`src/rlm_eval/metrics/beq_plus.py`), reusing the same
copy already vendored (with the same attribution) in
`MixtureOfMathExperts/scripts/beq/run_beq_plus.py`. That script drives it as a
batch/shard job (spins up one `AutoLeanServer` per shard run); this module wraps
the same algorithm behind a class that holds the Lean REPL server open across
many `score()` calls, which is what an online RL reward needs.

The Lean environment points at a fresh Mathlib v4.8.0-rc1 checkout (built via
`lake exe cache get`) under this project's own `repos/mathlib4`, pinned to match
the `internlm/Lean-Workbook` dataset's target toolchain -- see ai4math_training's
plan notes.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from lean_interact import AutoLeanServer, Command, LeanREPLConfig
from lean_interact.interface import (
    CommandResponse,
    LeanError,
    Pos,
    message_intersects_code,
)
from lean_interact.project import LocalProject
from lean_interact.utils import (
    clean_last_theorem_string,
    extract_last_theorem,
    indent_code,
    split_conclusion,
)

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
MATHLIB_ROOT = os.environ.get("MATHLIB_ROOT", str(PROJECT_ROOT / "repos" / "mathlib4"))
LEAN_INTERACT_CACHE_DIR = os.environ.get("LEAN_INTERACT_CACHE_DIR", str(Path.home() / ".cache" / "lean_interact"))
BASE_IMPORT = "import Mathlib\nset_option maxRecDepth 10000"


# ─────────────────────────────────────────────────────────────────────────────
# VENDORED VERBATIM from augustepoiroux/RLMEval src/rlm_eval/metrics/beq_plus.py
# (only reformatting of comments; the algorithm/tactics are unchanged). This is
# the published BEq+ metric. `check_theorem_equivalence` returns BOTH BEqL and
# BEq+ per direction in a single pass over the two directions.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BEqCPUResult:
    beql_unidirections: tuple[str | None, str | None] = (None, None)
    beq_plus_unidirections: tuple[str | None, str | None] = (None, None)

    def beql(self) -> bool:
        return all(proof is not None for proof in self.beql_unidirections)

    def beq_plus(self) -> bool:
        return all(proof is not None for proof in self.beq_plus_unidirections)


def update_tuple(t: tuple, idx: int, value) -> tuple:
    t_list = list(t)
    t_list[idx] = value
    return tuple(t_list)


def extract_exact_proof(lean_output: CommandResponse, proof_start_line: int | None = None) -> str | None:
    start = Pos(line=proof_start_line, column=0) if proof_start_line else None
    for message in lean_output.messages:
        if message_intersects_code(message, start, None):
            if message.severity == "error":
                return None
            if message.severity == "info" and message.data.startswith("Try this:"):
                return message.data.split("Try this:")[1].strip()
    return None


def check_theorem_equivalence(
    theorem1: str, theorem2: str, lean_server: AutoLeanServer, context_env: int, timeout_per_proof: int
) -> BEqCPUResult:
    base_thm_name = "base_theorem"
    reformulated_thm_name = "reformulated_theorem"

    def prove_all(tactics: list[str]) -> str:
        prove_independent = " ; ".join([f"(all_goals try {t})" for t in tactics])
        prove_combined = "all_goals (" + " ; ".join([f"(try {t})" for t in tactics]) + ")"
        return "all_goals intros\nfirst | (" + prove_independent + ") | (" + prove_combined + ")"

    solver_tactics_apply = ["tauto", "simp_all_arith!", "noncomm_ring", "exact?"]
    solver_tactics_have = ["tauto", "simp_all_arith!", "exact? using this"]
    proof_all_apply = prove_all(solver_tactics_apply)
    proof_all_have = prove_all(solver_tactics_have)

    res = BEqCPUResult()
    for i, (base_thm, reform_thm) in enumerate([(theorem1, theorem2), (theorem2, theorem1)]):
        try:
            formal_1_code = clean_last_theorem_string(base_thm, base_thm_name, add_sorry=True) + "\n\n"
            formal_2_start_line = formal_1_code.count("\n") + 1
            formal_2_code = f"{clean_last_theorem_string(reform_thm, reformulated_thm_name, add_sorry=False)} := by"
        except ValueError:
            break

        def check_proof_sub(proof: str, formal_code: str = formal_1_code + formal_2_code) -> str | None:
            prepended_proof = "\nintros\nsymm_saturate\n"
            try:
                lean_output = lean_server.run(
                    Command(cmd=formal_code + indent_code(prepended_proof + proof, 2), env=context_env),
                    timeout=timeout_per_proof,
                )
                if isinstance(lean_output, LeanError):
                    return None
                if proof == "sorry":
                    if lean_output.lean_code_is_valid(start_pos=Pos(line=formal_2_start_line, column=0)):
                        return proof
                    return None
                if lean_output.lean_code_is_valid(start_pos=Pos(line=formal_2_start_line, column=0), allow_sorry=False):
                    if proof == "exact?":
                        return extract_exact_proof(lean_output, proof_start_line=formal_2_start_line)
                    return proof
            except (TimeoutError, ConnectionAbortedError, json.JSONDecodeError):
                pass
            return None

        if check_proof_sub("sorry") is None:
            break

        # 1. `exact?`, and check it uses the base theorem in the proof.
        proof_exact = check_proof_sub("exact?")
        if proof_exact and base_thm_name in proof_exact:
            res.beql_unidirections = update_tuple(res.beql_unidirections, i, proof_exact)
            res.beq_plus_unidirections = update_tuple(res.beq_plus_unidirections, i, proof_exact)
            continue

        # 2. try to apply the base theorem directly
        proof_apply = check_proof_sub(f"apply {base_thm_name}\n" + proof_all_apply)
        if proof_apply:
            res.beq_plus_unidirections = update_tuple(res.beq_plus_unidirections, i, proof_apply)
            continue

        # 3. add the conclusion of the base theorem as a hypothesis
        provable_without_have = False
        try:
            res_without_have = lean_server.run(
                Command(cmd=formal_2_code + proof_all_have, env=context_env), timeout=timeout_per_proof
            )
            if isinstance(res_without_have, CommandResponse):
                provable_without_have = res_without_have.lean_code_is_valid(allow_sorry=False)
        except (TimeoutError, ConnectionAbortedError, json.JSONDecodeError):
            pass

        if not provable_without_have:
            idx_conclusion = split_conclusion(formal_1_code)
            if idx_conclusion:
                idx_end_conclusion = formal_1_code.rfind(":=")
                conclusion = formal_1_code[idx_conclusion:idx_end_conclusion].strip()
                have_stmt_proof = (
                    f"have {conclusion} := by\n"
                    + indent_code(f"apply_rules [{base_thm_name}]\n" + proof_all_apply, 2)
                    + "\n"
                )
                proof_have = check_proof_sub(have_stmt_proof + proof_all_have)
                if proof_have:
                    res.beq_plus_unidirections = update_tuple(res.beq_plus_unidirections, i, proof_have)
                    continue

        # 4. apply with tolerance on the conclusion (`convert`)
        for max_step in range(0, 5):
            proof_convert = check_proof_sub(
                f"convert (config := .unfoldSameFun) {base_thm_name} using {max_step}\n" + proof_all_apply
            )
            if proof_convert:
                res.beq_plus_unidirections = update_tuple(res.beq_plus_unidirections, i, proof_convert)
                break

        if not res.beq_plus_unidirections[i]:
            break

    return res
# ─────────────────────────── end vendored code ──────────────────────────────


def split_header_and_theorem(full_snippet: str) -> tuple[str, str]:
    """Split a full Lean file snippet (imports / namespace / variable / open /
    theorem) into (context, theorem_text). `context` has import lines and the
    `set_option maxRecDepth` line stripped (already covered by BASE_IMPORT),
    matching MixtureOfMathExperts/scripts/beq/run_beq_plus.py's header_context().
    """
    idx = extract_last_theorem(full_snippet)
    header, theorem_text = full_snippet[:idx], full_snippet[idx:]
    ctx_lines = []
    for line in header.splitlines():
        s = line.strip()
        if s.startswith("import "):
            continue
        if s.startswith("set_option maxRecDepth"):
            continue
        ctx_lines.append(line)
    context = "\n".join(ctx_lines).strip()
    return context, theorem_text


class BEqPlusScorer:
    """Persistent BEq+ / type-check scorer, backed by one long-lived Lean REPL
    server. Safe to call `score()` / `typecheck()` many times in a row --
    Mathlib is imported once, and per-example header contexts are cached."""

    def __init__(
        self,
        mathlib_root: str = MATHLIB_ROOT,
        cache_dir: str = LEAN_INTERACT_CACHE_DIR,
        timeout_per_proof: int = 60,
        env_timeout: int = 600,
    ):
        self.timeout_per_proof = timeout_per_proof
        self.env_timeout = env_timeout
        config = LeanREPLConfig(
            project=LocalProject(directory=mathlib_root, auto_build=False),
            cache_dir=cache_dir,
            verbose=False,
        )
        self.server = AutoLeanServer(config=config, max_total_memory=1.5, max_process_memory=None)
        base_out = self.server.run(Command(cmd=BASE_IMPORT), timeout=env_timeout, add_to_session_cache=True)
        base_env = getattr(base_out, "env", None)
        if base_env is None:
            raise RuntimeError(f"failed to import Mathlib base env: {base_out}")
        self.base_env = base_env
        self._env_cache: dict[str, int] = {"": base_env}

    def get_env(self, context: str) -> int | None:
        if context in self._env_cache:
            return self._env_cache[context]
        out = self.server.run(Command(cmd=context, env=self.base_env), timeout=self.env_timeout, add_to_session_cache=True)
        env = getattr(out, "env", None)
        if env is not None:
            self._env_cache[context] = env
        return env

    def typecheck(self, theorem_text: str, context: str = "") -> bool:
        """True iff `theorem_text` elaborates (with `sorry` closing the proof)
        against Mathlib + `context`. This is the syntactic / type-check reward
        signal -- it does NOT check semantic equivalence to anything."""
        env = self.get_env(context)
        if env is None:
            return False
        try:
            code = clean_last_theorem_string(theorem_text, "candidate_theorem", add_sorry=True)
        except ValueError:
            return False
        try:
            out = self.server.run(Command(cmd=code, env=env), timeout=self.timeout_per_proof)
        except (TimeoutError, ConnectionAbortedError, json.JSONDecodeError):
            return False
        if isinstance(out, LeanError):
            return False
        return out.lean_code_is_valid()

    def score(self, gold_full_snippet: str, pred_theorem_text: str) -> dict:
        """gold_full_snippet: full record text (imports+namespace+...+theorem),
        e.g. LoCoLib's `gold_standard_formal_theorem` field.
        pred_theorem_text: the model's raw generated theorem statement (no
        surrounding imports/namespace needed -- the gold record's context is
        reused for both sides, matching how BEq+ is meant to be run).

        Returns {"typecheck": bool, "beql": bool, "beq_plus": bool, "error": str|None}.
        """
        context, gold_theorem_text = split_header_and_theorem(gold_full_snippet)
        out = {"typecheck": False, "beql": False, "beq_plus": False, "error": None}
        env = self.get_env(context)
        if env is None:
            out["error"] = "header_env_failed"
            return out
        out["typecheck"] = self.typecheck(pred_theorem_text, context)
        try:
            res = check_theorem_equivalence(gold_theorem_text, pred_theorem_text, self.server, env, self.timeout_per_proof)
            out["beql"] = res.beql()
            out["beq_plus"] = res.beq_plus()
        except Exception as e:  # keep the server alive on a single bad record
            out["error"] = f"{type(e).__name__}: {e}"[:200]
        return out


if __name__ == "__main__":
    # Minimal self-test using two LoCoLib_V1 records: one exact restatement
    # (should score beq_plus=True) and one clearly different theorem (should
    # score beq_plus=False), plus a syntactically broken prediction.
    scorer = BEqPlusScorer()

    gold = (
        "import Mathlib\n"
        "namespace TestNS\n\n"
        "theorem add_zero_eq (n : ℕ) : n + 0 = n := by rfl"
    )
    pred_equiv = "theorem restated (n : ℕ) : n + 0 = n"
    pred_diff = "theorem restated (n : ℕ) : n + 1 = n"
    pred_broken = "theorem this is not valid lean :::"

    print("equivalent pair ->", scorer.score(gold, pred_equiv))
    print("different pair   ->", scorer.score(gold, pred_diff))
    print("broken pred      ->", scorer.score(gold, pred_broken))
