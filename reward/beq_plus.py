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
import time as _time
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
    remove_lean_comments,
    split_conclusion,
)

# ── DIRECTION SEMANTICS (read before using n_directions as a reward signal) ───
#
# `check_theorem_equivalence` iterates [(gold, pred), (pred, gold)] and, for each
# pair, proves the SECOND from the FIRST. So:
#
#   direction 0  proves pred from gold  ->  gold ⇒ pred  ->  pred is WEAKER
#   direction 1  proves gold from pred  ->  pred ⇒ gold  ->  pred is STRONGER
#
# The loop `break`s when a direction fails, and direction 0 runs first. That
# makes the ladder ASYMMETRIC in a way that matters for reward shaping:
#
#   n_directions == 1  <=>  gold ⇒ pred succeeded and pred ⇒ gold failed
#                           i.e. the ONLY observable one-direction state is
#                           "prediction is strictly weaker than the gold".
#   n_directions == 0  <=>  gold ⇒ pred failed, and pred ⇒ gold WAS NEVER TRIED.
#                           A strictly STRONGER prediction is therefore
#                           indistinguishable from unrelated garbage.
#
# Consequence: a reward that pays partial credit for `n_directions >= 1` pays it
# exclusively for weakening the statement, and pays nothing for strengthening it.
# Weakening is also the direction that trends toward vacuity (anything trivially
# true is implied by the gold). BEq+ guards the worst of this internally -- step 3
# of the cascade is skipped when the candidate is provable on its own -- but
# steps 2 and 4 (`apply` / `convert ... using 0..4`) are not gated that way.
#
# `BEqPlusScorer.score(..., probe_stronger=True)` closes the asymmetry by running
# a second, argument-swapped cascade when direction 0 fails, so `pred ⇒ gold` is
# observed on its own. It costs roughly one extra cascade per non-equivalent
# rollout, which is why it is OFF by default -- see BEQ_PROBE_STRONGER.
BEQ_PROBE_STRONGER = os.environ.get("BEQ_PROBE_STRONGER", "0") == "1"

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
MATHLIB_ROOT = os.environ.get("MATHLIB_ROOT", str(PROJECT_ROOT / "repos" / "mathlib4"))
LEAN_INTERACT_CACHE_DIR = os.environ.get("LEAN_INTERACT_CACHE_DIR", str(Path.home() / ".cache" / "lean_interact"))
BASE_IMPORT = "import Mathlib\nset_option maxRecDepth 10000"

# Per-Lean-process memory cap, in MB. See BEqPlusScorer.__init__ for why this is
# not optional. Set BEQ_MEMORY_LIMIT_MB=0 to disable (matching the upstream
# batch-scoring script's behaviour, only safe when nothing else shares the box).
BEQ_MATHLIB_BASELINE_MB = 4500
BEQ_MEMORY_LIMIT_MB = int(os.environ.get("BEQ_MEMORY_LIMIT_MB", "8000")) or None
if BEQ_MEMORY_LIMIT_MB is not None and BEQ_MEMORY_LIMIT_MB < BEQ_MATHLIB_BASELINE_MB:
    raise ValueError(
        f"BEQ_MEMORY_LIMIT_MB={BEQ_MEMORY_LIMIT_MB} is below the ~{BEQ_MATHLIB_BASELINE_MB}MB "
        "that `import Mathlib` needs; the Lean REPL would die on startup. "
        "Use a larger value, or 0 to disable the cap."
    )
# Per-proof-attempt timeout. The cascade makes up to ~18 Lean calls per pair
# (9 tactic attempts x 2 directions), so this multiplies out; 60s is the
# upstream default and is far too generous for an online reward loop.
BEQ_TIMEOUT_PER_PROOF = int(os.environ.get("BEQ_TIMEOUT_PER_PROOF", "30"))

# Repairs the stage-3 independence guard, which cannot fire without it. Changes
# BEq+ verdicts, so runs with and without it are not comparable.
BEQ_HAVE_GUARD_PATCH = os.environ.get("BEQ_HAVE_GUARD_PATCH", "0") == "1"

# BEq+ IMPLIES type-check -- verified with zero exceptions across 9,528 scored
# rollouts. So when the prediction does not elaborate, the equivalence cascade
# (up to ~18 Lean calls) cannot possibly fire and is pure waste. 23.4% of
# rollouts fail type-check, so skipping is a large fraction of the calls.
#
# This is metric-preserving by construction, but VERIFY on your data before
# trusting it: re-score a sample with BEQ_SKIP_CASCADE_ON_TYPECHECK_FAIL=0 and
# confirm every verdict matches. Set to 0 to restore the old behaviour.
BEQ_SKIP_CASCADE_ON_TYPECHECK_FAIL = os.environ.get(
    "BEQ_SKIP_CASCADE_ON_TYPECHECK_FAIL", "1") == "1"
# Accumulate per-phase wall-clock so the cost of the cascade vs. the type-check
# is measurable rather than assumed (see BEqPlusScorer.stats).
BEQ_TIME_PHASES = os.environ.get("BEQ_TIME_PHASES", "1") == "1"


# ─────────────────────────────────────────────────────────────────────────────
# VENDORED VERBATIM from augustepoiroux/RLMEval src/rlm_eval/metrics/beq_plus.py
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BEqCPUResult:
    beql_unidirections: tuple[str | None, str | None] = (None, None)
    beq_plus_unidirections: tuple[str | None, str | None] = (None, None)

    # ── ADDITIVE INSTRUMENTATION — NOT part of the published metric ───────────
    # These record what the cascade already decided and then discarded. Nothing
    # below changes a tactic, a timeout, or the control flow: every assignment
    # sits next to an existing branch, and `beql()` / `beq_plus()` read only the
    # two original fields. To verify that, re-score a pool and diff `beq_plus`
    # and `beql` against the previous run -- they must be identical.
    #
    # rungs[i]: which cascade rung closed direction i.
    #   1 = `exact?` (also the ONLY rung that sets beql)   2 = `apply`
    #   3 = `have <conclusion>`                            4 = `convert`
    # A lower rung is a tighter match: rung 1 means the statements are
    # definitionally interchangeable, rung 4 means only the conclusion survives
    # heavy congruence. This is the closeness ordinal the bools throw away.
    rungs: tuple[int | None, int | None] = (None, None)
    # convert_levels[i]: the `using k`, k in 0..4, when rung 4 fired. Higher k =
    # more tolerance was needed = further apart.
    convert_levels: tuple[int | None, int | None] = (None, None)
    # provable_alone[i]: rung 3's `provable_without_have` -- whether the SECOND
    # theorem of the pair proves on its own, with no reference to the first.
    # For direction 0 the second theorem is the PREDICTION, so this is a
    # free self-provability check on exactly the dead-band rollouts.
    #
    # IT IS A LOWER BOUND, NOT `self_prove`'s `provable`. Rung 3 proves with
    # `prove_all(["tauto", "simp_all_arith!", "exact? using this"])`, and there
    # is no `this` in scope at that point, so it is effectively tauto +
    # simp_all_arith!. `self_prove` additionally tries `noncomm_ring` and a bare
    # `exact?`. Expect this to MISS statements only those two can close.
    # It is also only computed when rungs 1 and 2 both failed and the `sorry`
    # gate passed -- how often that holds on the dead band is an empirical
    # question, which is what Phase 0 measures.
    provable_alone: tuple[bool | None, bool | None] = (None, None)
    # Why the direction loop stopped, which the bools conflate into n_directions=0:
    #   "unparseable"  clean_last_theorem_string raised (no theorem in the output)
    #   "sorry_gate"   the reformulated theorem would not elaborate beside the base
    #   "no_rung"      it elaborated but no rung closed -- the real dead band
    #   None           both directions completed
    stop_reason: str | None = None

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
            res.stop_reason = "unparseable"  # instrumentation only
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
            res.stop_reason = "sorry_gate"  # instrumentation only
            break

        # 1. `exact?`, and check it uses the base theorem in the proof.
        proof_exact = check_proof_sub("exact?")
        if proof_exact and base_thm_name in proof_exact:
            res.beql_unidirections = update_tuple(res.beql_unidirections, i, proof_exact)
            res.beq_plus_unidirections = update_tuple(res.beq_plus_unidirections, i, proof_exact)
            res.rungs = update_tuple(res.rungs, i, 1)  # instrumentation only
            continue

        # 2. try to apply the base theorem directly
        proof_apply = check_proof_sub(f"apply {base_thm_name}\n" + proof_all_apply)
        if proof_apply:
            res.beq_plus_unidirections = update_tuple(res.beq_plus_unidirections, i, proof_apply)
            res.rungs = update_tuple(res.rungs, i, 2)  # instrumentation only
            continue

        # 3. add the conclusion of the base theorem as a hypothesis
        provable_without_have = False
        try:
            # LOCAL PATCH vs upstream. `formal_2_code` ends in ":= by" and
            # `proof_all_have` begins with "all_goals intros", so upstream's bare
            # concatenation yields ":= byall_goals intros ...", which Lean
            # rejects. `lean_code_is_valid` is therefore always False without the
            # newline, `provable_without_have` always False, and this guard --
            # whose whole purpose is to suppress a `have`-stage false positive --
            # can never fire. With it, gold `forall n, n + 0 = n` vs the strictly
            # weaker `0 + 0 = 0` stops scoring equivalent at this stage.
            #
            # It does NOT close the whole false-positive class: only stage 3
            # consults the guard, and stage 4 (`convert`) has no independence
            # check, so such a pair can move one stage down and still score
            # equivalent.
            #
            # OFF BY DEFAULT because it changes BEq+ verdicts, and every recorded
            # number was produced without it. Turn it on together with the reward
            # default, under a new series tag.
            _sep = "\n" if BEQ_HAVE_GUARD_PATCH else ""
            res_without_have = lean_server.run(
                Command(cmd=formal_2_code + _sep + proof_all_have, env=context_env), timeout=timeout_per_proof
            )
            if isinstance(res_without_have, CommandResponse):
                provable_without_have = res_without_have.lean_code_is_valid(allow_sorry=False)
        except (TimeoutError, ConnectionAbortedError, json.JSONDecodeError):
            pass
        # instrumentation only -- the free self-provability read. See the field
        # comment on BEqCPUResult.provable_alone for what this does and does not
        # mean. Recorded here, BEFORE the `if`, so a True value is captured even
        # though the vendored code only uses it to skip rung 3.
        res.provable_alone = update_tuple(res.provable_alone, i, provable_without_have)

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
                    res.rungs = update_tuple(res.rungs, i, 3)  # instrumentation only
                    continue

        # 4. apply with tolerance on the conclusion (`convert`)
        for max_step in range(0, 5):
            proof_convert = check_proof_sub(
                f"convert (config := .unfoldSameFun) {base_thm_name} using {max_step}\n" + proof_all_apply
            )
            if proof_convert:
                res.beq_plus_unidirections = update_tuple(res.beq_plus_unidirections, i, proof_convert)
                res.rungs = update_tuple(res.rungs, i, 4)  # instrumentation only
                res.convert_levels = update_tuple(res.convert_levels, i, max_step)  # instrumentation only
                break

        if not res.beq_plus_unidirections[i]:
            res.stop_reason = "no_rung"  # instrumentation only
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


# A prediction from a CoT-distilled model is a self-contained file: its own
# `import` / `open` / `variable` / `namespace`, plus (upstream of the fence) a
# `<think>` block that `_clean_solution` does not always remove. Keep only the
# lines that are actually Lean context so a stray prose line can't break the env.
_CTX_KEEP_RE = re.compile(
    r"^\s*(open|variable|namespace|section|end|universe|set_option|notation|"
    r"local|scoped|attribute|noncomputable)\b")


def _standalone_context(header: str) -> str:
    """The Lean-context lines of a prediction's own header (imports already
    dropped by `split_header_and_theorem`)."""
    return "\n".join(ln for ln in header.splitlines()
                     if _CTX_KEEP_RE.match(ln)).strip()


def _union_context(gold_ctx: str, pred_ctx: str) -> str:
    """Gold context first, then any prediction-context line not already present
    (compared whitespace-insensitively). BEq+ needs one env in which BOTH the
    gold statement and the prediction statement elaborate; layering the
    prediction's own `open`/`variable`/`namespace` on top of the gold's is the
    pragmatic way to get there for a standalone prediction. A prediction `open`
    that makes a name in the gold statement ambiguous just costs that row a
    miss, not a crash."""
    seen, lines = set(), []
    for block in (gold_ctx, pred_ctx):
        for ln in block.splitlines():
            key = " ".join(ln.split())
            if not key or key in seen:
                continue
            seen.add(key)
            lines.append(ln)
    return "\n".join(lines).strip()


# ── candidate's OWN proof, kept intact (not sorry'd away) ──────────────────
#
# `clean_theorem_string`/`clean_last_theorem_string` (lean_interact.utils)
# unconditionally cut at the top-level `:=` via `split_implementation`,
# regardless of `add_sorry` -- they exist to STATE a theorem, never to check a
# proof someone wrote for it. Nothing upstream or in this file renames a
# theorem while leaving its proof body untouched, so `check_own_proof` below
# needs its own renamer.
_STANDARD_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
_AXIOMS_RE = re.compile(r"depends on axioms:\s*\[(.*?)\]", re.S)


def _rename_theorem_keep_proof(lean_code: str, new_name: str) -> str:
    """Like `clean_theorem_string`, but does NOT touch anything from `:=`
    onward -- the proof, however long, survives verbatim. Only the
    declaration keyword's NAME is swapped, so the renamed candidate can
    coexist in the same env as anything else without a name collision.
    """
    idx = extract_last_theorem(lean_code)  # raises ValueError if none found
    head, tail = lean_code[:idx], lean_code[idx:]
    clean_tail = remove_lean_comments(tail)
    if clean_tail is None:
        raise ValueError("Comment removal failed.")
    clean_tail = clean_tail.strip()
    m = re.search(r"\b(theorem|lemma|example)\s", clean_tail)
    if m is None:
        raise ValueError("Theorem declaration keyword not found.")
    clean_tail = clean_tail[m.start():]
    rest = re.sub(r"^[^\s]+", "", clean_tail).strip()      # drop the keyword
    if not clean_tail.startswith("example"):
        rest = re.sub(r"^[^\s:({\[]+", "", rest).strip()   # drop the name
    return head + f"theorem {new_name} " + rest


def _parse_axioms(messages) -> list[str] | None:
    """Pull the axiom list out of a `#print axioms <name>` info message.

    None means the message was never found (Lean didn't answer as expected,
    e.g. the declaration failed before `#print axioms` could run) -- treated
    as "unknown", never as "clean", by every caller.
    """
    for msg in messages:
        if getattr(msg, "severity", None) != "info":
            continue
        data = getattr(msg, "data", "")
        if "does not depend on any axioms" in data:
            return []
        match = _AXIOMS_RE.search(data)
        if match:
            return [a.strip() for a in match.group(1).split(",") if a.strip()]
    return None


class BEqPlusScorer:
    """Persistent BEq+ / type-check scorer, backed by one long-lived Lean REPL
    server. Safe to call `score()` / `typecheck()` many times in a row --
    Mathlib is imported once, and per-example header contexts are cached."""

    def __init__(
        self,
        mathlib_root: str = MATHLIB_ROOT,
        cache_dir: str = LEAN_INTERACT_CACHE_DIR,
        timeout_per_proof: int = BEQ_TIMEOUT_PER_PROOF,
        # Narval: Mathlib's ~4200 oleans live on Lustre, where a cold
        # `import Mathlib` can exceed the old hardcoded 600s. Override with
        # BEQ_ENV_TIMEOUT; staging mathlib4 to $SLURM_TMPDIR is the real fix.
        env_timeout: int = int(os.environ.get("BEQ_ENV_TIMEOUT", "600")),
        memory_hard_limit_mb: int | None = BEQ_MEMORY_LIMIT_MB,
    ):
        self.timeout_per_proof = timeout_per_proof
        self.probe_timeout = float(os.environ.get("BEQ_PROBE_TIMEOUT", "10"))
        self.env_timeout = env_timeout
        config = LeanREPLConfig(
            project=LocalProject(directory=mathlib_root, auto_build=False),
            cache_dir=cache_dir,
            memory_hard_limit_mb=memory_hard_limit_mb,
            verbose=False,
        )
        # max_total_memory=1.5 disables lean-interact's SYSTEM-wide guard
        self.server = AutoLeanServer(
            config=config,
            max_total_memory=1.5,
            max_process_memory=0.8 if memory_hard_limit_mb else None,
        )
        base_out = self.server.run(Command(cmd=BASE_IMPORT), timeout=env_timeout, add_to_session_cache=True)
        base_env = getattr(base_out, "env", None)
        if base_env is None:
            raise RuntimeError(f"failed to import Mathlib base env: {base_out}")
        self.base_env = base_env
        self._env_cache: dict[str, int] = {"": base_env}
        self.stats: dict[str, float] = {
            "score_calls": 0,
            "typecheck_calls": 0,
            "timeout": 0,
            "infra": 0,      # REPL died / connection aborted / malformed JSON
            "other_error": 0,
            "probe_calls": 0,
            # Cascade skipped because the prediction did not type-check, and so
            # could not have proved in either direction (see
            # BEQ_SKIP_CASCADE_ON_TYPECHECK_FAIL).
            "cascade_skipped": 0,
            # REPL restarts. Climbing here means AutoLeanServer is recycling the
            # Lean process often (memory), and each restart costs one rollout.
            "restarts": 0,
            "t_typecheck": 0.0,
            "t_cascade": 0.0,
        }

    def _note_error(self, exc: BaseException) -> str:
        """Classify a scorer failure and bump the matching counter."""
        if isinstance(exc, TimeoutError):
            kind = "timeout"
        elif isinstance(exc, (ConnectionAbortedError, json.JSONDecodeError,
                              BrokenPipeError, OSError, AttributeError, ValueError)):
            kind = "infra"
        else:
            kind = "other_error"
        self.stats[kind] += 1
        return kind

    def _restart(self) -> bool:
        """Re-establish the base Mathlib env after the REPL died or recycled.

        INVALIDATES THE CACHE FIRST, AND UNCONDITIONALLY. The first version
        cleared it only on a successful rebuild, so a restart that ALSO failed --
        the likely case, since the REPL is being recycled under memory pressure --
        left the worker holding ids from the dead process and handing out
        confident False verdicts against environments that no longer exist. An
        empty cache costs one re-elaboration per header; a stale one costs the
        rest of the run.
        """
        self.stats["restarts"] += 1
        self._env_cache = {}
        try:
            out = self.server.run(Command(cmd=BASE_IMPORT), timeout=self.env_timeout,
                                  add_to_session_cache=True)
        except Exception:
            return False
        env = getattr(out, "env", None)
        if env is None:
            return False
        self.base_env = env
        self._env_cache = {"": env}
        return True

    def _run(self, cmd: str, env: int | None, timeout: float | None,
             add_to_session_cache: bool = False):
        """The ONE place this class talks to Lean. Returns None on any failure.

        Every caller must treat None as "Lean did not answer", never as "the
        statement is wrong" -- that distinction is what `stats` exists to track.
        A broad `except Exception` is deliberate: lean_interact raises whatever
        the underlying subprocess failure happens to produce, and enumerating
        those types is what left the AttributeError uncaught.
        """
        try:
            return self.server.run(Command(cmd=cmd, env=env), timeout=timeout,
                                   add_to_session_cache=add_to_session_cache)
        except (TimeoutError, json.JSONDecodeError) as e:
            self._note_error(e)          # the REPL is alive, this command was not
            return None
        except Exception as e:
            self._note_error(e)
            self._restart()
            return None

    def get_env(self, context: str) -> int | None:
        if context in self._env_cache:
            return self._env_cache[context]
        out = self._run(context, self.base_env, self.env_timeout, add_to_session_cache=True)
        env = getattr(out, "env", None)
        if env is not None:
            self._env_cache[context] = env
        return env

    def typecheck_ex(self, theorem_text: str, context: str = "") -> tuple[bool, str | None]:
        """`(elaborates, error_kind)`.

        `error_kind` is None for an honest verdict and one of
        "timeout"/"infra"/"other_error" when the answer is "False because Lean
        failed", not "False because the statement is wrong". Callers that feed a
        reward MUST be able to tell those apart -- see BEqPlusScorer.stats.
        """
        self.stats["typecheck_calls"] += 1
        env = self.get_env(context)
        if env is None:
            self.stats["infra"] += 1
            return False, "infra"
        try:
            code = clean_last_theorem_string(theorem_text, "candidate_theorem", add_sorry=True)
        except ValueError:
            return False, None  # genuinely unparseable output, not a Lean failure
        out = self._run(code, env, self.timeout_per_proof)
        if out is None:
            return False, "infra"
        if isinstance(out, LeanError):
            return False, None
        return out.lean_code_is_valid(), None

    def typecheck_message(self, theorem_text: str, context: str = "") -> tuple[bool, str]:
        """`(elaborates, human-readable Lean diagnostics)`.

        `typecheck_ex` deliberately returns only an error CATEGORY, because a
        reward must distinguish "wrong statement" from "Lean broke". Multi-turn
        feedback needs the opposite: the actual message, so the policy can fix
        the statement. Same check, different projection -- the diagnostics are
        already on the response and were simply being dropped.
        """
        self.stats["typecheck_calls"] += 1
        env = self.get_env(context)
        if env is None:
            self.stats["infra"] += 1
            return False, "Lean environment unavailable."
        try:
            code = clean_last_theorem_string(theorem_text, "candidate_theorem", add_sorry=True)
        except ValueError:
            return False, ("Could not parse a single Lean theorem declaration from the "
                           "output. Emit exactly one `theorem ... := by sorry`.")
        out = self._run(code, env, self.timeout_per_proof)
        if out is None:
            return False, "Lean did not respond."
        if isinstance(out, LeanError):
            return False, str(getattr(out, "message", "Lean error"))
        if out.lean_code_is_valid():
            return True, ""
        errs = [f"{getattr(m, 'severity', 'error')}: {getattr(m, 'data', m)}"
                for m in (getattr(out, "messages", []) or [])
                if getattr(m, "severity", "") == "error"]
        return False, "\n".join(errs) if errs else "Statement failed to elaborate."

    def check_own_proof(self, theorem_text: str, context: str) -> dict:
        """Elaborate the candidate's OWN submission as written -- statement
        AND whatever proof it wrote, never sorry'd away.

        Returns {"type_correct", "proved", "sorry_used", "axioms",
        "error_kind"}.

        `type_correct` allows `sorry` -- a WARNING in Lean, not an error, so a
        file whose only proof is `sorry` still elaborates (matches the
        `type_correct` semantics documented in reward/reward.py:41-43).
        `proved` is the stronger claim reward/reward.py's table has specified
        since before this method existed (line 44-46): sorry-free elaboration
        AND not cheating with a false axiom, checked here via `#print axioms`
        against Lean/Mathlib's standard trio (`_STANDARD_AXIOMS`). Any axiom
        outside that set -- e.g. a smuggled `axiom foo : False`-style
        cheat -- fails `proved` even though Lean accepted the file cleanly.
        `axioms=None` means the axiom list could not be read (distinct from
        `[]`, clean), and never counts as clean.
        """
        self.stats["typecheck_calls"] += 1
        env = self.get_env(context)
        if env is None:
            self.stats["infra"] += 1
            return {"type_correct": False, "proved": False, "sorry_used": False,
                    "axioms": None, "error_kind": "infra"}
        try:
            code = _rename_theorem_keep_proof(theorem_text, "candidate_theorem")
        except ValueError:
            return {"type_correct": False, "proved": False, "sorry_used": False,
                    "axioms": None, "error_kind": None}
        code += "\n\n#print axioms candidate_theorem"
        out = self._run(code, env, self.timeout_per_proof)
        if out is None:
            return {"type_correct": False, "proved": False, "sorry_used": False,
                    "axioms": None, "error_kind": "infra"}
        if isinstance(out, LeanError):
            return {"type_correct": False, "proved": False, "sorry_used": False,
                    "axioms": None, "error_kind": None}
        type_correct = out.lean_code_is_valid(allow_sorry=True)
        sorry_free = out.lean_code_is_valid(allow_sorry=False)
        axioms = _parse_axioms(out.messages) if sorry_free else None
        proved = bool(sorry_free and axioms is not None
                      and set(axioms) <= _STANDARD_AXIOMS)
        return {"type_correct": type_correct, "proved": proved,
                "sorry_used": bool(type_correct and not sorry_free),
                "axioms": axioms, "error_kind": None}

    def _prove_direction(self, premise: str, goal: str, env: int) -> bool:
        """True iff `goal` can be proved from `premise` by the BEq+ cascade.

        Runs the vendored routine with the arguments in the order that puts the
        direction we care about FIRST, so the routine's early `break` cannot hide
        it. Only direction 0 of the result is read.
        """
        self.stats["probe_calls"] += 1
        res = check_theorem_equivalence(premise, goal, self.server, env, self.timeout_per_proof)
        return res.beq_plus_unidirections[0] is not None

    def score(self, gold_full_snippet: str, pred_theorem_text: str, probe_stronger: bool | None = None,
              *, pred_context: str | None = None) -> dict:
        """gold_full_snippet: full record text (imports+namespace+...+theorem),
        e.g. LoCoLib's `gold_standard_formal_theorem` field.
        pred_theorem_text: the model's raw generated theorem statement (no
        surrounding imports/namespace needed -- the gold record's context is
        reused for both sides, matching how BEq+ is meant to be run).
        pred_context: extra context lines the prediction brought its own (its
        `open`/`variable`/`namespace`). When given, the env is the UNION of the
        gold context and this, so a self-contained prediction still elaborates.
        `score_standalone` is the entry point that fills it in.

        `probe_stronger` overrides the BEQ_PROBE_STRONGER default; see the
        DIRECTION SEMANTICS note at the top of this module.

        Returns {"typecheck", "beql", "beq_plus", "beq_plus_fwd", "beq_plus_bwd",
                 "n_directions", "gold_implies_pred", "pred_implies_gold",
                 "semantic_signal", "error", "error_kind"}.

        `beq_plus` is the strict published metric: BOTH directions must prove.
        The per-direction flags are surfaced because `check_theorem_equivalence`
        already computes them internally, and one-direction-proves is a genuine
        intermediate state (the prediction implies the gold, or vice versa, but
        not both -- e.g. a strictly stronger or strictly weaker statement). That
        gives a graded "how close is this" signal instead of BEq+'s hard binary.
        `n_directions` in {0,1,2} is the convenient scalar form.

        NOTE the asymmetry documented at the top of this module: `n_directions`
        is faithful to the published metric but can only ever report the WEAKER
        one-direction state. `semantic_signal` is the direction-symmetric
        replacement -- identical to `n_directions` unless `probe_stronger` is on,
        in which case a strictly stronger prediction also scores 1 instead of
        being lumped in with garbage. Reward functions should prefer it.
        """
        if probe_stronger is None:
            probe_stronger = BEQ_PROBE_STRONGER
        self.stats["score_calls"] += 1
        context, gold_theorem_text = split_header_and_theorem(gold_full_snippet)
        if pred_context:
            context = _union_context(context, pred_context)
        out = {"typecheck": False, "beql": False, "beq_plus": False,
               "beq_plus_fwd": False, "beq_plus_bwd": False, "n_directions": 0,
               "gold_implies_pred": False, "pred_implies_gold": False,
               "semantic_signal": 0, "error": None, "error_kind": None,
               # Cascade instrumentation; see BEqCPUResult. `rung`/`convert_level`
               # describe direction 0 (gold => pred), the one that always runs.
               # `provable_alone` is the free self-provability read on the
               # PREDICTION and is the signal the dead-band ladder wants.
               "rung": None, "convert_level": None,
               "provable_alone": None, "stop_reason": None}
        env = self.get_env(context)
        if env is None:
            self.stats["infra"] += 1
            out["error"] = "header_env_failed"
            out["error_kind"] = "infra"
            self._restart()
            return out
        _t0 = _time.perf_counter() if BEQ_TIME_PHASES else 0.0
        out["typecheck"], tc_err = self.typecheck_ex(pred_theorem_text, context)
        if BEQ_TIME_PHASES:
            self.stats["t_typecheck"] += _time.perf_counter() - _t0
        if tc_err:
            out["error_kind"] = tc_err
        if BEQ_SKIP_CASCADE_ON_TYPECHECK_FAIL and not out["typecheck"]:
            self.stats["cascade_skipped"] += 1
            return out
        _t1 = _time.perf_counter() if BEQ_TIME_PHASES else 0.0
        try:
            res = check_theorem_equivalence(gold_theorem_text, pred_theorem_text, self.server, env, self.timeout_per_proof)
            out["beql"] = res.beql()
            out["beq_plus"] = res.beq_plus()
            out["beq_plus_fwd"] = res.beq_plus_unidirections[0] is not None
            out["beq_plus_bwd"] = res.beq_plus_unidirections[1] is not None
            out["n_directions"] = int(out["beq_plus_fwd"]) + int(out["beq_plus_bwd"])
            # Direction 0 is gold => pred; direction 1 is pred => gold.
            out["gold_implies_pred"] = out["beq_plus_fwd"]
            out["pred_implies_gold"] = out["beq_plus_bwd"]
            # Cascade instrumentation, direction 0 -- the one that always runs.
            out["rung"] = res.rungs[0]
            out["convert_level"] = res.convert_levels[0]
            out["provable_alone"] = res.provable_alone[0]
            out["stop_reason"] = res.stop_reason
            # Direction 1 is only ATTEMPTED when direction 0 succeeded, so a
            # strictly stronger prediction reads as 0 directions unless probed.
            if probe_stronger and not out["gold_implies_pred"] and out["typecheck"]:
                out["pred_implies_gold"] = self._prove_direction(pred_theorem_text, gold_theorem_text, env)
            out["semantic_signal"] = (
                2 if out["beq_plus"]
                else int(out["gold_implies_pred"] or out["pred_implies_gold"])
            )
        except Exception as e:  # keep the server alive on a single bad record
            out["error"] = f"{type(e).__name__}: {e}"[:200]
            out["error_kind"] = self._note_error(e)
            # The vendored cascade calls lean_server.run directly and catches only
            # three exception types, so a recycled REPL lands here. Recording it
            # is not enough: _env_cache still holds ids from the dead process, and
            # a worker that keeps those scores every later rollout 0.0. Restart.
            if out["error_kind"] == "infra":
                self._restart()
        if BEQ_TIME_PHASES:
            self.stats["t_cascade"] += _time.perf_counter() - _t1
        return out

    def score_standalone(self, gold_full_snippet: str, pred_full_snippet: str,
                         probe_stronger: bool | None = None) -> dict:
        """`score()` for a prediction that is a SELF-CONTAINED snippet -- its own
        `import` / `open` / `variable` / `namespace` and its own theorem name --
        rather than a bare theorem meant to slot into the gold's context.

        This is what a CoT-distilled model emits (its teacher was prompted for
        standalone Lean). `score()` would strip the prediction's header and
        elaborate the bare theorem in the GOLD's context, where it usually does
        not type-check; here the prediction keeps its header and the equivalence
        env is the union of the two. Same return shape as `score()`.
        """
        try:
            pred_header, pred_theorem_text = split_header_and_theorem(pred_full_snippet)
        except ValueError:
            # No theorem/lemma keyword in the model output -- nothing to score.
            self.stats["score_calls"] += 1
            return {"typecheck": False, "beql": False, "beq_plus": False,
                    "beq_plus_fwd": False, "beq_plus_bwd": False, "n_directions": 0,
                    "gold_implies_pred": False, "pred_implies_gold": False,
                    "semantic_signal": 0, "error": None, "error_kind": None,
                    "rung": None, "convert_level": None,
                    "provable_alone": None, "stop_reason": None}
        return self.score(gold_full_snippet, pred_theorem_text, probe_stronger,
                          pred_context=_standalone_context(pred_header))


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
