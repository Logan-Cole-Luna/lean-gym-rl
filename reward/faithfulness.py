#!/usr/bin/env python3
"""NL -> FL faithfulness: does the Lean proof carry out the English argument?

The one adapter between this repo and `scripts/AutoFaith`. Nothing else imports
AutoFaith, so upstream churn lands here and nowhere else.

WHAT THIS IS NOT. `scripts/eval/proof_alignment.py` compares two LEAN proofs'
goal states (FL <-> FL structure). This is the other axis, the one
`reward/reward.py`'s table calls "the only place the English enters the score":
the candidate's Lean proof against the informal proof it was handed.

THE PIPELINE, from AutoFaith's `faithful_metric/main.py`:

    informal statement + informal proof --(judge)--> NLBlock list
    candidate Lean proof             --(our Lean REPL)--> FLBlock list
    NLBlocks + FLBlocks              --(judge)--> score in [0,1]

TWO THINGS THIS DELIBERATELY DOES DIFFERENTLY FROM AUTOFAITH.

1. FL BLOCKS COME FROM OUR REPL. `FL_checkpoints.LeanREPL` spawns its own
   `lake exe repl` and reads a file INSIDE a built Lean project -- a writable
   project per worker. It sends `{"path":..., "allTactics": True}` and reads
   `tactic / proofState / goals / pos / endPos`; `lean_interact`'s `Tactic`
   exposes exactly those five fields, and `Command(cmd=..., all_tactics=True)`
   takes source text rather than a path. So we reuse the `BEqPlusScorer` REPL we
   already hold open and keep AutoFaith's pure-Python tree logic unchanged.
   `scripts/eval/proof_alignment.py` established this route.

2. ZERO FL BLOCKS IS UNKNOWN, NEVER ZERO. This is the load-bearing line in the
   file. A TERM-MODE proof (`:= rfl`, `:= fun x => ...`, `:= Nat.add_comm a b`)
   is valid Lean that emits NO tactics, so it yields no FLBlocks -- and
   `WholeProofJudge.judge` then takes its `_empty()` branch and returns a hard
   `score=0.0`. MEASURED: 48.7% of LoCoLib golds are term-mode (365/749), and
   43.8% of the RL policy's own proofs. Letting that reach the judge would score
   half of all perfectly faithful proofs 0.0. We check before calling it.

   That leaves a structural ~40-49% unknown rate on this dataset, which is why
   this is an OFFLINE metric and why coverage is reported as a first-class
   number. It is not fit to back a reward band until AutoFaith grows a term-mode
   path -- judging the proof TERM directly, with no block structure.

NEVER RAISES. Every entry point returns a result whose `score` is None on any
failure. `reward/reward_fn.py`'s `_never_raises` turns an escaping exception into
an all-zero dict, which would silently discard a correct BEq+ verdict along with
the faithfulness one.

NEVER HOLDS THE LEAN SEMAPHORE ACROSS A NETWORK CALL. Lean work and judge work
are separate steps; the caller owns `_lean_slot` and must release it before the
judge is called. `BEQ_MAX_CONCURRENT=1` is a thread-safety requirement, so
holding it across a multi-second generation would serialise the whole trainer
behind the judge.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

AUTOFAITH_ROOT = Path(os.environ.get(
    "AUTOFAITH_ROOT",
    Path(__file__).resolve().parents[1] / "scripts" / "AutoFaith" / "faithful_metric"))

JUDGE_URL = os.environ.get("FAITH_JUDGE_URL", "http://127.0.0.1:8732/generate")
JUDGE_TIMEOUT = float(os.environ.get("FAITH_JUDGE_TIMEOUT", "180"))
# The matrix of Lean round-trips is linear in this, and a runaway proof would
# otherwise dominate the run. `proof_alignment.py` caps at 24 for the same reason.
MAX_TACTICS = int(os.environ.get("FAITH_MAX_TACTICS", "24"))


# Every failure path here returns None rather than raising, which is required
# (see the module docstring) but also hides OUR OWN bugs as infrastructure
# failures -- an AttributeError in the adapter is indistinguishable from Lean
# timing out. FAITH_DEBUG=1 prints what was actually caught.
FAITH_DEBUG = os.environ.get("FAITH_DEBUG", "0") == "1"


def _debug(where: str, exc: BaseException) -> None:
    if FAITH_DEBUG:
        print(f"[faithfulness] {where}: {type(exc).__name__}: {exc}", flush=True)


@dataclass(frozen=True)
class FaithfulnessResult:
    """`score` is None whenever we could not find out. Never a number on failure.

    `error` is a taxonomy, not a message, so the validity gate can say WHICH
    failure is dominating rather than reporting one opaque unknown rate.
    """
    score: float | None
    error: str | None            # see _ERRORS
    n_nl_blocks: int = 0
    n_fl_blocks: int = 0
    seconds: float = 0.0
    strategy_match: bool | None = None
    summary: str = ""

    @property
    def known(self) -> bool:
        return self.score is not None


_ERRORS = (
    "no_autofaith",       # the clone is missing or unimportable
    "no_judge",           # judge server unreachable / errored
    "bad_json",           # judge returned something that is not the schema
    "no_declaration",     # the candidate has no extractable theorem
    "fl_extract_failed",  # Lean did not answer
    "fl_blocks_empty",    # TERM-MODE proof: no tactics, hence no blocks
    "nl_blocks_failed",   # NL block extraction raised or came back empty
)


# ---------------------------------------------------------------------------
# AutoFaith imports, done lazily and defensively.
#
# NEVER import `faithful_metric.judge`: it constructs a JudgeModel and calls
# generate_nl_blocks AT MODULE SCOPE, then writes nl_blocks.json. Importing it
# fires an OpenAI call with an empty key and writes a file into the clone.
# `faithful_metric.FL_checkpoints` is also avoided at the REPL level -- only its
# pure tree logic is reused, via _fl_blocks_from_tactics below.
# ---------------------------------------------------------------------------
def _autofaith():
    """Return the AutoFaith symbols we use, or None if unavailable."""
    if str(AUTOFAITH_ROOT) not in sys.path:
        sys.path.insert(0, str(AUTOFAITH_ROOT))
    try:
        from scorer.checkpoints import Checkpoint, FLBlock, FLTacticCategory  # noqa
        from scorer.prompt import (NL_CHECKPOINT_GENERATION_PROMPT,  # noqa
                                   WHOLE_PROOF_FAITHFULNESS_PROMPT)
        import FL_checkpoints as flc  # noqa
        return {"Checkpoint": Checkpoint, "FLBlock": FLBlock,
                "FLTacticCategory": FLTacticCategory, "flc": flc,
                "NL_PROMPT": NL_CHECKPOINT_GENERATION_PROMPT,
                "JUDGE_PROMPT": WHOLE_PROOF_FAITHFULNESS_PROMPT}
    except Exception as e:
        _debug("_autofaith", e)
        return None


# ---------------------------------------------------------------------------
# The judge: one HTTP call to our own FormalRx server.
# ---------------------------------------------------------------------------
def judge_generate(system: str | None, user: str, *, temperature: float = 0.0,
                   url: str = JUDGE_URL, timeout: float = JUDGE_TIMEOUT) -> str | None:
    """Raw text from the judge, or None. Never raises."""
    payload = {"user": user, "temperature": temperature}
    if system:
        payload["system"] = system
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("text")
    except Exception as e:
        _debug("judge_generate", e)
        return None


def _parse_json(text: str) -> dict | None:
    """Pull a JSON object out of the judge's reply. None if there isn't one.

    Mirrors AutoFaith's `_parse_json` but does not raise, and falls back to the
    outermost brace pair -- an unconstrained local model prepends prose more
    often than a structured-output API does.
    """
    import re
    t = re.sub(r"^```(?:json)?\s*", "", text.strip())
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# FL blocks, from OUR REPL. See note 1 in the module docstring.
# ---------------------------------------------------------------------------
def fl_blocks(scorer, lean_proof: str, context: str = "",
              max_tactics: int = MAX_TACTICS) -> tuple[list | None, str | None]:
    """(blocks, error). `([], None)` is a legitimate TERM-MODE result.

    Distinguishes "Lean did not answer" (`fl_extract_failed`) from "Lean answered
    and there were no tactics" (empty list, no error) -- the caller maps the
    second to `fl_blocks_empty`, and conflating them would hide the term-mode
    population inside the infrastructure-failure rate.
    """
    af = _autofaith()
    if af is None:
        return None, "no_autofaith"
    from lean_interact import Command

    from reward.beq_plus import split_header_and_theorem
    try:
        _ctx, decl = split_header_and_theorem(lean_proof)
    except Exception:
        return None, "no_declaration"

    env = scorer.get_env(context)
    if env is None:
        return None, "fl_extract_failed"
    try:
        # Not `scorer._run`: it builds its own Command and cannot pass
        # all_tactics. Same reason proof_alignment.py bypasses it.
        out = scorer.server.run(Command(cmd=decl, env=env, all_tactics=True),
                                timeout=scorer.timeout_per_proof)
    except Exception:
        return None, "fl_extract_failed"

    tactics = getattr(out, "tactics", None) or []
    if not tactics:
        return [], None                      # term-mode, or the proof failed
    tactics = tactics[:max_tactics]

    # lean_interact's `Tactic` carries exactly the five fields AutoFaith's
    # `_parse_raw_tactics` reads off the raw REPL dict, so its RawTactic is
    # constructible directly and every downstream tree function is reused as-is.
    flc = af["flc"]
    raw = [flc.RawTactic(tactic=t.tactic, proof_state=t.proof_state, goals=t.goals,
                         line=t.start_pos.line, column=t.start_pos.column,
                         end_line=t.end_pos.line, end_column=t.end_pos.column)
           for t in tactics]
    try:
        # `_build_tactic_tree` / `_convert_raw_tactic` are classmethods ON
        # LeanREPL, not module functions -- calling them off the module raises
        # AttributeError, which the broad catch below would report as a Lean
        # failure. Hence FAITH_DEBUG.
        roots = flc.LeanREPL._build_tactic_tree(raw)
        blocks: list = []
        # `_convert_raw_tactic` is an instance method only because it issues a
        # ProofStep for the after-state. Give it an instance whose `send` is our
        # server, so no second REPL is created.
        shim = _repl_shim(flc, scorer)
        for root in sorted(roots, key=lambda x: x.start):
            blocks.extend(flc.LeanREPL._convert_raw_tactic(shim, root))
        return blocks, None
    except Exception as e:
        _debug("fl_blocks", e)
        return None, "fl_extract_failed"


def _repl_shim(flc, scorer):
    """An AutoFaith `LeanREPL` whose `send` is OUR server and whose `__init__`
    spawns nothing.

    SUBCLASSED, not reimplemented: `_convert_raw_tactic` also reaches for
    `_is_internal` and `_goals_to_checkpoint` on self, and copying those here
    would fork the block semantics the moment upstream touched them. The only
    thing we override is where the bytes go.
    """
    class _Shim(flc.LeanREPL):
        def __init__(self, scorer):          # deliberately not super().__init__
            self.scorer = scorer

        def send(self, request: dict) -> dict:
            from lean_interact import ProofStep
            out = self.scorer.server.run(
                ProofStep(tactic=request["tactic"], proof_state=request["proofState"]),
                timeout=self.scorer.timeout_per_proof)
            goals = getattr(out, "goals", None)
            # AutoFaith's `_goals_to_checkpoint` takes `str | list[str]`; the
            # REPL's "no goals" sentinel is what `_is_internal` tests for.
            return {"goals": goals if goals else "no goals"}

    return _Shim(scorer)


# ---------------------------------------------------------------------------
# NL blocks, and the whole-proof judgement.
# ---------------------------------------------------------------------------
def nl_blocks(statement_nl: str, proof_nl: str, **kw) -> tuple[list | None, str | None]:
    """AutoFaith's NL-block extraction, run through OUR judge server.

    Not `NL_judge_model.JudgeModel`: that hardwires the OpenAI client, and its
    `_validate_transition_consistency` RAISES on adjacent-checkpoint mismatch
    (which is also why AutoFaith's own retry loop is unreachable). We keep the
    prompt and drop the raise -- an inconsistent chain is still a usable
    decomposition for a whole-proof judgement, and turning it into an exception
    would convert a judge wobble into a lost rollout.
    """
    af = _autofaith()
    if af is None:
        return None, "no_autofaith"
    prompt = (af["NL_PROMPT"].replace("{THEOREM_STATEMENT}", statement_nl)
              .replace("{NATURAL_LANGUAGE_PROOF}", proof_nl))
    raw = judge_generate(None, prompt, **kw)
    if raw is None:
        return None, "no_judge"
    obj = _parse_json(raw)
    if not isinstance(obj, dict):
        return None, "bad_json"
    blocks = obj.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        return None, "nl_blocks_failed"
    return blocks, None


def score_faithfulness(statement_nl: str, proof_nl: str, lean_proof: str, *,
                       scorer, context: str = "", **kw) -> FaithfulnessResult:
    """THE ENTRY POINT. Returns a result whose `score` is None on any failure.

    Order matters: Lean first (cheap, and its term-mode verdict short-circuits
    the whole judge cost), then the two judge calls.
    """
    t0 = time.time()
    el = lambda: time.time() - t0

    blocks_fl, err = fl_blocks(scorer, lean_proof, context)
    if err:
        return FaithfulnessResult(None, err, seconds=el())
    if not blocks_fl:
        # TERM-MODE. Not a faithfulness verdict -- see the module docstring.
        return FaithfulnessResult(None, "fl_blocks_empty", seconds=el())

    blocks_nl, err = nl_blocks(statement_nl, proof_nl, **kw)
    if err:
        return FaithfulnessResult(None, err, n_fl_blocks=len(blocks_fl), seconds=el())

    af = _autofaith()
    fl_json = json.dumps([b.to_dict() for b in blocks_fl], indent=2, ensure_ascii=False)
    nl_json = json.dumps(blocks_nl, indent=2, ensure_ascii=False)
    prompt = (af["JUDGE_PROMPT"].replace("{NL_BLOCKS}", nl_json)
              .replace("{FL_BLOCKS}", fl_json))
    raw = judge_generate(None, prompt, **kw)
    if raw is None:
        return FaithfulnessResult(None, "no_judge", len(blocks_nl), len(blocks_fl), el())
    obj = _parse_json(raw)
    if not isinstance(obj, dict) or not isinstance(obj.get("score"), (int, float)):
        return FaithfulnessResult(None, "bad_json", len(blocks_nl), len(blocks_fl), el())

    return FaithfulnessResult(
        score=min(1.0, max(0.0, float(obj["score"]))),
        error=None, n_nl_blocks=len(blocks_nl), n_fl_blocks=len(blocks_fl),
        seconds=el(), strategy_match=obj.get("strategy_match"),
        summary=str(obj.get("summary", ""))[:500])


def proof_follows_argument(informal_proof: str | None, lean_proof: str | None,
                           *, statement_nl: str = "", scorer=None,
                           context: str = "", **kw) -> float | None:
    """The `reward.reward.proof_follows_argument` signature, for the day this is
    wired in. Returns None -- unknown -- on every failure path, per that
    function's own stated requirement. Not called from the reward loop today.
    """
    if not informal_proof or not lean_proof or scorer is None:
        return None
    return score_faithfulness(statement_nl, informal_proof, lean_proof,
                              scorer=scorer, context=context, **kw).score
