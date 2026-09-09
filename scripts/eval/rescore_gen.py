#!/usr/bin/env python3
"""Re-score completions that were already generated, changing nothing but the
Lean environment.

WHAT THIS ISOLATES. `evaluate_checkpoints.py` does two things in one job:
generate, then score. So two eval runs differ in the model AND in whatever Lean
environment each job happened to stage, and a rate difference cannot be assigned
to either. This reads the completions back out of a `gen_*.jsonl` and re-scores
them, so the text is held FIXED by construction and the only live variable is
the environment this process runs in.

WHY IT WAS WRITTEN. results/REWARD_ARMS.md section 5 reports a ~15pp jump in
elaboration between the SFT resume point and step 10 of every RL arm, identical
across all three rewards and therefore not attributable to any of them. Pairing
the two `gen_*.jsonl` files on row index shows 500/760 rows have BYTE-IDENTICAL
completions, and 96 of those 500 (19.2%) got a different `typecheck` verdict in
the two jobs -- 91 in one direction and 5 in the other. An asymmetry that large
is not REPL flakiness. The candidate cause is the one CLAUDE.md already warns
about: `hpc/job_prelude.sh` defaults MATHLIB_TAR to the v4.8.0-rc1 tar, LoCoLib
needs v4.23, and a job that forgets the override stages the wrong Mathlib and
silently fails to elaborate anything that needs a newer module.

Run it twice, once per environment, on the SAME file. If the rate follows the
environment rather than the file, the jump is a scoring artifact.

NO GPU. Generation already happened; this is Lean only.

    # under whichever Mathlib the job stages
    python scripts/eval/rescore_gen.py --gen <gen_*.jsonl> --workers 8 \\
        --out results/rescore/<tag>.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_W = None
_STANDALONE = False


def _init(standalone: bool = False):
    global _W, _STANDALONE
    from reward.beq_plus import BEqPlusScorer
    _W = BEqPlusScorer()
    _STANDALONE = standalone
    print(f"[rescore] worker {os.getpid()} ready (standalone={standalone})", flush=True)


def _score(row: dict) -> dict:
    """One row, scored exactly the way evaluate_checkpoints.py scored it.

    THE MODE MUST MATCH THE ORIGINAL EVAL. `score()` strips the prediction's own
    header and scores it against the gold's context; `score_standalone()` keeps
    it and builds a union context. A model that emits a self-contained snippet
    (its own `import`s) scored through `score()` collapses to ~0%: MEASURED, the
    distilled series read 48.95% -> 0.39% and minif2f 18.9% -> 0.4% that way.
    That looks exactly like a real regression, which is why the mode is recorded
    in the output rather than assumed. `--standalone` mirrors
    `evaluate_checkpoints.py`'s flag of the same name.
    """
    try:
        r = (_W.score_standalone(row["gold"], row["pred"]) if _STANDALONE
             else _W.score(row["gold"], row["pred"]))
    except Exception as e:
        return {"i": row["i"], "typecheck": False, "beq_plus": False,
                "gold_implies_pred": False, "pred_implies_gold": False,
                "semantic_signal": 0,
                "error_kind": type(e).__name__, "error": str(e)[:300]}
    # THE SAME PER-EXAMPLE FIELDS `evaluate_checkpoints.py` STORES. Capturing
    # only typecheck/beq_plus produced records that carried the headline rates
    # but had lost the DIRECTIONAL detail, so `weaker_only_rate` could not be
    # re-derived and the outcome-breakdown figure silently showed a 0% band.
    # A rebuilt record has to be a drop-in replacement for the one it supersedes.
    return {"i": row["i"], "typecheck": bool(r.get("typecheck")),
            "beq_plus": bool(r.get("beq_plus")),
            "gold_implies_pred": bool(r.get("gold_implies_pred", False)),
            "pred_implies_gold": bool(r.get("pred_implies_gold", False)),
            "semantic_signal": int(r.get("semantic_signal", 0) or 0),
            "error_kind": r.get("error_kind"),
            "error": (r.get("error") or "")[:300]}


def _toolchain() -> str | None:
    """The Lean version MATHLIB_ROOT pins, or None if it cannot be read."""
    root = os.environ.get("MATHLIB_ROOT")
    f = Path(root, "lean-toolchain") if root else None
    try:
        return f.read_text().strip() if f and f.is_file() else None
    except Exception:
        return None


def _summarise(gen: Path, out_rows: list[dict], toolchain: str | None) -> dict:
    n = len(out_rows)
    n_tc = sum(r["typecheck"] for r in out_rows)
    n_beq = sum(r["beq_plus"] for r in out_rows)
    n_err = sum(bool(r["error_kind"]) for r in out_rows)
    n_weak = sum(1 for r in out_rows if r.get("gold_implies_pred") and not r["beq_plus"])
    return {"gen": str(gen), "n": n, "toolchain": toolchain,
            "weaker_only_rate": n_weak / n if n else 0.0,
            "standalone": bool(os.environ.get("RESCORE_STANDALONE")),
            "env": {k: os.environ.get(k) for k in
                    ("MATHLIB_ROOT", "MATHLIB_TAR", "LEAN_INTERACT_CACHE_DIR",
                     "SLURM_JOB_ID")},
            "typecheck_rate": n_tc / n if n else 0.0,
            "beq_plus_rate": n_beq / n if n else 0.0,
            "scorer_error_rate": n_err / n if n else 0.0,
            "per_example": out_rows}


def run_batch(args, toolchain: str | None) -> None:
    """Score every gen file in this chunk through ONE worker pool.

    THE POOL IS BUILT ONCE. Each worker imports Mathlib and holds a ~4.3GB REPL,
    which dominates the cost of a small file: a 244-row eval spends minutes on
    startup to do a couple of minutes of work. Files are handed to the same pool
    one after another, so that startup is paid once per CHUNK instead of once
    per FILE. Measured before this change: 318 tasks, each staging a 5.4GB tar
    and building 8 REPLs, for a median file of ~400 rows.

    A FILE THAT FAILS DOES NOT KILL THE CHUNK. `imap_unordered` with a timeout
    can raise on a wedged worker (that is why the timeout exists), and losing 12
    good files because the 13th hung would be worse than skipping it.
    """
    paths = [Path(l.strip()) for l in args.manifest.open() if l.strip()]
    mine = paths[args.chunk::args.num_chunks]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rescore] chunk {args.chunk}/{args.num_chunks}: {len(mine)} files, "
          f"toolchain={toolchain}", flush=True)

    todo = []
    for g in mine:
        label = g.name[len("gen_"):-len(".jsonl")]
        out = args.out_dir / f"{label}.{args.tag}.json"
        if out.exists():
            print(f"[rescore] cached: {out.name}", flush=True)
            continue
        todo.append((g, out))
    if not todo:
        print("[rescore] nothing to do", flush=True)
        return

    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init,
                                     initargs=(args.standalone,)) as pool:
        for k, (g, out) in enumerate(todo, 1):
            rows = [json.loads(l) for l in g.open() if l.strip()]
            if args.n:
                rows = rows[: args.n]
            print(f"[rescore] ({k}/{len(todo)}) {g.name}: {len(rows)} rows", flush=True)
            got, t1 = [], time.time()
            try:
                it = pool.imap_unordered(_score, rows)
                for i in range(len(rows)):
                    got.append(it.next(timeout=args.result_timeout))
                    if (i + 1) % 200 == 0:
                        print(f"[rescore]   {i+1}/{len(rows)} "
                              f"{(time.time()-t1)/(i+1):.2f}s/ex", flush=True)
            except Exception as e:
                print(f"[rescore] SKIPPING {g.name}: {type(e).__name__} after "
                      f"{len(got)}/{len(rows)} rows", flush=True)
                continue
            got.sort(key=lambda r: r["i"])
            out.write_text(json.dumps(_summarise(g, got, toolchain), indent=2))
            print(f"[rescore] -> {out.name} "
                  f"({(time.time()-t1)/max(len(rows),1):.2f}s/ex)", flush=True)
    print(f"[rescore] chunk done in {(time.time()-t0)/60:.1f} min", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, help="one gen_*.jsonl")
    # BATCHED MODE. One task, many files, ONE worker pool. Each worker holds a
    # ~4.3GB Mathlib REPL that takes minutes to build, so paying that per FILE
    # (318 times, for files as small as 244 rows) costs far more than the
    # scoring itself. Slicing a manifest instead amortises it over a chunk.
    ap.add_argument("--manifest", type=Path, help="file of gen paths, one per line")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--num-chunks", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, help="batched mode: where to write")
    ap.add_argument("--tag", default="v423")
    ap.add_argument("--n", type=int, default=0, help="0 = the whole file")
    ap.add_argument("--workers", type=int, default=8)
    # Each worker holds its own ~4.3GB Mathlib REPL, so --result-timeout is not
    # optional: a Pool whose worker dies holding a task hangs forever, and that
    # has cost this project ~9 GPU-hours before.
    ap.add_argument("--result-timeout", type=float, default=900.0)
    ap.add_argument("--out", type=Path)
    # A rescore whose whole purpose is to pin the toolchain must not run under
    # an unverified one. `stage_mathlib` returns 0 even when it did nothing, and
    # the caller then falls back to the shared checkout, which is v4.8 -- that
    # silently produced three outputs named `.v423.json` computed under v4.8.
    ap.add_argument("--standalone", action="store_true",
                    help="score via BEqPlusScorer.score_standalone, for series "
                         "whose predictions carry their own imports")
    ap.add_argument("--require-toolchain", default="",
                    help="abort unless MATHLIB_ROOT pins exactly this")
    args = ap.parse_args()

    toolchain = _toolchain()
    if args.require_toolchain and toolchain != args.require_toolchain:
        raise SystemExit(
            f"[rescore] REFUSING TO RUN: MATHLIB_ROOT={os.environ.get('MATHLIB_ROOT')} "
            f"pins {toolchain!r}, not the required {args.require_toolchain!r}. "
            "Staging almost certainly failed and fell back to the shared checkout; "
            "scoring here would mislabel the output.")

    os.environ["RESCORE_STANDALONE"] = "1" if args.standalone else ""
    if args.manifest:
        return run_batch(args, toolchain)
    if not args.gen or not args.out:
        raise SystemExit("[rescore] need --gen and --out, or --manifest and --out-dir")
    rows = [json.loads(l) for l in args.gen.open() if l.strip()]
    if args.n:
        rows = rows[: args.n]
    print(f"[rescore] {len(rows)} rows from {args.gen.name}", flush=True)
    # Recorded so the output says WHICH environment produced it. Without this
    # the file is just another set of rates and the whole comparison is lost.
    env = {k: os.environ.get(k) for k in
           ("MATHLIB_ROOT", "MATHLIB_TAR", "MATHLIB_TAR_FLAT",
            "LEAN_INTERACT_CACHE_DIR", "SLURM_JOB_ID")}
    print(f"[rescore] MATHLIB_ROOT={os.environ.get('MATHLIB_ROOT')} "
          f"toolchain={toolchain}", flush=True)

    t0, out = time.time(), []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init,
                                     initargs=(args.standalone,)) as pool:
        it = pool.imap_unordered(_score, rows)
        for k in range(len(rows)):
            out.append(it.next(timeout=args.result_timeout))
            if (k + 1) % 50 == 0:
                print(f"[rescore] {k+1}/{len(rows)} "
                      f"{(time.time()-t0)/(k+1):.2f}s/ex", flush=True)

    out.sort(key=lambda r: r["i"])
    n = len(out)
    n_tc = sum(r["typecheck"] for r in out)
    n_beq = sum(r["beq_plus"] for r in out)
    n_err = sum(bool(r["error_kind"]) for r in out)
    n_weak = sum(1 for r in out if r.get("gold_implies_pred") and not r["beq_plus"])
    rec = {"gen": str(args.gen), "n": n, "toolchain": toolchain,
           "weaker_only_rate": n_weak / n if n else 0.0,
           "standalone": args.standalone, "env": env,
           "typecheck_rate": n_tc / n, "beq_plus_rate": n_beq / n,
           "scorer_error_rate": n_err / n, "per_example": out}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2))
    print(f"[rescore] toolchain={toolchain}  typecheck {n_tc}/{n} "
          f"({100*n_tc/n:.1f}%)  beq_plus {n_beq}/{n} ({100*n_beq/n:.1f}%)  "
          f"errors {n_err}", flush=True)
    print(f"[rescore] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
