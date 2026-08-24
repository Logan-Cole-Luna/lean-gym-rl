#!/usr/bin/env python3
"""Score generated rollouts with BEq+ in a pool of parallel Lean REPLs.

Why a separate script with its own process pool: during TRAINING, Lean scoring
is pinned to one REPL per agent-loop worker and serialised
(reward/reward_fn.py's BEQ_MAX_CONCURRENT=1 is a thread-safety requirement, and
verl's FSDP param/optimizer offload already occupies ~37-42GB of host RAM, so
there is no room for more Mathlib instances). OFFLINE there is no FSDP and no
vLLM on the host, so the same work parallelises across processes and the whole
pass runs in a fraction of the time.

Each worker builds its OWN BEqPlusScorer (own Lean REPL, own Mathlib env), so
there is no shared state and no lean_interact session race. Budget ~4.3GB
resident per worker plus headroom for `exact?` searches, which is why
--workers defaults to 4 and BEQ_MEMORY_LIMIT_MB is capped per process.

Resumable by design: the output is append-only JSONL keyed by
(prompt_index, sample_index), and a rerun skips keys already present. A pass
over ~10k rollouts takes hours, and it must survive an interruption.

Usage:
    python scripts/score_rollouts.py \
        --rollouts data/rollouts/sft390_k8.jsonl \
        --out data/rollouts/sft390_k8.scored.jsonl --workers 4
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_scorer = None


def _init_worker(memory_limit_mb: int, timeout_per_proof: int, probe_stronger: bool) -> None:
    global _scorer
    os.environ["BEQ_MEMORY_LIMIT_MB"] = str(memory_limit_mb)
    os.environ["BEQ_TIMEOUT_PER_PROOF"] = str(timeout_per_proof)
    if probe_stronger:
        os.environ["BEQ_PROBE_STRONGER"] = "1"
    from reward.beq_plus import BEqPlusScorer

    _scorer = BEqPlusScorer(memory_hard_limit_mb=memory_limit_mb,
                            timeout_per_proof=timeout_per_proof)
    print(f"[score] worker {os.getpid()} ready (Mathlib loaded)", flush=True)


def _score_one(rec: dict) -> dict:
    from reward.reward_fn import _clean_solution

    pred = _clean_solution(rec["completion"])
    try:
        r = _scorer.score(rec["gold"], pred)
    except Exception as exc:  # a dead REPL must not kill the whole pass
        return {**{k: rec[k] for k in ("prompt_index", "sample_index")},
                "pred": pred, "typecheck": False, "beq_plus": False,
                "gold_implies_pred": False, "pred_implies_gold": False,
                "semantic_signal": 0, "error_kind": f"worker:{type(exc).__name__}"}
    return {
        "prompt_index": rec["prompt_index"], "sample_index": rec["sample_index"],
        "pred": pred,
        "typecheck": bool(r["typecheck"]), "beq_plus": bool(r["beq_plus"]),
        "gold_implies_pred": bool(r.get("gold_implies_pred", False)),
        "pred_implies_gold": bool(r.get("pred_implies_gold", False)),
        "semantic_signal": int(r.get("semantic_signal", 0)),
        "error_kind": r.get("error_kind"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--memory-limit-mb", type=int, default=6000,
                    help="per-REPL hard cap; workers*this must fit in host RAM")
    ap.add_argument("--timeout-per-proof", type=int, default=30)
    ap.add_argument("--probe-stronger", action="store_true",
                    help="also test pred=>gold alone (see beq_plus.py DIRECTION SEMANTICS)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--result-timeout", type=float, default=900,
                    help="seconds to wait for any single result before assuming a worker "
                         "died holding its task (see the loop below); 0 disables")
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.rollouts).read_text().splitlines() if l.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[int, int]] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                done.add((d["prompt_index"], d["sample_index"]))
            except Exception:
                continue
        print(f"[score] resuming: {len(done)} already scored")

    todo = [r for r in records if (r["prompt_index"], r["sample_index"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[score] {len(todo)} rollouts to score with {args.workers} Lean workers "
          f"(~{args.memory_limit_mb}MB cap each)")
    if not todo:
        print("[score] nothing to do")
        return

    ctx = mp.get_context("spawn")
    t0 = time.time()
    n_tc = n_beq = n_err = 0
    with out_path.open("a") as fh, ctx.Pool(
        args.workers, initializer=_init_worker,
        initargs=(args.memory_limit_mb, args.timeout_per_proof, args.probe_stronger),
    ) as pool:
        # DO NOT go back to a plain `for res in pool.imap_unordered(...)`.
        # A worker that dies (the memory cap kills them) is replaced by the Pool,
        # but the task it was holding is LOST, and imap_unordered then blocks
        # forever waiting for a result that will never arrive. Measured cost of
        # that: three jobs (score_pool slice 1, both passk runs) each finished
        # their real work and then sat hung until the walltime killed them --
        # ~9 GPU-hours burned, and because they exited TIMEOUT rather than
        # COMPLETED every `afterok` behind them was cancelled. Each file was
        # short by exactly ONE rollout of ~7,744.
        #
        # Driving the iterator by hand with a per-result timeout turns a silent
        # multi-hour hang into a clean exit that keeps 99.99% of the data. The
        # timeout is per RESULT, not per rollout, and results stream in from
        # `--workers` in parallel, so 15 min is far above the worst realistic
        # wait (18 Lean calls x 30s = 9 min for a single pair).
        # 0/negative means "wait forever". Passing timeout=0 straight through
        # would abort on the FIRST result, i.e. the exact opposite of the
        # documented behaviour.
        _result_timeout = args.result_timeout if args.result_timeout > 0 else None
        it = pool.imap_unordered(_score_one, todo, chunksize=1)
        i = 0
        while True:
            try:
                res = it.next(timeout=_result_timeout)
            except StopIteration:
                break
            except mp.TimeoutError:
                print(f"[score] STALLED: no result for {_result_timeout}s after "
                      f"{i}/{len(todo)}. A worker almost certainly died holding a task; "
                      f"its rollout is lost. Writing what we have and exiting cleanly.",
                      flush=True)
                break
            i += 1
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            fh.flush()  # the pass is hours long; never buffer results away
            n_tc += res["typecheck"]
            n_beq += res["beq_plus"]
            n_err += bool(res["error_kind"])
            if i % 100 == 0 or i == len(todo):
                el = time.time() - t0
                rate = i / el
                eta = (len(todo) - i) / rate / 60
                print(f"[score] {i}/{len(todo)}  typecheck {n_tc} ({100*n_tc/i:.1f}%)  "
                      f"beq+ {n_beq} ({100*n_beq/i:.1f}%)  err {n_err}  "
                      f"{rate:.2f}/s  ETA {eta:.0f}min", flush=True)

    print(f"[score] done in {(time.time()-t0)/60:.1f}min -> {out_path}")
    if n_err:
        print(f"[score] WARNING: {n_err} rollouts hit a Lean scorer failure and are "
              f"recorded as False; they are NOT verdicts about the model.")


if __name__ == "__main__":
    main()
