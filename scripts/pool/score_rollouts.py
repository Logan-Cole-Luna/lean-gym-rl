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
    python scripts/pool/score_rollouts.py \
        --rollouts data/rollouts/sft390_k8.jsonl \
        --out data/rollouts/sft390_k8.scored.jsonl --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: F401  -- repo root + every stage folder

import leanpool


def _score_one(rec: dict) -> dict:
    from reward.reward_fn import _clean_solution

    pred = _clean_solution(rec["completion"])
    try:
        r = leanpool.scorer().score(rec["gold"], pred)
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
        # Cascade instrumentation (see BEqCPUResult in reward/beq_plus.py).
        # Kept as None/bool rather than the float encoding _diagnostics uses --
        # this file is read by analysis code, not by verl's metric aggregator,
        # so `provable_alone: None` ("rung 3 never ran") stays distinguishable
        # from `False` ("it ran and the statement did not prove") without
        # needing a companion *_known key.
        "beql": bool(r.get("beql", False)),
        "rung": r.get("rung"),
        "convert_level": r.get("convert_level"),
        "provable_alone": r.get("provable_alone"),
        "stop_reason": r.get("stop_reason"),
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

    done = leanpool.resume_done(out_path)
    if done:
        print(f"[score] resuming: {len(done)} already scored")

    todo = [r for r in records if (r["prompt_index"], r["sample_index"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[score] {len(todo)} rollouts to score with {args.workers} Lean workers "
          f"(~{args.memory_limit_mb}MB cap each)")
    if not todo:
        print("[score] nothing to do")
        return

    n_tc = n_beq = n_err = 0

    def progress(i, res):
        nonlocal n_tc, n_beq, n_err
        n_tc += res["typecheck"]
        n_beq += res["beq_plus"]
        n_err += bool(res["error_kind"])
        return (f"typecheck {n_tc} ({100*n_tc/i:.1f}%)  "
                f"beq+ {n_beq} ({100*n_beq/i:.1f}%)  err {n_err}")

    leanpool.run(_score_one, todo, workers=args.workers,
                 memory_limit_mb=args.memory_limit_mb,
                 env={"BEQ_TIMEOUT_PER_PROOF": args.timeout_per_proof,
                      **({"BEQ_PROBE_STRONGER": 1} if args.probe_stronger else {})},
                 scorer_kwargs={"timeout_per_proof": args.timeout_per_proof},
                 result_timeout=args.result_timeout,
                 out_path=out_path, on_result=progress, tag="score")

    if n_err:
        print(f"[score] WARNING: {n_err} rollouts hit a Lean scorer failure and are "
              f"recorded as False; they are NOT verdicts about the model.")


if __name__ == "__main__":
    main()
