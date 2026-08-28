"""One place for the "pool of Lean workers" pattern.

Every offline Lean pass in this repo has the same shape: fork N processes, give
each its own `BEqPlusScorer` (so each pays the ~4.3GB Mathlib import once), and
stream results back. That shape had been copy-pasted into `score_rollouts.py` and
`probe_locolib_elab.py`, and the copies had already
drifted apart in four ways that all cost data:

  - fork instead of spawn,
  - no per-result timeout conversion, so `--result-timeout 0` aborted on the
    FIRST result rather than waiting forever,
  - bare `except Exception` around `it.next`, which swallows real worker errors
    as if they were stalls,
  - results accumulated in memory and written once at the end, so a walltime
    kill lost the whole pass.

The last three are exactly the failure modes the drain loop exists to prevent.
"""
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: F401  -- repo root + every stage folder

_scorer = None


def init_worker(memory_limit_mb: int, env: dict, kwargs: dict) -> None:
    """Pool initializer. Builds this worker's scorer and reports it ready.

    `env` goes into os.environ BEFORE `reward.beq_plus` is imported -- the BEQ_*
    knobs are read at import time, which is why the import is inside the
    function and not at module scope.
    """
    global _scorer
    os.environ["BEQ_MEMORY_LIMIT_MB"] = str(memory_limit_mb)
    for k, v in (env or {}).items():
        os.environ[k] = str(v)
    from reward.beq_plus import BEqPlusScorer

    _scorer = BEqPlusScorer(memory_hard_limit_mb=memory_limit_mb, **(kwargs or {}))
    print(f"[leanpool] worker {os.getpid()} ready (Mathlib loaded)", flush=True)


def scorer():
    """This worker's scorer. Only valid inside a pool built by `run()`."""
    if _scorer is None:
        raise RuntimeError("leanpool.scorer() called outside a leanpool worker")
    return _scorer


def run(fn, todo, *, workers: int, memory_limit_mb: int = 8000,
        env: dict | None = None, scorer_kwargs: dict | None = None,
        result_timeout: float = 900, out_path=None, on_result=None,
        tag: str = "leanpool", report_every: int = 100) -> list:
    """Map `fn` over `todo` across `workers` Lean processes, streaming results.

    Results are appended to `out_path` as JSONL and flushed per result, because
    these passes run for hours and a walltime kill must not cost the work
    already done. `on_result(i, res)` may return a short progress string.

    DO NOT go back to a plain `for res in pool.imap_unordered(...)`. A worker
    that dies (the memory cap kills them on purpose) is replaced by the Pool,
    but the task it was holding is LOST, and imap_unordered then blocks forever
    waiting for a result that will never arrive. Measured cost: three jobs
    (score_pool slice 1, both passk runs) each finished their real work and then
    sat hung until walltime -- ~9 GPU-hours, and because they exited TIMEOUT
    rather than COMPLETED, every `afterok` behind them was cancelled. Each file
    was short by exactly ONE rollout of ~7,744.

    Driving the iterator by hand with a per-result timeout turns a silent
    multi-hour hang into a clean exit that keeps 99.99% of the data. The timeout
    is per RESULT, not per rollout, and results stream in from `workers` in
    parallel, so 15 min is far above the worst realistic wait (18 Lean calls x
    30s = 9 min for a single pair).
    """
    # 0/negative means "wait forever". Passing timeout=0 straight to `it.next`
    # would abort on the FIRST result, i.e. the exact opposite of that.
    timeout = result_timeout if result_timeout and result_timeout > 0 else None

    # spawn, not fork: a forked worker inherits the parent's partially built
    # state, and the Lean REPL does not survive it.
    ctx = mp.get_context("spawn")
    t0 = time.time()
    results = []
    fh = open(out_path, "a") if out_path else None
    try:
        with ctx.Pool(workers, initializer=init_worker,
                      initargs=(memory_limit_mb, env, scorer_kwargs)) as pool:
            it = pool.imap_unordered(fn, todo, chunksize=1)
            i = 0
            while True:
                try:
                    res = it.next(timeout=timeout)
                except StopIteration:
                    break
                except mp.TimeoutError:
                    print(f"[{tag}] STALLED: no result for {timeout}s after "
                          f"{i}/{len(todo)}. A worker almost certainly died holding a "
                          f"task; its item is lost. Keeping what we have and exiting "
                          f"cleanly.", flush=True)
                    break
                i += 1
                results.append(res)
                if fh is not None:
                    fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                    fh.flush()  # the pass is hours long; never buffer results away
                extra = on_result(i, res) if on_result else ""
                if i % report_every == 0 or i == len(todo):
                    el = time.time() - t0
                    rate = i / el if el else 0.0
                    eta = (len(todo) - i) / rate / 60 if rate else 0.0
                    print(f"[{tag}] {i}/{len(todo)}  {extra}  "
                          f"{rate:.2f}/s  ETA {eta:.0f}min", flush=True)
    finally:
        if fh is not None:
            fh.close()
    print(f"[{tag}] done in {(time.time()-t0)/60:.1f}min, {len(results)} results")
    return results


def resume_done(out_path, key=("prompt_index", "sample_index")) -> set:
    """Keys already present in an output JSONL, so a rerun skips them."""
    done = set()
    p = Path(out_path)
    if not p.exists():
        return done
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            done.add(tuple(d[k] for k in key))
        except Exception:
            continue
    return done
