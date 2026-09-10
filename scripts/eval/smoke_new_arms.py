#!/usr/bin/env python3
"""Do the four `outcome`-family entry points survive real Lean and real rollouts?

WHY THIS RUNS BEFORE A TRAINING JOB, NOT AFTER. A reward that raises does not
lose its rollout, it loses the JOB: verl marks the agent-loop task failed, the
GRPO step never closes, and the run sits on a GPU until walltime. That has cost
this project ~10 GPU-hours once already. Every `compute_score_*` carries
`@_never_raises`, so the observable symptom of a broken new arm is not a crash --
it is a silent stream of all-zero dicts with `scorer_error=1`, which trains the
policy on noise. This checks for that directly, on CPU, for the price of a few
minutes of Lean.

WHAT IT ASSERTS, beyond "did not raise":

  * the KEY SET is identical across all four arms and across every row. verl
    aggregates each key of `reward_extra_info` into a metric, and a ragged
    schema is a mid-run failure that no single call reveals.
  * `outcome` and `outcome_brevity` return the SAME score on a row whose
    brevity is 1.0, and differ only where it is not -- the arms are meant to be
    one term apart, and this is the only place that is checked end to end.
  * the `scorer_error` rate is ~0. A high rate here means Lean is misconfigured
    (wrong Mathlib) and the training job would burn its walltime scoring zeros.
  * with NO judge reachable, the faith arms degrade to exactly `outcome`, and
    say so via `faith_known=0`. That is the documented silent-degradation mode,
    so it is asserted rather than assumed.

    python scripts/eval/smoke_new_arms.py --gen <gen_*.jsonl> --n 24
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ARMS = ("compute_score_outcome", "compute_score_outcome_brevity",
        "compute_score_outcome_faith", "compute_score_outcome_v2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", type=Path, default=Path("results/smoke_new_arms.json"))
    args = ap.parse_args()

    import reward.reward_fn as rf
    from reward.reward import OUTCOMES

    rows = [json.loads(l) for l in args.gen.open() if l.strip()][: args.n]
    print(f"[smoke] {len(rows)} rows from {args.gen.name}", flush=True)
    print(f"[smoke] W_BREVITY_BAND_ARM={rf.W_BREVITY_BAND_ARM} "
          f"W_FAITH_BAND={rf.W_FAITH_BAND} "
          f"judge={os.environ.get('FAITH_JUDGE_URL', '(default, expected down)')}",
          flush=True)

    # extra_info as the training corpus supplies it, so the faith arms take the
    # same path they will take in the loop rather than a `no_informal` shortcut.
    import pandas as pd
    val = pd.read_parquet("data_locolib/val_proof_nl.parquet")
    info = {i: dict(v) for i, v in enumerate(val["extra_info"])}

    out, keysets, t0 = [], {}, time.time()
    for k, r in enumerate(rows, 1):
        rec = {"i": r["i"]}
        for arm in ARMS:
            d = getattr(rf, arm)("locolib", r["completion"], r["gold"],
                                 info.get(r["i"]))
            keysets.setdefault(arm, set()).add(frozenset(d))
            rec[arm] = d
        out.append(rec)
        if k % 5 == 0:
            print(f"[smoke] {k}/{len(rows)} {(time.time()-t0)/k:.1f}s/row", flush=True)

    fail = []
    # 1. one key set, everywhere.
    allk = {frozenset(ks) for arm in ARMS for ks in keysets[arm]}
    if len(allk) != 1:
        fail.append(f"ragged schema: {len(allk)} distinct key sets across arms/rows")
    # 2. scorer errors
    err = sum(r["compute_score_outcome"]["scorer_error"] for r in out) / max(len(out), 1)
    if err > 0.10:
        fail.append(f"scorer_error rate {100*err:.1f}% -- Lean is likely misconfigured")
    # 3. brevity is the only difference between outcome and outcome_brevity
    for r in out:
        a, b = r["compute_score_outcome"], r["compute_score_outcome_brevity"]
        if a["brevity"] >= 1.0 and abs(a["score"] - b["score"]) > 1e-9:
            fail.append(f"row {r['i']}: brevity=1.0 but scores differ "
                        f"{a['score']} vs {b['score']}")
        if a["brevity"] < 1.0 and b["score"] > a["score"] + 1e-9:
            fail.append(f"row {r['i']}: brevity term RAISED the score")
    # 4. with no judge, the faith arms must equal their non-faith twins
    nofaith = sum(r["compute_score_outcome_faith"]["faith_known"] for r in out)
    if nofaith == 0:
        for r in out:
            if abs(r["compute_score_outcome_faith"]["score"]
                   - r["compute_score_outcome"]["score"]) > 1e-9:
                fail.append(f"row {r['i']}: no judge, yet faith arm != outcome")

    n = len(out)
    print(f"\n[smoke] {n} rows, {time.time()-t0:.0f}s total")
    print(f"[smoke] scorer_error {100*err:.1f}%   keys per call: {len(next(iter(allk)))}")
    for arm in ARMS:
        s = [r[arm]["score"] for r in out]
        print(f"  {arm:32s} mean {sum(s)/n:.4f}  min {min(s):.2f}  max {max(s):.2f}")
    hist = Counter(OUTCOMES[int(r["compute_score_outcome"]["outcome_code"])] for r in out)
    print(f"[smoke] outcomes: {dict(hist)}")
    print(f"[smoke] faith_known on {nofaith}/{n} rows "
          f"(0 expected with no judge running)")
    print(f"[smoke] brevity fired (<1.0) on "
          f"{sum(r['compute_score_outcome']['brevity'] < 1.0 for r in out)}/{n} rows")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"n": n, "scorer_error_rate": err, "failures": fail,
         "per_example": out}, indent=2))
    if fail:
        print("\n[smoke] FAILURES:")
        for f in fail[:20]:
            print("  -", f)
        raise SystemExit(1)
    print("\n[smoke] all assertions hold")


if __name__ == "__main__":
    main()
