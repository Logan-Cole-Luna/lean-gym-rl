#!/usr/bin/env python3
"""Measure what the BEq+ reward actually fires on, against human labels.

BEq+ is the semantic gate every arm in `reward/reward_fn.py` is built on, but
nothing in this repo has ever measured its error rate. `PAug/ProofNetVerif`
(released with BEq+, Poiroux et al. EMNLP 2025) is 3752 (gold, prediction)
pairs carrying a HUMAN `correct` label, which makes it the one place that
number can be established.

Two numbers matter, and they are not symmetric:

    FALSE POSITIVE RATE   beq_plus fires where the human says the prediction is
                          wrong. This is the reward-hacking surface: every FP is
                          a full 1.0 (gated) or a jump to the 0.50/0.85 rungs
                          (outcome) paid for a wrong answer, and the policy will
                          find them.
    RECALL                beq_plus misses where the human says correct. Costly
                          but honest -- it flattens the reward and slows
                          learning, it does not teach the wrong thing.

ProofNetVerif is STATEMENT-level: every `lean4_prediction` is `sorry`d, so this
calibrates the BEq+ half of the ladder only. It says nothing about
`check_own_proof` / the `proved` column.

THE GOLD GATE RUNS FIRST AND IS NOT OPTIONAL. If the gold statements do not
elaborate against the local Mathlib, every cascade fails for that reason and the
"false negatives" measured are Mathlib drift, not BEq+ behaviour. `--gate-only`
runs just that check; the full run aborts below `--min-gold-elab`.

    # is this Mathlib even compatible? ~1 min
    python scripts/eval/beq_calibration.py --gate-only --n 200

    # the real run, balanced on the human label
    python scripts/eval/beq_calibration.py --n 600 --out results/beq_calibration

    # shard across hosts
    python scripts/eval/beq_calibration.py --shard 0 --num-shards 8 --out ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reward.beq_plus import BEqPlusScorer, split_header_and_theorem


def load_rows(splits: tuple[str, ...]) -> list[dict]:
    """ProofNetVerif rows. `datasets` is already a requirement of this repo."""
    from datasets import load_dataset
    ds = load_dataset("PAug/ProofNetVerif")
    rows = []
    for split in splits:
        for r in ds[split]:
            rows.append({"id": r["id"], "split": split,
                         "header": r["lean4_src_header"],
                         "gold": r["lean4_formalization"],
                         "pred": r["lean4_prediction"],
                         "correct": bool(r["correct"])})
    return rows


def stratified(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Half true / half false, and never two predictions for the same problem
    within a label. ProofNetVerif carries ~10.4 predictions per problem, so an
    unstratified draw would sample a handful of problems many times over and
    report a confidence interval far tighter than the data supports."""
    rng = random.Random(seed)
    out = []
    for label in (True, False):
        pool = [r for r in rows if r["correct"] is label]
        rng.shuffle(pool)
        seen, keep = set(), []
        for r in pool:                      # one prediction per problem first
            if r["id"] not in seen:
                seen.add(r["id"])
                keep.append(r)
        keep += [r for r in pool if r not in keep]   # then top up if n demands
        out += keep[: n // 2]
    rng.shuffle(out)
    return out


def gold_gate(scorer: BEqPlusScorer, rows: list[dict], limit: int) -> dict:
    """Do the GOLD statements elaborate here? Cheap, and decides everything.

    Gates EVERY distinct problem, not a sample: one elaboration is milliseconds
    against a cascade's seconds, and the per-problem verdict is needed later to
    separate BEq+'s own misses from Mathlib drift. A pair whose GOLD does not
    elaborate cannot possibly prove in either direction, so it is a guaranteed
    false negative for a reason that has nothing to do with BEq+.
    """
    seen, ok_ids, bad_ids, lat = set(), set(), set(), []
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        ctx, thm = split_header_and_theorem(r["header"] + "\n" + r["gold"])
        t0 = time.perf_counter()
        try:
            good, _err = scorer.typecheck_ex(thm, ctx)
        except Exception:
            good = False
        lat.append(time.perf_counter() - t0)
        (ok_ids if good else bad_ids).add(r["id"])
        if len(seen) >= limit:
            break
    lat.sort()
    return {"checked": len(seen), "elaborated": len(ok_ids),
            "rate": len(ok_ids) / max(len(seen), 1),
            "median_s": lat[len(lat) // 2] if lat else 0.0,
            "ok_ids": sorted(ok_ids), "bad_ids": sorted(bad_ids)}


def run(scorer: BEqPlusScorer, rows: list[dict], out_jsonl: Path,
        max_seconds: float | None = None) -> list[dict]:
    """Score pairs, appending each verdict to `out_jsonl` as it lands.

    Incremental because a full cascade can take minutes: the run must be
    interruptible, resumable and shardable without losing what it already paid
    Lean for. `max_seconds` stops cleanly on a budget rather than being killed,
    so a partial sample is still a usable -- and still label-balanced -- sample.
    """
    recs, t_start = [], time.time()
    fh = out_jsonl.open("a")
    for i, r in enumerate(rows, 1):
        gold_full = r["header"] + "\n" + r["gold"]
        t0 = time.perf_counter()
        try:
            s = scorer.score(gold_full, r["pred"])
        except Exception as e:                 # never lose the whole run to one row
            s = {"error_kind": type(e).__name__, "beq_plus": False, "typecheck": False}
        dt = time.perf_counter() - t0
        recs.append({
            # Identifies the (problem, prediction) PAIR, not the problem:
            # ProofNetVerif carries ~10.4 predictions per problem, so `id`
            # alone cannot dedup a merge. Stable across runs, unlike wall-clock.
            "pair": hashlib.sha1((r["id"] + "\x00" + r["pred"]).encode()).hexdigest()[:16],
            "id": r["id"], "split": r["split"], "human": r["correct"],
            "beq_plus": bool(s.get("beq_plus")), "typecheck": bool(s.get("typecheck")),
            "n_directions": int(s.get("n_directions") or 0),
            "gold_implies_pred": bool(s.get("gold_implies_pred")),
            "pred_implies_gold": bool(s.get("pred_implies_gold")),
            "rung": int(s.get("rung") or 0),
            "gold_elaborates": r.get("gold_elaborates", True),
            "convert_level": int(s.get("convert_level") or 0),
            "error_kind": s.get("error_kind"), "seconds": dt,
        })
        fh.write(json.dumps(recs[-1]) + "\n")
        fh.flush()
        if i % 10 == 0 or i == len(rows):
            fp = sum(x["beq_plus"] and not x["human"] for x in recs)
            tp = sum(x["beq_plus"] and x["human"] for x in recs)
            print(f"[calib] {i}/{len(rows)}  tp={tp} fp={fp}  "
                  f"{sum(x['seconds'] for x in recs)/i:.2f}s/pair  "
                  f"elapsed {(time.time()-t_start)/60:.1f}m", flush=True)
        if max_seconds and time.time() - t_start > max_seconds:
            print(f"[calib] budget of {max_seconds}s reached at {i}/{len(rows)}; "
                  f"reporting on what completed", flush=True)
            break
    fh.close()
    return recs


def report(recs: list[dict]) -> dict:
    """Metrics over whatever records are passed. Called twice: once on the whole
    sample, once on the gold-elaborating subset (see `gold_gate`)."""
    pos = [r for r in recs if r["human"]]
    neg = [r for r in recs if not r["human"]]
    tp = sum(r["beq_plus"] for r in pos)
    fp = sum(r["beq_plus"] for r in neg)
    fn, tn = len(pos) - tp, len(neg) - fp
    prec = tp / max(tp + fp, 1)
    rec = tp / max(len(pos), 1)
    lat = sorted(r["seconds"] for r in recs)
    q = lambda p: lat[int(p * (len(lat) - 1))] if lat else 0.0
    return {
        "n": len(recs), "n_pos": len(pos), "n_neg": len(neg),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "false_positive_rate": fp / max(len(neg), 1),
        "accuracy": (tp + tn) / max(len(recs), 1),
        # A one-direction hit is the `gated` arm's 0.25 step, so its own
        # precision is a separate reward-design question from BEq+'s.
        "one_dir_on_neg": sum(r["n_directions"] == 1 for r in neg) / max(len(neg), 1),
        "one_dir_on_pos": sum(r["n_directions"] == 1 for r in pos) / max(len(pos), 1),
        "typecheck_rate": sum(r["typecheck"] for r in recs) / max(len(recs), 1),
        "scorer_error_rate": sum(bool(r["error_kind"]) for r in recs) / max(len(recs), 1),
        "latency": {"mean": sum(lat) / max(len(lat), 1), "p50": q(.5),
                    "p90": q(.9), "p99": q(.99), "max": lat[-1] if lat else 0.0},
        # Which cascade rung produced each false positive. If FPs concentrate in
        # rung 4 (`convert ... using k`), capping convert_level is a one-line
        # hardening with a measured justification behind it.
        "rung_hist_fp": dict(Counter(r["rung"] for r in neg if r["beq_plus"])),
        "rung_hist_tp": dict(Counter(r["rung"] for r in pos if r["beq_plus"])),
        "convert_level_fp": dict(Counter(r["convert_level"] for r in neg
                                         if r["beq_plus"] and r["rung"] == 4)),
    }


def merge(out: Path) -> dict:
    """Combine every `records.shard*.jsonl` under `out` into one report.

    Sharding without this is a trap: each shard writes a report over its own
    ~1/N of the sample, and reading one of those as "the" false-positive rate
    silently divides the evidence by N.
    """
    recs, seen, dupes = [], set(), 0
    # `records.shard*.jsonl` ONLY. A broader glob would sweep in any other
    # records file sitting in the same directory -- a partial run from a
    # different --num-shards, a serial latency run -- and those cover
    # overlapping slices of the same stratified sample, so the merged counts
    # would silently double-weight whatever both touched.
    for f in sorted(out.glob("records.shard*.jsonl")):
        for line in f.open():
            if not line.strip():
                continue
            r = json.loads(line)
            # Shards partition the sample, so a duplicate means two runs with
            # different --num-shards landed here. Drop it rather than
            # double-weight that pair. Older records predate the `pair` key and
            # fall back to (id, prediction-free) identity, which CANNOT
            # distinguish two predictions for one problem -- rerun them.
            key = r.get("pair") or (r["id"], r["split"], r["seconds"])
            if key in seen:
                dupes += 1
                continue
            seen.add(key)
            recs.append(r)
    if not recs:
        raise SystemExit(f"[calib] no records.shard*.jsonl under {out}")
    rep = report(recs)
    clean = [r for r in recs if r.get("gold_elaborates", True)]
    rep["gold_elaborating_only"] = report(clean) if clean else None
    rep["merged_from"] = sorted(f.name for f in out.glob("records.shard*.jsonl"))
    rep["duplicates_dropped"] = dupes
    rep["unkeyed_records"] = sum(1 for r in recs if not r.get("pair"))
    # Latency is per-pair wall clock, so it only means anything alongside the
    # number of REPLs that were competing. Prefer what a shard recorded; fall
    # back to the shard COUNT, never to 1 -- mislabelling a 5-REPL run as serial
    # would understate the reward loop's real throughput by the contention factor.
    shards = [json.loads(f.read_text()) for f in out.glob("report.shard*.json")]
    rep["concurrency"] = max([s.get("concurrency") for s in shards
                              if s.get("concurrency")] or [len(rep["merged_from"])])
    gates = [s["gold_gate"] for s in shards if s.get("gold_gate")]
    if gates:
        rep["gold_gate"] = max(gates, key=lambda g: g["checked"])
    rep["config"] = next((s["config"] for s in shards if s.get("config")), None)
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="pairs to score, balanced on the human label")
    ap.add_argument("--splits", default="valid,test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/beq_calibration"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="combine every records*.jsonl under --out into one "
                         "report.json and exit; no Lean, no scoring")
    ap.add_argument("--gate-n", type=int, default=100)
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="wall-clock budget for the scoring phase; stop cleanly "
                         "and report on the pairs that completed")
    ap.add_argument("--min-gold-elab", type=float, default=0.90,
                    help="abort if fewer gold statements than this elaborate")
    args = ap.parse_args()

    if args.merge:
        rep = merge(args.out)
        c = rep.get("gold_elaborating_only") or rep
        print(f"[calib] merged {rep['n']} pairs from {len(rep['merged_from'])} shard files"
              + (f", dropped {rep['duplicates_dropped']} duplicates" if rep['duplicates_dropped'] else "")
              + (f"  [WARNING: {rep['unkeyed_records']} records predate the `pair` key "
                 f"and could not be deduped reliably]" if rep['unkeyed_records'] else ""))
        print(f"  precision {c['precision']:.3f}  recall {c['recall']:.3f}  "
              f"FPR {c['false_positive_rate']:.3f}  -> {args.out / 'report.json'}")
        return

    rows = load_rows(tuple(args.splits.split(",")))
    print(f"[calib] {len(rows)} rows, {len({r['id'] for r in rows})} problems, "
          f"{sum(r['correct'] for r in rows)} labelled correct", flush=True)

    scorer = BEqPlusScorer()
    gate = gold_gate(scorer, rows, args.gate_n)
    print(f"[calib] GOLD GATE: {gate['elaborated']}/{gate['checked']} elaborate "
          f"({100*gate['rate']:.1f}%), median {gate['median_s']:.2f}s", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gold_gate.json").write_text(json.dumps(gate, indent=2))
    if args.gate_only:
        return
    if gate["rate"] < args.min_gold_elab:
        raise SystemExit(
            f"[calib] ABORT: only {100*gate['rate']:.1f}% of GOLD statements elaborate "
            f"against this Mathlib. Every cascade would fail for that reason, and the "
            f"result would measure Mathlib drift, not BEq+. Check MATHLIB4_TAG.")

    ok_ids = set(gate["ok_ids"])
    for r in rows:
        r["gold_elaborates"] = r["id"] in ok_ids
    sample = stratified(rows, args.n, args.seed)
    if args.num_shards > 1:
        sample = sample[args.shard::args.num_shards]
        print(f"[calib] shard {args.shard}/{args.num_shards}: {len(sample)} pairs", flush=True)

    t0 = time.time()
    tag = f".shard{args.shard}" if args.num_shards > 1 else ""
    recs = run(scorer, sample, args.out / f"records{tag}.jsonl", args.max_seconds)
    rep = report(recs)
    # The headline number. A pair whose gold does not elaborate is a guaranteed
    # miss for a Mathlib-version reason, and folding those into "recall" would
    # blame BEq+ for dataset drift.
    clean = [r for r in recs if r["gold_elaborates"]]
    rep["gold_elaborating_only"] = report(clean) if clean else None
    rep["wall_seconds"] = time.time() - t0
    rep["scorer_stats"] = dict(scorer.stats)
    rep["gold_gate"] = gate
    rep["config"] = {k: os.environ.get(k) for k in
                     ("BEQ_PROBE_STRONGER", "BEQ_TIMEOUT_PER_PROOF",
                      "BEQ_SKIP_CASCADE_ON_TYPECHECK_FAIL", "BEQ_HAVE_GUARD_PATCH")}
    # Latency is per-pair WALL CLOCK, so it is inflated by however many REPLs
    # were competing for the box. Record the count, or the number reads as a
    # single-REPL figure and understates the reward loop's real throughput.
    rep["concurrency"] = args.num_shards

    (args.out / f"report{tag}.json").write_text(json.dumps(rep, indent=2))

    print("\n" + "=" * 62)
    print(f"  BEq+ vs human labels   n={rep['n']}  ({rep['n_pos']}+ / {rep['n_neg']}-)")
    print("=" * 62)
    print(f"  precision {rep['precision']:.3f}   recall {rep['recall']:.3f}   f1 {rep['f1']:.3f}")
    c = rep.get("gold_elaborating_only")
    if c:
        print(f"  gold elaborates only (n={c['n']}): precision {c['precision']:.3f}  "
              f"recall {c['recall']:.3f}  f1 {c['f1']:.3f}  FPR {c['false_positive_rate']:.3f}")
    print(f"  FALSE POSITIVE RATE   {rep['false_positive_rate']:.3f}   <- the hacking surface")
    print(f"  confusion   tp={rep['tp']} fp={rep['fp']} fn={rep['fn']} tn={rep['tn']}")
    print(f"  latency  mean {rep['latency']['mean']:.2f}s  p50 {rep['latency']['p50']:.2f}s  "
          f"p90 {rep['latency']['p90']:.2f}s  p99 {rep['latency']['p99']:.2f}s")
    print(f"  rung of true positives  {rep['rung_hist_tp']}")
    print(f"  rung of FALSE positives {rep['rung_hist_fp']}")
    print(f"  wall {rep['wall_seconds']/60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
