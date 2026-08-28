#!/usr/bin/env python3
"""Pick the best checkpoint by BEq+, with the paired test that says whether the
difference is real.

Why this exists: the 200-step SFT->RL run saved 8 checkpoints and every one of
them scored at or below the SFT policy it started from, but that was only visible
after the fact. Two separate mistakes are easy to make here and this script
blocks both:

  1. **Selecting on the wrong metric.** verl's per-run validation number is the
     mean of that run's reward function. A reward that pays for type-check and
     structural similarity can rise while BEq+ falls -- which is exactly what
     happened. Checkpoint choice must key off BEq+ itself. (Reward functions now
     emit `acc` = BEq+ so `val-core/<data_source>/acc/mean@1` is the real metric
     during training; this script is the offline counterpart.)

  2. **Selecting on noise.** Every checkpoint is scored on the same pinned
     validation slice, so the comparison is PAIRED and McNemar's exact test
     applies. At n=400 the minimum detectable BEq+ difference is ~5.6pp; picking
     the argmax over 8 checkpoints that all sit inside the noise band is just
     overfitting the validation set.

Usage:
    python scripts/eval/select_checkpoint.py                       # vs the best SFT run
    python scripts/eval/select_checkpoint.py --baseline sft-step390
    python scripts/eval/select_checkpoint.py --metric typecheck
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"

# Produced before the chat-template fix; see scripts/misc/compare_results.py.
STALE = {"ablation_comparison.json"}
NOT_A_COMPARISON = {"rung_probe.json"}


def load_records(results_dir: Path) -> dict[str, dict]:
    """label -> record, newest file wins for duplicate labels."""
    merged: dict[str, dict] = {}
    paths = [
        p for p in results_dir.glob("eval_*.json")
        if p.name not in STALE and p.name not in NOT_A_COMPARISON
    ]
    for p in sorted(paths, key=lambda x: x.stat().st_mtime):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        for label, rec in data.items():
            if isinstance(rec, dict) and "per_example" in rec:
                rec["_file"] = p.name
                merged[label] = rec
    return merged


def mcnemar_exact(a: list[dict], b: list[dict], key: str) -> tuple[int, int, float]:
    """(lost, gained, two-sided p) for paired binary outcomes a -> b."""
    n = min(len(a), len(b))
    lost = sum(1 for i in range(n) if a[i][key] and not b[i][key])
    gained = sum(1 for i in range(n) if not a[i][key] and b[i][key])
    disc = lost + gained
    if disc == 0:
        return lost, gained, 1.0
    k = min(lost, gained)
    p = 2.0 * sum(comb(disc, i) for i in range(k + 1)) / (2 ** disc)
    return lost, gained, min(p, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--baseline", default=None,
                    help="label to compare against (default: highest-BEq+ label containing 'sft')")
    ap.add_argument("--metric", default="beq_plus", choices=["beq_plus", "typecheck"])
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    recs = load_records(Path(args.results_dir))
    if not recs:
        raise SystemExit(f"No eval_*.json records in {args.results_dir}")

    rate_key = "beq_plus_rate" if args.metric == "beq_plus" else "typecheck_rate"

    baseline = args.baseline
    if baseline is None:
        sft = {k: v for k, v in recs.items() if "sft" in k.lower() and "rl" not in k.lower()}
        if not sft:
            raise SystemExit("No SFT record found; pass --baseline explicitly.")
        baseline = max(sft, key=lambda k: sft[k][rate_key])
    if baseline not in recs:
        raise SystemExit(f"Unknown baseline {baseline!r}. Known: {', '.join(sorted(recs))}")

    base_rec = recs[baseline]
    base_ex = base_rec["per_example"]
    print(f"baseline: {baseline}  ({args.metric} {100*base_rec[rate_key]:.1f}%, "
          f"n={base_rec.get('n', len(base_ex))})\n")

    rows = []
    for label, rec in recs.items():
        if label == baseline:
            continue
        ex = rec["per_example"]
        if len(ex) != len(base_ex):
            # Different eval sizes are not paired; skip rather than compare
            # a 400-example run against an 80-example one.
            continue
        lost, gained, p = mcnemar_exact(base_ex, ex, args.metric)
        rows.append((label, rec, 100 * (rec[rate_key] - base_rec[rate_key]), lost, gained, p))

    rows.sort(key=lambda r: -r[1][rate_key])

    hdr = f"{'checkpoint':<44} {args.metric+'%':>9} {'Δ pp':>7} {'lost':>5} {'gain':>5} {'p':>8}"
    print(hdr)
    print("-" * len(hdr))
    for label, rec, delta, lost, gained, p in rows:
        mark = "*" if p < args.alpha else " "
        print(f"{label:<44} {100*rec[rate_key]:>8.1f}% {delta:>+7.1f} "
              f"{lost:>5} {gained:>5} {p:>8.4f}{mark}")
    print(f"\n* = paired difference vs baseline significant at alpha={args.alpha} "
          "(McNemar exact, two-sided)")

    if not rows:
        print("\nNothing comparable to the baseline at the same eval size.")
        return

    best_label, best_rec, best_delta, _, _, best_p = rows[0]
    print("\n" + "=" * 72)
    if best_rec[rate_key] <= base_rec[rate_key]:
        print(f"BEST = {baseline} (the baseline). No checkpoint beat it on {args.metric}.")
        print("Do not ship an RL checkpoint on the strength of a within-noise argmax.")
    elif best_p < args.alpha:
        print(f"BEST = {best_label}  ({100*best_rec[rate_key]:.1f}%, {best_delta:+.1f} pp, p={best_p:.4f})")
        print("Improvement over the baseline is statistically significant.")
    else:
        print(f"BEST (argmax) = {best_label}  ({100*best_rec[rate_key]:.1f}%, "
              f"{best_delta:+.1f} pp, p={best_p:.4f})")
        print("NOT significant -- this is inside the noise band, so it is a tie with")
        print(f"{baseline}, not a win. Enlarge the eval set or train longer before claiming it.")
    print("=" * 72)

    # A drift diagnostic the aggregate rates cannot show: if the RL policy is
    # replacing correct statements with strictly weaker ones, `weaker_only_rate`
    # rises while beq_plus falls. That points at the reward's one-direction rung
    # (reward/beq_plus.py DIRECTION SEMANTICS), not at exploration or KL.
    have_weaker = [(l, r) for l, r in recs.items() if "weaker_only_rate" in r]
    if have_weaker:
        print("\nweaker-only rate (gold => pred proves, pred => gold does not):")
        for label, rec in sorted(have_weaker, key=lambda kv: -kv[1]["weaker_only_rate"]):
            print(f"  {label:<44} {100*rec['weaker_only_rate']:>6.1f}%")


if __name__ == "__main__":
    main()
