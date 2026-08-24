#!/usr/bin/env python3
"""Fit compute_score_placebo's constants to a policy's measured group geometry.

The placebo is the control that separates "the reward taught something" from
"GRPO's update noise moved the policy". That only works if its ADVANTAGE
GEOMETRY matches the real arm's: same fraction of groups that can produce a
gradient, same typical within-group spread. Those are properties of the POLICY,
not of the code, so the constants must be re-fitted whenever the policy changes
(a new model scale, a new SFT run). Shipping 0.5B's constants with a 3B policy
would give a control that is not a control.

Measured on sft-step390, the shipped defaults (0.37/0.30) ran +12.8% on
informative-group rate and +21.1% on within-group sigma -- i.e. the control was
meaningfully more damaging than the arm it was controlling for.

The placebo model: a per-PROMPT gate at probability g (so a prompt is either
"at the ability edge" or hopeless, persistently), and for gated prompts a
per-ROLLOUT Bernoulli at probability p. Both are hash-derived, hence
deterministic. This script grid-searches (g, p) to match two targets:

  * informative-group fraction  -- how often GRPO gets any gradient at all
  * mean within-group std       -- how large the advantages are when it does

Usage:
    python scripts/calibrate_placebo.py --scored data_3b/rollouts/xxx.scored.jsonl
    # then, for the placebo arm:
    export BEQ_PLACEBO_GROUP_P=... BEQ_PLACEBO_ROLLOUT_P=...
"""
from __future__ import annotations

import argparse
import collections
import json
from math import comb
from pathlib import Path


def group_stats(counts: dict[int, float], k: int, total: float) -> tuple[float, float, float]:
    """(informative fraction, mean within-group std, mean reward) for a
    histogram of winners-per-group."""
    inf = sum(v for c, v in counts.items() if 0 < c < k) / total
    sd = sum(v * ((c / k) * (1 - c / k)) ** 0.5 for c, v in counts.items()) / total
    mean = sum(v * (c / k) for c, v in counts.items()) / total
    return inf, sd, mean


def placebo_hist(g: float, p: float, k: int, n: int = 100_000) -> dict[int, float]:
    """Expected winners-per-group histogram under gate g + Bernoulli p."""
    h: dict[int, float] = collections.defaultdict(float)
    h[0] += (1 - g) * n
    for j in range(k + 1):
        h[j] += g * n * comb(k, j) * (p ** j) * ((1 - p) ** (k - j))
    return h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing --out file (see the guard below)")
    ap.add_argument("--metric", default="beq_plus")
    ap.add_argument("--out", default="", help="write shell exports here")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.scored).read_text().splitlines() if l.strip()]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["prompt_index"]].append(r)
    k = max(len(v) for v in by.values())
    groups = [v for v in by.values() if len(v) == k]
    emp = collections.Counter(sum(bool(x[args.metric]) for x in v) for v in groups)
    P = float(len(groups))

    t_inf, t_sd, t_mean = group_stats(emp, k, P)
    print(f"[calib] {len(groups)} groups at k={k}")
    print(f"[calib] TARGET (measured policy): informative {t_inf:.4f}  "
          f"within-group sd {t_sd:.4f}  mean reward {t_mean:.4f}")
    starved = emp[0]
    print(f"[calib]   starved {starved}/{int(P)} ({100*starved/P:.1f}%)  "
          f"saturated {emp[k]}/{int(P)} ({100*emp[k]/P:.1f}%)")

    # MEAN REWARD IS PART OF THE OBJECTIVE -- do not drop it back out.
    # Fitting only (informative, within-group sd) leaves the problem DEGENERATE:
    # within-group sd is symmetric under p <-> 1-p, so two settings score
    # identically and the grid returns whichever it reaches first. That is not
    # cosmetic. At k=8 a 2/8 group yields advantages {+0.75 x2, -0.25 x6} and a
    # 6/8 group yields {+0.25 x6, -0.75 x2}: same spread, MIRRORED skew, so the
    # two branches push the policy asymmetrically. It bit us for real -- the same
    # geometry fitted the base 3B series at p=0.20 (mean reward 0.11 against a
    # measured 0.36) and the mid-trained series at p=0.79 (mean 0.40 against a
    # measured 0.37), i.e. the two "matched" controls were mirror images of each
    # other. Mean reward is the asymmetric statistic, so including it picks the
    # branch and the fit becomes unique.
    best = None
    for gi in range(1, 101):
        for pi in range(1, 100):
            g, p = gi / 100, pi / 100
            a, b, _ = group_stats(placebo_hist(g, p, k), k, 100_000.0)
            m = g * p
            # relative error on all three targets, so none dominates by scale
            err = ((a / t_inf - 1) ** 2 + (b / t_sd - 1) ** 2 + (m / t_mean - 1) ** 2) ** 0.5
            if best is None or err < best[0]:
                best = (err, g, p, a, b, m)
    _, g, p, a, b, m = best
    print(f"[calib] BEST FIT: BEQ_PLACEBO_GROUP_P={g:.2f}  BEQ_PLACEBO_ROLLOUT_P={p:.2f}")
    print(f"[calib]   informative {a:.4f} ({100*(a/t_inf-1):+.1f}%)  "
          f"within-group sd {b:.4f} ({100*(b/t_sd-1):+.1f}%)  "
          f"mean reward {m:.4f} ({100*(m/t_mean-1):+.1f}%)")
    if max(abs(a / t_inf - 1), abs(b / t_sd - 1), abs(m / t_mean - 1)) > 0.05:
        print("[calib] WARNING: fit is worse than 5% on one target. The gate+Bernoulli "
              "family may not reproduce this policy's group shape; report the "
              "residual alongside any result that leans on this control.")

    # Never silently overwrite constants an already-trained arm was fitted with:
    # doing so retroactively changes what a finished control means.
    if args.out and Path(args.out).exists() and not args.force:
        print(f"[calib] {args.out} already exists -- NOT overwriting. Existing arms "
              f"were trained against it; pass --force only if nothing depends on it.")
        print(f"[calib] would have written GROUP_P={g:.2f} ROLLOUT_P={p:.2f}")
        return
    if args.out:
        Path(args.out).write_text(
            f"export BEQ_PLACEBO_GROUP_P={g:.2f}\n"
            f"export BEQ_PLACEBO_ROLLOUT_P={p:.2f}\n")
        print(f"[calib] wrote {args.out}")


if __name__ == "__main__":
    main()
