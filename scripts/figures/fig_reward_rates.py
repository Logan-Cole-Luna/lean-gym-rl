#!/usr/bin/env python3
"""How often does each reward pay each of its values, over the benchmark set?

The question behind it is signal quality, not correctness. Under GRPO the
advantage is the reward minus the group mean, so any rollout sitting at the same
value as the rest of its group contributes NOTHING to the gradient. A reward can
be perfectly calibrated (see `scripts/eval/reward_audit.py`) and still be almost
useless if it pays one value to 90% of the batch.

Scored WITHOUT Lean. `results/hpc_ablation/proved_*.json` already holds the
per-rollout verdicts -- `beq_plus`, the statement typecheck, own-proof
elaboration, `proved` -- and the eval JSON holds `semantic_signal`. Those are fed
into the REAL ladder (`reward_for` / `gated_reward` / `typecheck_reward`), never
a reimplementation of it: substituting recorded Lean verdicts for fresh ones is
sound, re-deriving the table by hand would not be.

    python scripts/figures/fig_reward_rates.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from figstyle import ACCENT, BASELINE, BLUE, CONTROL, GREEN, ORANGE, PINK, SKY, use_style
from matplotlib import pyplot as plt

from reward.reward import (OUTCOMES, gated_reward, outcome_for, reward_for,
                           signals_from_score, typecheck_reward)

HPC = Path(os.path.expanduser("~/Downloads/results_sft"))

# The rollout set every reward is scored over. One policy, three rewards, so any
# difference between the bars is the REWARD and not the model.
BENCH = ("rl3b_locolib_proof_lr6_outcome", 90,
         Path("results/hpc_ablation/proved_outcome90.json"))

ARM_VALUES = {                       # every value each arm can pay
    "outcome": [0.00, 0.05, 0.15, 0.30, 0.50, 0.85],
    "outcome_prefix": [0.00, 0.05, 0.15, 0.30, 0.50, 0.85],
    "gated": [0.00, 0.25, 1.00],
    "typecheck": [0.00, 1.00],
}
ARM_COLOR = {"outcome": BLUE, "outcome_prefix": SKY, "gated": GREEN, "typecheck": ORANGE}
ARM_LABEL = {"outcome": "outcome (shipped)", "outcome_prefix": "outcome (pre-fix wiring)",
             "gated": "gated", "typecheck": "type-check"}


def load_bench(arm: str, step: int, proved_json: Path) -> list[dict]:
    """Join the saved own-proof verdicts with the eval's `semantic_signal`.

    `semantic_signal` is only needed by `gated`, for its one-direction step --
    the one rung that is not derivable from `beq_plus` alone.
    """
    recs = json.loads(proved_json.read_text())
    g = glob.glob(str(HPC / "**" / f"eval_{arm}-step{step}_n*.json"), recursive=True)
    sem = {}
    if g:
        per = list(json.loads(Path(g[0]).read_text()).values())[0]["per_example"]
        sem = {e["i"]: e.get("semantic_signal", 0) for e in per}
    for r in recs:
        r["semantic_signal"] = sem.get(r["i"], 1 if r["beq_plus"] else 0)
    return recs


def score(recs: list[dict]) -> list[dict]:
    """One row per rollout: the value each arm pays, and the outcome row."""
    out = []
    for x in recs:
        # The shapes `BEqPlusScorer.score()` and `check_own_proof()` return.
        r = {"typecheck": x["typecheck_stmt"], "beq_plus": x["beq_plus"],
             "semantic_signal": x["semantic_signal"], "error_kind": x["error_kind"]}
        pc = {"type_correct": x["type_correct"], "proved": x["proved"],
              "sorry_used": x["sorry_used"], "error_kind": x["error_kind"]}
        s = signals_from_score(r, pc)
        # The same rollout under the OLD column-2 wiring, where `type_correct`
        # came from BEq+'s statement check. Kept side by side because the fix is
        # spec-correct but changes the SHAPE of the signal, not just its level.
        s_pre = signals_from_score(r, pc, typecheck_from_own_proof=False)
        out.append({
            "i": x["i"], "outcome": outcome_for(s), "outcome_pre": outcome_for(s_pre),
            "outcome_score": reward_for(s),
            "outcome_prefix_score": reward_for(s_pre),
            "gated_score": gated_reward(r),
            # The typecheck arm elaborates the submission's OWN proof.
            "typecheck_score": typecheck_reward(x["type_correct"], x["error_kind"]),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    args = ap.parse_args()
    use_style()
    args.out.mkdir(parents=True, exist_ok=True)

    arm, step, pj = BENCH
    rows = score(load_bench(arm, step, pj))
    n = len(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.7),
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    # --- how often each arm pays each of its values ------------------------
    # Stacked, not grouped: the outcome arm's values are 0.05 apart at the low
    # end, so grouped bars on a 0-1 reward axis overlap into mush. One row per
    # arm reads as "where does this reward's mass sit", which is the question.
    arms = ("outcome", "outcome_prefix", "gated", "typecheck")
    stats = {}
    for a in arms:
        vals = [r[f"{a}_score"] for r in rows]
        rates = [(v, sum(abs(x - v) < 1e-9 for x in vals) / n) for v in ARM_VALUES[a]]
        stats[a] = {"rates": rates, "modal": max(r for _, r in rates),
                    "mean": sum(vals) / n,
                    "distinct": sum(1 for _, r in rates if r > 0.005)}

    # Darker = a higher reward value, so the eye reads left-to-right as "worse
    # to better" within every row regardless of which values that arm uses.
    shade = lambda v: plt.cm.viridis(0.12 + 0.76 * v)
    ypos = np.arange(len(arms))[::-1]
    for yi, a in zip(ypos, arms):
        left = 0.0
        for v, rate in stats[a]["rates"]:
            if rate <= 0:
                continue
            ax1.barh(yi, 100 * rate, 0.62, left=left, color=shade(v),
                     edgecolor="white", linewidth=1.2)
            if rate > 0.035:
                ax1.text(left + 50 * rate, yi, f"{v:.2f}\n{100*rate:.0f}%",
                         ha="center", va="center", fontsize=8, fontweight="bold",
                         color="white" if v < 0.62 else "#1A1A1A")
            left += 100 * rate
        ax1.text(101, yi, f"mean {stats[a]['mean']:.3f}", va="center", fontsize=8.5,
                 fontweight="bold", color=ARM_COLOR[a])
    ax1.set_yticks(ypos, [ARM_LABEL[a] for a in arms], fontsize=9.5)
    ax1.set_xlim(0, 118)
    ax1.set_xticks([0, 20, 40, 60, 80, 100])
    ax1.set_xlabel("% of the benchmark set  (segment label = reward value paid)")
    ax1.set_title(f"How often each reward pays each value   n={n} rollouts, one fixed policy")
    ax1.grid(axis="y", visible=False)
    # The single biggest segment in each row is mass with no within-group
    # gradient: every rollout there has the same advantage as its neighbours.
    ax1.annotate("the widest segment in a row is mass with no within-group gradient",
                 xy=(0.5, -0.19), xycoords="axes fraction", ha="center",
                 fontsize=8.5, color=BASELINE, style="italic")

    # --- the ladder rows underneath ----------------------------------------
    hist = [sum(r["outcome"] == o for r in rows) / n for o in OUTCOMES]
    hist_pre = [sum(r["outcome_pre"] == o for r in rows) / n for o in OUTCOMES]
    y = np.arange(len(OUTCOMES))
    h = 0.38
    ax2.barh(y - h/2, [100 * v for v in hist], h, color=BLUE, label="shipped")
    ax2.barh(y + h/2, [100 * v for v in hist_pre], h, color=SKY, label="pre-fix")
    ax2.set_yticks(y, [o.replace("_", " ") for o in OUTCOMES], fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("% of the benchmark set")
    ax2.set_title("Which ladder row they land on")
    for i, (a_, b_) in enumerate(zip(hist, hist_pre)):
        for off, v, c in ((-h/2, a_, BLUE), (h/2, b_, SKY)):
            if v > 0.004:
                ax2.text(100 * v + 0.8, i + off, f"{100*v:.0f}", va="center",
                         fontsize=7.5, fontweight="bold", color=c)
    ax2.set_xlim(0, max(max(hist), max(hist_pre)) * 124)
    ax2.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out / "reward_rates.png")
    plt.close(fig)

    summary = {"benchmark": f"{arm}@step{step}", "n": n,
               "outcome_row_rates": dict(zip(OUTCOMES, hist)),
               "outcome_row_rates_prefix": dict(zip(OUTCOMES, hist_pre)),
               "arms": {a: {"mean_reward": s["mean"],
                            "modal_mass": s["modal"],
                            "distinct_values_used": s["distinct"],
                            "rates": {f"{v:.2f}": r for v, r in s["rates"]}}
                        for a, s in stats.items()}}
    (args.out / "reward_rates.json").write_text(json.dumps(summary, indent=2))

    print(f"benchmark: {arm} @ step{step},  n={n} rollouts\n")
    print(f"{'arm':11s} {'mean':>7} {'modal mass':>11} {'values used':>12}   rates")
    for a, s in stats.items():
        rr = "  ".join(f"{v:.2f}:{100*r:.0f}%" for v, r in s["rates"] if r > 0.005)
        print(f"{a:11s} {s['mean']:>7.3f} {100*s['modal']:>10.1f}% "
              f"{s['distinct']:>12d}   {rr}")
    print(f"\nladder rows (shipped):" + "  ".join(
        f" {o}:{100*h:.1f}%" for o, h in zip(OUTCOMES, hist) if h > 0))
    print(f"ladder rows (pre-fix): " + "  ".join(
        f" {o}:{100*h:.1f}%" for o, h in zip(OUTCOMES, hist_pre) if h > 0))
    print(f"\n" + "  ".join(
        f"{o}:{100*h:.1f}%" for o, h in zip(OUTCOMES, hist) if h > 0))
    print(f"\n-> {args.out / 'reward_rates.png'}")


if __name__ == "__main__":
    main()
