#!/usr/bin/env python3
"""Training-time curves: reward and loss per GRPO step, one line per arm.

Source is `results/train/train_metrics/<project>/<arm>.<jobid>.jsonl` -- verl's
FileLogger output, one `{"step", "data"}` row per optimiser step. Runs are
discovered off disk (evalio.discover_train_runs) and coloured to match the eval
figures (figstyle.arm_style), so `arm_trajectories.png` and `training_curves.png`
read together.

    python scripts/figures/fig_training.py
    make plots                       # runs this alongside make_figures.py

Two panels, because those are the two questions this answers:

  reward   critic/rewards/mean, the mean reward the batch earned that step, with
           the periodic validation reward (val-core/<ds>/reward/mean@1) overlaid.
  loss     actor/loss (total) and actor/pg_loss (the policy-gradient term). In
           GRPO neither is a descending curve -- pg_loss is an advantage-weighted
           ratio with mean ~0 by construction -- so this panel is a stability
           read, not a "going down is good" one.

Both raw and an EMA are drawn: batch 16 makes the per-step reward noisy enough
that the trend is otherwise hard to see.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: F401  -- repo root + every stage folder

import evalio
import figstyle as fs
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "results" / "figures"

REWARD_KEY = "critic/rewards/mean"
LOSS_KEYS = (("actor/loss", "total", "-", 1.0),
             ("actor/pg_loss", "policy-gradient", "--", 0.55))


def ema(ys: list[float], alpha: float = 0.35) -> list[float]:
    """Exponential moving average; alpha is the weight on the newest point."""
    out: list[float] = []
    m: float | None = None
    for y in ys:
        m = y if m is None else alpha * y + (1 - alpha) * m
        out.append(m)
    return out


def val_reward_series(steps: dict[int, dict]) -> tuple[list[int], list[float]]:
    """(xs, ys) for the validation reward, whatever data_source it is keyed under."""
    key = next((k for k in next(iter(steps.values()), {})
                if k.startswith("val-core/") and k.endswith("/reward/mean@1")), None)
    # The key only appears on test_freq steps, so scan every step for it, not row 0.
    if key is None:
        for d in steps.values():
            key = next((k for k in d
                        if k.startswith("val-core/") and k.endswith("/reward/mean@1")), None)
            if key:
                break
    return evalio.train_series(steps, key) if key else ([], [])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="",
                    help="comma-separated experiment names to draw (default: all on disk)")
    ap.add_argument("--max-step", type=int, default=0, help="clip the x-axis at this step")
    ap.add_argument("--alpha", type=float, default=0.35, help="EMA weight on the newest point")
    ap.add_argument("--out-dir", default=str(OUTDIR))
    args = ap.parse_args()

    runs = evalio.discover_train_runs()
    want = {a.strip() for a in args.arms.split(",") if a.strip()}
    if want:
        runs = {e: s for e, s in runs.items() if e in want}
    runs = {e: s for e, s in runs.items() if s}
    if not runs:
        sys.exit("no training metrics under results/train/train_metrics/")
    clip = args.max_step or None
    print(f"[fig_training] runs: "
          + ", ".join(f"{e}({min(s)}..{max(s)})" for e, s in sorted(runs.items())))

    fs.use_style()
    fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(12.4, 4.6))

    reward_ends, loss_ends = {}, {}
    for exp, steps in sorted(runs.items()):
        if clip:
            steps = {s: d for s, d in steps.items() if s <= clip}
            if not steps:
                continue
        st = fs.arm_style(exp)
        color = st["color"]
        name = fs.end_label(st)

        # ---- reward ----
        xs, ys = evalio.train_series(steps, REWARD_KEY)
        if xs:
            ax_r.plot(xs, ys, color=color, lw=1.0, alpha=0.28, zorder=2)
            sm = ema(ys, args.alpha)
            ax_r.plot(xs, sm, color=color, lw=2.0, marker=st["marker"],
                      markevery=max(1, len(xs) // 10), markeredgecolor="white",
                      markeredgewidth=0.7, label=st["label"], zorder=3)
            reward_ends[name] = (xs[-1], sm[-1], color)
        vx, vy = val_reward_series(steps)
        if vx:
            ax_r.plot(vx, vy, ls="none", marker="*", markersize=15, color=color,
                      markeredgecolor="white", markeredgewidth=0.8, zorder=4)

        # ---- loss ----
        for key, _lab, ls, a in LOSS_KEYS:
            lx, ly = evalio.train_series(steps, key)
            if not lx:
                continue
            sm = ema(ly, args.alpha)
            ax_l.plot(lx, ly, color=color, lw=0.9, alpha=0.22 * a, zorder=2)
            ax_l.plot(lx, sm, color=color, lw=2.0 if ls == "-" else 1.5, ls=ls,
                      alpha=a, label=st["label"] if ls == "-" else None, zorder=3)
            if ls == "-":
                loss_ends[name] = (lx[-1], sm[-1], color)

    ax_r.set_xlabel("GRPO step")
    ax_r.set_ylabel("mean reward per rollout")
    ax_r.set_title("Training reward  (line = EMA, faint = per step, ★ = validation)")
    ax_r.set_ylim(bottom=0)
    fs.finish(ax_r, end_labels=reward_ends, legend_loc="upper left")

    ax_l.set_xlabel("GRPO step")
    ax_l.set_ylabel("actor loss")
    ax_l.set_title("Training loss  (solid = total, dashed = policy-gradient)")
    fs.finish(ax_l, end_labels=loss_ends, legend_loc="upper left")

    fig.suptitle("GRPO training curves  (verl step metrics)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "training_curves.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig_training] wrote {p}")


if __name__ == "__main__":
    main()
