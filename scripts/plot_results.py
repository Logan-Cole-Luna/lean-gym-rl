#!/usr/bin/env python3
"""Parse verl training logs into per-step metrics, plot them, and quantify what
each reward function actually did at the training level.

verl's console logger emits one `step:N - k:v - k:v ...` line per step. This
scrapes those lines out of the run logs in logs/, groups them by arm, and
produces:

  results/training_curves.png    reward / type-check / entropy / response-length
  results/reward_impact.md       per-arm quantitative summary table
  results/training_metrics.csv   the tidy parsed data, for any further analysis

Usage:
    python scripts/plot_results.py                  # auto-discover logs/
    python scripts/plot_results.py --logs a.log b.log
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The arm that produced a log line is identified by the experiment_name echoed
# by run_grpo.sh (set -x) near the top of each log.
EXPERIMENT_RE = re.compile(r"^\+ experiment_name=(\S+)", re.M)
STEP_LINE_RE = re.compile(r"step:(\d+) - (.*)$")

# Metrics we care about, and how to label them.
WANTED = {
    "critic/rewards/mean": "reward",
    "critic/score/mean": "score",
    "actor/entropy": "entropy",
    "actor/pg_loss": "pg_loss",
    "actor/grad_norm": "grad_norm",
    "response_length/mean": "resp_len",
    "val-core/lean_workbook/acc/mean@1": "val_metric",
}

# What each arm's reward function actually measures -- crucial for reading the
# plots, since `reward` is NOT on a common scale across arms.
REWARD_SEMANTICS = {
    "compute_score_typecheck_only": "type-check pass rate (1.0 = all rollouts elaborate)",
    "compute_score_composite": "0.1*typecheck + 0.9*BEq+ (strict, binary BEq+)",
    "compute_score_shaped": "0.15 typecheck + 0.35 one-direction + 0.50 both (graded)",
}


def arm_label(experiment_name: str) -> str:
    """Compress verl experiment_name into a short readable arm label."""
    n = experiment_name
    # Check prefixes BEFORE the reward-name suffixes: `rl_from_sft_compute_score_shaped`
    # ends with `compute_score_shaped` and would otherwise be mislabelled as the
    # from-scratch shaped arm, which is exactly the comparison this plot exists
    # to make.
    if n.startswith("rl_from_sft"):
        return "SFT → RL (shaped)"
    if n.startswith("curriculum_p1"):
        return "curriculum phase-1 (type-check)"
    if n.startswith("curriculum_"):
        return "curriculum (warm-start + shaped)"
    if n.endswith("compute_score_typecheck_only"):
        return "typecheck-only"
    if n.endswith("compute_score_composite"):
        return "composite (strict BEq+)"
    if n.endswith("compute_score_shaped"):
        return "shaped (from scratch)"
    return n


def _parse_step_line(sm: re.Match) -> dict:
    row = {"step": int(sm.group(1))}
    for kv in sm.group(2).split(" - "):
        if ":" not in kv:
            continue
        k, _, v = kv.partition(":")
        k = k.strip()
        if k not in WANTED:
            continue
        # verl prints some values as np.float64(...) / np.int64(...)
        nm = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", v.strip())
        if nm:
            row[WANTED[k]] = float(nm.group(0))
    return row


def parse_log(path: Path) -> list[tuple[str, list[dict]]]:
    """A single log may contain SEVERAL arms back-to-back (`make train` runs
    both ablation arms into one file), so split on each `experiment_name=` echo
    and attribute the step lines that follow it to that arm."""
    text = path.read_text(errors="replace")
    marks = [(m.start(), m.group(1)) for m in EXPERIMENT_RE.finditer(text)]
    if not marks:
        return []

    blocks: list[tuple[str, list[dict]]] = []
    for i, (pos, experiment) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        rows = []
        for line in text[pos:end].splitlines():
            sm = STEP_LINE_RE.search(line)
            if sm:
                row = _parse_step_line(sm)
                if len(row) > 1:
                    rows.append(row)
        if rows:
            blocks.append((experiment, rows))
    return blocks


def collect(log_paths: list[Path]) -> dict[str, list[dict]]:
    """arm label -> rows. Later logs for the same arm override earlier steps,
    so a resumed run's steps replace the crashed attempt's."""
    by_arm: dict[str, dict[int, dict]] = {}
    for p in sorted(log_paths):
        for experiment, rows in parse_log(p):
            label = arm_label(experiment)
            by_arm.setdefault(label, {})
            for r in rows:
                by_arm[label][r["step"]] = r
    return {k: [v[s] for s in sorted(v)] for k, v in by_arm.items()}


def write_csv(data: dict[str, list[dict]], out: Path) -> None:
    cols = ["arm", "step"] + sorted({k for rows in data.values() for r in rows for k in r if k != "step"})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for arm, rows in data.items():
            for r in rows:
                w.writerow({"arm": arm, **r})


def plot(data: dict[str, list[dict]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("reward", "Training reward\n(NOT comparable across arms - see note)"),
        ("entropy", "Policy entropy\n(collapse = no exploration left)"),
        ("pg_loss", "Policy-gradient loss\n(0 = advantages vanished, no learning)"),
        ("resp_len", "Mean response length (tokens)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, (key, title) in zip(axes.ravel(), panels):
        for i, (arm, rows) in enumerate(sorted(data.items())):
            xs = [r["step"] for r in rows if key in r]
            ys = [r[key] for r in rows if key in r]
            if xs:
                ax.plot(xs, ys, marker="o", ms=3, lw=1.6, label=arm, color=colors[i % len(colors)])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=8)

    fig.suptitle(
        "BEq+ RL ablation - training dynamics\n"
        "Each arm's 'reward' is its own objective, so curves are NOT directly comparable; "
        "see results/reward_impact.md for the common-metric comparison.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def quantify(data: dict[str, list[dict]], out: Path) -> None:
    lines = [
        "# Per-reward impact at the training level",
        "",
        "Parsed from verl's per-step console metrics (`results/training_metrics.csv`).",
        "",
        "**Read the `reward` column with care**: each arm optimises its own reward",
        "function, so these values are on different scales and are NOT comparable to",
        "each other. Only the columns below them (entropy, pg_loss, response length)",
        "and the separate common-metric eval (`results/ablation_comparison.json`)",
        "support cross-arm comparison.",
        "",
        "| arm | steps | reward first→last | peak | entropy first→last | dead steps | ‑ starved | ‑ saturated | resp_len first→last |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, rows in sorted(data.items()):
        if not rows:
            continue
        def series(k):
            return [r[k] for r in rows if k in r]
        rw, en, rl = series("reward"), series("entropy"), series("resp_len")
        pg = series("pg_loss")

        # A "dead" step contributes no gradient (all rollouts in every group tied).
        # Split by WHY: near-zero reward means the reward never fired (starved);
        # near-peak reward means every rollout already succeeds (saturated).
        peak_val = max(rw) if rw else 0.0
        dead = starved = saturated = 0
        for r in rows:
            if "pg_loss" not in r or abs(r["pg_loss"]) >= 1e-8:
                continue
            dead += 1
            rv = r.get("reward", 0.0)
            if rv <= 0.05 * max(peak_val, 1e-9):
                starved += 1
            elif peak_val and rv >= 0.95 * peak_val:
                saturated += 1

        f = lambda s: f"{s[0]:.3f}→{s[-1]:.3f}" if s else "n/a"
        peak = f"{peak_val:.3f}" if rw else "n/a"
        lines.append(
            f"| {arm} | {len(rows)} | {f(rw)} | {peak} | {f(en)} | "
            f"{dead}/{len(pg) if pg else 0} | {starved} | {saturated} | {f(rl)} |"
        )

    lines += [
        "",
        "## What each arm's reward measures",
        "",
    ]
    for fn, meaning in REWARD_SEMANTICS.items():
        lines.append(f"- `{fn}`: {meaning}")
    lines += [
        "",
        "## Why `pg_loss == 0` matters",
        "",
        "GRPO's advantage is computed *within* each rollout group. If every sample in a",
        "group earns the same reward, the advantage is zero and the step contributes no",
        "gradient -- a **dead step**. Two opposite situations produce dead steps and are",
        "indistinguishable in the loss alone, so the table splits them by reward level:",
        "",
        "- **starved** (reward ~= 0): the reward never fired for *any* rollout. The",
        "  signal is too sparse for the policy to get off the ground. This is the",
        "  failure mode of the strict composite reward.",
        "- **saturated** (reward ~= peak): *every* rollout already succeeds, so there is",
        "  nothing further to optimise. This is where the type-check-only arm ends up --",
        "  and, since its objective is trivially satisfiable, where reward hacking lives.",
        "",
        "Starved steps early in training are the problem the shaped/curriculum arms are",
        "designed to fix: they exist to make rollout groups *differ* in reward so that",
        "an advantage, and therefore a gradient, exists at all.",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="*", default=None)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results"))
    args = ap.parse_args()

    paths = [Path(p) for p in args.logs] if args.logs else sorted((PROJECT_ROOT / "logs").glob("*.log"))
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise SystemExit("No logs found (looked in logs/*.log)")

    data = collect(paths)
    if not data:
        raise SystemExit("No parseable step metrics in those logs")
    for arm, rows in sorted(data.items()):
        print(f"  {arm:36s} {len(rows):>3} steps")

    out_dir = Path(args.out_dir)
    write_csv(data, out_dir / "training_metrics.csv")
    quantify(data, out_dir / "reward_impact.md")
    plot(data, out_dir / "training_curves.png")


if __name__ == "__main__":
    main()
