#!/usr/bin/env python3
"""The LoCoLib SFT comparison: does mid-training help, and does the other corpus transfer?

Three series, all scored on the SAME pinned LoCoLib validation slice, so they are
directly pairable:

    plain SFT              SFT on LoCoLib from the base model
    midtrain -> SFT        continued pre-training on Lean statements first
    other-corpus (OOD)     the Lean-Workbook model, never trained on LoCoLib

    python3 scripts/figures/fig_locolib.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: F401  -- repo root + every stage folder

import glob
import json
import math
import re

import matplotlib.pyplot as plt

import figstyle as fs

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/figures/locolib_sft.png"
N_EVAL = 999

SERIES = [
    ("sft3blocolib",     "plain SFT",          fs.BLUE,   "D"),
    ("sft3blocomt",      "midtrain → SFT", fs.GREEN,  "o"),
    ("sft3b_on_locolib", "other corpus (OOD)", fs.CONTROL, "s"),
]


def load(label: str) -> dict[int, list[dict]]:
    out = {}
    for f in glob.glob(str(REPO / f"results/eval/{label}/eval_{label}-step*_n{N_EVAL}.json")):
        step = int(re.search(r"-step(\d+)_", f).group(1))
        out[step] = next(iter(json.load(open(f)).values()))["per_example"]
    return dict(sorted(out.items()))


def rate(pe: list[dict], key: str) -> float:
    return 100.0 * sum(1 for r in pe if r.get(key)) / len(pe)


def mcnemar(a: list[dict], b: list[dict], key: str = "beq_plus") -> tuple[int, int, float]:
    """Paired exact test. Returns (b gains, b loses, p)."""
    n = min(len(a), len(b))
    gain = sum(1 for x, y in zip(a[:n], b[:n]) if not x.get(key) and y.get(key))
    lost = sum(1 for x, y in zip(a[:n], b[:n]) if x.get(key) and not y.get(key))
    m = gain + lost
    if not m:
        return gain, lost, 1.0
    p = 2 * sum(math.comb(m, i) for i in range(min(gain, lost) + 1)) / 2 ** m
    return gain, lost, min(1.0, p)


def main() -> None:
    fs.use_style()
    data = {label: load(label) for label, *_ in SERIES}
    drawn = {k: v for k, v in data.items() if v}
    if not drawn:
        sys.exit("no LoCoLib evals found")

    panels = [("beq_plus", "BEq+ (%)", "Semantic equivalence"),
              ("typecheck", "compiles (%)", "Syntax")]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for ax, (metric, ylab, title) in zip(axes, panels):
        ends, ys_all = {}, []
        for label, name, color, marker in SERIES:
            steps = data[label]
            if not steps:
                continue
            xs = sorted(steps)
            ys = [rate(steps[s], metric) for s in xs]
            ys_all += ys
            ax.plot(xs, ys, color=color, marker=marker, label=name,
                    markeredgecolor="white", markeredgewidth=0.7, zorder=3)
            ends[name] = (xs[-1], ys[-1], color)
        ax.set_xlabel("SFT step")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        pad = max(3.0, (max(ys_all) - min(ys_all)) * 0.15)
        ax.set_ylim(max(0, min(ys_all) - pad), max(ys_all) + pad)
        fs.finish(ax, end_labels=ends, legend_loc="center right")

    fig.suptitle(f"LoCoLib validation, greedy, n={N_EVAL}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")

    # The figure shows the trajectories; the paired tests are what license the
    # claim, so print them next to it rather than leaving them to be re-derived.
    best = {label: steps[max(steps)] for label, steps in drawn.items()}
    name_of = {label: name for label, name, *_ in SERIES}
    print(f"\nlast checkpoint of each series, paired on the same {N_EVAL} rows:")
    for label, pe in best.items():
        print(f"  {name_of[label]:20} step {max(drawn[label]):>3}  "
              f"BEq+ {rate(pe,'beq_plus'):5.1f}%  compiles {rate(pe,'typecheck'):5.1f}%")
    base = "sft3blocolib"
    if base in best:
        print("\nagainst plain SFT (McNemar exact, BEq+):")
        for label, pe in best.items():
            if label == base:
                continue
            g, l, p = mcnemar(best[base], pe)
            print(f"  {name_of[label]:20} gains {g:3}  loses {l:3}  p={p:.4f}")


if __name__ == "__main__":
    main()
