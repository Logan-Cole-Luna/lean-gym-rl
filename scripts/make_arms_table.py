#!/usr/bin/env python3
"""Regenerate the checkpoint table at the top of arms.md from cached results.

Written between the ARMS-TABLE markers so `make arms-table` can be re-run as
evals land.

PASS@1 IS THE REFERENCE MEASUREMENT. Cells are `BEq+ / compiles`, BOTH as pass@1
on the same sampled rollouts (T=1.15, 250 pinned prompts). That is the only
basis on which the two verdicts are comparable: the previous table paired greedy
BEq+ (T=0, 400 rows) against pass@32 (T=1.15, 250 rows), so a within-cell gap
mixed a decoder change, a prompt-set change and a k change all at once, and the
header had to warn readers not to read it.

Greedy T=0 evals are NOT dropped -- they are denser in step (every 10) and are
what the McNemar tests and `arm_trajectories.png` use. They are just not the
axis this table compares on.

Rows are `evalio.STEP_GRID` -- 10/30/50/90 -- and nothing else. Off-grid
checkpoints are still trained and still evaluated (typecheck runs to 150), they
are just not what results are reported at. The grid exists because the previous
table showed whatever each arm happened to have: `guided` was the only arm with
a step-70 pass@k, so row 70 was a row five of six arms could never fill.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalio

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
N_PREFIX = 400
GRID = evalio.STEP_GRID
START, END = "<!-- ARMS-TABLE:START -->", "<!-- ARMS-TABLE:END -->"

ARMS = [("rl3b_gated", "gated"), ("rl3b_guided", "guided"),
        ("rl3b_gated_edge", "gated_edge"), ("rl3b_selfprove", "selfprove"),
        ("rl3b_typecheck", "typecheck"), ("rl3b_v2_placebo", "placebo")]

METRICS = ("beq_plus", "typecheck")


def pass1_by_arm(metric: str) -> dict[str, dict[int, float]]:
    """arm -> step -> pass@1 (%)."""
    pk = evalio.load_passk(metric=metric)
    return {a: {s: v for s, v in
                ((s, evalio.passk_at(d, 1)) for s, d in steps.items()) if v is not None}
            for a, steps in pk.items() if a != "__baseline__"}


def baseline_passk(metric: str) -> dict[int, float]:
    base = evalio.load_passk(metric=metric).get("__baseline__", {})
    d = base.get(evalio.BASELINE_LABEL)
    return {r["k"]: 100 * r["rate"] for r in d["curve"]} if d else {}


def main() -> None:
    p1 = {m: pass1_by_arm(m) for m in METRICS}
    curve = {m: baseline_passk(m) for m in METRICS}

    sft = next(iter(json.loads((RESULTS / "eval_sft3b-step93_n1000.json").read_text()).values()))
    pe = sft["per_example"]
    greedy = {
        "beq_plus": (100 * sum(bool(x["beq_plus"]) for x in pe[:N_PREFIX]) / N_PREFIX,
                     100 * sft["beq_plus_rate"]),
        "typecheck": (100 * sum(bool(x["typecheck"]) for x in pe[:N_PREFIX]) / N_PREFIX,
                      100 * sft["typecheck_rate"]),
    }

    steps = [s for s in GRID
             if any(s in p1[m].get(a, {}) for m in METRICS for a, _ in ARMS)]

    def cell(a: str, s: int) -> str:
        vals = [p1[m].get(a, {}).get(s) for m in METRICS]
        if all(v is None for v in vals):
            return ""
        return " / ".join(f"{v:.1f}" if v is not None else "–" for v in vals)

    def row0(k: int) -> str:
        return " / ".join(f"{curve[m][k]:.1f}" if k in curve[m] else "–" for m in METRICS)

    names = [n for _, n in ARMS]
    lines = [
        START,
        "",
        "## Starting point: every arm resumes from here",
        "",
        "`sft3b-step93` is the checkpoint all six arms initialise from, so it is",
        "step 0 for all of them.",
        "",
        "| measurement | BEq+ | compiles |",
        "|---|---|---|",
        f"| **pass@1**, T=1.15, 250 prompts (**the reference**) | **{curve['beq_plus'].get(1, 0):.1f}%** "
        f"| **{curve['typecheck'].get(1, 0):.1f}%** |",
        f"| pass@32, same rollouts | {curve['beq_plus'].get(32, 0):.1f}% "
        f"| {curve['typecheck'].get(32, 0):.1f}% |",
        f"| greedy T=0, n={N_PREFIX} | {greedy['beq_plus'][0]:.1f}% | {greedy['typecheck'][0]:.1f}% |",
        f"| greedy T=0, n=1000 | {greedy['beq_plus'][1]:.1f}% | {greedy['typecheck'][1]:.1f}% |",
        "",
        "Greedy beats pass@1 by "
        f"{greedy['beq_plus'][0] - curve['beq_plus'].get(1, 0):+.1f}pp on BEq+ because a single",
        "T=1.15 sample is worse than the argmax. That gap is the decoder, not training.",
        f"The n=1000 and n={N_PREFIX} greedy numbers differ by "
        f"{greedy['beq_plus'][0] - greedy['beq_plus'][1]:+.1f}pp on the *same* checkpoint",
        "(the prefix is a slightly easier slice), which is why rates are never differenced",
        "across different n.",
        "",
        "## Every checkpoint, every arm",
        "",
        "Each cell is **BEq+ / compiles**, both as **pass@1** in %.",
        "Blank = not run, `–` = that half not measured.",
        "",
        "- Both halves come from the **same** sampled rollouts (T=1.15, 250 pinned",
        "  prompts, k=32 draws), so the gap within a cell IS meaningful: it is one",
        "  policy, one decoder, two verdicts on the same outputs.",
        f"- Rows are the reporting grid, {'/'.join(str(s) for s in GRID)}. Every arm is",
        "  evaluated at these steps and no others, so columns compare like with like.",
        "- Greedy T=0 BEq+ is also measured at these steps and backs the McNemar",
        "  tests; see `results/figures/arm_trajectories.png`.",
        "",
        "| step | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
        "| **0 (SFT)** | " + " | ".join([row0(1)] * len(names)) + " |",
    ]
    for s in steps:
        lines.append(f"| {s} | " + " | ".join(cell(a, s) for a, _ in ARMS) + " |")
    lines += ["", END]
    table = "\n".join(lines)

    p = ROOT / "arms.md"
    s = p.read_text()
    if START in s and END in s:
        s = s[:s.index(START)] + table + s[s.index(END) + len(END):]
    else:
        i = s.index("\n---\n")
        s = s[:i] + "\n" + table + "\n" + s[i:]
    p.write_text(s)
    print(f"[arms-table] {len(steps)} steps x {len(ARMS)} arms -> arms.md")


if __name__ == "__main__":
    main()
