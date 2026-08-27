#!/usr/bin/env python3
"""Regenerate every figure in results/figures/ from the cached eval JSONs.

Reads only `results/eval_*.json` and `results/passk_*.json`, so it is cheap,
deterministic, and safe to re-run after any eval lands: `make figures`.

Style is matched to Interplay-LM-Reasoning (arXiv:2512.07783) -- see
scripts/figstyle.py for how it was recovered and why the palette order is fixed.

Figures, and the paper panel each mirrors:

  1. arm_trajectories   BEq+ vs training step, every arm         (their panel 1)
  2. passk              pass@k, baseline vs mid-trained          (their panel 2)
  3. retention_gain     what each arm kept vs converted          (their panel 4)
  4. proxy_vs_semantic  type-check against BEq+, the exploit     (no direct analogue)

EVERY RATE IS COMPUTED ON THE COMMON PREFIX of per_example, never read from
`*_rate`. Arms were evaluated at different --n-eval, and `*_rate` describes
whatever n that file was written at; mixing them silently compares different
example sets.
"""
from __future__ import annotations

import argparse
import glob
import json
import textwrap
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evalio
import figstyle as fs
import matplotlib.pyplot as plt

RESULTS = PROJECT_ROOT / "results"
FIGDIR = RESULTS / "figures"
BASELINE_LABEL = "sft3b-step93"


def load_arm(arm: str) -> dict[int, list[dict]]:
    return evalio.load_arm_steps(arm)


def baseline() -> list[dict]:
    return evalio.load_baseline()["per_example"]


def rate(pe: list[dict], key: str, n: int) -> float:
    return evalio.rate(pe, key, n)


def fig_arm_trajectories(arms: dict[str, dict[int, list[dict]]], n: int, metric="beq_plus"):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    base = rate(baseline(), metric, n)
    ends = {}
    for arm, steps in arms.items():
        if not steps:
            continue
        st = fs.ARM_STYLE[arm]
        # Anchor at step 0. Every arm resumes from the SAME SFT checkpoint, so
        # the lines must share a left endpoint; without it each arm appears to
        # start wherever its first eval happens to land and a step-10 drop reads
        # as a starting level.
        xs = [0] + sorted(steps)
        ys = [base] + [rate(steps[s], metric, n) for s in sorted(steps)]
        # markevery skips the step-0 anchor: it is the same point for every arm,
        # and six stacked markers there would read as a data cluster.
        ax.plot(xs, ys, color=st["color"], marker=st["marker"],
                markevery=list(range(1, len(xs))),
                ls=st.get("linestyle", "-"), label=st["label"],
                markeredgecolor="white", markeredgewidth=0.7, zorder=3)
        ends[st.get("end_label") or st["label"].split(" (")[0]] = (xs[-1], ys[-1], st["color"])
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("BEq+ (%), greedy decode" if metric == "beq_plus"
                  else "type-check (%), greedy decode")
    ax.set_title(f"Semantic accuracy over training  (greedy T=0, paired, n={n})")
    lo = min(min(rate(st[k], metric, n) for k in st) for st in arms.values() if st)
    ax.set_ylim(max(0, lo - 6), base + 8)   # fit the data; 0-60 left half the panel empty
    fs.finish(ax, end_labels=ends, legend_loc="lower left",
              hline=(base, f"SFT baseline {base:.1f}%"))
    return fig


def fig_passk(grid: tuple[int, ...] = evalio.STEP_GRID, metric: str = "beq_plus"):
    """pass@k against k, one line per model: the SFT baselines plus each arm.

    COLOUR IS BY ENTITY, NOT BY FILE ORDER. The first version indexed
    fs.CATEGORICAL by enumerate(files), so adding an arm repainted every existing
    series and the baselines took gated's blue and selfprove's green. Arms now use
    their fs.ARM_STYLE colour -- the same one they carry in every other figure --
    and the baselines are neutral greys, which also keeps them visually
    subordinate to the thing being tested.

    One checkpoint per arm (the latest on the reporting grid), because a line per
    checkpoint would be ~30 lines and pass@k curves for adjacent steps overlap.
    """
    by = load_passk_by_step(metric)
    baselines = []
    for f in sorted(RESULTS.glob("passk_sft3b*_k*.json")):
        d = json.loads(f.read_text())
        if d.get("metric", "beq_plus") != metric:
            continue
        if "-step" in d["label"] and d["label"].split("-step")[0] in fs.ARM_STYLE:
            continue
        baselines.append(d)
    arms = {a: v for a, v in by.items()
            if a != "__baseline__" and a not in globals().get("HIDE", set())}
    if not baselines and not arms:
        return None

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ks = []
    for i, d in enumerate(baselines):
        xs = [r["k"] for r in d["curve"]]
        ys = [100 * r["rate"] for r in d["curve"]]
        ks = xs or ks
        color = fs.BASELINE if i == 0 else fs.CONTROL
        ax.plot(xs, ys, color=color, marker="*", markersize=9, ls="-." if i == 0 else ":",
                lw=1.8, label=d["label"], markeredgecolor="white",
                markeredgewidth=0.6, zorder=2)
        ax.annotate(f"{ys[-1]:.1f}", xy=(xs[-1], ys[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.5, fontweight="bold",
                    color=color, va="center")
    for arm, steps in sorted(arms.items()):
        avail = [s for s in steps if s in grid]
        if not avail:
            continue
        st_i = max(avail)
        d = steps[st_i]
        stl = fs.ARM_STYLE[arm]
        xs = [r["k"] for r in d["curve"]]
        ys = [100 * r["rate"] for r in d["curve"]]
        ks = xs or ks
        ax.plot(xs, ys, color=stl["color"], marker=stl["marker"],
                ls=stl.get("linestyle", "-"),
                label=f"{stl['label'].split(' (')[0]} @{st_i}",
                markeredgecolor="white", markeredgewidth=0.7, zorder=3)
        ax.annotate(f"{ys[-1]:.1f}", xy=(xs[-1], ys[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.5, fontweight="bold",
                    color=stl["color"], va="center")
    if not ks:
        plt.close(fig)
        return None
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlim(min(ks) * 0.85, max(ks) * 1.55)
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k BEq+ (%)")
    ax.set_title("pass@k — how much the ceiling moves, per arm", fontsize=12)
    fs.finish(ax, legend_loc="upper left")
    return fig


def fig_retention_gain(arms, n, step_of):
    base_pe = baseline()[:n]
    correct = sum(bool(x["beq_plus"]) for x in base_pe)
    wrong = n - correct
    names, kept, gained, colors, hatches = [], [], [], [], []
    for arm, step in step_of.items():
        pe = arms.get(arm, {}).get(step)
        if pe is None:
            continue
        pe = pe[:n]
        st = fs.ARM_STYLE[arm]
        names.append(textwrap.fill(
            st.get("end_label") or st["label"].split(" (")[0], width=12))
        kept.append(100 * sum(1 for a, b in zip(base_pe, pe)
                              if a["beq_plus"] and b["beq_plus"]) / correct)
        gained.append(100 * sum(1 for a, b in zip(base_pe, pe)
                                if not a["beq_plus"] and b["beq_plus"]) / wrong)
        colors.append(st["color"])
        hatches.append(st.get("hatch", ""))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    shown = next(iter(step_of.values()))
    # LIMITER IS READ OFF THE ARMS ACTUALLY DRAWN. It used to be read off
    # `step_of`, which carries every arm including ones this panel skipped for
    # lacking the step -- so the caption named `selfprove` as the constraint on a
    # chart selfprove is not in.
    drawn = [a for a in step_of if arms.get(a, {}).get(shown) is not None]
    limiter = min(drawn, key=lambda a: max(arms[a])) if drawn else None
    cap = (f"step capped by the shortest arm, "
           f"{fs.ARM_STYLE[limiter]['label'].split(' (')[0]}" if limiter else "")
    fig.suptitle(
        f"All arms at GRPO step {shown}, paired against the SFT baseline"
        + (f"\n({cap})" if cap else ""),
        fontsize=10.5, fontweight="bold")
    for ax, vals, title, ylab in (
        (axes[0], kept, f"Retention — of the {correct} the baseline got right", "kept (%)"),
        (axes[1], gained, f"Gain — of the {wrong} it got wrong", "converted (%)"),
    ):
        bars = ax.bar(range(len(names)), vals, color=colors, width=0.62,
                      edgecolor="white", linewidth=1.5, zorder=3, hatch=hatches)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9.5, fontweight="bold", color="#333333")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8.5)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, min(max(vals) * 1.28, 108) if vals else 1)
    # The ceiling every 0.5B arm was stuck under -- the point of the right panel.
    axes[1].axhspan(4.9, 7.3, color=fs.ACCENT, alpha=0.10, zorder=1)
    axes[1].annotate("4.9–7.3%: the band every 0.5B\narm and signal sat inside",
                     xy=(0.5, 0.97), xycoords="axes fraction", fontsize=8.5,
                     fontweight="bold", color=fs.ACCENT, ha="center", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.90))   # keep the suptitle off the panel titles
    return fig


def fig_proxy_vs_semantic(arms, n):
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for arm in ("rl3b_v2_placebo", "rl3b_typecheck", "rl3b_gated"):
        if arm in globals().get("HIDE", set()):
            continue
        steps = arms.get(arm) or {}
        if not steps:
            continue
        st = fs.ARM_STYLE[arm]
        xs = sorted(steps)
        ax.plot([rate(steps[s], "typecheck", n) for s in xs],
                [rate(steps[s], "beq_plus", n) for s in xs],
                color=st["color"], marker=st["marker"], ls=st.get("linestyle", "-"),
                label=st["label"], markeredgecolor="white", markeredgewidth=0.7,
                alpha=0.35 if arm == "rl3b_v2_placebo" else 1.0,
                zorder=2 if arm == "rl3b_v2_placebo" else 3)
    tc = arms.get("rl3b_typecheck") or {}
    if tc:
        last = max(tc)
        fs.callout(ax, "type-check arm, step 10 → %d:\nproxy +10pp, semantics −32pp" % last,
                   xy=(rate(tc[last], "typecheck", n), rate(tc[last], "beq_plus", n)),
                   xytext=(rate(tc[last], "typecheck", n) - 14,
                           rate(tc[last], "beq_plus", n) + 11))
    b = baseline()
    ax.plot(rate(b, "typecheck", n), rate(b, "beq_plus", n), marker="*",
            markersize=16, color=fs.BASELINE, ls="none", label="SFT baseline")
    ax.set_xlabel("type-check (%)  — the cheap proxy")
    ax.set_ylabel("BEq+ (%)  — what we care about")
    ax.set_title("Optimising the proxy moves AWAY from the goal")
    # UPPER left, not lower: the placebo's step-30/50/90 points sit at low
    # type-check AND low BEq+, i.e. exactly where a lower-left legend lands, and
    # the box hid most of the control's trajectory.
    fs.finish(ax, legend_loc="upper left")
    return fig


def load_passk_by_step(metric: str = "beq_plus") -> dict[str, dict[int, dict]]:
    """arm -> step -> curve, plus '__baseline__' -> {0: sft record}.

    Kept as a thin adapter over evalio.load_passk because the figures want the
    baseline at step 0, while evalio keys baselines by label so several can
    coexist.
    """
    pk = evalio.load_passk(metric=metric)
    base = pk.pop("__baseline__", {})
    if evalio.BASELINE_LABEL in base:
        pk["__baseline__"] = {0: base[evalio.BASELINE_LABEL]}
    return pk


def fig_arm_trajectories_pass1(grid: tuple[int, ...] = evalio.STEP_GRID):
    """THE reference trajectory figure: pass@1 on BEq+ and on compiling, side by side.

    Why pass@1 and not greedy. The two verdicts have to be read against each
    other -- "did RL buy compiling code, and at what cost in semantics" is the
    whole question -- and that comparison is only sound when both come off the
    SAME rollouts. Greedy BEq+ (T=0, 400 rows) against sampled compiling (T=1.15,
    250 rows) differs in decoder AND prompt set, so a gap between them measures
    the harness, not the policy. Both panels here are one decoder, one prompt
    set, one draw per prompt; only the verdict changes.

    Greedy is still evaluated and still denser in step -- see fig_arm_trajectories,
    which backs the McNemar tests. It is just not the axis arms are compared on.
    """
    panels = [("beq_plus", "BEq+ pass@1 (%)", "Semantics — equivalent to the gold"),
              ("typecheck", "compiles pass@1 (%)", "Syntax — Lean accepts it")]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    drew = False
    for ax, (metric, ylab, kind) in zip(axes, panels):
        by = load_passk_by_step(metric)
        base = by.get("__baseline__", {}).get(0)
        b = evalio.passk_at(base, 1) if base else None
        arms = {a: evalio.on_grid(v, grid)
                for a, v in by.items()
                if a != "__baseline__" and a not in globals().get("HIDE", set())}
        ends, ys_all = {}, []
        for arm, steps in sorted(arms.items()):
            if not steps:
                continue
            st = fs.ARM_STYLE[arm]
            pts = [(s, evalio.passk_at(steps[s], 1)) for s in sorted(steps)]
            pts = [(s, y) for s, y in pts if y is not None]
            if not pts:
                continue
            if b is not None:
                pts = [(0, b)] + pts        # every arm resumes from the SFT policy
            ax.plot(*zip(*pts), color=st["color"], marker=st["marker"],
                    ls=st.get("linestyle", "-"), label=st["label"],
                    markevery=list(range(1 if b is not None else 0, len(pts))),
                    markeredgecolor="white", markeredgewidth=0.7, zorder=3)
            ends[st.get("end_label") or st["label"].split(" (")[0]] = (pts[-1][0], pts[-1][1], st["color"])
            ys_all += [y for _, y in pts]
            drew = True
        if not ys_all:
            continue
        ax.set_xlabel("GRPO step")
        ax.set_ylabel(ylab)
        ax.set_title(kind)
        ax.set_ylim(max(0, min(ys_all) - 8), min(102, max(ys_all + ([b] if b else [])) + 10))
        fs.finish(ax, end_labels=ends, legend_loc="lower left",
                  hline=(b, f"SFT {b:.1f}%") if b is not None else None)
    if not drew:
        return None
    fig.suptitle("pass@1 — one sample per prompt, T=1.15, 250 pinned prompts",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_arm_trajectories_passk(k: int = 32, k_lo: int = 1, metric: str = "beq_plus"):
    """Same figure as arm_trajectories, with pass@k on y instead of pass@1.

    Identical layout, palette, markers and x-axis to arm_trajectories so the two
    read side by side -- but they are NOT the same measurement, and an earlier
    version of this docstring wrongly said arm_trajectories "is pass@1".

    arm_trajectories comes from evaluate_checkpoints.py, which decodes GREEDILY
    (temperature=0.0) over 400 val prompts. This figure samples at temperature
    1.15 over the 250 pinned prompts. On sft3b-step93 that is 41.2% vs 32.4% --
    8.8pp apart, because greedy beats a single sample from a temp-1.15
    distribution. Do not read a step-for-step difference between the two figures
    as sharpening.

    So the sharpening-vs-capability comparison is made INSIDE this figure, where
    both numbers come from the same sampled rollouts: pass@1 dotted, pass@32
    solid, one colour per arm. pass@1 moving while pass@32 does not is
    sharpening; pass@32 moving is a capability gain.
    """
    by = load_passk_by_step(metric)
    arms = {a: evalio.on_grid(v, globals().get("GRID", evalio.STEP_GRID))
            for a, v in by.items()
            if a != "__baseline__" and a not in globals().get("HIDE", set())}
    arms = {a: v for a, v in arms.items() if v}
    if not arms:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    base = by.get("__baseline__", {}).get(0)
    b = next((100 * r["rate"] for r in base["curve"] if r["k"] == k), None) if base else None
    b_lo = next((100 * r["rate"] for r in base["curve"] if r["k"] == k_lo), None) if base else None
    ends, lows = {}, []
    for arm, steps in arms.items():
        st = fs.ARM_STYLE[arm]
        xs = sorted(steps)
        for kk, anchor, ls, alpha, lw in ((k, b, st.get("linestyle", "-"), 1.0, 2.0),
                                          (k_lo, b_lo, ":", 0.75, 1.6)):
            ys = [next((100 * r["rate"] for r in steps[x]["curve"] if r["k"] == kk), None)
                  for x in xs]
            pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if not pts:
                continue
            # Step-0 anchor: all arms resume from the same SFT checkpoint. See
            # the note in fig_arm_trajectories -- markevery hides the shared point.
            if anchor is not None:
                pts = [(0, anchor)] + pts
            ax.plot(*zip(*pts), color=st["color"], marker=st["marker"], ls=ls, lw=lw,
                    alpha=alpha, markeredgecolor="white", markeredgewidth=0.7,
                    markevery=list(range(1 if anchor is not None else 0, len(pts))),
                    markersize=6 if kk == k else 4.5,
                    label=st["label"] if kk == k else None, zorder=3)
            lows += [y for _, y in pts]
            if kk == k:
                ends[st.get("end_label") or st["label"].split(" (")[0]] = (pts[-1][0], pts[-1][1], st["color"])
    ax.set_xlabel("GRPO step")
    if metric == "beq_plus":
        ax.set_ylabel("BEq+ (%)")
        ax.set_title("Sharpening vs capability  (sampled T=1.15, 250 pinned prompts)")
    else:
        ax.set_ylabel("compiles (%)")
        ax.set_title("Compiling: pass@32 is already saturated  (sampled T=1.15, 250 prompts)")
    top = max(lows + ([b] if b else []))
    ax.set_ylim(max(0, min(lows) - 6), top + 8)
    fs.finish(ax, end_labels=ends, legend_loc="lower left",
              hline=(b, f"SFT baseline {b:.1f}%") if b is not None else None)
    # Second legend for the k encoding: colour is the arm, line style is k.
    # ax.legend() REPLACES the axes legend, so the arm legend fs.finish() just
    # built has to be re-added as a standalone artist -- the previous version
    # called add_artist() on the NEW legend instead, which silently deleted the
    # arm legend and left the figure identifying six arms by end label alone.
    # Bottom-right, so the k legend does not sit on the baseline annotation.
    from matplotlib.lines import Line2D
    arm_leg = ax.get_legend()
    ax.legend(handles=[Line2D([], [], color="#555555", ls="-", lw=2.0,
                              label=f"pass@{k} — capability"),
                       Line2D([], [], color="#555555", ls=":", lw=1.6,
                              label=f"pass@{k_lo} — single sample")],
              loc="lower right", fontsize=8.5)
    if arm_leg is not None:
        ax.add_artist(arm_leg)
    return fig


def step_seconds() -> dict[str, list[tuple[int, float]]]:
    """arm -> [(step, seconds/step)], from checkpoint mtimes.

    WHY MTIMES AND NOT THE LOGS. verl writes `timing_s/step` to stdout, but only
    three arms have it: the rest logged to stderr in a form that carries no
    metrics, and those are precisely the expensive arms we most want to time.
    Consecutive checkpoint mtimes divided by SAVE_FREQ covers every arm.

    CROSS-VALIDATED before use: for rl3b_typecheck this gives 31s/step against a
    logged median of 26s, and the 5s gap is the ~50s checkpoint save amortised
    over 10 steps. So the estimate is step time PLUS save overhead -- slightly
    inclusive, consistent, and the right number for "what does a step cost me".

    Deltas spanning a chunk boundary include SLURM queue time (one gated interval
    is 5,223s against a 962s median) and are dropped at 3x the median.
    """
    import os, glob as _g
    out: dict[str, list[tuple[int, float]]] = {}
    root = PROJECT_ROOT / "checkpoints" / "beqplus_rl_poc"
    for arm in fs.ARM_STYLE:
        ds = []
        for d in _g.glob(str(root / arm / "global_step_*")):
            m = re.search(r"global_step_(\d+)$", d)
            if m:
                ds.append((int(m.group(1)), os.path.getmtime(d)))
        ds.sort()
        if len(ds) < 3:
            continue
        per = [((s1), (t1 - t0) / (s1 - s0)) for (s0, t0), (s1, t1) in zip(ds, ds[1:])]
        vals = sorted(v for _, v in per)
        med = vals[len(vals) // 2]
        out[arm] = [(s, v) for s, v in per if v < 3 * med]
    return out


# The Lean-free cost of a GRPO step, measured. Taken from the placebo arm, which
# runs the identical model, batch and optimiser and makes ZERO Lean calls, so its
# step time IS the floor every other arm pays before any scoring happens.
#
# THIS IS WHY typecheck AND placebo COST THE SAME. verl logs no `timing_s/reward`
# key -- the reward runs inside the agent loop and lands in `timing_s/gen`. From
# the two arms that logged timings (150 steps each):
#
#     arm         gen     old_log_prob  ref   update_actor  update_weights  step
#     placebo     4.43s      3.55s     5.54s     9.03s          2.73s      26.12s
#     typecheck   6.36s      3.23s     5.42s     8.23s          2.71s      25.70s
#
# Every non-gen term is identical, as it must be. The ENTIRE Lean type-check bill
# is the gen delta: 6.36 - 4.43 = ~1.9 s/step, about 7% of the step, which rounds
# away against run-to-run variance. At 24 agent-loop workers and 128 rollouts per
# step that is ~0.36s of Lean per rollout -- one elaboration against a header env
# already cached for that prompt.
#
# gated's ~962 s/step is therefore ~935s of Lean, or ~175 Lean-seconds per rollout
# at 24 workers. With BEQ_TIMEOUT_PER_PROOF=30 and up to 18 calls in the cascade,
# that is ~6 timed-out proof attempts per rollout -- exactly what you expect when
# most rollouts are NOT equivalent to the gold and the cascade runs to exhaustion.
# So the ~37x is real, and it is not "type-checking is cheap and BEq+ is dear" by
# a constant factor: it is 1 fast call versus 6 half-minute searches.
def _no_lean_floor(raw: dict[str, list[tuple[int, float]]]) -> float | None:
    xs = raw.get("rl3b_v2_placebo")
    if not xs:
        return None
    vals = sorted(v for _, v in xs)
    return vals[len(vals) // 2]


def fig_runtime(max_step: int):
    """What each reward costs in wall-clock -- the latency view.

    Left: seconds per GRPO step. Right: cumulative hours to reach a given step,
    on the same x-axis as arm_trajectories, so cost and benefit line up.

    THE ONE FIGURE NOT ON THE REPORTING GRID, deliberately. Per-step cost is
    derived from the mtime delta between CONSECUTIVE checkpoints, so it is a
    measurement of the run, not a result at a step. Subsetting to 10/30/50/90
    would average each value over 20-40 steps instead of 10 and would integrate
    the cumulative curve from four points -- less accurate, for a number that is
    not being compared across arms at a matched step. Only `max_step` applies.
    """
    raw = step_seconds()
    floor = _no_lean_floor(raw)
    data = {a: [(s, v) for s, v in xs if s <= max_step] for a, xs in raw.items()
            if a not in globals().get("HIDE", set())}
    data = {a: xs for a, xs in data.items() if xs}
    if not data:
        return None
    order = sorted(data, key=lambda a: -sum(v for _, v in data[a]) / len(data[a]))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    # HORIZONTAL LOLLIPOP, NOT BARS. The range is ~30x, so a linear bar chart
    # makes the cheap arms invisible and a LOG bar chart is worse: bar length
    # stops encoding ratio once the baseline is not zero. A dot encodes by
    # POSITION, which is legitimate on a log axis, and the stem stays decorative.
    ax = axes[0]
    names = [fs.ARM_STYLE[a]["label"] for a in order]
    meds = [sorted(v for _, v in data[a])[len(data[a]) // 2] for a in order]
    colors = [fs.ARM_STYLE[a]["color"] for a in order]
    y = range(len(order))
    ax.hlines(list(y), 1, meds, color=colors, lw=2.2, alpha=0.45, zorder=2)
    ax.plot(meds, list(y), "o", ms=11, color="none",
            markerfacecolor="none", zorder=3)
    for i, (v, c) in enumerate(zip(meds, colors)):
        ax.plot([v], [i], "o", ms=10, color=c, markeredgecolor="white",
                markeredgewidth=1.4, zorder=4)
        share = f"  ({100 * max(v - floor, 0.0) / v:.0f}% Lean)" if floor else ""
        ax.annotate((f"{v/60:.1f} min" if v >= 60 else f"{v:.0f} s") + share,
                    xy=(v, i), xytext=(11, 0), textcoords="offset points",
                    va="center", fontsize=9.5, fontweight="bold", color="#333333")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(14, max(meds) * 9.0)   # room for the "N min (97% Lean)" labels
    ax.set_xlabel("seconds per GRPO step  (log)")
    ax.set_title("Cost of one GRPO step", fontsize=11)
    ax.grid(axis="y", visible=False)
    # THE FLOOR MAKES THE FIGURE ANSWER ITS OWN QUESTION. Without it, "type-check
    # costs the same as a reward that never opens Lean" looks like a measurement
    # bug. Against the floor it reads correctly: type-check sits ON the Lean-free
    # cost of a step, so its Lean bill is ~0 at this resolution, and gated is 30x
    # above the floor because a failed BEq+ cascade burns ~6 proof timeouts.
    if floor:
        ax.axvline(floor, color=fs.BASELINE, lw=1.6, ls="-.", zorder=2)
        ax.annotate(f"{floor:.0f}s: model + optimiser,\nbefore any Lean call",
                    xy=(floor, 0.985), xycoords=("data", "axes fraction"),
                    xytext=(6, 0), textcoords="offset points",
                    ha="left", va="top", fontsize=8.5, fontweight="bold",
                    color=fs.BASELINE,
                    path_effects=[__import__("matplotlib.patheffects", fromlist=["x"])
                                  .withStroke(linewidth=2.6, foreground="white")])
    # THE NUMBERS IN THIS CALLOUT ARE LOGGED, NOT DERIVED FROM THE LOLLIPOPS.
    # type-check's Lean bill is ~2s against a ~31s step, which is inside the
    # mtime resolution -- a ratio computed from (32 - 31) is noise amplified to
    # four digits. verl's own `timing_s/gen` resolves it; see _no_lean_floor.
    if len(meds) >= 2 and floor:
        ax.annotate("Same step time, opposite reasons.\n"
                    "type-check adds ~2 s of Lean per step\n"
                    "(verl gen: 6.4s vs 4.4s with no Lean).\n"
                    "A failed BEq+ cascade burns ~6 proof\n"
                    "timeouts, so gated adds ~935 s.",
                    xy=(0.62, 0.21), xycoords="axes fraction", ha="center", va="center",
                    fontsize=8.8, fontweight="bold", color=fs.ACCENT,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=fs.ACCENT, lw=1.4, ls="--"))

    ax = axes[1]
    for arm, xs in data.items():
        st = fs.ARM_STYLE[arm]
        cum, tot = [], 0.0
        prev = 0
        for step, sec in xs:
            tot += sec * (step - prev) / 3600
            prev = step
            cum.append((step, tot))
        ax.plot(*zip(*cum), color=st["color"], marker=st["marker"],
                ls=st.get("linestyle", "-"), label=st["label"],
                markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("cumulative GPU-hours")
    ax.set_title("What it costs to get there", fontsize=11)
    # Lower right: the curves all rise left-to-right, so upper left is where the
    # data is, and the legend was sitting on gated AND on the y-axis label.
    fs.finish(ax, legend_loc="lower right")
    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400,
                    help="common prefix every rate is computed over")
    ap.add_argument("--steps", default=",".join(str(s) for s in evalio.STEP_GRID),
                    help="the reporting grid: only these GRPO steps appear in any figure. "
                         "Defaults to evalio.STEP_GRID. Off-grid checkpoints stay on disk "
                         "and stay evaluated -- they are just not what results are shown at.")
    ap.add_argument("--hide", default="rl3b_v2_placebo",
                    help="comma-separated arms to omit from every figure. The placebo is "
                         "hidden by default -- `typecheck` now carries the 'what a cheap "
                         "signal does' role in the figures. NOTE the two are NOT "
                         "interchangeable: typecheck is an informative-but-exploitable "
                         "reward, the placebo is calibrated zero-information noise, and "
                         "only the placebo can support a claim that RL helped rather than "
                         "drifted. Its numbers stay in arms.md and compare_arms.py. Pass "
                         "--hide '' to draw it again.")
    ap.add_argument("--out-dir", default=str(FIGDIR))
    args = ap.parse_args()

    fs.use_style()
    grid = tuple(int(s) for s in args.steps.split(",") if s.strip())
    hide = {a.strip() for a in args.hide.split(",") if a.strip()}
    globals()["GRID"] = grid
    globals()["HIDE"] = hide
    if hide:
        print(f"[figures] hidden from every figure: {', '.join(sorted(hide))}")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    arms = {a: evalio.on_grid(load_arm(a), grid) for a in fs.ARM_STYLE if a not in hide}
    have = {a: s for a, s in arms.items() if s}
    print(f"[figures] arms with evals: {', '.join(f'{a}({len(s)})' for a, s in have.items())}")

    figs = {
        "arm_trajectories_pass1": fig_arm_trajectories_pass1(grid),
        "arm_trajectories": fig_arm_trajectories(have, args.n),
        "passk": fig_passk(grid),
        "proxy_vs_semantic": fig_proxy_vs_semantic(have, args.n),
        "arm_trajectories_passk": fig_arm_trajectories_passk(k=32),
        "arm_trajectories_passk_typecheck": fig_arm_trajectories_passk(k=32, metric="typecheck"),
        "runtime": fig_runtime(max(grid)),
    }
    # Best available step per arm for the decomposition panel.
    # MATCHED STEPS. An earlier version took each arm's best-available step, which
    # put gated at 30 against the placebo at 150 -- i.e. compared one arm near its
    # peak with the control at its worst. Use the latest step every arm has.
    #
    # ONE-POINT ARMS DO NOT SET THE STEP. `selfprove_t30` is a two-checkpoint
    # probe, and on the reporting grid it has exactly one point (step 10). Letting
    # it into the intersection dragged the whole panel back to step 10, where the
    # arms have not diverged and the comparison says nothing. Arms with a single
    # grid point are still drawn if they happen to have the chosen step.
    ranked = {a: set(v) for a, v in have.items() if len(v) >= 2} or {
        a: set(v) for a, v in have.items()}
    common = set.intersection(*ranked.values())
    step = max(common) if common else min(grid)
    step_of = {a: step for a in have}
    print(f"[figures] retention/gain at the latest step all arms share: {step}")
    if step_of:
        figs["retention_gain"] = fig_retention_gain(have, args.n, step_of)

    for name, fig in figs.items():
        if fig is None:
            print(f"[figures] {name}: SKIPPED (no data)")
            continue
        p = out / f"{name}.png"
        fig.savefig(p)
        plt.close(fig)
        print(f"[figures] wrote {p}")


if __name__ == "__main__":
    main()
