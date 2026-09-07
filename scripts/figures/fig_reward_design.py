#!/usr/bin/env python3
"""Figures for the proof-length (brevity) reward and the BEq+ calibration.

Every panel is generated from measured data, not illustration:

  gold_lengths.png    LoCoLib gold proof-body lengths, and what the naive
                      `rfind(":=")` splitter does to them
  brevity_response.png the brevity term's response surface at the measured gold
                      percentiles, symmetric vs the shipped asymmetric band
  alignment_padding.png why proof_alignment.py uses F1 and not the raw DP total
  beq_calibration.png BEq+ against ProofNetVerif's human labels, plus latency

    python scripts/figures/fig_reward_design.py --calib results/beq_calibration
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from figstyle import ACCENT, BASELINE, BLUE, CONTROL, GREEN, ORANGE, PINK, SKY, use_style
from lean_interact.utils import split_implementation
from matplotlib import pyplot as plt

from reward.reward import (BREVITY_BAND, BREVITY_FREE_LONG, BREVITY_FREE_SHORT,
                           BREVITY_SOFTEN, BREVITY_ZERO_LONG,
                           BREVITY_ZERO_SHORT, brevity_for)

LOCOLIB = Path(__file__).resolve().parents[3] / "PartitionAndProve" / "datasets_training" / "LoCoLib"
_THEOREM_RE = re.compile(r"^\s*(theorem|lemma)\s", re.M)


def gold_bodies() -> tuple[list[int], list[int]]:
    """(split_implementation lengths, rfind lengths) over every labelled gold."""
    good, naive = [], []
    for f in sorted(LOCOLIB.glob("train_*.jsonl")):
        if "unlabelled" in f.name:
            continue
        for line in f.open(errors="replace"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            formal = (r.get("formal_proof") or "").strip()
            if not formal:
                continue
            m = list(_THEOREM_RE.finditer(formal))
            if not m:
                continue
            thm = formal[m[-1].start():].strip()
            i = split_implementation(thm)
            if i is None:
                continue
            good.append(len(thm[i + 2:].strip()))
            naive.append(len(thm[thm.rfind(":=") + 2:].strip()) if ":=" in thm else 0)
    return good, naive


def fig_gold_lengths(out: Path) -> dict:
    good, naive = gold_bodies()
    q = lambda v, p: float(np.percentile(v, p))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    bins = np.logspace(0, math.log10(max(good)), 60)
    ax1.hist(good, bins=bins, color=BLUE, alpha=0.85, label="split_implementation (correct)")
    ax1.hist(naive, bins=bins, color=ORANGE, alpha=0.55, label='rfind(":=") (naive)')
    ax1.set_xscale("log")
    ax1.set_xlabel("gold proof body (characters, log)")
    ax1.set_ylabel("golds")
    ax1.set_title(f"LoCoLib gold proof lengths (n={len(good):,})")
    top = ax1.get_ylim()[1]
    for p, c, frac in ((50, GREEN, 0.62), (90, PINK, 0.40)):
        ax1.axvline(q(good, p), color=c, ls="--", lw=1.6)
        ax1.annotate(f"p{p}={q(good, p):.0f}c", xy=(q(good, p) * 1.15, top * frac),
                     color=c, fontsize=9.5, fontweight="bold", ha="left", va="center",
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=c, lw=1.1))
    ax1.legend(loc="upper right")

    under = np.array(good) - np.array(naive)
    frac = float((under > 40).mean())
    ax2.hist(under[under > 0], bins=np.logspace(0, math.log10(max(under.max(), 2)), 50),
             color=ACCENT, alpha=0.8)
    ax2.set_xscale("log")
    ax2.set_xlabel('chars the naive splitter LOSES from the body')
    ax2.set_ylabel("golds")
    ax2.set_title(f"{100*frac:.1f}% undercounted by >40 chars")
    ax2.annotate(f"worst case\n{under.max():,} chars lost",
                 xy=(under.max(), 1), xytext=(under.max() * 0.06, ax2.get_ylim()[1] * 0.5),
                 fontsize=9, fontweight="bold", color=ACCENT,
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ACCENT, lw=1.4, ls="--"),
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4, ls="--"))
    fig.tight_layout()
    fig.savefig(out / "gold_lengths.png")
    plt.close(fig)
    return {"n": len(good),
            "pct": {str(p): q(good, p) for p in (10, 25, 50, 75, 90, 99)},
            "naive_pct": {str(p): q(naive, p) for p in (50, 90, 99)},
            "undercount_gt40_frac": frac, "worst_undercount": int(under.max())}


def _symmetric(pred, gold, soften=BREVITY_SOFTEN,
               free=math.log(1.5), zl=math.log(4.0), zs=math.log(6.0)):
    """The band that was tried FIRST and rejected -- kept only to draw it."""
    d = math.log((pred + soften) / (gold + soften))
    if abs(d) <= free:
        return 1.0
    if d > 0:
        return max(0.0, 1.0 - (d - free) / (zl - free))
    return max(0.0, 1.0 - (-d - free) / (zs - free))


def fig_brevity(out: Path, golds=(5, 28, 67, 170, 405, 1452)) -> None:
    ratios = np.logspace(math.log10(0.03), math.log10(30), 400)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cols = [BLUE, GREEN, ORANGE, PINK, SKY, CONTROL]
    for g, c in zip(golds, cols):
        ax1.plot(ratios, [brevity_for(max(1, int(r * g)), g) for r in ratios],
                 color=c, lw=2.0, label=f"gold {g}c")
        ax2.plot(ratios, [_symmetric(max(1, int(r * g)), g) for r in ratios], color=c, lw=2.0)
    for ax, title in ((ax1, "shipped: asymmetric dead band"),
                      (ax2, "rejected: symmetric band")):
        ax.set_xscale("log")
        ax.set_xlabel("candidate proof length / gold length")
        ax.set_title(title)
        ax.axvline(1.0, color=BASELINE, lw=1.2, ls="-.")
        ax.set_ylim(-0.03, 1.08)
    ax1.axvspan(1 / 3, 1.5, color=GREEN, alpha=0.10)
    ax2.axvspan(1 / 1.5, 1.5, color=GREEN, alpha=0.10)
    ax1.set_ylabel("brevity b  (1.0 = no penalty)")
    ax1.legend(loc="lower center", ncol=2, fontsize=8)
    ax2.annotate("`by omega` beating a\n61c gold is docked here",
                 xy=(0.13, _symmetric(8, 61)), xytext=(0.045, 0.42),
                 fontsize=8.5, fontweight="bold", color=ACCENT,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACCENT, lw=1.3, ls="--"),
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3, ls="--"))
    fig.suptitle(f"Brevity response — costs at most {BREVITY_BAND:.2f}, and only on SOLVED",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "brevity_response.png")
    plt.close(fig)


def fig_alignment(out: Path, n_gold=5, p_hit=0.25, trials=600) -> dict:
    """Raw DP total vs F1 as the candidate pads its proof with extra checkpoints."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pa", Path(__file__).resolve().parents[1] / "eval" / "proof_alignment.py")
    pa = importlib.util.module_from_spec(spec)
    sys.modules["pa"] = pa
    spec.loader.exec_module(pa)

    rng = np.random.default_rng(0)
    ks = [2, 3, 5, 8, 12, 16, 20, 25, 30]
    raw, f1s = [], []
    for k in ks:
        tot = f1 = 0.0
        for _ in range(trials):
            t = (rng.random((k, n_gold)) < p_hit).astype(float).tolist()
            r = pa.alignment_f1(t)
            tot += r["matched"]
            f1 += r["f1"]
        raw.append(tot / trials)
        f1s.append(f1 / trials)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(ks, raw, color=ACCENT, marker="s", label="raw DP total (calculate_score)")
    ax.set_xlabel("checkpoints the candidate emits  (gold has 5)")
    ax.set_ylabel("raw alignment total", color=ACCENT)
    ax.tick_params(axis="y", colors=ACCENT)
    ax2 = ax.twinx()
    ax2.plot(ks, f1s, color=BLUE, marker="D", label="F1 (shipped)")
    ax2.set_ylabel("alignment F1", color=BLUE)
    ax2.tick_params(axis="y", colors=BLUE)
    ax2.grid(False)
    ax.axvline(n_gold, color=BASELINE, ls="-.", lw=1.4)
    ax.annotate("gold length", xy=(n_gold, raw[0]), xytext=(n_gold + 0.6, raw[0]),
                fontsize=9, fontweight="bold", color=BASELINE)
    ax.set_title("Padding a proof buys raw score; F1 removes the incentive",
                 fontsize=12)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")
    fig.tight_layout()
    fig.savefig(out / "alignment_padding.png")
    plt.close(fig)
    return {"ks": ks, "raw": raw, "f1": f1s}


def _serial_latency(path: Path) -> dict | None:
    """Per-pair latency from a ONE-REPL run, if one is sitting beside the report.

    Kept separate from the sharded numbers rather than merged into them: the two
    measure different things, and averaging a contended run with an uncontended
    one produces a figure that describes neither.
    """
    if not path.exists():
        return None
    lat = sorted(json.loads(l)["seconds"] for l in path.open() if l.strip())
    if not lat:
        return None
    q = lambda p: lat[int(p * (len(lat) - 1))]
    return {"mean": sum(lat)/len(lat), "p50": q(.5), "p90": q(.9),
            "p99": q(.99), "max": lat[-1]}


def fig_new_rewards(out: Path, nr: Path, faith: Path) -> dict | None:
    """The two terms added this session: what they reach, and what they pay.

    Both are gated to the `SOLVED` row, so the honest denominator is the solved
    subset -- not the val slice. Drawing it as a funnel makes the gap between
    "we built a term" and "the term touches anything" impossible to miss.
    """
    import glob as _glob
    sp = nr / "summary.json"
    if not sp.exists():
        print(f"[fig] no new-reward summary at {sp}; skipping")
        return None
    S = json.loads(sp.read_text())
    F = json.loads((faith / "summary.json").read_text()) if (faith / "summary.json").exists() else {}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # --- panel 1: how much of the test set each term actually touches -------
    arm = "outcome" if "outcome" in S else next(iter(S))
    v = S[arm]
    n = v["n"]
    fm = F.get("matched", {})
    pool = F.get("pool", {})
    term_share = pool.get("term_mode_share", 0.0)
    faith_cov = (1 - term_share) * fm.get("coverage", 0.0)

    groups = [
        ("brevity", [("val slice", 100.0, CONTROL),
                     ("SOLVED\n(reachable)", 100 * v["reachable"] / n, SKY),
                     ("actually\nfires", 100 * v["fires"] / n, ACCENT)]),
        ("faithfulness", [("val slice", 100.0, CONTROL),
                          ("tactic-mode\n(reachable)", 100 * (1 - term_share), SKY),
                          ("judge returned\na score", 100 * faith_cov, BLUE)]),
    ]
    x, labels, colors, vals = [], [], [], []
    pos = 0.0
    for gname, stages in groups:
        for lab, val, col in stages:
            x.append(pos); labels.append(lab); vals.append(val); colors.append(col)
            pos += 1
        pos += 0.7
    ax1.bar(x, vals, 0.82, color=colors)
    for xi, val in zip(x, vals):
        ax1.text(xi, val + 2, f"{val:.1f}%", ha="center", fontweight="bold", fontsize=9)
    ax1.set_xticks(x, labels, fontsize=7.4, rotation=18, ha="right")
    ax1.set_ylim(0, 118)
    ax1.set_ylabel("% of the 760-row val slice")
    ax1.set_title("What each new term reaches")
    for gname, xc in (("brevity", 1.0), ("faithfulness", 4.7)):
        ax1.text(xc, 110, gname, ha="center", fontweight="bold", fontsize=10.5,
                 color=BASELINE)

    # --- panel 2: the brevity distribution, all vs reachable ----------------
    recs = [json.loads(l) for f in _glob.glob(str(nr / "brevity.*.jsonl"))
            for l in open(f) if l.strip()]
    allb = [r["b"] for r in recs if r["b"] is not None]
    solb = [r["b"] for r in recs if r["b"] is not None and r["solved"]]
    bins = np.linspace(0, 1, 21)
    ax2.hist(allb, bins=bins, color=CONTROL, alpha=0.85,
             label=f"all rollouts (n={len(allb)})")
    ax2.hist(solb, bins=bins, color=GREEN,
             label=f"SOLVED — the only rows it can charge (n={len(solb)})")
    ax2.set_yscale("log")
    ax2.set_xlabel("brevity  b   (1.0 = no deduction)")
    ax2.set_ylabel("rollouts (log)")
    ax2.set_title("Brevity: every reachable rollout already sits at b = 1")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.annotate(f"{100*sum(1 for x in allb if x < 1)/len(allb):.1f}% of ALL rollouts are "
                 f"charged\nbut 0% of the SOLVED ones",
                 xy=(0.55, 0.45), xycoords="axes fraction", fontsize=9,
                 fontweight="bold", color=ACCENT,
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ACCENT, lw=1.3, ls="--"))
    fig.tight_layout()
    fig.savefig(out / "new_rewards.png")
    plt.close(fig)
    return {"brevity": S, "faithfulness_coverage": faith_cov}


def fig_proved_trajectory(out: Path, d: Path) -> dict | None:
    """The ladder's top row across training -- the column no eval reported."""
    import glob as _glob
    import re as _re
    files = _glob.glob(str(d / "*.json"))
    if not files:
        print(f"[fig] no proved-trajectory data under {d}; skipping")
        return None
    series: dict[str, dict[int, dict]] = {}
    for f in files:
        m = _re.search(r"lr6_(\w+)-step(\d+)", f)
        if not m:
            continue
        series.setdefault(m.group(1), {})[int(m.group(2))] = \
            json.loads(Path(f).read_text())["summary"]
    if not series:
        return None

    COL = {"gated": BLUE, "outcome": GREEN, "typecheck": ORANGE}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.4), sharex=True)
    for arm in sorted(series):
        st = sorted(series[arm])
        for ax, key in ((ax1, "beq_plus_rate"), (ax2, "solved_rate")):
            ax.plot(st, [100 * series[arm][s_][key] for s_ in st],
                    color=COL.get(arm, CONTROL), marker="o", ms=4, label=arm)
    ax1.set_title("BEq+ rate — what the evals reported")
    ax2.set_title("SOLVED rate — what they did not")
    for ax in (ax1, ax2):
        ax.set_xlabel("RL step")
        ax.set_ylabel("% of the val slice")
        ax.legend(fontsize=9)
    # Same absolute span on both, so "flat" cannot be manufactured by a zoomed axis.
    lo = min(100 * series[a][s_][k] for a in series for s_ in series[a]
             for k in ("beq_plus_rate", "solved_rate"))
    hi = max(100 * series[a][s_][k] for a in series for s_ in series[a]
             for k in ("beq_plus_rate", "solved_rate"))
    ax1.set_ylim(lo - 3, hi + 3)
    ax2.set_ylim(lo - 3, hi + 3)
    fig.suptitle("Neither metric moves over 90 steps (n=250 per point)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "proved_trajectory.png")
    plt.close(fig)
    return {a: {str(s_): series[a][s_] for s_ in series[a]} for a in series}


def fig_faithfulness(out: Path, summary_path: Path, records_glob: str) -> dict | None:
    """NL->FL faithfulness: does it discriminate, and what does it cover?"""
    import glob as _glob
    if not summary_path.exists():
        print(f"[fig] no faithfulness summary at {summary_path}; skipping that panel")
        return None
    S = json.loads(summary_path.read_text())
    recs = [json.loads(l) for f in _glob.glob(records_glob) for l in open(f) if l.strip()]
    if not recs:
        return None

    ARMS = [a for a in ("matched", "mismatched", "padded") if a in S]
    COL = {"matched": GREEN, "mismatched": ACCENT, "padded": ORANGE}
    LBL = {"matched": "matched\n(own NL proof)", "mismatched": "mismatched\n(negative control)",
           "padded": "padded\n(vacuous `have`s)"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # --- score distributions: the discrimination evidence --------------------
    bins = [0.0, 0.25, 0.35, 0.75, 0.95]        # the rubric's observed levels
    w = 0.26
    x = np.arange(len(bins))
    for k, arm in enumerate(ARMS):
        vals = [r["score"] for r in recs if r["arm"] == arm and r["score"] is not None]
        n = len(vals)
        share = [100 * sum(abs(v - b) < 1e-6 for v in vals) / max(n, 1) for b in bins]
        ax1.bar(x + (k - 1) * w, share, w, color=COL[arm],
                label=f"{arm} (n={n}, mean {S[arm]['mean']:.2f})")
    ax1.set_xticks(x, [f"{b:.2f}" for b in bins])
    ax1.set_xlabel("faithfulness score returned by the judge")
    ax1.set_ylabel("% of scored rollouts")
    ax1.set_title("The negative control separates — but the matched arm has a zero mode")
    ax1.legend(loc="upper right", fontsize=8.5)

    # --- coverage waterfall: the number that decides reward-viability -------
    pool = S.get("pool", {})
    term = pool.get("term_mode_share", 0.0)
    sub = S["matched"]["coverage"]
    stages = [("all rollouts", 100.0, CONTROL),
              ("tactic-mode\n(has a `by` block)", 100 * (1 - term), SKY),
              ("judge returned\na usable score", 100 * (1 - term) * sub, BLUE)]
    ax2.bar([s[0] for s in stages], [s[1] for s in stages],
            color=[s[2] for s in stages], width=0.62)
    for i, (_, v, _c) in enumerate(stages):
        ax2.text(i, v + 1.6, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=10)
    ax2.set_ylim(0, 116)
    ax2.set_ylabel("% of the population")
    ax2.set_title("Coverage: what fraction can be scored at all")
    ax2.tick_params(axis="x", labelsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "faithfulness.png")
    plt.close(fig)
    return S


def fig_reward_audit(out: Path, summary_path: Path) -> dict | None:
    """What each arm pays for a known input, and where the ladder is flat."""
    if not summary_path.exists():
        print(f"[fig] no reward-audit summary at {summary_path}; skipping that panel")
        return None
    S = json.loads(summary_path.read_text())
    order = [c for c in ["empty", "prose", "bad_proof", "trivial_true", "trivial_rfl",
                         "gold_sorry", "gold"] if c in S]
    label = {"empty": "empty", "prose": "prose", "bad_proof": "broken proof",
             "trivial_true": "`True`", "trivial_rfl": "`n = n := rfl`",
             "gold_sorry": "gold, sorry'd", "gold": "GOLD"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(order))
    w = 0.27
    for i, (arm, c) in enumerate((("outcome", BLUE), ("gated", GREEN), ("typecheck", ORANGE))):
        vals = [S[c_]["mean_" + arm] for c_ in order]
        ax1.bar(x + (i - 1) * w, vals, w, color=c, label=f"{arm} arm")
        # Stagger the value labels by series: at w=0.27 three adjacent 1.00s
        # collide into "1.001.00" on any case where two arms agree.
        for xi, v in zip(x + (i - 1) * w, vals):
            if v > 0.005:
                ax1.text(xi, v + 0.015 + 0.055 * (i % 2), f"{v:.2f}", ha="center",
                         fontsize=7.5, fontweight="bold", color=("outcome", "gated",
                         "typecheck").__getitem__(i) and ["#1F3A4D", "#0B5B47", "#8A6200"][i])
    ax1.set_xticks(x, [label[c] for c in order], fontsize=9)
    ax1.set_ylabel("mean reward")
    ax1.set_ylim(0, 1.14)
    ax1.set_title("What each arm pays for a known input")
    ax1.legend(loc="upper left", fontsize=8.5)
    # The two hacks are the story: they compile, prove, and formalise nothing.
    for c_ in ("trivial_rfl", "trivial_true"):
        if c_ in order:
            ax1.axvspan(order.index(c_) - 0.45, order.index(c_) + 0.45,
                        color=ACCENT, alpha=0.07, zorder=0)

    # Does the outcome the reward assigned match what the table says it should?
    ok = [100 * S[c_]["matched_expected"] for c_ in order]
    # Three states, because they mean different things: agrees, agrees except
    # for stragglers, and never agrees. Painting a 31/32 the same red as a 0/29
    # would say the ladder is equally wrong in both, which it is not.
    cols = [GREEN if v >= 99 else (ACCENT if v < 1 else ORANGE) for v in ok]
    ax2.barh(np.arange(len(order)), ok, color=cols)
    ax2.set_yticks(np.arange(len(order)), [label[c] for c in order], fontsize=9)
    ax2.set_xlim(0, 108)
    ax2.set_xlabel("% landing on the outcome the table specifies")
    ax2.set_title("Agreement with reward.py's table")
    for i, (c_, v) in enumerate(zip(order, ok)):
        hist = S[c_]["outcome_hist"]
        # Name the row it landed on INSTEAD -- for a partial miss that is the
        # minority outcome, not the majority one, which is what went wrong.
        wrong = {k: n for k, n in hist.items() if k != S[c_]["expected"]}
        got = max(wrong, key=wrong.get) if wrong else ""
        txt = f"{v:.0f}%" + (f"  → {got}" if wrong else "")
        # Inside the bar in white when it is long enough to hold the label,
        # outside in the bar's own colour when it is not.
        inside = v > 45
        ax2.text(v - 2 if inside else v + 2, i, txt,
                 va="center", ha="right" if inside else "left",
                 fontsize=8, fontweight="bold",
                 color="white" if inside else cols[i])
    fig.tight_layout()
    fig.savefig(out / "reward_audit.png")
    plt.close(fig)
    return S


def fig_calibration(out: Path, report: Path) -> dict | None:
    if not report.exists():
        print(f"[fig] no calibration report at {report}; skipping that panel")
        return None
    rep = json.loads(report.read_text())
    core = rep.get("gold_elaborating_only") or rep
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))

    cm = np.array([[core["tp"], core["fn"]], [core["fp"], core["tn"]]], float)
    ax1.imshow(cm / cm.sum(), cmap="Blues", vmin=0, vmax=0.6)
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f"{int(cm[i, j])}\n{100*cm[i,j]/cm.sum():.1f}%",
                     ha="center", va="center", fontweight="bold",
                     color="white" if cm[i, j] / cm.sum() > 0.3 else "#222")
    ax1.set_xticks([0, 1], ["BEq+ fires", "BEq+ silent"])
    ax1.set_yticks([0, 1], ["human: correct", "human: wrong"])
    ax1.set_title(f"BEq+ vs human label (n={core['n']})")
    ax1.grid(False)

    bars = [("precision", core["precision"], BLUE), ("recall", core["recall"], GREEN),
            ("F1", core["f1"], SKY),
            ("false-positive\nrate", core["false_positive_rate"], ACCENT)]
    ax2.bar([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars])
    for i, b in enumerate(bars):
        ax2.text(i, b[1] + 0.02, f"{b[1]:.3f}", ha="center", fontweight="bold", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Reward firing quality")

    # Two series, because they answer different questions. The RL loop runs ONE
    # REPL (BEQ_MAX_CONCURRENT=1), so that is the per-rollout cost a training
    # step actually pays; the sharded figure is throughput under contention.
    # Log scale: the max is a ~6x outlier over p99 and would flatten the rest.
    lat = rep["latency"]
    ks = ["mean", "p50", "p90", "p99", "max"]
    serial = _serial_latency(report.parent / "latency.serial-run.jsonl")
    x = np.arange(len(ks))
    w = 0.38 if serial else 0.6
    nrepl = rep.get("concurrency", 1)
    ax3.bar(x - (w/2 if serial else 0), [lat[k] for k in ks], w, color=ORANGE,
            label=f"{nrepl} REPLs (this run)")
    if serial:
        ax3.bar(x + w/2, [serial[k] for k in ks], w, color=BLUE,
                label="1 REPL (the RL-loop config)")
    ax3.set_yscale("log")
    ax3.set_xticks(x, ks)
    for i, k in enumerate(ks):
        ax3.text(i - (w/2 if serial else 0), lat[k]*1.08, f"{lat[k]:.0f}", ha="center",
                 fontweight="bold", fontsize=8)
        if serial:
            ax3.text(i + w/2, serial[k]*1.08, f"{serial[k]:.0f}", ha="center",
                     fontweight="bold", fontsize=8, color=BLUE)
    ax3.set_ylabel("seconds per scored pair (log)")
    ax3.set_title("BEq+ latency per pair")
    ax3.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "beq_calibration.png")
    plt.close(fig)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    ap.add_argument("--calib", type=Path, default=Path("results/beq_calibration_full"))
    ap.add_argument("--audit", type=Path, default=Path("results/reward_audit_n200"))
    ap.add_argument("--faith", type=Path, default=Path("results/faithfulness_n300"))
    ap.add_argument("--traj", type=Path, default=Path("results/proved_trajectory"))
    args = ap.parse_args()
    use_style()
    args.out.mkdir(parents=True, exist_ok=True)

    stats = {"gold_lengths": fig_gold_lengths(args.out)}
    fig_brevity(args.out)
    stats["alignment_padding"] = fig_alignment(args.out)
    nrw = fig_new_rewards(args.out, Path("results/new_rewards"), args.faith)
    if nrw:
        stats["new_rewards"] = nrw
    traj = fig_proved_trajectory(args.out, args.traj)
    if traj:
        stats["proved_trajectory"] = traj
    fth = fig_faithfulness(args.out, args.faith / "summary.json",
                           str(args.faith / "records.*.jsonl"))
    if fth:
        stats["faithfulness"] = fth
    aud = fig_reward_audit(args.out, args.audit / "summary.json")
    if aud:
        stats["reward_audit"] = {k: {kk: v[kk] for kk in
                                     ("n", "expected", "matched_expected", "mean_outcome",
                                      "mean_gated", "mean_typecheck", "outcome_hist")}
                                 for k, v in aud.items()}
    rep = fig_calibration(args.out, args.calib / "report.json")
    if rep:
        stats["calibration"] = {k: rep[k] for k in
                                ("n", "precision", "recall", "f1",
                                 "false_positive_rate", "latency")}
    (args.out / "reward_design_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2)[:1600])
    print(f"\n[fig] wrote -> {args.out}")


if __name__ == "__main__":
    main()
