"""Shared figure style, matched to Interplay-LM-Reasoning (arXiv:2512.07783).

WHERE THE STYLE COMES FROM. Their plotting scripts are not in the repo -- only
stale `.pyc` files under `scripts/composition/__pycache__/` -- so the style was
recovered by sampling `assets/findings.png` directly. The dominant saturated
colours came back as #009E73, #0072B2, #E69F00, i.e. they use **Okabe-Ito**, the
standard colourblind-safe qualitative palette. Everything else here (pastel
panel tints, bold axis labels, dotted grid, markers on every point, in-axes
legend, value labels over bars, red dashed callouts) is read off that figure.

PALETTE ORDER IS NOT ARBITRARY -- do not reshuffle it. Assigned in fixed order
and validated: blue -> green -> orange -> pink -> sky passes the lightness band,
chroma floor, CVD-separation and normal-vision checks. Two other orderings that
look equally reasonable put #CC79A7 next to #009E73, which collapses to
deutan dE 7.6 -- below the 8.0 floor. If you add a series, re-validate.

Three of the five sit under 3:1 contrast against the surface, which obliges
"relief": every series is DIRECT-LABELLED at its line end, not identified by
colour alone. That is why `finish()` draws end labels rather than relying on the
legend.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# Okabe-Ito, in the validated order.
BLUE, GREEN, ORANGE, PINK, SKY = "#0072B2", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"
CATEGORICAL = [BLUE, GREEN, ORANGE, PINK, SKY]

# The control is deliberately NOT categorical: it is a reference, so it is grey
# and dashed. Only the CORRECTED fit (0.49/0.74) is plotted -- the original
# 0.55/0.20 fit came from a degenerate objective and is kept on disk for
# provenance, not shown -- so "placebo" in every figure means the corrected one.
# Grey fails the chroma floor by design: it must read as recessive, and the dash
# pattern is the secondary encoding that keeps it distinguishable.
CONTROL = "#8C8C8C"
BASELINE = "#3A3A3A"   # SFT reference line, and the neutral bar in bar charts
ACCENT = "#BE0000"     # the paper's red, for callouts only -- never a series

# Per-arm assignment. Stable across every figure so a colour always means the
# same arm -- colour follows the entity, never its rank in the current plot.
ARM_STYLE = {
    "rl3b_gated":      dict(color=BLUE,    marker="D", label="gated (BEq+)"),
    "rl3b_selfprove":  dict(color=GREEN,   marker="o", label="selfprove (gold-free)"),
    "rl3b_typecheck":  dict(color=ORANGE,  marker="s", label="type-check only"),
    "rl3b_guided":     dict(color=PINK,    marker="^", label="guided (BEq+ + similarity)"),
    "rl3b_gated_edge": dict(color=SKY,     marker="v", label="gated, edge pool"),
    "rl3b_v2_placebo": dict(color=CONTROL, marker="s", label="placebo", linestyle="--"),
    "rl3b_selfprove_t30": dict(color=GREEN, marker="P", label="selfprove (30s probe)", linestyle=":"),
    # THE lr6 SERIES: same reward, same pool, ACTOR_LR=1e-6 + 0.05 warmup instead
    # of the accidental 1e-5. Each keeps its twin's colour and marker, because
    # colour follows the entity, and is dash-dotted so the corrected run is
    # distinguishable from the one it corrects. Arms at different learning rates
    # are not comparable as *rewards*; plotting a pair together is a statement
    # about the LEARNING RATE, which is the one thing that differs.
    # `end_label` overrides the direct label drawn at the line end. Without it,
    # make_figures derives one as label.split(" (")[0], which maps both
    # "gated, edge pool" and "gated, edge pool (lr 1e-6)" onto the SAME key --
    # they share one dict entry and whichever is plotted last silently erases
    # the other's label.
    "rl3b_lr6_gated_edge": dict(color=SKY, marker="v", linestyle="-.",
                                label="gated, edge pool (lr 1e-6)",
                                end_label="edge pool, lr 1e-6", hatch="//"),
    "rl3b_lr6_typecheck": dict(color=ORANGE, marker="s", linestyle="-.",
                               label="type-check only (lr 1e-6)",
                               end_label="type-check, lr 1e-6", hatch="//"),
    # THE POOL-CONTROLLED PAIR. This is rl3b_lr6_gated_edge with ONE variable
    # changed: the reward. Same edge pool, same 1e-6 + warmup. It is the only
    # place in the set where reward quality is isolated from prompt selection,
    # so it must be plotted whenever rl3b_lr6_gated_edge is.
    "rl3b_lr6edge_typecheck": dict(color=ORANGE, marker="P", linestyle="-.",
                                   label="type-check, edge pool (lr 1e-6)",
                                   end_label="type-check, edge", hatch="\\\\"),
}


def end_label(style: dict) -> str:
    """The short label drawn at a line's right-hand end.

    Falls back to the legend label with any parenthetical dropped. That fallback
    is why `end_label` exists at all: it collapses "gated, edge pool" and
    "gated, edge pool (lr 1e-6)" onto the same string, and the end-label map is
    keyed on that string, so one arm silently erases the other. Set `end_label`
    explicitly on any arm whose legend label shares a prefix with another's.

    Lives here, beside ARM_STYLE, because the rule and the dict it reads have to
    change together -- it used to be inlined at four call sites in make_figures.
    """
    return style.get("end_label") or style["label"].split(" (")[0]

PANEL_TINT = {"blue": "#EAF0F8", "peach": "#FDF0E7", "green": "#EAF4EC", "amber": "#FDF6E3"}


def use_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelsize": 12, "axes.labelweight": "bold",
        "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.edgecolor": "#9BB0C9", "axes.linewidth": 1.2,
        "axes.grid": True, "grid.color": "#C9D3E0", "grid.linestyle": ":",
        "grid.linewidth": 0.8, "grid.alpha": 0.9,
        "axes.axisbelow": True,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "xtick.color": "#333333", "ytick.color": "#333333",
        "legend.fontsize": 9, "legend.framealpha": 0.92,
        "legend.edgecolor": "#9BB0C9", "legend.fancybox": False,
        "lines.linewidth": 2.0, "lines.markersize": 6,
    })


def finish(ax, *, end_labels: dict[str, tuple[float, float, str]] | None = None,
           legend_loc: str = "best", tint: str | None = None,
           hline: tuple[float, str] | None = None) -> None:
    """Legend + reference line + direct end-labels.

    Legend and end labels are both drawn, not either: the legend carries identity
    for every series, and the end labels supply the "relief" the contrast check
    requires for the three hues that sit under 3:1 against white.

    `hline` draws the SFT reference line and its right-edge label, and RESERVES
    that label's box against the end labels. It lives here rather than in each
    figure because it was copy-pasted into both trajectory figures, and because
    only this function knows where the end labels are going -- when three arms
    finish near the baseline, their labels printed straight through it.
    """
    if tint:
        ax.set_facecolor(PANEL_TINT[tint])
    handles, _ = ax.get_legend_handles_labels()
    if len(handles) >= 2:
        ax.legend(loc=legend_loc)
    # End labels get a white halo: they sit over gridlines and sometimes over a
    # neighbouring series, and an unhaloed bold label on a dotted grid is the
    # collision the palette validator cannot see.
    # COLLISION IN BOTH AXES. The first version compared y alone, in data units,
    # so two arms whose lines END at different steps still collided: "gated, edge
    # pool" finishing at step 50 ran its text straight through "guided" at step
    # 70, three points below it. Labels are boxed in DISPLAY space -- width
    # estimated from the string, height from the font -- and only shifted when the
    # boxes actually intersect.
    fig = ax.figure
    scale = fig.dpi / 72.0
    fontsize, pad = 8.5, 6.0
    h = (fontsize + 2.5) * scale
    y_top = ax.transData.transform((0, ax.get_ylim()[1]))[1]
    placed: list[tuple[float, float, float]] = []
    if hline is not None:
        hy, htext = hline
        ax.axhline(hy, color=BASELINE, lw=1.6, ls="-.", zorder=1)
        ax.annotate(htext, xy=(0.99, hy), xycoords=("axes fraction", "data"),
                    xytext=(0, 6), textcoords="offset points", ha="right",
                    fontsize=9, fontweight="bold", color=BASELINE)
        hx1 = ax.transAxes.transform((0.99, 0))[0]
        placed.append((hx1 - len(htext) * 9 * 0.62 * scale, hx1,
                       ax.transData.transform((0, hy))[1] + 6 * scale))
    for name, (x, y, color) in sorted((end_labels or {}).items(), key=lambda kv: -kv[1][1]):
        px, py = ax.transData.transform((x, y))
        x0 = px + pad * scale
        x1 = x0 + (len(name) * fontsize * 0.62 + pad) * scale
        def hits(cy):
            return any(not (x1 < qx0 or x0 > qx1) and abs(cy - qy) < h
                       for qx0, qx1, qy in placed)
        dy = 0.0
        while hits(py + dy) and py + dy + h < y_top:
            dy += h
        while hits(py + dy):          # ran out of headroom -- go down instead
            dy -= h
        placed.append((x0, x1, py + dy))
        # LEADER LINE when the label has been pushed far from its point. With
        # three or four arms the de-collision shift is small and the label still
        # reads as belonging to the nearest line. With six it is not: in the
        # pass@1 panel the lr6 arms pushed "gated", "guided" and "gated, edge
        # pool" 8pp above their own line ends, where they sat directly beside
        # OTHER arms' lines. A displaced label without a connector does not just
        # look untidy, it attributes the wrong value to the wrong series.
        # Threshold is one label height: below that the label still touches its
        # own line and a connector would be visual noise.
        leader = dict(arrowstyle="-", color=color, lw=0.8, alpha=0.55,
                      shrinkA=1.0, shrinkB=2.0) if abs(dy) > h else None
        ax.annotate(name, xy=(x, y), xytext=(pad, dy / scale), textcoords="offset points",
                    va="center", ha="left", fontsize=fontsize, fontweight="bold", color=color,
                    arrowprops=leader, zorder=2,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])


def callout(ax, text, xy, xytext, color=ACCENT):
    """The paper's red dashed annotation box."""
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=9.5, fontweight="bold",
                color=color, ha="center",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color,
                          lw=1.4, ls="--"),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.4, ls="--"))
