#!/usr/bin/env python3
"""Two-arm cut of arm_trajectories_pass1: gated (BEq+) vs type-check only.

The presentation figure -- the reward-quality contrast with nothing else in the
frame. Everything comes from make_figures.fig_arm_trajectories_pass1; this only
sets the HIDE set, one ARM_STYLE entry, the step grid and the titles, all of
which that function reads at call time. It is a wrapper, not a fork, so changes
to the underlying figure still flow through.

    python scripts/fig_gated_vs_typecheck.py            # the 1e-5 arms
    python scripts/fig_gated_vs_typecheck.py --series lr6  # the corrected LR

WHAT "gated" IS HERE. The line is the rl3b_gated_edge run. That arm's reward IS
compute_score_gated -- byte-identical to the `gated` arm; the only difference is
the prompt pool (data_3b/train_edge.parquet, prompts scoring 1-7/8 at k=8). The
legend names the REWARD, so it is accurate. The pool belongs in the caption, as
does n=250, which the trimmed suptitle no longer carries.
"""
import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import figstyle as fs
import make_figures as mf

ap = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--series", default="", choices=["", "lr6"],
                help="'' = the accidental 1e-5 arms; 'lr6' = ACTOR_LR=1e-6 + warmup")
args = ap.parse_args()

PREFIX = f"{args.series}_" if args.series else ""
SUFFIX = f"_{args.series}" if args.series else ""
GATED = f"rl3b_{PREFIX}gated_edge"
TYPECHECK = f"rl3b_{PREFIX}typecheck"
KEEP = {TYPECHECK, GATED}
OUT = REPO / f"results/figures/arm_trajectories_pass1_gated_vs_typecheck{SUFFIX}.png"
PANEL_TITLES = ["Semantic Equivalence", "Syntax"]
# The LR belongs in the title. These two series are NOT comparable to each other
# as rewards -- they differ in the optimiser -- so a figure that does not say
# which one it is invites exactly that comparison.
SUPTITLE = "pass@1, T=1.15" + (", lr 1e-6" if args.series == "lr6" else "")

# Drop "edge" from the label and give it the canonical gated styling, so the
# figure reads as the two-reward comparison it is.
fs.ARM_STYLE[GATED] = dict(color=fs.BLUE, marker="D", label="gated (BEq+)")
fs.ARM_STYLE[TYPECHECK] = dict(color=fs.ORANGE, marker="s", label="type-check only")

fs.use_style()
mf.HIDE = {a for a in fs.ARM_STYLE if a not in KEEP}

# CUT BOTH ARMS AT THE LAST STEP THE GATED ARM ACTUALLY HAS.
# type-check has pass@k at every grid step; gated_edge trails, because its
# pass@k jobs land one checkpoint at a time. Drawing each arm to its own end
# puts unopposed steps of type-check decay in the frame, which the eye reads as
# part of the comparison when there is nothing beside it. Derived from what is
# on disk, so this extends itself as pass@k jobs land.
steps_have = set()
for f in (REPO / "results").glob("passk_" + GATED + "-step*_k32.json"):
    m = re.search(r"step(\d+)", f.name)
    if m:
        steps_have.add(int(m.group(1)))
if not steps_have:
    sys.exit("no pass@k files found for " + GATED)
last = max(steps_have)
grid = tuple(s for s in mf.evalio.STEP_GRID if s <= last)
print(GATED + " has pass@k at " + str(sorted(steps_have))
      + "; cutting grid at " + str(last) + " -> " + str(grid))

mf.GRID = grid
fig = mf.fig_arm_trajectories_pass1(grid)
if fig is None:
    sys.exit("no data drawn")

# Retitle after the fact rather than forking the figure function.
axes = fig.axes[:len(PANEL_TITLES)]
if len(axes) != len(PANEL_TITLES):
    sys.exit("expected %d panels, got %d" % (len(PANEL_TITLES), len(fig.axes)))
for ax, title in zip(axes, PANEL_TITLES):
    ax.set_title(title)
fig.suptitle(SUPTITLE, fontsize=13, fontweight="bold")
fig.tight_layout()

fig.savefig(OUT)
print("wrote " + str(OUT))
