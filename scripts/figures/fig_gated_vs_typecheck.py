#!/usr/bin/env python3
"""Two-arm cut of arm_trajectories_pass1: gated (BEq+) vs type-check only.

The presentation figure -- the reward-quality contrast with nothing else in the
frame. Everything comes from make_figures.fig_arm_trajectories_pass1; this only
sets the HIDE set, one ARM_STYLE entry, the step grid and the titles, all of
which that function reads at call time. It is a wrapper, not a fork, so changes
to the underlying figure still flow through.

    python scripts/figures/fig_gated_vs_typecheck.py            # the 1e-5 arms
    python scripts/figures/fig_gated_vs_typecheck.py --series lr6  # the corrected LR

WHAT "gated" IS HERE. The line is the rl3b_gated_edge run. That arm's reward IS
compute_score_gated -- byte-identical to the `gated` arm; the only difference is
the prompt pool (data_3b/train_edge.parquet, prompts scoring 1-7/8 at k=8). The
legend names the REWARD, so it is accurate. The pool belongs in the caption, as
does n=250, which the trimmed suptitle no longer carries.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: F401  -- repo root + every stage folder

import evalio
import figstyle as fs
import make_figures as mf

ap = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--series", default="", choices=["", "lr6"],
                help="'' = the accidental 1e-5 arms; 'lr6' = ACTOR_LR=1e-6 + warmup")
ap.add_argument("--source", default="greedy", choices=["greedy", "passk"],
                help="greedy = eval_*.json (every arm, every 10 steps); "
                     "passk = passk_*.json (pass@1 of the sampled curve)")
ap.add_argument("--typecheck-arms", default="",
                help="comma-separated override of the type-check arms to draw")
args = ap.parse_args()

GATED = f"rl3b_{args.series}_gated_edge" if args.series else "rl3b_gated_edge"

# DRAW BOTH TYPE-CHECK ARMS, BECAUSE THE POOL IS THE POINT.
# The gated line is always the EDGE-pool run. There are two type-check arms at
# 1e-6 and they differ from each other only in the prompt pool:
#
#   rl3b_lr6_typecheck      full pool  -- differs from gated in reward AND pool
#   rl3b_lr6edge_typecheck  edge pool  -- differs from gated in reward ONLY
#
# Pairing gated against the full-pool arm alone showed BEq+ ahead by 3.2pp and
# read as a reward effect. The pool-matched arm puts the two rewards dead level
# (25 wins each, p=1.0, no step separated by more than 0.7pp), i.e. the 3.2pp
# was the POOL. Drawing only one of them makes the figure argue for whichever
# was picked, so it draws both and the reader can see the pool move and the
# reward move separately.
DEFAULT_TCS = {"": ["rl3b_typecheck"],
               "lr6": ["rl3b_lr6_typecheck", "rl3b_lr6edge_typecheck"]}[args.series]
TYPECHECKS = [a for a in args.typecheck_arms.split(",") if a] or DEFAULT_TCS
KEEP = {GATED, *TYPECHECKS}
_tag = (f"_{args.series}" if args.series else "") + \
       ("_custom" if TYPECHECKS != DEFAULT_TCS else "")
_stem = {"greedy": "arm_trajectories_greedy_gated_vs_typecheck",
         "passk": "arm_trajectories_pass1_gated_vs_typecheck"}[args.source]
OUT = REPO / ("results/figures/" + _stem + _tag + ".png")
PANEL_TITLES = ["Semantic Equivalence", "Syntax"]
# The LR belongs in the title. These two series are NOT comparable to each other
# as rewards -- they differ in the optimiser -- so a figure that does not say
# which one it is invites exactly that comparison.
SUPTITLE = ("greedy T=0, n=1000" if args.source == "greedy" else "pass@1, T=1.15") \
           + (", lr 1e-6" if args.series == "lr6" else "")

# "edge" stays out of the visible labels, per how this figure is captioned; the
# pool is named as "same pool" / "full pool", which is the distinction that
# actually matters to a reader and does not require knowing what the edge pool
# is. `end_label` is set explicitly because these labels share a prefix and the
# end-label map is keyed on that string -- see figstyle.ARM_STYLE.
fs.ARM_STYLE[GATED] = dict(color=fs.BLUE, marker="D",
                           label="gated (BEq+)", end_label="gated (BEq+)")
_TC_STYLE = {
    "rl3b_typecheck":         dict(color=fs.ORANGE, marker="s", linestyle="-",
                                   label="type-check only",
                                   end_label="type-check"),
    "rl3b_lr6_typecheck":     dict(color=fs.ORANGE, marker="s", linestyle="-",
                                   label="type-check, full pool",
                                   end_label="type-check,\nfull pool"),
    "rl3b_lr6edge_typecheck": dict(color=fs.ORANGE, marker="P", linestyle="--",
                                   label="type-check, same pool as gated",
                                   end_label="type-check,\nsame pool"),
}
for _a in TYPECHECKS:
    if _a in _TC_STYLE:
        fs.ARM_STYLE[_a] = _TC_STYLE[_a]

fs.use_style()
mf.HIDE = {a for a in fs.ARM_STYLE if a not in KEEP}

# CUT BOTH ARMS AT THE LAST STEP THE GATED ARM ACTUALLY HAS.
# type-check has pass@k at every grid step; gated_edge trails, because its
# pass@k jobs land one checkpoint at a time. Drawing each arm to its own end
# puts unopposed steps of type-check decay in the frame, which the eye reads as
# part of the comparison when there is nothing beside it. Derived from what is
# on disk, so this extends itself as pass@k jobs land.
# evalio.load_passk owns the pass@k index, including the metric filter inside
# each JSON that keeps the `_tc` records separate. Globbing the filenames here
# bypassed that and hardcoded k=32, so a job at another k read as "no data".
if args.source == "greedy":
    arms = {a: mf.load_arm(a) for a in KEEP}
    empty = [a for a, v in arms.items() if not v]
    if empty:
        print("WARNING: no eval records for " + ", ".join(sorted(empty)))
    arms = {a: v for a, v in arms.items() if v}
    for a, v in sorted(arms.items()):
        print(f"  {a:26} greedy steps {sorted(v)}")
    fig = mf.fig_greedy_two_panel(arms, 1000)
else:
    # AN ARM WITH NO pass@k DATA MUST NOT VANISH QUIETLY.
    # This path is pass@k-driven, so an arm with eval JSONs but no passk_*.json
    # simply does not draw -- which is how rl3b_lr6edge_typecheck stayed
    # invisible while sitting on disk, and how a default swap emptied the
    # type-check side of this figure with no error. Say what is missing.
    _by = evalio.load_passk(metric="beq_plus")
    _missing = [a for a in KEEP if not _by.get(a)]
    if _missing:
        print("WARNING: no pass@k records for " + ", ".join(sorted(_missing)))
        print("         those arms will be ABSENT. Generate them with:")
        for a in sorted(_missing):
            print(f"           for s in 10 30 50 90; do sbatch --export=ALL,"
                  f"CKPT_DIR=$PWD/checkpoints/merged/{a}-step$s,LABEL={a}-step$s "
                  f"hpc/passk.slurm; done")

    steps_have = set(_by.get(GATED, {}))
    if not steps_have:
        sys.exit("no pass@k records found for " + GATED)
    # CUT EVERY ARM AT THE LAST STEP THE GATED ARM ACTUALLY HAS, so unopposed
    # steps of type-check decay do not sit in the frame with nothing beside them.
    last = max(steps_have)
    grid = tuple(x for x in evalio.STEP_GRID if x <= last)
    print(GATED + " has pass@k at " + str(sorted(steps_have))
          + "; cutting grid at " + str(last) + " -> " + str(grid))
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
