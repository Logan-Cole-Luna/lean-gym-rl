#!/usr/bin/env python3
"""Read verl's per-step training metrics for an arm.

WHY THIS EXISTS. verl's `console` logger prints one `step:N - key:val - ...`
line per step from inside the TaskRunner Ray actor, and Ray's log forwarding
dedups and drops those under load -- which is why most arms in results/ have no
readable reward, KL, entropy or clip_ratio trace at all. configs/run_grpo.sh now
also enables verl's `file` backend, which writes {"step": N, "data": {...}}
JSONL unbuffered. This reads it back.

ONE FILE PER SLURM JOB. FileLogger opens its path with mode "wb", which
truncates, and arms run as chained afterany chunks with resume_mode=auto -- so
run_grpo.sh gives each job its own file and this globs the prefix and merges.
Later chunks win on a repeated step, since a resumed chunk re-runs the step it
was killed in.

    python scripts/figures/read_train_metrics.py rl3b_lr6_gated_edge
    python scripts/figures/read_train_metrics.py rl3b_lr6_gated_edge --keys actor/kl_loss,actor/entropy
    python scripts/figures/read_train_metrics.py rl3b_lr6_gated_edge --csv out.csv
"""
import argparse
import csv
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path(os.getenv("VERL_FILE_LOGGER_ROOT", REPO / "results/train_metrics"))

# The handful worth looking at first: is the reward moving, is the policy
# staying near the SFT reference, is it still exploring, is it hitting the cap.
DEFAULT_KEYS = ("critic/score/mean", "critic/rewards/mean", "actor/kl_loss",
                "actor/entropy", "actor/grad_norm", "response_length/mean",
                "response_length/clip_ratio")


def load(experiment: str, root: Path = DEFAULT_ROOT, project: str = "beqplus_rl_poc") -> dict[int, dict]:
    """step -> metrics, merged across this experiment's per-job files."""
    files = sorted((root / project).glob(f"{experiment}.*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"no metric files for {experiment!r} under {root / project}\n"
                         f"(runs started before the file logger landed have none -- "
                         f"only stdout, which is why this script exists)")
    by_step: dict[int, dict] = {}
    for f in files:                       # mtime order, so a resumed chunk wins
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                  # a chunk killed mid-write leaves a partial tail
            by_step[int(rec["step"])] = rec["data"]
    print(f"[metrics] {experiment}: {len(by_step)} steps from {len(files)} file(s)")
    return by_step


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", help="e.g. rl3b_lr6_gated_edge")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--project", default="beqplus_rl_poc")
    ap.add_argument("--keys", default=",".join(DEFAULT_KEYS),
                    help="comma-separated metric keys; 'all' for every key present")
    ap.add_argument("--every", type=int, default=10, help="print every Nth step")
    ap.add_argument("--csv", type=Path, help="also write the full series here")
    args = ap.parse_args()

    by_step = load(args.experiment, args.root, args.project)
    steps = sorted(by_step)
    present = sorted({k for s in steps for k in by_step[s]})
    keys = present if args.keys == "all" else [k for k in args.keys.split(",") if k in by_step[steps[-1]]]
    missing = [] if args.keys == "all" else [k for k in args.keys.split(",") if k not in present]

    print(f"{'step':>6}" + "".join(f"{k.split('/')[-1]:>14}" for k in keys))
    for s in steps:
        if s % args.every and s != steps[-1]:
            continue
        row = "".join(f"{by_step[s].get(k, float('nan')):>14.4f}" if isinstance(by_step[s].get(k), (int, float))
                      else f"{'-':>14}" for k in keys)
        print(f"{s:>6}{row}")
    if missing:
        print(f"[metrics] not logged by this run: {', '.join(missing)}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["step"] + present)
            for s in steps:
                w.writerow([s] + [by_step[s].get(k, "") for k in present])
        print(f"[metrics] wrote {args.csv} ({len(steps)} rows x {len(present)} keys)")


if __name__ == "__main__":
    main()
