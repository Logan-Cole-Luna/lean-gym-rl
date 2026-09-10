#!/usr/bin/env python3
"""Firing rates and distributions for the two reward terms added this session.

A reward term is only worth its complexity if it FIRES. Both terms here reach
the score on the `SOLVED` row alone, so the population that matters is not the
val slice -- it is the solved subset of it, and that subset is ~24%.

    brevity        pays a deduction when the proof body's length is far from the
                   gold's. Pure string work: no Lean, no judge.
    faithfulness   pays the 0.15 band when the Lean proof follows the English
                   argument. Needs both a Lean REPL and a judge, so its rates
                   come from `faithfulness_eval.py`; this script covers brevity
                   and joins the two into one firing-rate table.

`proved_rate.py` output supplies which rollouts are SOLVED, so no rescoring is
needed -- the join is on the val-row index.

    python scripts/eval/new_reward_rates.py --out results/new_rewards
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reward.reward import BREVITY_BAND, brevity_for
from reward.reward_fn import _proof_body_len

HPC = Path(os.path.expanduser("~/Downloads/results_sft"))


def load(arm: str, step: int) -> tuple[list[dict], dict[int, dict]]:
    g = glob.glob(str(HPC / "**" / f"gen_{arm}-step{step}_n*.jsonl"), recursive=True)
    if not g:
        return [], {}
    rows = [json.loads(l) for l in open(g[0]) if l.strip()]
    # Which rows are SOLVED, from the already-measured proved rates.
    pj = Path(f"results/proved_trajectory/{arm}-step{step}.json")
    solved = {}
    if pj.exists():
        for e in json.loads(pj.read_text())["per_example"]:
            solved[e["i"]] = e
    return rows, solved


def analyse(arm: str, step: int) -> dict | None:
    rows, solved = load(arm, step)
    if not rows:
        return None
    out = []
    for r in rows:
        pc, gc = _proof_body_len(r["pred"]), _proof_body_len(r["gold"])
        b = brevity_for(pc, gc)
        e = solved.get(r["i"])
        out.append({"i": r["i"], "b": b, "pred_chars": pc, "gold_chars": gc,
                    "beq_plus": bool(r.get("beq_plus")),
                    "proved": bool(e["proved"]) if e else None,
                    "solved": bool(e and e["proved"] and e["beq_plus"])})
    known = [x for x in out if x["b"] is not None]
    # The reachable population: brevity only touches the SOLVED row.
    reach = [x for x in known if x["solved"]]
    fires = [x for x in reach if x["b"] < 1.0]
    measured = [x for x in out if x["solved"] is not None]
    return {
        "arm": arm, "step": step, "n": len(out),
        "n_solved_measured": len(measured),
        "solved": sum(x["solved"] for x in measured),
        "b_measurable": len(known) / max(len(out), 1),
        "reachable": len(reach),
        "fires": len(fires),
        "firing_rate": len(fires) / max(len(reach), 1),
        "mean_b_on_reachable": (sum(x["b"] for x in reach) / len(reach)) if reach else None,
        "mean_deduction": (BREVITY_BAND * (1 - sum(x["b"] for x in reach) / len(reach))
                           if reach else 0.0),
        "records": out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/new_rewards"))
    ap.add_argument("--step", type=int, default=90)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for arm in ("rl3b_locolib_proof_lr6_gated", "rl3b_locolib_proof_lr6_outcome",
                "rl3b_locolib_proof_lr6_typecheck"):
        a = analyse(arm, args.step)
        if not a:
            continue
        (args.out / f"brevity.{arm}-step{args.step}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in a.pop("records")))
        summary[arm.split("lr6_")[1]] = a

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{'arm':11s} {'n':>5} {'solved':>7} {'reachable':>10} {'fires':>6} "
          f"{'firing rate':>12} {'mean b':>8} {'mean deduction':>15}")
    for arm, v in summary.items():
        print(f"{arm:11s} {v['n']:>5} {v['solved']:>7} {v['reachable']:>10} "
              f"{v['fires']:>6} {100*v['firing_rate']:>11.1f}% "
              f"{v['mean_b_on_reachable']:>8.4f} {v['mean_deduction']:>15.4f}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
