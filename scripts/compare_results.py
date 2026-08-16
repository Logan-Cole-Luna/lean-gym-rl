#!/usr/bin/env python3
"""Aggregate cached evaluation results into one comparison table.

Each `scripts/evaluate_checkpoints.py` run writes a JSON keyed by checkpoint
label. Re-scoring every checkpoint just to add one new row is wasteful -- BEq+
runs a bidirectional Lean tactic search per example and takes tens of minutes.
So: evaluate only the NEW checkpoint, then merge its JSON with the cached ones
here.

STALE RESULTS ARE EXCLUDED BY NAME. `results/ablation_comparison.json` was
produced before the chat-template bug was fixed in evaluate_checkpoints.py --
it fed raw prompts to vLLM while both verl's RL rollouts and the SFT trainer
apply the chat template during training, so every model was scored
off-distribution (SFT came out at 0.0% when it is actually ~33%). Those numbers
must never be mixed into a comparison. If you add new result files, make sure
they were produced *after* that fix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"

# Produced before the chat-template fix -- see module docstring.
STALE = {"ablation_comparison.json"}
# Not a checkpoint comparison (reward-rung occupancy probe).
NOT_A_COMPARISON = {"rung_probe.json"}

# Reading order for the table: pipeline stage, then reward.
ORDER = [
    "base (untrained)",
    "typecheck_only-step30",
    "composite-step30",
    "guided-step30",
    "curriculum_p1_typecheck-step15",
    "sft-step30",
    "rl_from_sft-step15",
]

DESCRIPTIONS = {
    "base (untrained)": "no training",
    "typecheck_only-step30": "RL from scratch, type-check reward",
    "composite-step30": "RL from scratch, 0.1*tc + 0.9*BEq+",
    "guided-step30": "RL from scratch, guided (similarity + BEq+)",
    "curriculum_p1_typecheck-step15": "curriculum phase 1 only",
    "sft-step30": "SFT on (informal -> gold formal)",
    "rl_from_sft-step15": "SFT -> RL, shaped reward",
}


def load(paths: list[Path]) -> tuple[dict, dict]:
    """label -> record, and label -> list of (file, beq_rate) for duplicates."""
    merged: dict[str, dict] = {}
    seen: dict[str, list] = {}
    for p in sorted(paths, key=lambda x: x.stat().st_mtime):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        for label, rec in data.items():
            if not isinstance(rec, dict) or "beq_plus_rate" not in rec:
                continue
            seen.setdefault(label, []).append((p.name, rec["beq_plus_rate"]))
            merged[label] = rec  # newest file wins
    return merged, seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--extra", nargs="*", default=[], help="additional result JSONs")
    args = ap.parse_args()

    d = Path(args.results_dir)
    paths = [p for p in d.glob("*.json") if p.name not in STALE and p.name not in NOT_A_COMPARISON]
    paths += [Path(x) for x in args.extra]
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise SystemExit(f"No result JSONs in {d}")

    merged, seen = load(paths)
    if not merged:
        raise SystemExit("No usable records found")

    print(f"cached result files used ({len(paths)}):")
    for p in sorted(paths):
        print(f"  - {p.name}")
    if STALE:
        print(f"excluded as stale (pre chat-template fix): {', '.join(sorted(STALE))}")

    ordered = [l for l in ORDER if l in merged] + [l for l in merged if l not in ORDER]

    print("\n" + "=" * 86)
    print(f"{'checkpoint':<38} {'type-check%':>12} {'BEq+%':>8}   {'n':>4}  description")
    print("-" * 86)
    for label in ordered:
        r = merged[label]
        print(f"{label:<38} {100*r['typecheck_rate']:>11.1f}% {100*r['beq_plus_rate']:>7.1f}% "
              f"{r.get('n', '?'):>5}  {DESCRIPTIONS.get(label, '')}")
    print("=" * 86)

    # Surface re-measurement spread: identical checkpoints scored in different
    # runs differ slightly because vLLM batching is not bit-deterministic even at
    # temperature 0. Differences below this spread are not real.
    dupes = {k: v for k, v in seen.items() if len({round(x[1], 4) for x in v}) > 1}
    if dupes:
        print("\nre-measurement spread (same checkpoint, different eval runs):")
        for label, entries in dupes.items():
            vals = ", ".join(f"{100*v:.1f}% ({f})" for f, v in entries)
            spread = 100 * (max(v for _, v in entries) - min(v for _, v in entries))
            print(f"  {label}: {vals}   -> spread {spread:.1f} pp")
        print("  Treat differences smaller than this spread as noise.")


if __name__ == "__main__":
    main()
