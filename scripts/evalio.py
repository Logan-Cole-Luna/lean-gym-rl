"""Canonical readers for cached eval / pass@k results.

WHY THIS EXISTS. Four scripts had grown their own copy of "glob the eval JSONs,
pull per_example, compute a rate" -- make_figures, make_arms_table, compare_arms
and select_checkpoint -- and they had already drifted apart in two ways that
matter:

  * whether a step evaluated at several n keeps the largest or the first found;
  * whether the rate is recomputed over a fixed prefix or read from `*_rate`.

The second is the dangerous one. `*_rate` describes whatever n its own file was
written at, and arms in this project were evaluated at n=400 and n=1000, so
mixing the two silently compares different example sets. Everything here
recomputes over an explicit prefix and never reads `*_rate`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
BASELINE_LABEL = "sft3b-step93"


def load_arm_steps(arm: str, results: Path = RESULTS) -> dict[int, list[dict]]:
    """step -> per_example, keeping the LARGEST n available for each step."""
    out: dict[int, list[dict]] = {}
    for p in results.glob(f"eval_{arm}-step*_n*.json"):
        m = re.search(r"-step(\d+)_n\d+\.json$", p.name)
        if not m:
            continue
        rec = next(iter(json.loads(p.read_text()).values()))
        pe = rec.get("per_example")
        if not pe:
            continue
        step = int(m.group(1))
        if step not in out or len(pe) > len(out[step]):
            out[step] = pe
    return out


def load_baseline(results: Path = RESULTS, label: str = BASELINE_LABEL) -> dict:
    """The full eval record for the SFT checkpoint every arm starts from."""
    f = results / f"eval_{label}_n1000.json"
    return next(iter(json.loads(f.read_text()).values()))


def rate(per_example: list[dict], key: str = "beq_plus", n: int | None = None) -> float:
    """Percentage over the first n records. Never reads `*_rate` -- see module docstring."""
    pe = per_example[:n] if n else per_example
    if not pe:
        return float("nan")
    return 100 * sum(bool(x[key]) for x in pe) / len(pe)


def load_passk(results: Path = RESULTS, metric: str = "beq_plus") -> dict[str, dict[int, dict]]:
    """arm -> step -> pass@k record, for one metric.

    Arms are keyed by step. Baselines (anything not matching `rl3b_*-step<N>`)
    land under '__baseline__' keyed by their label, so several can coexist --
    `sft3b-step93` and `sft3bmt` are different models, and an earlier version
    collapsed them onto the same slot.

    FILTER ON `metric`. The same label now has a record per metric (`beq_plus`
    from the pass@k job, `typecheck` derived from the same scored rollouts), and
    keying by label alone lets whichever file globs last silently win.
    """
    out: dict[str, dict[int, dict]] = {}
    for f in results.glob("passk_*_k*.json"):
        d = json.loads(f.read_text())
        if d.get("metric", "beq_plus") != metric:
            continue
        lab = d.get("label", "")
        # `sft3b-step93` also matches "<name>-step<N>", so keying on the pattern
        # alone files the BASELINE as an arm called "sft3b". Arms are the rl3b_*
        # runs; everything else is a baseline at step 0.
        m = re.match(r"(rl3b_[A-Za-z0-9_]+)-step(\d+)$", lab)
        if m:
            out.setdefault(m.group(1), {})[int(m.group(2))] = d
        else:
            out.setdefault("__baseline__", {})[lab] = d
    return out


def passk_at(record: dict, k: int) -> float | None:
    v = next((r["rate"] for r in record["curve"] if r["k"] == k), None)
    return None if v is None else 100 * v


# THE REPORTING GRID. Every arm is evaluated at these steps and only these, so
# tables and figures compare like with like. It exists because the grid used to
# be whatever each arm happened to have: `guided` was the only arm with a
# step-70 pass@k, which put a row in arms.md that five of six arms could never
# fill, and it made the greedy trajectory (every 10 steps) silently denser than
# the sampled one (every 20-40).
#
# Steps OUTSIDE the grid are still evaluated and still on disk -- `typecheck`
# runs to 150 -- they are just not what results are reported at. Cost figures are
# the deliberate exception: see fig_runtime.
STEP_GRID = (10, 30, 50, 90)


def on_grid(steps: dict, grid: tuple[int, ...] = STEP_GRID) -> dict:
    """Subset a {step: value} mapping to the reporting grid."""
    return {s: v for s, v in steps.items() if s in grid}
