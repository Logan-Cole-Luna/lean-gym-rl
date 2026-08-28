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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
import os

# Series identity. Override for another model, size or corpus; the defaults name
# the series currently on disk. `BASELINE_LABEL` is a fallback only -- readers
# call `discover_baseline_label()`, which prefers whatever SFT dir is actually
# on disk (see below) and only lands here when nothing matches.
RUN_PREFIX = os.environ.get("RUN_PREFIX", "rl3b")
BASELINE_LABEL = os.environ.get("BASELINE_LABEL", "sft3b-step93")
BASELINE_PREFIX = os.environ.get("BASELINE_PREFIX", "sft")


def load_arm_steps(arm: str, results: Path = RESULTS) -> dict[int, list[dict]]:
    """step -> per_example, keeping the LARGEST n available for each step."""
    out: dict[int, list[dict]] = {}
    for p in (results / "eval" / arm).glob(f"eval_{arm}-step*_n*.json"):
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


_EVAL_RE = re.compile(r"^eval_(?P<label>.+)-step(?P<step>\d+)_n(?P<n>\d+)\.json$")


def _eval_steps(d: Path) -> dict[int, int]:
    """step -> largest n, over eval_<dir>-step<N>_n<n>.json in one result dir."""
    out: dict[int, int] = {}
    for p in d.glob(f"eval_{d.name}-step*_n*.json"):
        m = _EVAL_RE.match(p.name)
        if not m or m["label"] != d.name:
            continue
        s, n = int(m["step"]), int(m["n"])
        out[s] = max(out.get(s, 0), n)
    return out


def discover_arms(run_prefix: str = RUN_PREFIX, results: Path = RESULTS) -> list[str]:
    """Arm directories under results/eval/ that carry at least one eval JSON.

    An arm is a directory named `<run_prefix>_*` holding an
    `eval_<name>-step<N>_n<n>.json`. `archive/` and the baseline SFT dir do not
    match the prefix, so they fall out. This is what lets `make figures` track
    whatever series is currently on disk instead of a hardcoded roster.
    """
    root = results / "eval"
    if not root.is_dir():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name == "archive":
            continue
        if run_prefix and not d.name.startswith(run_prefix + "_"):
            continue
        if _eval_steps(d):
            out.append(d.name)
    return out


def discover_baseline_label(prefix: str = BASELINE_PREFIX,
                            run_prefix: str = RUN_PREFIX,
                            results: Path = RESULTS) -> str | None:
    """The SFT checkpoint the arms resume from, as `<dir>-step<N>`.

    `BASELINE_LABEL` in the environment wins outright. Otherwise: the most
    recently written `<prefix>*` directory that is NOT itself an arm, taken at
    its latest step. Returns None when nothing matches, so callers can fall back.
    """
    env = os.environ.get("BASELINE_LABEL")
    if env:
        return env
    root = results / "eval"
    if not root.is_dir():
        return None
    best: tuple[float, str] | None = None
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name == "archive" or not d.name.startswith(prefix):
            continue
        if run_prefix and d.name.startswith(run_prefix + "_"):
            continue
        steps = _eval_steps(d)
        if not steps:
            continue
        mtime = max(p.stat().st_mtime
                    for p in d.glob(f"eval_{d.name}-step*_n*.json"))
        if best is None or mtime > best[0]:
            best = (mtime, f"{d.name}-step{max(steps)}")
    return best[1] if best else None


def union_steps(arms: dict[str, dict]) -> tuple[int, ...]:
    """Every GRPO step present across the given `arm -> {step: ...}` maps, sorted."""
    return tuple(sorted({s for steps in arms.values() for s in steps}))


def load_baseline(results: Path = RESULTS, label: str | None = None) -> dict:
    """The full eval record for the SFT checkpoint every arm starts from.

    `label` defaults to `discover_baseline_label()` (env override, else the
    newest SFT dir on disk), then to the module `BASELINE_LABEL`. The largest
    available n is used -- baselines across series have been written at n=1000,
    n=999 and n=760, and hardcoding one silently broke the read for the others.
    """
    label = label or discover_baseline_label(results=results) or BASELINE_LABEL
    d = results / "eval" / label.split("-step")[0]
    files = sorted(d.glob(f"eval_{label}_n*.json"),
                   key=lambda p: int(re.search(r"_n(\d+)\.json$", p.name).group(1)))
    if not files:
        raise FileNotFoundError(f"no baseline eval for {label!r} under {d}")
    return next(iter(json.loads(files[-1].read_text()).values()))


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
    for f in results.glob("eval/*/passk_*_k*.json"):
        d = json.loads(f.read_text())
        if d.get("metric", "beq_plus") != metric:
            continue
        lab = d.get("label", "")
        # `sft3b-step93` also matches "<name>-step<N>", so keying on the pattern
        # alone files the BASELINE as an arm called "sft3b". Arms are the rl3b_*
        # runs; everything else is a baseline at step 0.
        m = re.match(rf"({re.escape(RUN_PREFIX)}_[A-Za-z0-9_]+)-step(\d+)$", lab)
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


# ---- training-time step metrics (verl FileLogger) -------------------------------

TRAIN_METRICS = RESULTS / "train" / "train_metrics"
_JOB_SUFFIX_RE = re.compile(r"\.[^.]+\.jsonl$")


def discover_train_runs(root: Path = TRAIN_METRICS) -> dict[str, dict[int, dict]]:
    """experiment name -> {step: metrics dict}, merged across per-job files.

    verl's FileLogger opens its path with mode "wb" (truncates), and arms here
    run as chained afterany chunks that each get their own
    `<project>/<experiment>.<jobid>.jsonl`. Reading an arm therefore means
    globbing the prefix and letting a later job's row for a step win. Each line
    is `{"step": N, "data": {...}}`; the experiment name is the eval arm label,
    so figstyle.arm_style() paints the training and eval curves the same colour.
    """
    runs: dict[str, dict[int, dict]] = {}
    if not root.is_dir():
        return runs
    for f in sorted(root.glob("*/*.jsonl")):
        exp = _JOB_SUFFIX_RE.sub("", f.name)
        d = runs.setdefault(exp, {})
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            d[int(r["step"])] = r.get("data", r)
    return runs


def train_series(steps: dict[int, dict], key: str) -> tuple[list[int], list[float]]:
    """(xs, ys) for one metric key over sorted steps, skipping steps that lack it."""
    xs: list[int] = []
    ys: list[float] = []
    for s in sorted(steps):
        v = steps[s].get(key)
        if v is not None:
            xs.append(s)
            ys.append(float(v))
    return xs, ys
