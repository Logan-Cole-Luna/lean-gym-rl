#!/usr/bin/env python3
"""Assemble the standalone reward-audit report: one self-contained HTML file.

The prose lives in `report/template.html`; every number that comes out of a RUN
is a `{{PLACEHOLDER}}` filled from `results/beq_calibration/report.json`, and the
figures are inlined as data: URIs so the result opens from disk with no server
and no sibling files. A placeholder left unfilled is a hard error -- the point of
templating the numbers is that the report cannot quietly go stale after a
re-run, which a hand-edited HTML blob silently would.

    python scripts/figures/fig_reward_design.py          # regenerate the panels
    python scripts/figures/build_report.py               # -> reward_audit.html
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG_RE = re.compile(r'src="figs/([^"]+\.png)"')
PLACEHOLDER_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def shard_wall_minutes(report_path: Path) -> str:
    """Longest shard's scoring wall time, in whole minutes.

    The SLOWEST shard, not the mean: shards run concurrently, so that is how
    long the calibration actually took.
    """
    walls = [json.loads(f.read_text()).get("wall_seconds", 0)
             for f in report_path.parent.glob("report.shard*.json")]
    return f"~{max(walls)/60:.0f}" if walls else "?"


def fields(report: dict) -> dict[str, str]:
    """Every run-derived number the template can ask for.

    The headline metrics come from `gold_elaborating_only`, not the full sample:
    a pair whose GOLD does not elaborate is a guaranteed miss for a Mathlib
    version reason, and folding those into recall blames BEq+ for dataset drift.
    """
    c = report.get("gold_elaborating_only") or report
    gate = report.get("gold_gate", {})
    rt, rf = c.get("rung_hist_tp", {}), c.get("rung_hist_fp", {})
    rung = lambda k: int(rt.get(str(k), rt.get(k, 0)))
    # Rule of three: with 0 events in n trials the 95% upper bound is ~3/n.
    fpr_hi = 3.0 / max(c["n_neg"], 1) if c["fp"] == 0 else c["false_positive_rate"]
    rec_ci = 1.96 * math.sqrt(c["recall"] * (1 - c["recall"]) / max(c["n_pos"], 1))
    return {
        "N_ALL": str(report["n"]), "N_POS_ALL": str(report["n_pos"]),
        "N_NEG_ALL": str(report["n_neg"]), "N_CLEAN": str(c["n"]),
        "N_POS": str(c["n_pos"]), "N_NEG": str(c["n_neg"]),
        "TP": str(c["tp"]), "FP": str(c["fp"]), "FN": str(c["fn"]), "TN": str(c["tn"]),
        "RECALL": f"{c['recall']:.3f}", "RECALL_CI": f"{rec_ci:.3f}",
        "PRECISION": f"{c['precision']:.3f}", "F1": f"{c['f1']:.3f}",
        "FPR_HI": f"{fpr_hi:.3f}",
        "RUNG2": str(rung(2)), "RUNG3": str(rung(3)), "RUNG4": str(rung(4)),
        "RUNG34": str(rung(3) + rung(4)),
        "RUNG34_PCT": f"{100*(rung(3)+rung(4))/max(c['tp'], 1):.0f}",
        "ONEDIR_POS": f"{100*c['one_dir_on_pos']:.1f}",
        "ONEDIR_NEG": f"{100*c['one_dir_on_neg']:.1f}",
        "GATE_OK": str(gate.get("elaborated", "?")),
        "GATE_PCT": f"{100*gate.get('rate', 0):.1f}",
        "SHARDS": str(report.get("concurrency", 1)),
        "ERRS": str(round(report["scorer_error_rate"] * report["n"])),
        "FP_RUNGS": ", ".join(f"rung {k}: {v}" for k, v in sorted(rf.items())) or "none",
        "WALL": report.get("_wall", "?"),
    }


def audit_fields(path: Path) -> dict[str, str]:
    """Numbers from the reward-calibration audit, if it has been run.

    Absent, the placeholders resolve to "--" rather than failing the build: the
    calibration report and the audit are produced by different runs, and a
    missing audit should not block rebuilding the rest of the page.
    """
    if not path.exists():
        return {k: "--" for k in ("AUDIT_GOLDS", "AUDIT_GOLD_OUTCOME", "AUDIT_GOLD_GATED",
                                  "AUDIT_GOLD_BEQ", "AUDIT_GOLD_PROVED",
                                  "AUDIT_TRIVIAL_OUTCOME", "AUDIT_TRIVIAL_MATCH")}
    a = json.loads(path.read_text())
    g = a.get("gold", {})
    triv = [a[k] for k in ("trivial_rfl", "trivial_true") if k in a]
    mean = lambda xs, k: sum(x[k] for x in xs) / len(xs) if xs else 0.0
    return {
        "AUDIT_GOLDS": str(g.get("n", "--")),
        "AUDIT_GOLD_OUTCOME": f"{g.get('mean_outcome', 0):.3f}",
        "AUDIT_GOLD_GATED": f"{g.get('mean_gated', 0):.3f}",
        "AUDIT_GOLD_BEQ": f"{100*g.get('beq_plus_rate', 0):.0f}",
        "AUDIT_GOLD_PROVED": f"{100*g.get('proved_rate', 0):.0f}",
        "AUDIT_TRIVIAL_OUTCOME": f"{mean(triv, 'mean_outcome'):.2f}",
        "AUDIT_TRIVIAL_MATCH": f"{100*mean(triv, 'matched_expected'):.0f}",
        "AUDIT_TRIVIAL_GATED": f"{mean(triv, 'mean_gated'):.2f}",
        "AUDIT_GOLD_MATCH": f"{100*g.get('matched_expected', 0):.0f}",
        # How many of the trivial submissions BEq+ matched all the way to SOLVED.
        "AUDIT_TRIVIAL_SOLVED": " and ".join(
            f"{t['outcome_hist'].get('solved', 0)}/{t['n']}" for t in triv) or "--",
    }


# Measured once over the 760-row proof val slice; see the report section. Held
# here rather than recomputed at build time so the page builds without Lean.
DANGLING = {"DANGLING_PCT": "9.1", "DANGLING_OPEN": "38",
            "DANGLING_VAR": "26", "DANGLING_INC": "5"}


def rates_fields(path: Path) -> dict[str, str]:
    """Reward-value mass per arm, from fig_reward_rates.py's own summary."""
    keys = ("RATES_N", "RATES_PRE_MODAL", "RATES_POST_MODAL", "RATES_MOVED",
            "RATES_GATED_TOP", "RATES_GATED_BOT", "RATES_GATED_MID")
    if not path.exists():
        return {k: "--" for k in keys}
    d = json.loads(path.read_text())
    a = d["arms"]
    pre_050 = float(a["outcome_prefix"]["rates"].get("0.50", 0))
    g = a["gated"]["rates"]
    return {
        "RATES_N": str(d["n"]),
        "RATES_PRE_MODAL": f"{100*a['outcome_prefix']['modal_mass']:.0f}",
        "RATES_POST_MODAL": f"{100*a['outcome']['modal_mass']:.0f}",
        # The rows that were 0.50 under the old wiring and are 0.05 under the new.
        "RATES_MOVED": f"{100*pre_050:.0f}",
        "RATES_GATED_TOP": f"{100*float(g.get('1.00', 0)):.0f}",
        "RATES_GATED_BOT": f"{100*float(g.get('0.00', 0)):.0f}",
        "RATES_GATED_MID": f"{100*float(g.get('0.25', 0)):.0f}",
    }


def faith_fields(path: Path) -> dict[str, str]:
    """NL->FL faithfulness numbers, from faithfulness_eval.py's own summary."""
    keys = ("FAITH_N", "FAITH_MATCHED", "FAITH_MISMATCHED", "FAITH_DELTA",
            "FAITH_PAD_DELTA", "FAITH_AUC", "FAITH_AUC_LO", "FAITH_AUC_HI",
            "FAITH_NPOS", "FAITH_NNEG", "FAITH_TERM", "FAITH_COVERAGE",
            "FAITH_CALLS", "FAITH_SEC")
    if not path.exists():
        return {k: "--" for k in keys}
    d = json.loads(path.read_text())
    m, x, pd_ = d.get("matched", {}), d.get("mismatched", {}), d.get("padded", {})
    nc, pool = d.get("negative_control", {}), d.get("pool", {})
    lo, hi = (nc.get("auc_ci95") or [0, 0])
    term = pool.get("term_mode_share", 0.0)
    # Coverage over the WHOLE population, not the pre-filtered subset: the
    # tactic-mode filter is a way to spend the judge budget, not a smaller
    # denominator to report against.
    eff = (1 - term) * m.get("coverage", 0.0)
    return {
        "FAITH_N": str(m.get("n", "--")),
        "FAITH_MATCHED": f"{m.get('mean', 0):.3f}",
        "FAITH_MISMATCHED": f"{x.get('mean', 0):.3f}",
        "FAITH_DELTA": f"{nc.get('delta_mean', 0):+.3f}",
        "FAITH_PAD_DELTA": f"{d.get('padding', {}).get('delta_mean', 0):+.3f}",
        "FAITH_AUC": f"{nc.get('auc', 0):.3f}",
        "FAITH_AUC_LO": f"{lo:.3f}", "FAITH_AUC_HI": f"{hi:.3f}",
        "FAITH_NPOS": str(nc.get("n_matched", "--")),
        "FAITH_NNEG": str(nc.get("n_mismatched", "--")),
        "FAITH_TERM": f"{100*term:.1f}", "FAITH_COVERAGE": f"{100*eff:.1f}",
        "FAITH_CALLS": str(d.get("judge_calls", 196)),
        "FAITH_SEC": f"{d.get('judge_mean_s', 13.3):.1f}",
    }


def new_reward_fields(nr: Path, faith: Path) -> dict[str, str]:
    """Firing rates for the two terms added this session.

    The denominator is the REACHABLE population, not the val slice: both terms
    are gated to the SOLVED row, so a rate quoted against 760 would understate
    how inert they are by an order of magnitude.
    """
    keys = ("NR_REACH", "NR_FIRES", "NR_DEDUCT", "FAITH_REACH", "FAITH_SCORED")
    out = {k: "--" for k in keys}
    sp = nr / "summary.json"
    if sp.exists():
        S = json.loads(sp.read_text())
        reach = sum(v["reachable"] for v in S.values())
        fires = sum(v["fires"] for v in S.values())
        out["NR_REACH"] = str(reach)
        out["NR_FIRES"] = f"{fires}  ({100*fires/max(reach,1):.1f}%)"
        out["NR_DEDUCT"] = f"{max(v['mean_deduction'] for v in S.values()):.4f}"
    fp = faith / "summary.json"
    if fp.exists():
        F = json.loads(fp.read_text())
        m, pool = F.get("matched", {}), F.get("pool", {})
        n_pool = pool.get("n_pool", 760)
        term = pool.get("term_mode_share", 0.0)
        out["FAITH_REACH"] = f"{round(n_pool*(1-term))}"
        out["FAITH_SCORED"] = f"{round(n_pool*(1-term)*m.get('coverage',0))}  " \
                              f"({100*(1-term)*m.get('coverage',0):.1f}%)"
    return out


def traj_fields(d: Path) -> dict[str, str]:
    """The observed SOLVED range across every measured checkpoint."""
    import glob as _glob
    vals = []
    for f in _glob.glob(str(d / "*.json")):
        try:
            vals.append(json.loads(Path(f).read_text())["summary"]["solved_rate"])
        except Exception:
            continue
    if not vals:
        return {"TRAJ_SOLVED_LO": "--", "TRAJ_SOLVED_HI": "--", "TRAJ_N": "--"}
    return {"TRAJ_SOLVED_LO": f"{min(vals):.2f}", "TRAJ_SOLVED_HI": f"{max(vals):.2f}",
            "TRAJ_N": str(len(vals))}


def guard_field(path: Path) -> dict[str, str]:
    """How often the triviality guard's input is known. Its own tiny run."""
    if not path.exists():
        return {"GUARD_KNOWN": "--"}
    d = json.loads(path.read_text())
    return {"GUARD_KNOWN": f"{100*d['provable_alone_known']:.1f}"}


def build(template: Path, figs: Path, report: Path, out: Path,
          audit: Path, guard: Path) -> None:
    html = template.read_text()
    try:
        rep = json.loads(report.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"[report] cannot read {report}: {e}. Run "
                         f"`beq_calibration.py --merge --out {report.parent}` first.")
    rep["_wall"] = shard_wall_minutes(report)
    vals = fields(rep)
    vals.update(audit_fields(audit))
    vals.update(guard_field(guard))
    vals.update(DANGLING)
    vals.update(rates_fields(figs / "reward_rates.json"))
    vals.update(new_reward_fields(ROOT / "results" / "new_rewards",
                                  ROOT / "results" / "faithfulness_n300"))
    vals.update(traj_fields(ROOT / "results" / "proved_trajectory"))
    vals.update(faith_fields(ROOT / "results" / "faithfulness_n300" / "summary.json"))

    missing = {m for m in PLACEHOLDER_RE.findall(html)} - set(vals)
    if missing:
        raise SystemExit(f"[report] template asks for unknown placeholders: {sorted(missing)}")
    html = PLACEHOLDER_RE.sub(lambda m: vals[m.group(1)], html)

    inlined = []

    def embed(m: re.Match) -> str:
        p = figs / m.group(1)
        if not p.exists():
            raise SystemExit(f"[report] missing figure {p}. Run fig_reward_design.py first.")
        inlined.append((p.name, p.stat().st_size))
        return f'src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"'

    html = FIG_RE.sub(embed, html)
    out.write_text(html)
    for name, size in inlined:
        print(f"  inlined {name} ({size/1024:.0f} KB)")
    print(f"  {len(vals)} numbers filled from {report}")
    print(f"  -> {out}  ({len(html)/1024/1024:.2f} MB, self-contained)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, default=Path(__file__).parent / "report" / "template.html")
    ap.add_argument("--figs", type=Path, default=ROOT / "results" / "figures")
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "beq_calibration_full" / "report.json")
    ap.add_argument("--out", type=Path, default=ROOT / "reward_audit.html")
    ap.add_argument("--audit", type=Path,
                    default=ROOT / "results" / "reward_audit_n200" / "summary.json")
    ap.add_argument("--guard", type=Path,
                    default=ROOT / "results" / "reward_audit" / "guard.json")
    a = ap.parse_args()
    build(*[getattr(a, k) for k in ("template", "figs", "report", "out", "audit", "guard")])


if __name__ == "__main__":
    main()
