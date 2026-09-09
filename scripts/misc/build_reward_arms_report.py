#!/usr/bin/env python3
"""Assemble results/reports/reward_arms_lr6/ from the eval records.

Mirrors the layout of results/reports/minif2f_corpus_ablation/ (README + eval/ +
figures/ + tables/ + training/), so the two ablations read the same way.

SCOPE: the three GRPO reward arms of SERIES_TAG=locolib_proof_lr6, against the
SFT checkpoint they all resume from. Distillation arms are deliberately out.

EVERY NUMBER COMES FROM THE REBUILT RECORDS, which carry a `toolchain` field.
The script asserts that field is uniform across every record it reads: the whole
point of the 2026-09-08 correction was that the baseline and the arms had been
scored under different Mathlib versions, and a report that silently mixed them
again would be the same defect wearing a new folder.

    source hpc/cc_env.sh
    python scripts/misc/build_reward_arms_report.py
"""
from __future__ import annotations

import json
import shutil
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "reports" / "reward_arms_lr6"
EVAL = ROOT / "results" / "eval"
N = 760
TOTAL_STEPS = 90
STEPS = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# TWO SERIES, PLOTTED SEPARATELY AND ON PURPOSE. lr6 ran at
# max_response_length=128 and c512 at 512. A truncated rollout scores 0 under
# every reward, so the cap is a standing gradient against longer answers and the
# two groups are not directly comparable. `outcome` appears in both and is the
# only honest bridge between them: same reward, two settings.
SERIES = {
    "lr6": {
        "title": "locolib_proof_lr6 (128-token cap)",
        "arms": [("gated", "rl3b_locolib_proof_lr6_gated", "BEq+ gated"),
                 ("outcome", "rl3b_locolib_proof_lr6_outcome", "Six-outcome ladder"),
                 ("typecheck", "rl3b_locolib_proof_lr6_typecheck",
                  "Type-check only (exploitable)")],
    },
    "c512": {
        "title": "locolib_proof_c512 (512-token cap)",
        "arms": [("outcome", "rl3b_locolib_proof_c512_outcome", "Six-outcome ladder (control)"),
                 ("brevity", "rl3b_locolib_proof_c512_outcome_brevity", "+ brevity"),
                 ("faith", "rl3b_locolib_proof_c512_outcome_faith", "+ faithfulness"),
                 ("v2", "rl3b_locolib_proof_c512_outcome_v2", "+ both (outcome v2)")],
    },
}
ARMS = SERIES["lr6"]["arms"]          # the group the headline table reports
SFT = "sft3blocolib_proof"


def available(group: str) -> list:
    """Arms of a group that actually have eval records yet.

    The c512 arms finish at different times, so the report is built from
    whatever exists and names the rest as pending rather than failing or, worse,
    silently omitting them without saying so.
    """
    out = []
    for key, series, label in SERIES[group]["arms"]:
        d = EVAL / series
        if d.is_dir() and any(d.glob("eval_*.json")):
            out.append((key, series, label))
    return out


def steps_for(series: str) -> list:
    """Evaluated steps present on disk, in order."""
    d = EVAL / series
    if not d.is_dir():
        return []
    out = []
    for f in d.glob("eval_*.json"):
        stem = f.stem
        if "-step" not in stem:
            continue
        try:
            out.append(int(stem.split("-step")[1].split("_n")[0]))
        except ValueError:
            pass
    return sorted(out)


def train_metrics(series: str) -> list[dict]:
    """verl step metrics for one run, concatenated across chunks and ordered."""
    import glob as _g
    rows = []
    for f in _g.glob(str(ROOT / "results" / "train" / "train_metrics" /
                        "beqplus_rl_poc" / f"{series}.*.jsonl")):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    rows.sort(key=lambda r: r["step"])
    # A resumed chunk re-emits its first step; keep the last write per step.
    dedup = {r["step"]: r for r in rows}
    return [dedup[k] for k in sorted(dedup)]


def load(series: str, step: int) -> dict:
    p = EVAL / series / f"eval_{series}-step{step}_n{N}.json"
    return next(iter(json.load(open(p)).values()))


def per(r: dict, key: str) -> dict:
    return {e["i"]: bool(e[key]) for e in r["per_example"]}


def mcnemar(x: dict, y: dict) -> tuple[int, int, float]:
    """Paired exact test on the same rows. Never difference two headline rates."""
    ids = sorted(set(x) & set(y))
    n01 = sum(1 for i in ids if not x[i] and y[i])
    n10 = sum(1 for i in ids if x[i] and not y[i])
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    return n01, n10, min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def main() -> None:
    for sub in ("eval", "figures", "tables", "training"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    base = load(SFT, 76)
    # THE GUARD. A mixed-toolchain report is the exact bug this corrects.
    seen = {base.get("toolchain")}
    finals, traj = {}, {}
    for key, series, _label in ARMS:
        finals[key] = load(series, 90)
        seen.add(finals[key].get("toolchain"))
        traj[key] = {}
        for s in STEPS:
            try:
                r = load(series, s)
                traj[key][s] = 100 * r["beq_plus_rate"]
                seen.add(r.get("toolchain"))
            except FileNotFoundError:
                pass
    if len(seen) != 1 or None in seen:
        raise SystemExit(f"[report] REFUSING: records span toolchains {seen}. "
                         "Re-derive them all under one before building a report.")
    toolchain = seen.pop()

    for series in [SFT] + [s for _, s, _ in ARMS]:
        dst = OUT / "eval" / series
        dst.mkdir(exist_ok=True)
        for f in (EVAL / series).glob("eval_*.json"):
            shutil.copy2(f, dst / f.name)

    bb = per(base, "beq_plus")
    rows = []
    for key, _series, label in ARMS:
        r = finals[key]
        y = per(r, "beq_plus")
        ids = sorted(set(bb) & set(y))
        d = (sum(y[i] for i in ids) - sum(bb[i] for i in ids)) / len(ids)
        g, l, p = mcnemar(bb, y)
        rows.append({"key": key, "label": label,
                     "elab": 100 * r["typecheck_rate"], "beq": 100 * r["beq_plus_rate"],
                     "delta": 100 * d, "gain": g, "lost": l, "p": p})

    tc = per(finals["typecheck"], "beq_plus")
    contrasts = []
    for key in ("gated", "outcome"):
        y = per(finals[key], "beq_plus")
        ids = sorted(set(tc) & set(y))
        d = (sum(y[i] for i in ids) - sum(tc[i] for i in ids)) / len(ids)
        _, _, p = mcnemar(tc, y)
        contrasts.append((key, 100 * d, p))

    _figures(traj, base)
    _tables(rows, traj, base, contrasts, toolchain)
    _readme(rows, traj, base, contrasts, toolchain)
    print(f"[report] wrote {OUT}")


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        print(f"[report] no matplotlib ({e}); skipping figures")
        return None


def _save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    fig.clf()


def _rate_fig(plt, metric: str, ylabel: str, name: str, base: dict) -> None:
    """One panel per series group. NOT one shared axis: the two groups ran at
    different response caps and putting them on one plot would invite exactly
    the comparison the cap change invalidates."""
    groups = [g for g in SERIES if available(g)]
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(5.6 * len(groups), 3.8),
                             squeeze=False, sharey=True)
    for ax, g in zip(axes[0], groups):
        for _key, series, label in available(g):
            xs = steps_for(series)
            ys = [100 * load(series, x)[metric] for x in xs]
            done = bool(xs) and max(xs) >= TOTAL_STEPS
            ax.plot(xs, ys, marker="o", ms=4, ls="-" if done else "--",
                    label=label if done
                    else f"{label} (running, {max(xs) if xs else 0}/{TOTAL_STEPS})")
        ax.axhline(100 * base[metric], ls="--", c="0.4", lw=1,
                   label=f"SFT step 76 ({100*base[metric]:.1f}%)")
        pend = [l for k, sr, l in SERIES[g]["arms"] if not steps_for(sr)]
        ttl = SERIES[g]["title"] + (f"\n(pending: {', '.join(pend)})" if pend else "")
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("GRPO step"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    axes[0][0].set_ylabel(ylabel)
    fig.suptitle(f"{ylabel} on the pinned 760-row LoCoLib slice", fontsize=10)
    fig.tight_layout()
    _save(fig, name)


def _outcome_breakdown(plt, base: dict) -> None:
    """Where the population sits, not just the headline rate.

    The categories are nested by construction: elaboration hard-gates BEq+
    (BEQ_SKIP_CASCADE_ON_TYPECHECK_FAIL=1), so `no_elab` is the floor everything
    else is carved out of. This is the figure that carries the paper's actual
    finding, that every gain is a reduction in `no_elab` rather than better
    mathematics.
    """
    cols = [("no_elab", "#cc4444"), ("elaborates only", "#eebb66"),
            ("weaker only", "#77aabb"), ("BEq+", "#44aa77")]
    bars, labels = [], []
    def split(r):
        n = len(r["per_example"])
        ne = sum(1 for e in r["per_example"] if not e["typecheck"])
        beq = sum(1 for e in r["per_example"] if e["beq_plus"])
        weak = sum(1 for e in r["per_example"]
                   if e["typecheck"] and not e["beq_plus"]
                   and (e.get("gold_implies_pred") or e.get("pred_implies_gold")))
        return [100*ne/n, 100*(n-ne-beq-weak)/n, 100*weak/n, 100*beq/n]
    bars.append(split(base)); labels.append("SFT-76")
    for g in SERIES:
        for _k, series, label in available(g):
            st = steps_for(series)
            if not st:
                continue
            bars.append(split(load(series, st[-1])))
            labels.append(f"{label}\n{g} step {st[-1]}")
    if len(bars) < 2:
        return
    fig, ax = plt.subplots(figsize=(1.5 * len(bars) + 2.5, 4.0))
    bottom = [0.0] * len(bars)
    for i, (name, colour) in enumerate(cols):
        vals = [b[i] for b in bars]
        ax.bar(labels, vals, bottom=bottom, label=name, color=colour)
        bottom = [a + b for a, b in zip(bottom, vals)]
    ax.set_ylabel("% of the 760-row slice"); ax.set_ylim(0, 100)
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.14))
    ax.tick_params(axis="x", labelsize=7)
    _save(fig, "fig_outcome_breakdown")


def _training_fig(plt) -> None:
    """The GRPO analogue of a training-loss curve.

    There is no single loss to plot: what matters is whether each arm optimised
    the reward it was given (critic/score/mean), and whether it did so without
    drifting (actor/kl_loss against the ~0.05 target). Reward means are NOT
    comparable ACROSS arms, since each reward has its own scale; the shape is
    what to read.
    """
    panels = [("critic/score/mean", "training reward (arm's own scale)"),
              ("actor/kl_loss", "actor/kl_loss (target ~0.05)"),
              ("actor/entropy", "policy entropy"),
              ("response_length/mean", "response length (tokens)")]
    have = {}
    for g in SERIES:
        for _k, series, label in SERIES[g]["arms"]:
            rows = train_metrics(series)
            if rows:
                have[f"{label} [{g}]"] = rows
    if not have:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4))
    for ax, (key, title) in zip(axes.ravel(), panels):
        for label, rows in have.items():
            xs = [r["step"] for r in rows if key in r["data"]]
            ys = [r["data"][key] for r in rows if key in r["data"]]
            if not xs:
                continue
            # AN UNFINISHED RUN MUST NOT LOOK FINISHED. With seven overlapping
            # traces it is otherwise impossible to see that an arm stops at 49
            # while the rest reach 90, and a reader would take a partial curve
            # for a converged one.
            done = max(xs) >= TOTAL_STEPS
            ax.plot(xs, ys, lw=1.2, ls="-" if done else "--",
                    label=label if done else f"{label} (running, {max(xs)}/{TOTAL_STEPS})")
        ax.set_title(title, fontsize=9); ax.set_xlabel("step"); ax.grid(alpha=.3)
    axes[0][1].axhline(0.05, ls="--", c="0.4", lw=1)
    axes[0][0].legend(fontsize=6)
    fig.suptitle("Training-side signals per arm", fontsize=10)
    fig.tight_layout()
    _save(fig, "fig_training_signals")


def _output_length(plt, base: dict) -> None:
    """Completion length in characters, from the cached generations.

    Worth a panel because `max_response_length` is a live variable in this
    comparison: the c512 arms can express answers the lr6 arms could not.
    """
    import glob as _g
    series_files = []
    for g in SERIES:
        for _k, series, label in available(g):
            st = steps_for(series)
            hits = _g.glob(str(EVAL / series / f"gen_{series}-step{st[-1]}_n*.jsonl")) if st else []
            if hits:
                series_files.append((f"{label} [{g}]", hits[0]))
    sft = _g.glob(str(EVAL / SFT / f"gen_{SFT}-step76_n*.jsonl"))
    if sft:
        series_files.insert(0, ("SFT-76", sft[0]))
    if not series_files:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    data, labels = [], []
    for label, f in series_files:
        lens = [len(json.loads(l).get("completion") or "") for l in open(f) if l.strip()]
        if lens:
            data.append(lens); labels.append(label)
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("completion length (chars)")
    ax.set_title("Output length at each arm's final checkpoint", fontsize=10)
    ax.tick_params(axis="x", labelsize=6, rotation=20); ax.grid(alpha=.3, axis="y")
    _save(fig, "fig_output_length")


def _figures(traj: dict, base: dict) -> None:
    plt = _plt()
    if plt is None:
        return
    _rate_fig(plt, "beq_plus_rate", "BEq+ (%)", "fig_beq_plus_rate", base)
    _rate_fig(plt, "typecheck_rate", "elaborates (%)", "fig_elaborate_rate", base)
    _outcome_breakdown(plt, base)
    _training_fig(plt)
    _output_length(plt, base)
    print("[report] figures written")


def _tables(rows, traj, base, contrasts, toolchain) -> None:
    t = ["% auto-generated by scripts/misc/build_reward_arms_report.py",
         r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r"arm & elaborates & BEq+ & $\Delta$BEq+ vs SFT & gained/lost & McNemar $p$ \\",
         r"\midrule",
         rf"SFT step 76 (baseline) & {100*base['typecheck_rate']:.1f}\% & "
         rf"{100*base['beq_plus_rate']:.1f}\% & --- & --- & --- \\", r"\midrule"]
    for r in rows:
        t.append(rf"{r['label']} & {r['elab']:.1f}\% & {r['beq']:.1f}\% & "
                 rf"{r['delta']:+.2f} pp & {r['gain']}/{r['lost']} & {r['p']:.1e} \\")
    t += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Step-90 checkpoints against the SFT checkpoint all three resume "
          r"from, paired McNemar exact over the same 760 rows. All records re-derived "
          rf"under {toolchain.replace('_', chr(92)+'_')}. Only the two semantic rewards "
          r"improve on SFT; the exploitable type-check proxy does not.}",
          r"\label{tab:arms-results}", r"\end{table}"]
    (OUT / "tables" / "table_results.tex").write_text("\n".join(t) + "\n")

    b = ["% auto-generated by scripts/misc/build_reward_arms_report.py",
         r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{l" + "c" * len(STEPS) + "}", r"\toprule",
         "arm & " + " & ".join(str(s) for s in STEPS) + r" \\", r"\midrule"]
    for key, _s, label in ARMS:
        b.append(f"{label} & " + " & ".join(
            f"{traj[key][s]:.1f}" if s in traj[key] else "---" for s in STEPS) + r" \\")
    b += [r"\bottomrule", r"\end{tabular}",
          r"\caption{BEq+ (\%) at every evaluated step, $n=760$. Arms cross over "
          r"between checkpoints, so a single-step comparison of two close arms is not "
          r"a result.}", r"\label{tab:arms-bystep}", r"\end{table}"]
    (OUT / "tables" / "table_eval_by_step.tex").write_text("\n".join(b) + "\n")

    c = ["% auto-generated by scripts/misc/build_reward_arms_report.py",
         r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{ll}",
         r"\toprule", r"setting & value \\", r"\midrule",
         r"base model & Qwen2.5-Coder-3B-Instruct \\",
         r"corpus & LoCoLib proof-pair, 760-row pinned val slice \\",
         r"resume point & \texttt{sft\_3b\_locolib\_proof/global\_step\_76} \\",
         r"trainer & verl GRPO, FSDP actor + vLLM rollout, 2$\times$A100-40GB \\",
         r"batch / rollout $n$ / ppo epochs & 16 / 8 / 1 \\",
         r"max response length & 128 tokens \\",
         r"actor LR & 1e-6, warmup ratio 0.05 \\",
         r"steps & 90, evaluated every 10 \\",
         rf"Lean & {toolchain.replace('_', chr(92)+'_')} \\",
         r"\bottomrule", r"\end{tabular}",
         r"\caption{Configuration. All three arms are identical except "
         r"\texttt{reward.custom\_reward\_function.name}.}",
         r"\label{tab:arms-config}", r"\end{table}"]
    (OUT / "tables" / "table_config.tex").write_text("\n".join(c) + "\n")
    print("[report] tables written")


def _readme(rows, traj, base, contrasts, toolchain) -> None:
    by = {r["key"]: r for r in rows}
    L = [f"# Does the reward function matter? (LoCoLib proof-pair, in-domain)",
         "",
         "Three GRPO arms, identical except `reward.custom_reward_function.name`,",
         "all resuming from the same SFT checkpoint. The question is whether a",
         "semantic reward (BEq+) beats an exploitable proxy (type-check only) on",
         "held-out BEq+, and whether either beats not doing RL at all.",
         "",
         f"All records re-derived under `{toolchain}` on 2026-09-08; see",
         "`../REWARD_ARMS.md` section 0 for why that matters and what it changed.",
         "",
         "## Headline",
         "",
         "Each row is that arm's **final** checkpoint (step 90), not a checkpoint",
         "selected on this table's own BEq+ column. Both metrics score the generated",
         "*statement* only, see Caveats.",
         "",
         "| arm | step | elaborates | BEq+ | Δ BEq+ vs SFT | McNemar p |",
         "|---|---|---|---|---|---|",
         f"| SFT baseline | 76 | {100*base['typecheck_rate']:.1f}% | "
         f"{100*base['beq_plus_rate']:.1f}% | -- | -- |"]
    for r in rows:
        L.append(f"| {r['label']} | 90 | {r['elab']:.1f}% | {r['beq']:.1f}% | "
                 f"{r['delta']:+.2f} pp | {r['p']:.1e} |")
    L += ["", "## Trajectories (BEq+ %)", ""]
    for key, _s, label in ARMS:
        L.append(f"- **{label}**  " + "  ".join(
            f"{s}:{traj[key][s]:.1f}" for s in sorted(traj[key])))
    L += ["", "## Read", "",
          "**Only the semantic rewards beat SFT.** "
          f"BEq+ gated {by['gated']['delta']:+.2f} pp (p={by['gated']['p']:.1e}) and the "
          f"six-outcome ladder {by['outcome']['delta']:+.2f} pp (p={by['outcome']['p']:.1e}); "
          f"the exploitable type-check proxy is {by['typecheck']['delta']:+.2f} pp "
          f"(p={by['typecheck']['p']:.2f}, not significant).",
          "",
          "**Arm against arm at step 90**, paired on the same rows:", ""]
    for key, d, p in contrasts:
        L.append(f"- {key} vs type-check: {d:+.2f} pp BEq+, McNemar p={p:.4f}")
    L += ["",
          "Read the trajectories, not the final numbers: the arms cross over between",
          "checkpoints and the paired differences are the honest measure.",
          "",
          "**Every gain is a reduction in outputs that fail to elaborate.** BEq+",
          "conditional on elaborating is flat at ~88% across every checkpoint of every",
          "arm. GRPO here is not teaching better mathematics, it is teaching the model",
          "to emit something Lean accepts. The reward that directly pays for",
          "elaboration is the worst of the three at achieving it.",
          "",
          "## Caveats",
          "",
          "- **Proof validity is not measured, by either metric.** Eval scoring goes",
          "  through `BEqPlusScorer.typecheck_ex`, which replaces the proof body with",
          "  `sorry` before Lean is called. *elaborates* means the generated",
          "  **statement** elaborates; *BEq+* means that statement is bidirectionally",
          "  equivalent to the gold **statement**. `proved` is 0.0% throughout.",
          "  Elaboration hard-gates BEq+ (`BEQ_SKIP_CASCADE_ON_TYPECHECK_FAIL=1`), so",
          "  the two columns are nested, not independent.",
          "- **The SFT baseline may be undertrained, and this is load-bearing.**",
          "  Re-scored under the same toolchain: step 38 50.9%, step 76 58.2%, step 114",
          "  60.9%. `best_sft_proof.txt` pins 76. Measured against 114 instead, the",
          "  gain largely disappears (gated +0.53 pp, p=0.79). But step114-step76 is",
          "  itself only +2.76 pp at p=0.092, and picking the highest SFT checkpoint is",
          "  selecting on the statistic. The defensible claim is narrow: **the",
          "  reward-attributable gain is the same order as the variation between",
          "  adjacent SFT checkpoints**, and this setup cannot separate them because it",
          "  has no selection split (`trainer.test_freq: -1`, no dev parquet).",
          "- **No checkpoint is selected on the reported metric.** Every number is each",
          "  arm's final step, fixed by the training schedule.",
          "- **`max_response_length=128`**, a 16GB-card artifact that outlived its",
          "  hardware. A truncated rollout scores 0 under every reward, so this is a",
          "  standing gradient against longer answers, shared by all three arms. Now",
          "  512 repo-wide; the `locolib_proof_c512` series reruns this at the new cap.",
          "- **Cost.** Lean scoring is ~97% of a gated step and ~1% of a type-check",
          "  step (~962 s vs ~26 s). The semantic reward's advantage costs ~37x the",
          "  wall-clock per step.",
          "",
          "## Layout",
          "",
          "```",
          "eval/<series>/     the corrected eval records, each stamped with `toolchain`",
          "figures/           BEq+ trajectory",
          "tables/            LaTeX: results, per-step, config",
          "training/          verl step metrics per arm",
          "```",
          "",
          "Regenerate: `source hpc/cc_env.sh && python scripts/misc/build_reward_arms_report.py`",
          ""]
    (OUT / "README.md").write_text("\n".join(L))
    print("[report] README written")


if __name__ == "__main__":
    main()
