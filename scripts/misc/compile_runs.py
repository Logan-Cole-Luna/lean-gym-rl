#!/usr/bin/env python3
"""One table of every completed run in this project, on both eval slices.

    source hpc/cc_env.sh
    python scripts/misc/compile_runs.py

Writes results/RUNS.md and results/tables/table_all_runs.tex. The per-experiment
folders (results/distilled_vs_baseline*, results/beqok_*, results/minif2f_*)
each answer one question; this answers "what have we run, and what came out".

Every rate is the run's FINAL checkpoint, never a checkpoint argmaxed over the
column being reported: there is no selection split in this setup, so selecting
on the reported metric would inflate both the effect and its p-value. Read the
per-experiment folders' table_eval_by_step.tex for the trajectories.

Both metrics score the generated STATEMENT only. BEqPlusScorer.typecheck_ex
replaces the model's proof body with `sorry` before calling Lean, so `elab` is a
statement-elaboration rate and `BEq+` is statement equivalence to the gold.
Proof validity is not measured anywhere in this pipeline.
"""
from __future__ import annotations

import ast, json, math, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL, LOGS = ROOT / "results" / "eval", ROOT / "logs"

# The two pinned slices. LoCoLib is the corpus most runs were tuned on; miniF2F
# is competition problems no run has seen, so it measures transfer.
SLICES = {"locolib": 760, "minif2f": 244}
BASE = {"locolib": "sft3blocolib_proof_base", "minif2f": "minif2f_base"}

# One row per training run. `loco`/`mf2f` are the eval-series labels; "" means
# the run was never evaluated on that slice. `note` records why, when the reason
# is not simply "not run yet".
RUNS = [
    dict(name="Untrained base", fam="control", corpus="---", target="---", cot="---",
         loco="sft3blocolib_proof_base", mf2f="minif2f_base", logs=[], data=None,
         note="Qwen2.5-Coder-3B-Instruct, zero-shot"),

    dict(name="Gold, capped", fam="gold SFT", corpus="LoCoLib", target="gold", cot="no",
         loco="sft3blocolib_proof", mf2f="minif2f_capped",
         logs=["sft_1934200.out"], data="data_locolib"),
    dict(name="Gold, full data", fam="gold SFT", corpus="LoCoLib", target="gold", cot="no",
         loco="sft3blocolib_proof_uncapped", mf2f="minif2f_uncapped",
         logs=["sft_2186409.out", "sft_2153038.out", "sft_2169809.out",
               "sft_2169810.out"], data="data_locolib_uncapped"),
    dict(name="Gold, matched rows", fam="gold SFT", corpus="LoCoLib", target="gold", cot="no",
         loco="sft3blocolib_proof_matchedgold", mf2f="minif2f_matchedgold",
         logs=["sft_2078523.out", "sft_2065394.out", "sft_2073801.out"], data="data_locolib_matched_gold",
         note="same instances as Distilled, gold target instead of the teacher's"),
    dict(name="Gold target + CoT", fam="gold SFT", corpus="LoCoLib", target="gold", cot="yes",
         loco="sft3blocolib_proof_cotgold", mf2f="minif2f_cotgold",
         logs=["sft_2201744.out", "sft_2201745.out"], data="data_locolib_cotgold",
         note="teacher's reasoning trace, gold statement as the target"),

    dict(name="Distilled, const LR", fam="distilled SFT", corpus="LoCoLib", target="teacher",
         cot="yes", loco="sft3blocolib_proof_distilled", mf2f="minif2f_distilled_constlr",
         logs=["sft_2153042.out", "sft_2064959.out", "sft_2073796.out"], data="data_locolib_distilled"),
    dict(name="Distilled, cosine LR", fam="distilled SFT", corpus="LoCoLib", target="teacher",
         cot="yes", loco="sft3blocolib_proof_distilledcos", mf2f="minif2f_distilled",
         logs=["sft_2205189.out", "sft_2205190.out"], data="data_locolib_distilled"),
    dict(name="Distilled, BEq+-clean", fam="distilled SFT", corpus="LoCoLib", target="teacher",
         cot="yes", loco="sft3blocolib_proof_beqok", mf2f="minif2f_beqok",
         logs=["sft_2201739.out", "sft_2201741.out"], data="data_locolib_distilled_beqok",
         note="teacher targets filtered to those BEq+ says match the gold"),

    dict(name="Distilled, cosine LR, val folded", fam="distilled SFT", corpus="LoCoLib",
         target="teacher", cot="yes", loco="", mf2f="minif2f_distilledcos225",
         logs=["sft_2250596.out", "sft_2250597.out"], data="data_locolib_distilled",
         note="LoCoLib val slice folded into training, so LoCoLib is no longer held out"),
    dict(name="BEq+-clean, cosine LR, val folded", fam="distilled SFT", corpus="LoCoLib",
         target="teacher", cot="yes", loco="", mf2f="minif2f_beqokcos",
         logs=["sft_2250599.out", "sft_2250600.out"], data="data_locolib_distilled_beqok",
         note="as above; the BEq+-filtering ablation at matched settings"),

    dict(name="Distilled, no CoT", fam="trace ablation", corpus="LoCoLib", target="teacher",
         cot="no", loco="", mf2f="minif2f_locolib_nocot",
         logs=["sft_2328157.out", "sft_2328158.out"], data="data_locolib_distilled_nocot",
         note="teacher Lean only, `<think>` block stripped; val folded into training"),
    dict(name="Mizar distilled", fam="corpus ablation", corpus="Mizar", target="teacher",
         cot="yes", loco="", mf2f="minif2f_mizar",
         logs=["sft_2328154.out", "sft_2328155.out"], data="data_mizar_distilled",
         note="different corpus; LoCoLib was never its training or eval domain"),
    dict(name="Mizar distilled, no CoT", fam="corpus ablation", corpus="Mizar",
         target="teacher", cot="no", loco="", mf2f="minif2f_mizar_nocot",
         logs=["sft_2328642.out", "sft_2328643.out"], data="data_mizar_distilled_nocot"),

    # GRPO arms resume from the gold-capped SFT checkpoint, so they are only
    # comparable to each other and to that baseline.
    dict(name="GRPO, BEq+ gated", fam="GRPO", corpus="LoCoLib", target="reward", cot="no",
         loco="rl3b_locolib_proof_lr6_gated", mf2f="", logs=[], data=None,
         note="resumes from Gold, capped; ACTOR_LR=1e-6"),
    dict(name="GRPO, outcome ladder", fam="GRPO", corpus="LoCoLib", target="reward", cot="no",
         loco="rl3b_locolib_proof_lr6_outcome", mf2f="", logs=[], data=None,
         note="resumes from Gold, capped; ACTOR_LR=1e-6"),
    dict(name="GRPO, type-check", fam="GRPO", corpus="LoCoLib", target="reward", cot="no",
         loco="rl3b_locolib_proof_lr6_typecheck", mf2f="", logs=[], data=None,
         note="exploitable ablation; resumes from Gold, capped; ACTOR_LR=1e-6"),
]

_CFG = re.compile(r"^\{'model': \{.*\}$")


def sft_config(logs):
    """LR schedule, epochs and train file, read out of verl's config dump."""
    for nm in logs:
        p = LOGS / nm
        if not p.exists():
            continue
        for ln in p.read_text(errors="replace").splitlines():
            if not _CFG.match(ln):
                continue
            try:
                c = ast.literal_eval(ln)
            except (ValueError, SyntaxError):
                continue
            return dict(sched=c["optim"]["lr_scheduler_type"], lr=c["optim"]["lr"],
                        epochs=c["trainer"]["total_epochs"],
                        train_file=c["data"]["train_files"])
    return {}


def evals(label, n):
    """Every checkpoint of one eval series, keyed by step."""
    d = EVAL / label
    if not label or not d.is_dir():
        return {}
    out = {}
    pat = re.compile(rf"^eval_{re.escape(label)}-step(\d+)_n{n}\.json$")
    for f in sorted(d.glob(f"eval_{label}-step*_n{n}.json")):
        m = pat.match(f.name)
        if m:
            out[int(m.group(1))] = next(iter(json.load(open(f)).values()))
    return out


def mcx(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def paired(A, B):
    """McNemar exact on the per-example BEq+ outcome, over the shared rows."""
    a = {r["i"]: bool(r["beq_plus"]) for r in A["per_example"]}
    b = {r["i"]: bool(r["beq_plus"]) for r in B["per_example"]}
    k = sorted(set(a) & set(b))
    return mcx(sum(a[i] and not b[i] for i in k), sum(b[i] and not a[i] for i in k))


def rows():
    base = {s: evals(BASE[s], n) for s, n in SLICES.items()}
    out = []
    for r in RUNS:
        rec = dict(r)
        rec["cfg"] = sft_config(r["logs"])
        f = rec["cfg"].get("train_file")
        rec["train_rows"] = train_rows(f) if f else None
        for s, n in SLICES.items():
            ev = evals(r["loco"] if s == "locolib" else r["mf2f"], n)
            if not ev:
                rec[s] = None
                continue
            st = max(ev)                       # final checkpoint, never an argmax
            e = ev[st]
            b0 = base[s].get(0)
            rec[s] = dict(step=st, n_ckpt=len(ev), beq=e["beq_plus_rate"] * 100,
                          elab=e["typecheck_rate"] * 100,
                          delta=None if not b0 or r["fam"] == "control" else
                          (e["beq_plus_rate"] - b0["beq_plus_rate"]) * 100,
                          p=None if not b0 or r["fam"] == "control" else paired(e, b0))
        out.append(rec)
    return out


def train_rows(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return None
    import pandas as pd
    return len(pd.read_parquet(p))


def cell(d, key):
    return "---" if not d else f"{d[key]:.1f}"


def main():
    rs = rows()
    md = ["# Every completed run\n",
          "One row per training run, both eval slices, final checkpoint. Generated by",
          "`scripts/misc/compile_runs.py`; the per-experiment folders under `results/`",
          "carry the figures, trajectories and paired tests behind each line.\n",
          "**What the columns mean.** `elab` is the rate at which the generated",
          "*statement* elaborates; `BEq+` is the rate at which it is bidirectionally",
          "equivalent to the gold *statement*. Scoring replaces the model's proof body",
          "with `sorry` before Lean is called (`BEqPlusScorer.typecheck_ex`), so **proof",
          "validity is not measured** by either column. `Δ` is BEq+ against the untrained",
          "base on the same slice, `p` its paired McNemar exact.\n",
          "**Every row is that run's final checkpoint**, fixed by the training schedule.",
          "Nothing here is argmaxed over the column it is reported in: there is no",
          "selection split in this setup (`trainer.test_freq: -1`, no dev parquet), so",
          "selecting the best step would inflate both the effect and the p-value.\n",
          "| run | corpus | target | CoT | rows | LR sched | LoCoLib-760 step | elab | BEq+ | Δ | p | miniF2F-244 step | elab | BEq+ | Δ | p |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        L, M = r["locolib"], r["minif2f"]
        md.append("| " + " | ".join([
            r["name"], r["corpus"], r["target"], r["cot"],
            f"{r['train_rows']:,}" if r["train_rows"] else "---",
            r["cfg"].get("sched", "---"),
            "---" if not L else str(L["step"]), cell(L, "elab"), cell(L, "beq"),
            "---" if not L or L["delta"] is None else f"{L['delta']:+.1f}",
            "---" if not L or L["p"] is None else f"{L['p']:.2g}",
            "---" if not M else str(M["step"]), cell(M, "elab"), cell(M, "beq"),
            "---" if not M or M["delta"] is None else f"{M['delta']:+.1f}",
            "---" if not M or M["p"] is None else f"{M['p']:.2g}",
        ]) + " |")

    md += ["", "## Trajectories\n",
           "Checkpoints evaluated per run, BEq+ % at each. Runs cross over between",
           "checkpoints, so a single-step comparison of two close runs is not a result.\n"]
    for s, n in SLICES.items():
        md.append(f"**{s} (n={n})**\n")
        for r in rs:
            ev = evals(r["loco"] if s == "locolib" else r["mf2f"], n)
            if len(ev) > 1:
                md.append(f"- {r['name']}  " + "  ".join(
                    f"{k}:{ev[k]['beq_plus_rate']*100:.1f}" for k in sorted(ev)))
        md.append("")

    grpo = [r for r in rs if r["fam"] == "GRPO" and r["locolib"]]
    sft0 = next((r for r in rs if r["name"] == "Gold, capped"), None)
    if grpo and sft0 and sft0["locolib"]:
        ref = evals(sft0["loco"], SLICES["locolib"])
        ref = ref[max(ref)]
        md += ["## GRPO arms against the SFT checkpoint they resume from\n",
               f"All three arms resume from *{sft0['name']}* "
               f"(step {sft0['locolib']['step']}, {sft0['locolib']['beq']:.1f}% BEq+), so",
               "that checkpoint, not the untrained model, is their baseline.\n",
               "| arm | step | elab | BEq+ | Δ vs SFT | p |", "|---|---|---|---|---|---|"]
        for r in grpo:
            ev = evals(r["loco"], SLICES["locolib"])
            e = ev[max(ev)]
            md.append(f"| {r['name']} | {r['locolib']['step']} | {r['locolib']['elab']:.1f} | "
                      f"{r['locolib']['beq']:.1f} | "
                      f"{r['locolib']['beq'] - sft0['locolib']['beq']:+.1f} | "
                      f"{paired(e, ref):.2g} |")
        md.append("")
        # the arms against each other: the reward comparison this repo exists for
        md += ["Arm against arm, same slice, same step:\n",
               "| pair | Δ BEq+ | p |", "|---|---|---|"]
        by = {r["name"]: r for r in grpo}
        for x, y in (("GRPO, BEq+ gated", "GRPO, type-check"),
                     ("GRPO, BEq+ gated", "GRPO, outcome ladder"),
                     ("GRPO, outcome ladder", "GRPO, type-check")):
            if x in by and y in by:
                ex, ey = evals(by[x]["loco"], SLICES["locolib"]), evals(by[y]["loco"], SLICES["locolib"])
                ex, ey = ex[max(ex)], ey[max(ey)]
                md.append(f"| {x.replace('GRPO, ','')} vs {y.replace('GRPO, ','')} | "
                          f"{by[x]['locolib']['beq'] - by[y]['locolib']['beq']:+.1f} | "
                          f"{paired(ex, ey):.2g} |")
        md.append("")

    md += ["## Notes\n"]
    for r in rs:
        if r.get("note"):
            md.append(f"- **{r['name']}** -- {r['note']}")
    md += ["",
           "A run with `---` on the LoCoLib slice either folded that slice into its own",
           "training (so it is no longer held out) or was never evaluated there; the",
           "note on the run says which. The GRPO arms all resume from *Gold, capped*, so",
           "compare them to that row and to each other, not to the SFT runs above them.",
           "",
           "Regenerate: `source hpc/cc_env.sh && python scripts/misc/compile_runs.py`"]
    (ROOT / "results" / "RUNS.md").write_text("\n".join(md) + "\n")

    # LaTeX: the miniF2F slice only, which is the one every run shares
    body = ["\\begin{tabular}{llccrcc}", "\\toprule",
            "run & corpus & CoT & rows & step & elaborates & BEq+ \\\\", "\\midrule"]
    for r in rs:
        M = r["minif2f"]
        if not M:
            continue
        body.append(" & ".join([
            r["name"].replace("+", "$+$"), r["corpus"], r["cot"],
            f"{r['train_rows']:,}" if r["train_rows"] else "---",
            str(M["step"]), f"{M['elab']:.1f}\\%", f"{M['beq']:.1f}\\%"]) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    tabs = ROOT / "results" / "tables"
    tabs.mkdir(parents=True, exist_ok=True)
    (tabs / "table_all_runs.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\small\n" + "\n".join(body) +
        "\n\\caption{Every completed run on the pinned 244-row miniF2F slice, at each "
        "run's final checkpoint. Both columns score the generated statement only; the "
        "model's proof is replaced with \\texttt{sorry} before Lean is called, so proof "
        "validity is not measured.}\n\\label{tab:all-runs}\n\\end{table}\n")
    print(f"[runs] {len(rs)} runs -> results/RUNS.md, results/tables/table_all_runs.tex")


if __name__ == "__main__":
    main()
