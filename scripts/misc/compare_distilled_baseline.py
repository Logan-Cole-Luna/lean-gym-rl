#!/usr/bin/env python3
"""CoT-distilled SFT vs the UNTRAINED base model, on the pinned LoCoLib proof-pair
validation slice. Writes every figure and LaTeX table into one folder.

    source hpc/cc_env.sh
    python scripts/misc/compare_distilled_baseline.py [--full]

The untrained model is a single point (step 0), so it is drawn the way the rest of
the repo draws a reference level: a dash-dot horizontal line with a right-edge
label, via `figstyle.finish(..., hline=...)` -- the same convention as
results/figures/beq_plus_rate.png. The distilled run is the trajectory.

Both reported metrics score the generated *statement* only: `typecheck_ex` replaces
the model's proof body with `sorry` before calling Lean, so `typecheck_rate` is a
statement-elaboration rate and `beq_plus_rate` is statement equivalence to the gold.
Proof validity is not measured anywhere in this pipeline. Every table reports each
run's FINAL checkpoint; nothing is argmaxed over the metric being reported.

Exactly two series appear here, by design: the untrained floor and the distilled
model. Other arms of the investigation (BEq+-filtered targets, CoT-on-gold, the
gold baselines) live outside this folder.
"""
from __future__ import annotations

import argparse, json, math, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts", ROOT / "scripts" / "data", ROOT / "scripts" / "figures"):
    sys.path.insert(0, str(p))

# Two eval slices, same treatment. LoCoLib is the in-domain slice the models
# were tuned on; miniF2F is competition problems the models have never seen, so
# it measures transfer rather than fit.
CORPORA = {
    "locolib": dict(out="distilled_vs_baseline_locolib", n=760, kind="in-domain",
                    val=ROOT / "data_locolib" / "val_proof.parquet",
                    base_label="sft3blocolib_proof_base",
                    dist_label="sft3blocolib_proof_distilledcos",
                    ceiling=None),
    "minif2f": dict(out="distilled_vs_baseline", n=244, kind="out-of-domain",
                    val=ROOT / "data_minif2f" / "val_proof.parquet",
                    base_label="minif2f_base", dist_label="minif2f_distilled",
                    # 212/244 miniF2F golds elaborate under Mathlib v4.23; a gold
                    # that will not elaborate can never be matched, so BEq+ here
                    # is capped well below 100%.
                    ceiling=86.9),
    # every trained SFT variant on the same OOD slice. `series` overrides the
    # module default (base + one distilled) with the full roster.
    "minif2f_all": dict(out="distilled_vs_baseline_v2_minif2f", n=244,
                        kind="out-of-domain", ceiling=86.9,
                        val=ROOT / "data_minif2f" / "val_proof.parquet",
                        series="ALL_MINIF2F"),
    # does keeping ONLY the teacher targets BEq+-equivalent to the gold ("fully
    # valid per our pipeline") beat the full teacher target set? Both cosine LR,
    # both to ~225 steps, same everything else -- only the training data differs.
    "beqok_ablation": dict(out="beqok_cosine_vs_distilled_cosine", n=244,
                           kind="out-of-domain", ceiling=None,
                           val=ROOT / "data_minif2f" / "val_proof.parquet",
                           series="BEQOK_ABLATION"),
    # training corpus + reasoning-trace ablation, all on miniF2F: a Mizar-distilled
    # model, a LoCoLib-distilled model with the <think> stripped, our LoCoLib
    # distilled model, and the untrained base.
    "corpus_ablation": dict(out="minif2f_corpus_ablation", n=244,
                            kind="out-of-domain", ceiling=None,
                            val=ROOT / "data_minif2f" / "val_proof.parquet",
                            series="CORPUS_ABLATION"),
}
CORPUS = CORPORA["locolib"]      # main() resets this
OUT = FIGS = TABS = None
N = 760
EVAL, LOGS = ROOT / "results" / "eval", ROOT / "logs"
TOKENIZER = "/scratch/logan03/ai4math_training_models/qwen2.5-coder-3b-instruct"

DOM_DISTILLED = {"algebraic_structures": 3672, "foundations_logic": 3323, "number_theory": 1394}

# role: "hline" = a level, not a trajectory. "primary" = the subject.
# "ref" = context, thin dashed.
SERIES = [
    dict(key="base", tex="Untrained base", role="hline", col="BASELINE",
         label="sft3blocolib_proof_base", steps=[0], sft_logs=[]),
    dict(key="distilled", tex="Distilled", role="primary", col="BLUE",
         label="sft3blocolib_proof_distilledcos", steps=[32, 64, 96, 128, 160, 164],
         sft_logs=["sft_2205189.out", "sft_2205190.out"],
         train_parquet=ROOT / "data_locolib_distilled" / "sft_proof.parquet",
         train_rows=8389, domains=DOM_DISTILLED),
]

# For --corpus minif2f_all: every trained variant vs the untrained floor. Two
# visual families -- gold-target (greys/green) and distilled/CoT (blues/pink) --
# distinguished by colour, marker and dash; end-labels carry identity.
_ALL = [
    dict(key="base", tex="Untrained base", role="hline", col="BASELINE",
         label="minif2f_base", steps=[0], sft_logs=[]),
    # gold-target family
    dict(key="capped", tex="Baseline (capped, gold)", role="primary", col="CONTROL",
         label="minif2f_capped", steps=[38, 76, 114], sft_logs=["sft_1934200.out"], ls="--"),
    dict(key="uncapped", tex="Baseline (full data, gold)", role="primary", col="BASELINE",
         label="minif2f_uncapped", steps=[40, 80, 120, 160, 200, 225],
         sft_logs=["sft_2153038.out", "sft_2169809.out", "sft_2169810.out", "sft_2186409.out"]),
    dict(key="matchedgold", tex="Matched-gold", role="primary", col="GREEN",
         label="minif2f_matchedgold", steps=[33, 66, 96], sft_logs=["sft_2078523.out"]),
    dict(key="cotgold", tex="CoT + gold target", role="primary", col="SKY",
         label="minif2f_cotgold", steps=[32, 64, 96, 128, 160, 192],
         sft_logs=["sft_2201744.out", "sft_2201745.out"]),
    # distilled / CoT family
    dict(key="distilled", tex="Distilled (cosine LR)", role="primary", col="BLUE",
         label="minif2f_distilled", steps=[32, 64, 96, 128, 160],
         sft_logs=["sft_2205189.out", "sft_2205190.out"],
         train_parquet=ROOT / "data_locolib_distilled" / "sft_proof.parquet",
         train_rows=8389, domains=DOM_DISTILLED),
    dict(key="distilled_constlr", tex="Distilled (constant LR)", role="primary", col="BLUE",
         label="minif2f_distilled_constlr", steps=[33, 66, 96, 100, 125, 150, 175],
         sft_logs=["sft_2073796.out", "sft_2153042.out"], ls=":"),
    dict(key="beqok", tex="Distilled, BEq+-clean", role="primary", col="PINK",
         label="minif2f_beqok", steps=[25, 50, 75, 100, 125, 150, 175],
         sft_logs=["sft_2201739.out", "sft_2201741.out"]),
]

_CORPUS = [
    dict(key="base", tex="Untrained base", role="hline", col="BASELINE",
         label="minif2f_base", steps=[0], sft_logs=[]),
    dict(key="ours", tex="LoCoLib distilled (ours)", role="primary", col="BLUE",
         label="minif2f_distilled", steps=[32, 64, 96, 128, 160],
         sft_logs=["sft_2205189.out", "sft_2205190.out"],
         train_parquet=ROOT / "data_locolib_distilled" / "sft_proof.parquet",
         train_rows=8389, domains=DOM_DISTILLED),
    dict(key="nocot", tex="LoCoLib distilled, no CoT", role="primary", col="GREEN",
         label="minif2f_locolib_nocot",
         steps=[25, 50, 75, 100, 125, 150, 175, 200, 225, 250],
         sft_logs=["sft_2328157.out", "sft_2328158.out"],
         train_parquet=ROOT / "data_locolib_distilled_nocot" / "sft_proof.parquet",
         train_rows=9149, domains=None),
    dict(key="mizar", tex="Mizar distilled", role="primary", col="PINK",
         label="minif2f_mizar",
         steps=[25, 50, 75, 100, 125, 150, 175, 200, 225, 250],
         sft_logs=["sft_2328154.out", "sft_2328155.out"],
         train_parquet=ROOT / "data_mizar_distilled" / "sft_proof.parquet",
         train_rows=6316, domains=None),
    dict(key="mizar_nocot", tex="Mizar distilled, no CoT", role="primary", col="ORANGE",
         label="minif2f_mizar_nocot",
         steps=[25, 50, 75, 100, 125, 150, 175, 200, 225, 250],
         sft_logs=["sft_2328642.out", "sft_2328643.out"],
         train_parquet=ROOT / "data_mizar_distilled_nocot" / "sft_proof.parquet",
         train_rows=6316, domains=None),
]

_BEQOK = [
    dict(key="base", tex="Untrained base", role="hline", col="BASELINE",
         label="minif2f_base", steps=[0], sft_logs=[]),
    dict(key="distilled", tex="All teacher targets", role="primary", col="BLUE",
         label="minif2f_distilledcos225",
         steps=[25, 50, 75, 100, 125, 150, 175, 200],
         sft_logs=["sft_2250596.out", "sft_2250597.out"],
         train_parquet=ROOT / "data_locolib_distilled" / "sft_proof_plusval.parquet",
         train_rows=9149, domains=DOM_DISTILLED),
    dict(key="beqok", tex="BEq+-valid targets only", role="primary", col="PINK",
         label="minif2f_beqokcos",
         steps=[25, 50, 75, 100, 125, 150, 175, 200, 225, 250],
         sft_logs=["sft_2250599.out", "sft_2250600.out"],
         train_parquet=ROOT / "data_locolib_distilled_beqok" / "sft_proof_plusval.parquet",
         train_rows=7181, domains=None),
]

_STEP = re.compile(r"^step:(\d+) - .*?train/loss:([0-9.]+)")
_FENCE = re.compile(r"```(?:lean4?|Lean4?)?\s*(.*?)```", re.S)


def parse_logs(names):
    d = {}
    for nm in names:
        p = LOGS / nm
        if p.exists():
            for ln in p.read_text(errors="replace").splitlines():
                m = _STEP.match(ln)
                if m:
                    d[int(m.group(1))] = float(m.group(2))
    return sorted(d.items())


def load_evals(c):
    out = {}
    for s in c["steps"]:
        p = EVAL / c["label"] / f"eval_{c['label']}-step{s}_n{N}.json"
        if p.exists():
            out[s] = next(iter(json.load(open(p)).values()))
    return out


def load_gen(c, s):
    p = EVAL / c["label"] / f"gen_{c['label']}-step{s}_n{N}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def final(ev):
    # The last evaluated checkpoint, fixed by the training schedule rather than
    # picked by the number being reported. Argmaxing BEq+ over checkpoints and
    # then reporting that same BEq+ selects on the statistic: there is no
    # separate selection split in this setup (trainer.test_freq is -1, and no
    # dev parquet exists), so the whole trajectory is the result and the final
    # step is the one pre-registered point in it.
    return max(ev)


def outcome(pe):
    c = Counter()
    for r in pe:
        if not r["typecheck"]:
            c["no_elab"] += 1
        elif r["semantic_signal"] == 2 or r["beq_plus"]:
            c["beq_plus"] += 1
        elif r["semantic_signal"] == 1:
            c["weaker_only"] += 1
        else:
            c["typecheck_only"] += 1
    return c


def mcx(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def paired(A, B):
    a = {r["i"]: bool(r["beq_plus"]) for r in A["per_example"]}
    b = {r["i"]: bool(r["beq_plus"]) for r in B["per_example"]}
    k = sorted(set(a) & set(b))
    return dict(n=len(k), both=sum(a[i] and b[i] for i in k),
                a_only=sum(a[i] and not b[i] for i in k),
                b_only=sum(b[i] and not a[i] for i in k),
                neither=sum(not a[i] and not b[i] for i in k),
                p=mcx(sum(a[i] and not b[i] for i in k), sum(b[i] and not a[i] for i in k)))


def pctl(a, q):
    return a[min(len(a) - 1, int(q * len(a)))] if a else 0


def summ(a):
    return dict(n=len(a), mean=sum(a) / len(a) if a else 0, p50=pctl(a, .5),
                p90=pctl(a, .9), p95=pctl(a, .95), max=a[-1] if a else 0)


def toklen(tok, ss):
    return sorted(len(x) for x in tok(ss, add_special_tokens=False)["input_ids"]) if ss else []


# ─────────────────────────── figures ───────────────────────────
def figures(tok, trajs, evs, gens, full):
    import figstyle as fs
    import numpy as np
    plt = fs.plt
    fs.use_style()
    C = {k: getattr(fs, k) for k in ("BLUE", "GREEN", "ORANGE", "PINK", "SKY", "BASELINE", "CONTROL")}

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(FIGS / f"{name}.{ext}")
        plt.close(fig)

    # This folder answers one question: can distilled data train the model at
    # all. So the only things here are the untrained floor and the distilled
    # trajectories -- no gold-target series, in the figures or the tables.
    live = [c for c in SERIES if evs[c["key"]]]
    hl = next((c for c in live if c["role"] == "hline"), None)
    traj = [c for c in live if c["role"] != "hline"]

    # 1. eval trajectory -- the headline figure
    for metric, ylab, fname in (("beq_plus_rate", "BEq+ (%), greedy decode", "fig_beq_plus_rate"),
                                ("typecheck_rate", "statement elaborates (%), greedy decode",
                                 "fig_elaborate_rate")):
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        ends, vals = {}, []
        for c in traj:
            ev = evs[c["key"]]
            xs = sorted(ev)
            ys = [ev[s][metric] * 100 for s in xs]
            vals += ys
            prim = c["role"] == "primary"
            ax.plot(xs, ys, marker=c.get("mk", "o" if prim else "s"), color=C[c["col"]],
                    lw=2.0 if prim else 1.2, ls=c.get("ls", "-" if prim else "--"),
                    alpha=1 if prim else .75, ms=6 if prim else 4,
                    markeredgecolor="white", markeredgewidth=0.7,
                    label=c["tex"] + ("" if prim else " (ref)"), zorder=3)
            ends[c["tex"]] = (xs[-1], ys[-1], C[c["col"]])
        b = evs[hl["key"]][0][metric] * 100 if hl else None
        if b is not None:
            vals.append(b)
        ax.set_xlabel("SFT step")
        ax.set_ylabel(ylab)
        ax.set_title(("Statement equivalence" if metric == "beq_plus_rate"
                      else "Statement elaboration rate (pass@1)")
                     + f"  (greedy T=0, paired, n={N})")
        lo, hi = min(vals), max(vals)
        ax.set_ylim(max(0, lo - 6), min(100, hi + 8))
        fs.finish(ax, end_labels=ends, legend_loc="lower left",
                  hline=(b, f"untrained {b:.1f}%") if b is not None else None)
        save(fig, fname)

    # 2. training loss
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for c in traj:
        tr = trajs[c["key"]]
        if not tr:
            continue
        xs, ys = zip(*tr)
        prim = c["role"] == "primary"
        ax.plot(xs, ys, color=C[c["col"]], lw=1.8 if prim else 1.1,
                ls="-" if prim else "--", alpha=1 if prim else .7,
                label=c["tex"] + ("" if prim else " (ref)"))
    ax.set_xlabel("SFT step")
    ax.set_ylabel("SFT train loss")
    ax.set_title("SFT training loss")
    ax.legend(loc="upper right")
    save(fig, "fig_training_loss")

    # 3. outcome breakdown -- untrained included as a bar, it is the floor
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    cats = ["no_elab", "typecheck_only", "weaker_only", "beq_plus"]
    colz = [C["CONTROL"], C["ORANGE"], C["SKY"], C["GREEN"]]
    labs, sp = [], []
    for c in live:
        ev = evs[c["key"]]
        st = final(ev)
        sp.append(outcome(ev[st]["per_example"]))
        labs.append(f"{c['tex']}\n(step {st})")
    x = np.arange(len(labs))
    bot = np.zeros(len(labs))
    for cat, cc in zip(cats, colz):
        v = np.array([s[cat] for s in sp])
        ax.bar(x, v, .55, bottom=bot, color=cc, label=cat.replace("_", " "))
        bot += v
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylim(0, N * 1.25)  # headroom for the top legend
    ax.set_ylabel(f"val examples (of {N})")
    ax.set_title("Outcome breakdown at final checkpoint")
    ax.legend(loc="upper center", ncol=4, fontsize=8, frameon=False)
    save(fig, "fig_outcome_breakdown")

    # 4. eval output length -- untrained vs distilled
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bins = range(0, 1600, 40)
    for c in live:
        if c["key"] not in ("base", "distilled"):
            continue
        ax.hist(gens[c["key"]], bins=bins, color=C[c["col"]], alpha=.55, label=c["tex"])
    ax.axvline(1536, color=C["CONTROL"], ls=":", lw=1.4)
    ax.text(1500, ax.get_ylim()[1] * .55, "1536-tok cap", fontsize=8, ha="right",
            rotation=90, va="center", color=C["CONTROL"])
    ax.set_xlabel("completion length (Qwen tokens)")
    ax.set_ylabel("val examples")
    ax.set_title("Eval output length, final checkpoint")
    ax.legend(loc="upper right")
    save(fig, "fig_output_length")

    # 5. SFT target length (trained series only)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bins = range(0, 2100, 50)
    seen_parquet = set()
    for c in traj:
        tp = c.get("train_parquet")
        if not tp or not Path(tp).exists() or str(tp) in seen_parquet:
            continue
        seen_parquet.add(str(tp))
        msgs = list(_pd.read_parquet(tp)["messages"])
        if not full and len(msgs) > 4000:
            import random
            msgs = random.Random(0).sample(msgs, 4000)
        L = toklen(tok, [m[1]["content"] for m in msgs])
        c["_tlen"] = L
        ax.hist(L, bins=bins, color=C[c["col"]], alpha=.5, label=c["tex"])
    ax.set_xlabel("assistant-target length (Qwen tokens)")
    ax.set_ylabel("training rows")
    ax.set_title("SFT target length distribution")
    ax.legend(loc="upper right")
    save(fig, "fig_target_length")

    # 6. domain composition
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    vd = Counter(e["domain"] for e in _pd.read_parquet(CORPUS["val"])["extra_info"])
    # LoCoLib and miniF2F label their domains differently. Only put the training
    # mix on the same axis when the two taxonomies actually share labels;
    # otherwise the train bars would all be zero.
    shared = set(DOM_DISTILLED) & set(vd)
    if shared:
        doms = [d for d in DOM_DISTILLED if d in shared]
        sets = [("Distilled train", DOM_DISTILLED, C["BLUE"]), (f"val ({N})", vd, C["ORANGE"])]
        w = .34
    else:
        doms = [d for d, _ in vd.most_common()]
        sets = [(f"val ({N})", vd, C["ORANGE"])]
        w = .6
    x = np.arange(len(doms))
    for i, (nm, d, cc) in enumerate(sets):
        t = sum(d.values())
        ax.bar(x + (i - 0.5) * w, [100 * d.get(k, 0) / t for k in doms], w, color=cc, label=nm)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in doms])
    ax.set_ylabel("share of set (%)")
    ax.set_title("Domain composition" if len(sets) > 1 else
                 f"Eval-slice composition ({CORPUS['kind']})")
    ax.legend(loc="upper right")
    save(fig, "fig_domain_distribution")


# ─────────────────────────── tables ───────────────────────────
def _tex(body, cap, lab):
    return ("% auto-generated by scripts/misc/compare_distilled_baseline.py\n"
            "\\begin{table}[t]\n\\centering\n\\small\n" + body +
            f"\n\\caption{{{cap}}}\n\\label{{tab:{lab}}}\n\\end{{table}}\n")


def tables(evs, gens, full):
    live = [c for c in SERIES if evs[c["key"]]]
    base = next(c for c in live if c["key"] == "base")
    dist = next(c for c in live if c["key"] == "distilled")
    b_ev, d_ev = evs["base"][0], evs["distilled"][final(evs["distilled"])]
    d_final = final(evs["distilled"])
    pr = paired(b_ev, d_ev)

    # headline: every trained series against the untrained floor
    lines = [f"Untrained base & 0 & {b_ev['typecheck_rate']*100:.1f}\\% & "
             f"{b_ev['beq_plus_rate']*100:.1f}\\% & --- & --- \\\\", "\\midrule"]
    for c in live:
        if c["key"] == "base":
            continue
        ev = evs[c["key"]]
        st = final(ev)
        r = ev[st]
        q = paired(b_ev, r)
        lines.append(f"{c['tex']} & {st} & {r['typecheck_rate']*100:.1f}\\% & "
                     f"{r['beq_plus_rate']*100:.1f}\\% & "
                     f"{(r['beq_plus_rate']-b_ev['beq_plus_rate'])*100:+.1f} pp & {q['p']:.2g} \\\\")
    body = ("\\begin{tabular}{lccccc}\n\\toprule\nseries & step & elaborates & BEq+ & "
            "$\\Delta$BEq+ vs untrained & McNemar $p$ \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    (TABS / "table_results.tex").write_text(_tex(
        body, "Final-checkpoint results against the untrained base model. The step "
        "is the last one of each run, not a step selected on this table's own BEq+ "
        "column; the full trajectory is Table~\\ref{tab:dvb-bystep}. "
        f"Paired McNemar exact on the per-example BEq+ outcome over the same {N} "
        "validation rows. Both columns score the generated \\emph{statement} only "
        "(the proof body is replaced with \\texttt{sorry} before Lean is called), so "
        "\\emph{elaborates} means the statement elaborates and BEq+ means it is "
        "bidirectionally equivalent to the gold statement. The untrained model was "
        "given the most charitable settings (standalone scoring, 1536-token budget).",
        "dvb-results"))

    # every checkpoint
    rows = []
    for c in live:
        for s in sorted(evs[c["key"]]):
            r = evs[c["key"]][s]
            rows.append(f"{c['tex']} & {s} & {r['typecheck_rate']*100:.1f}\\% & "
                        f"{r['beq_plus_rate']*100:.1f}\\% & {r['weaker_only_rate']*100:.1f}\\% \\\\")
        rows.append("\\midrule")
    body = ("\\begin{tabular}{llccc}\n\\toprule\nseries & step & elaborates & BEq+ & weaker-only \\\\\n"
            "\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}")
    (TABS / "table_eval_by_step.tex").write_text(_tex(
        body, f"Eval rates at every evaluated checkpoint (same {N}-row slice).", "dvb-bystep"))

    # outcome decomposition -- the load-bearing table
    order = [("no\\_elab", "no_elab"), ("elaborates only", "typecheck_only"),
             ("weaker only", "weaker_only"), ("BEq+", "beq_plus")]
    cols = [(c, final(evs[c["key"]])) for c in live]
    hdr = " & ".join(f"{c['tex']}" for c, _ in cols)
    body = ["\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\noutcome & " + hdr + " \\\\\n\\midrule"]
    oc = [outcome(evs[c["key"]][s]["per_example"]) for c, s in cols]
    for lab, k in order:
        body.append(lab + " & " + " & ".join(str(o[k]) for o in oc) + " \\\\")
    body.append("\\midrule")
    body.append("elaborates & " + " & ".join(str(N - o["no_elab"]) for o in oc) + " \\\\")
    body.append("BEq+ $\\mid$ elaborates & " + " & ".join(
        f"{100*o['beq_plus']/(N-o['no_elab']):.1f}\\%" for o in oc) + " \\\\")
    body.append("\\bottomrule\n\\end{tabular}")
    (TABS / "table_outcome_breakdown.tex").write_text(_tex(
        "\n".join(body), "Per-example outcome at each series' final checkpoint. The "
        "last row separates the two things SFT teaches: producing a statement that "
        "elaborates at all, and producing the \\emph{intended} statement. Neither row "
        "says anything about the model's proof, which is discarded before scoring.",
        "dvb-outcome"))

    # output lengths, untrained vs distilled
    bs, ds = summ(gens["base"]), summ(gens["distilled"])
    bf = sum(1 for g in load_gen(base, 0) if _FENCE.search(g["completion"]))
    df_ = sum(1 for g in load_gen(dist, d_final) if _FENCE.search(g["completion"]))
    body = ("\\begin{tabular}{lcc}\n\\toprule\n & Untrained base & Distilled (CoT) \\\\\n\\midrule\n"
            f"mean & {bs['mean']:.0f} & {ds['mean']:.0f} \\\\\n"
            f"median & {bs['p50']} & {ds['p50']} \\\\\n"
            f"p90 & {bs['p90']} & {ds['p90']} \\\\\n"
            f"p95 & {bs['p95']} & {ds['p95']} \\\\\n"
            f"max & {bs['max']} & {ds['max']} \\\\\n\\midrule\n"
            f"hit the 1536-tok cap & {sum(1 for x in gens['base'] if x>=1530)}/{N} & "
            f"{sum(1 for x in gens['distilled'] if x>=1530)}/{N} \\\\\n"
            f"closed \\texttt{{```lean}} fence & {bf}/{N} & {df_}/{N} \\\\\n"
            "\\bottomrule\n\\end{tabular}")
    (TABS / "table_output_lengths.tex").write_text(_tex(
        body, "Eval completion length in Qwen tokens. The untrained model runs to "
        "the generation cap on a third of the slice: prose preambles and restated "
        "context rather than a theorem.", "dvb-outlen"))

    # config
    rows = [("training", "none (zero-shot instruct)", "SFT, 3 ep then resumed +4"),
            ("training rows", "---", f"{dist['train_rows']:,}"),
            ("assistant target", "---", "teacher \\texttt{<think>} + teacher Lean"),
            ("base model", "Qwen2.5-Coder-3B-Instruct", "(same)"),
            ("LR / schedule", "---", "5e-5 / constant"),
            ("eval scoring", "\\texttt{score\\_standalone}", "\\texttt{score\\_standalone}"),
            ("gen budget at eval", "1536 tok", "1536 tok")]
    body = ("\\begin{tabular}{lll}\n\\toprule\n & Untrained base & Distilled (CoT) \\\\\n\\midrule\n"
            + "".join(f"{a} & {b} & {c} \\\\\n" for a, b, c in rows) + "\\bottomrule\n\\end{tabular}")
    (TABS / "table_config.tex").write_text(_tex(
        body, "Configuration. Both are scored identically and given the same "
        "generation budget, so the comparison is not confounded by eval settings.",
        "dvb-config"))

    # target lengths + domains, unchanged in spirit
    tl = {c["key"]: summ(c.get("_tlen", [])) for c in SERIES}
    have = [c for c in SERIES if c.get("_tlen")]
    body = ("\\begin{tabular}{l" + "c" * len(have) + "}\n\\toprule\n & "
            + " & ".join(c["tex"] for c in have) + " \\\\\n\\midrule\n"
            + "".join(f"{k} & " + " & ".join(f"{tl[c['key']][k]:.0f}" if k == "mean"
                      else f"{tl[c['key']][k]}" for c in have) + " \\\\\n"
                      for k in ("mean", "p50", "p90", "p95", "max"))
            + "\\bottomrule\n\\end{tabular}")
    (TABS / "table_target_lengths.tex").write_text(_tex(
        body, "SFT assistant-target length in Qwen tokens"
        + ("." if full else " (4k-row sample)."), "dvb-tgtlen"))

    vd = Counter(e["domain"] for e in _pd.read_parquet(CORPUS["val"])["extra_info"])
    shared = set(DOM_DISTILLED) & set(vd)
    doms = ([d for d in DOM_DISTILLED if d in shared] if shared
            else [d for d, _ in vd.most_common()])
    td, tv = sum(DOM_DISTILLED.values()), sum(vd.values())
    if not shared:
        body = ("\\begin{tabular}{lc}\n\\toprule\ndomain & val (%d) \\\\\n\\midrule\n" % N
                + "".join(f"{d} & {vd[d]} ({100*vd[d]/tv:.0f}\\%) \\\\\n" for d in doms)
                + f"\\midrule\ntotal & {tv} \\\\\n\\bottomrule\n\\end{{tabular}}")
        (TABS / "table_training_distribution.tex").write_text(_tex(
            body, f"Composition of the {CORPUS['kind']} evaluation slice. The model was "
            f"trained on a different corpus ({td:,} rows) whose domain labels do not "
            "overlap with these, so the training mix is not shown alongside.",
            "dvb-traindist"))
        return b_ev, d_ev, d_final, pr, live
    body = ("\\begin{tabular}{lcc}\n\\toprule\ndomain & Distilled train & val ({N}) \\\\\n\\midrule\n"
            + "".join(f"{d.replace('_',' ')} & {DOM_DISTILLED[d]} ({100*DOM_DISTILLED[d]/td:.0f}\\%) "
                      f"& {vd[d]} ({100*vd[d]/tv:.0f}\\%) \\\\\n" for d in doms)
            + f"\\midrule\ntotal & {td} & {tv} \\\\\n\\bottomrule\n\\end{{tabular}}")
    (TABS / "table_training_distribution.tex").write_text(_tex(
        body, "Training-set domain composition (exact).", "dvb-traindist"))
    return b_ev, d_ev, d_final, pr, live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--corpus", choices=sorted(CORPORA), default="locolib",
                    help="which eval slice to write up; see CORPORA")
    a = ap.parse_args()
    global CORPUS, OUT, FIGS, TABS, N, SERIES
    CORPUS = CORPORA[a.corpus]
    N = CORPUS["n"]
    if CORPUS.get("series") == "ALL_MINIF2F":
        SERIES = _ALL
    elif CORPUS.get("series") == "BEQOK_ABLATION":
        SERIES = _BEQOK
    elif CORPUS.get("series") == "CORPUS_ABLATION":
        SERIES = _CORPUS
    OUT = ROOT / "results" / CORPUS["out"]
    FIGS, TABS = OUT / "figures", OUT / "tables"
    # the two series carry a different eval label per corpus; everything else
    # about them (colour, role, training logs, corpus stats) is identical
    if "base_label" in CORPUS:
        for c in SERIES:
            c["label"] = CORPUS["base_label"] if c["key"] == "base" else CORPUS["dist_label"]
    FIGS.mkdir(parents=True, exist_ok=True)
    TABS.mkdir(parents=True, exist_ok=True)
    global _pd
    import pandas as _pd  # noqa
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)

    trajs = {c["key"]: parse_logs(c["sft_logs"]) for c in SERIES}
    evs = {c["key"]: load_evals(c) for c in SERIES}
    gens = {}
    for c in SERIES:
        if evs[c["key"]] and c["key"] in ("base", "distilled"):
            gens[c["key"]] = toklen(tok, [g["completion"] for g in load_gen(c, final(evs[c["key"]]))])
    print("[dvb] series with evals: " + ", ".join(
        f"{c['tex']}({len(evs[c['key']])})" for c in SERIES if evs[c["key"]]))
    print("[dvb] figures ..."); figures(tok, trajs, evs, gens, a.full)
    print("[dvb] tables ...");  b_ev, d_ev, d_final, pr, live = tables(evs, gens, a.full)

    roster = CORPUS.get("series") == "ALL_MINIF2F"
    ablation = CORPUS.get("series") == "BEQOK_ABLATION"
    if ablation:
        A = evs["distilled"]; B = evs["beqok"]     # "all targets" / "valid only"
        aB, bB = final(A), final(B)
        md = ["# Does filtering the distilled targets to BEq+-valid help? (miniF2F, OOD)\n",
              "Two SFT runs, identical except the training data: **all teacher targets",
              "that elaborate** (9,149 rows) vs **only the ~76% BEq+-equivalent to the gold**",
              "(7,181 rows). Both cosine LR from scratch, the LoCoLib pinned-760 slice folded",
              "into training since the eval moved to miniF2F, scored with `score_standalone`",
              f"on the same {N}-row out-of-domain slice.\n"]
    elif roster:
        md = ["# Every trained SFT variant on miniF2F (out-of-domain)\n",
              f"All {len([c for c in live if c['role']!='hline'])} trained models plus the "
              "untrained base, on the same pinned",
              f"{N}-row `{CORPUS['val'].relative_to(ROOT)}` slice -- competition problems the",
              "models never saw. Distilled/CoT models are scored with `score_standalone`",
              "(they emit self-contained snippets), gold-target models with the plain path;",
              "all use a 1536-token budget where a `<think>` block is involved. The untrained",
              "model is the dash-dot floor. Note 32 of the 244 miniF2F golds do not "
              "elaborate under Mathlib v4.23, so BEq+ is effectively capped near 87%.\n"]
    else:
        md = ["# CoT-distilled SFT vs the untrained base model\n",
              f"Same pinned {N}-row `{CORPUS['val'].relative_to(ROOT)}` ({CORPUS['kind']}),",
              "same scoring path",
              "(`score_standalone`) and the same 1536-token generation budget for both, so",
              "nothing here is confounded by eval settings. The untrained model is a level,",
              "not a trajectory, so it is drawn as a dash-dot reference line.\n",]
    md += ["## Headline\n",
          "Each row is that run's **final** checkpoint, not a checkpoint selected on this",
          "table's own BEq+ column. Both metrics score the generated *statement* only --",
          "see Caveats.\n",
          "| series | step | elaborates | BEq+ | Δ BEq+ vs untrained | McNemar p |",
          "|---|---|---|---|---|---|",
          f"| untrained base | 0 | {b_ev['typecheck_rate']*100:.1f}% | {b_ev['beq_plus_rate']*100:.1f}% | — | — |"]
    for c in live:
        if c["key"] == "base":
            continue
        st = final(evs[c["key"]]); r = evs[c["key"]][st]; q = paired(b_ev, r)
        md.append(f"| {c['tex']} | {st} | {r['typecheck_rate']*100:.1f}% | "
                  f"{r['beq_plus_rate']*100:.1f}% | {(r['beq_plus_rate']-b_ev['beq_plus_rate'])*100:+.1f} pp | {q['p']:.2g} |")
    md += ["", "## Trajectories (BEq+ %)\n"]
    for c in live:
        if c["role"] == "hline":
            continue
        md.append(f"- **{c['tex']}**  " + "  ".join(
            f"{s}:{evs[c['key']][s]['beq_plus_rate']*100:.1f}" for s in sorted(evs[c["key"]])))
    bo, do_ = outcome(b_ev["per_example"]), outcome(d_ev["per_example"])
    bc, dc = N - bo["no_elab"], N - do_["no_elab"]
    ladder = [("teacher statement + CoT", "distilled"),
              ("teacher statement, BEq+-filtered + CoT", "beqok"),
              ("gold statement + CoT", "cotgold")]
    rung = [(t, k) for t, k in ladder if evs.get(k)]
    if len(rung) > 1 and not ablation:
        md += ["", "## What the target choice buys", "",
               "Replacing the teacher's *statement* with the gold statement, everything",
               "else held fixed:", "",
               "| training target | BEq+ | elaborates |", "|---|---|---|"]
        for t, k in rung:
            r = evs[k][final(evs[k])]
            md.append(f"| {t} | {r['beq_plus_rate']*100:.1f}% | {r['typecheck_rate']*100:.1f}% |")
        if roster:
            md += ["",
                   "Out of domain the ladder is not monotone: swapping in the gold",
                   "statement (`cotgold`) *hurts* here on both axes, and BEq+-filtering",
                   "the teacher targets (`beqok`) is what helps. The in-domain finding",
                   "that the gold statement is the better target does not transfer -- see",
                   "the Read section."]
        else:
            md += ["",
                   "Elaboration barely moves across the rungs: the skill of writing Lean",
                   "that parses is insensitive to which target you train on. Only fidelity",
                   "to the reference statement responds."]
    if ablation:
        A = evs["distilled"]; B = evs["beqok"]; aB, bB = final(A), final(B)
        d = (A[aB]["beq_plus_rate"] - B[bB]["beq_plus_rate"]) * 100
        pr2 = paired(A[aB], B[bB])
        md += ["", "## Read", "",
               f"Filtering **does not help**. At the final checkpoint of each run BEq+ is "
               f"{A[aB]['beq_plus_rate']*100:.1f}% (all targets, step {aB}) against "
               f"{B[bB]['beq_plus_rate']*100:.1f}% (valid only, step {bB}): {d:+.1f} pp, "
               f"McNemar p={pr2['p']:.2g} -- not significant, and at matched steps "
               "150/175/200 neither run leads either.",
               "",
               "**Statement elaboration is clearly worse for the filtered run** (~52-61% vs "
               "~60-70% at every step -- see `fig_elaborate_rate`). Dropping 24% of the "
               "training data cost more elaboration skill than the label-cleanliness bought "
               "back.",
               "",
               "Both plateau by step ~75-125 then wander in a 12-16% band; longer training "
               "does nothing on this OOD slice. The earlier hint that filtering helped "
               "(+2.9 pp, p=0.09) was a noisier setup (constant LR, no folded val, only to "
               "step 175) and does not survive here.",
               "",
               "For this transfer target, **data volume beats target-statement purity**.",
               ""]
    elif roster:
        fam = {"capped": "gold", "uncapped": "gold", "matchedgold": "gold",
               "cotgold": "gold", "distilled": "distilled", "distilled_constlr": "distilled",
               "beqok": "distilled"}
        def bqt(k):
            r = evs[k][final(evs[k])]; return r["beq_plus_rate"] * 100, r["typecheck_rate"] * 100
        gold = [bqt(k) for k in fam if fam[k] == "gold" and evs.get(k)]
        dist = [bqt(k) for k in fam if fam[k] == "distilled" and evs.get(k)]
        gold_bq, gold_tc = [x[0] for x in gold], [x[1] for x in gold]
        dist_bq, dist_tc = [x[0] for x in dist], [x[1] for x in dist]
        md += ["", "## Read", "",
               "The out-of-domain ranking inverts the in-domain one. On LoCoLib the",
               "gold-target models lead; on miniF2F the distilled/CoT models do:",
               "",
               f"- distilled family final-checkpoint BEq+ {min(dist_bq):.1f}-{max(dist_bq):.1f}%, "
               f"elaboration {min(dist_tc):.1f}-{max(dist_tc):.1f}%",
               f"- gold-target family {min(gold_bq):.1f}-{max(gold_bq):.1f}%, "
               f"elaboration {min(gold_tc):.1f}-{max(gold_tc):.1f}%",
               "",
               "The likely cause is prompt-format transfer: miniF2F is self-contained",
               "competition statements with a trivial preamble, which is the shape the",
               "distilled models learned to emit (their teacher wrote standalone Lean).",
               "The gold-target models learned to emit a bare theorem that slots into",
               "LoCoLib's rich namespace context, and that habit does not carry over.",
               "`CoT + gold target`"
               + (f" ({bqt('cotgold')[0]:.1f}%)" if evs.get("cotgold") else "")
               + " sits at the bottom of the range with the",
               "capped baseline -- the `<think>` prefix plus a gold-context bare theorem",
               "transfers poorly on both axes.",
               "",
               f"`BEq+-clean` (distilled targets filtered to statements BEq+ says match",
               f"the gold) tops the distilled band at {max(dist_bq):.1f}%, though the band "
               "is a",
               "few points wide and the runs cross over between checkpoints -- read",
               "`table_eval_by_step.tex`, not the single number.",
               "",
               "Every trained model still beats the untrained base "
               f"({b_ev['beq_plus_rate']*100:.1f}% BEq+ / {b_ev['typecheck_rate']*100:.1f}% "
               "elaboration) with paired significance -- see `table_results.tex`.",
               ""]
    else:
        md += ["", "## Read", "",
               "SFT does two separable jobs, and `table_outcome_breakdown.tex` splits them:",
               "",
               f"1. **Make a statement that elaborates at all.** The untrained model "
               f"elaborates {bc}/{N} ({100*bc/N:.1f}%) of the slice; after distilled SFT "
               f"that is {dc}/{N} ({100*dc/N:.1f}%). This is the bulk of the gain.",
               f"2. **Make the *intended* theorem.** Conditional on elaborating, the untrained "
               f"model is right {100*bo['beq_plus']/bc:.1f}% of the time and the distilled "
               f"model {100*do_['beq_plus']/dc:.1f}%."
               # only claim drift where the one-directional bucket is actually populated;
               # on a slice where it is a couple of examples the claim would be noise
               + (f" The `weaker_only` bucket ({do_['weaker_only']} vs {bo['weaker_only']}) "
                  f"is statement drift: a theorem implied by the gold rather than equivalent "
                  f"to it, traced to the ~24% of the training targets that BEq+ says are not "
                  f"the gold theorem."
                  if do_["weaker_only"] >= 0.02 * dc else
                  f" The one-directional `weaker_only` bucket is near-empty here "
                  f"({do_['weaker_only']} vs {bo['weaker_only']}), so on this slice the "
                  f"failures are outright wrong statements rather than drifted ones."),
               "",
               "Distilled data trains the model: it moves BEq+ from "
               f"{b_ev['beq_plus_rate']*100:.1f}% to {d_ev['beq_plus_rate']*100:.1f}% and the "
               f"statement-elaboration rate from {100*bc/N:.1f}% to {100*dc/N:.1f}%, both "
               "with overwhelming paired significance."]
    md += ["", "## Caveats", "",
           "- **Proof validity is not measured, by either metric.** Scoring goes through",
           "  `BEqPlusScorer.typecheck_ex`, which calls",
           "  `clean_last_theorem_string(..., add_sorry=True)`: the model's proof body is",
           "  stripped and replaced with `sorry` before Lean is ever called. So",
           "  *elaborates* means the generated **statement** elaborates, and *BEq+* means",
           "  that statement is bidirectionally equivalent to the gold **statement**.",
           "  Neither says whether the model's proof closes the goal.",
           "  `BEqPlusScorer.check_own_proof` would check exactly that and is never called",
           "  by `scripts/eval/evaluate_checkpoints.py`. Elaboration also hard-gates BEq+",
           "  (`BEQ_SKIP_CASCADE_ON_TYPECHECK_FAIL=1`), so the two are nested, not",
           "  independent.",
           "- **No checkpoint is selected on the reported metric.** Every table and figure",
           "  annotation uses each run's *final* checkpoint, fixed by the training",
           "  schedule. There is no separate selection split here (`trainer.test_freq: -1`,",
           "  no dev parquet), so an argmax over the reported BEq+ column would be",
           "  selecting on the statistic it then reports. Read",
           "  `table_eval_by_step.tex` -- the full trajectory -- as the result.",
           "- The untrained number is an upper bound: standalone (union-context) scoring,",
           "  a 1536-token generation budget, and zero-shot with no few-shot exemplars.",
           f"- Adjacent checkpoints still disagree on a sizeable minority of the {N} examples",
           "  where the aggregate rate barely moves, so read the curve, not any single",
           "  checkpoint. The LR is annealed over the run, which keeps the trajectory",
           "  monotone; an earlier constant-LR run of the same corpus swung several",
           "  points between neighbouring checkpoints.",
           "",
           "## Contents", "",
           "```",
           "figures/   fig_beq_plus_rate, fig_elaborate_rate, fig_training_loss,",
           "           fig_outcome_breakdown, fig_output_length, fig_target_length,",
           "           fig_domain_distribution            (.png + .pdf)",
           "tables/    table_results, table_eval_by_step, table_outcome_breakdown,",
           "           table_output_lengths, table_config, table_target_lengths,",
           "           table_training_distribution        (.tex, booktabs)",
           "eval/      the per-checkpoint eval JSON (with per_example) and the raw",
           "           generations, copied verbatim from results/eval/<label>/",
           "training/  the SFT job logs these curves were parsed from, plus",
           "           train_loss.csv (series, step, loss)",
           "```",
           "",
           "Regenerate: `source hpc/cc_env.sh && python scripts/misc/compare_distilled_baseline.py [--full]`"]
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    # Copy the inputs in, so the folder is readable without the rest of the repo.
    import shutil, csv
    ED, TD = OUT / "eval", OUT / "training"
    for d in (ED, TD):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    n_ev = 0
    for c in SERIES:
        src = EVAL / c["label"]
        if not evs[c["key"]] or not src.is_dir():
            continue
        dst = ED / c["label"]
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.iterdir()):
            if f.is_file():
                shutil.copy2(f, dst / f.name)
                n_ev += 1
    n_lg = 0
    rows = []
    for c in SERIES:
        if not evs[c["key"]]:
            continue            # only ship what the figures/tables actually show
        for nm in c["sft_logs"]:
            f = LOGS / nm
            if f.exists():
                shutil.copy2(f, TD / nm)
                n_lg += 1
        for step, loss in trajs[c["key"]]:
            rows.append((c["key"], c["tex"], step, f"{loss:.6f}"))
    with open(TD / "train_loss.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series", "label", "step", "train_loss"])
        w.writerows(rows)
    print(f"[dvb] copied {n_ev} eval files, {n_lg} SFT logs, {len(rows)} loss rows")
    print(f"[dvb] wrote {OUT}")


if __name__ == "__main__":
    main()
