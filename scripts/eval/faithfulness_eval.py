#!/usr/bin/env python3
"""NL -> FL faithfulness over real rollouts, with the controls that decide
whether the metric means anything. OFFLINE ONLY.

`reward/reward.py:362-369` makes three things non-optional before a judged
metric may feed a reward, and this script exists to produce them:

    * an UNKNOWN return of None, so a judge that fails does not silently award
      the band                                     -> `unknown_rate`, by cause
    * a validity gate on the unknown rate          -> `coverage`
    * a NEGATIVE CONTROL, scoring a proof against a DIFFERENT problem's informal
      proof, because "unlike BEq+ this will be judged rather than derived, and is
      therefore gameable by exactly the policy being trained"  -> --arm mismatched

COVERAGE IS THE HEADLINE, NOT A FOOTNOTE. AutoFaith's FL side is built from
`allTactics`, and a term-mode proof (`:= rfl`) emits none. 48.7% of LoCoLib golds
and 43.8% of the RL policy's proofs are term-mode, so ~half of every population
here is structurally unscorable. That is reported, never imputed as 0.

NO DATA REGENERATION. `prepare_locolib.py` interpolates the informal proof into
the PROMPT, and the prompt is stored verbatim in every `gen_*.jsonl`, so both
judge inputs parse back out (verified 760/760). The parquet change that an
in-loop judge would need is not required here.

    # needs scripts/eval/faithfulness_judge_server.py up on FAITH_JUDGE_URL
    python scripts/eval/faithfulness_eval.py --gen <gen_*.jsonl> --n 60 \
        --arm matched --shard 0 --num-shards 4 --out results/faithfulness
    python scripts/eval/faithfulness_eval.py --merge --out results/faithfulness
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reward.beq_plus import BEqPlusScorer, split_header_and_theorem
from reward.faithfulness import score_faithfulness

# `INSTRUCTION_PROOF` in scripts/data/prepare_locolib.py:44-50.
PROMPT_RE = re.compile(
    r"Statement:\s*(.*?)\s*Proof:\s*(.*?)\s*Lean 4 theorem and proof:", re.S)

ARMS = ("matched", "mismatched", "padded")


def recover(prompt: str) -> tuple[str, str] | None:
    """(informal statement, informal proof) out of the stored prompt."""
    m = PROMPT_RE.search(prompt)
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def is_tactic_mode(lean_proof: str) -> bool:
    """Does the proof body contain a `by` block, i.e. can it produce FL blocks?

    The cheap string form of the term-mode split. `fl_blocks` gives the real
    verdict from Lean, but pre-filtering with this lets a run spend its judge
    budget on the scorable population instead of rediscovering the same 50%.
    """
    try:
        _ctx, decl = split_header_and_theorem(lean_proof)
        from lean_interact.utils import split_implementation
        i = split_implementation(decl)
        if i is None:
            return False
        return "by" in re.findall(r"\b\w+\b", decl[i + 2:])
    except Exception:
        return False


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]{4,}", s.lower()))


def pair_mismatched(rows: list[dict], seed: int) -> dict[int, int]:
    """Row index -> the index whose informal proof it will be scored against.

    Not a plain shuffle: a random partner can be near-identical on a dataset with
    related lemmas, which would understate the separation and flatter the metric.
    Pick, per row, the least token-similar candidate from a sample.
    """
    rng = random.Random(seed)
    out = {}
    toks = {i: _tokens(r["_statement"]) for i, r in enumerate(rows)}
    for i in range(len(rows)):
        cands = rng.sample([j for j in range(len(rows)) if j != i],
                           min(24, len(rows) - 1))
        out[i] = min(cands, key=lambda j: len(toks[i] & toks[j]) /
                     max(len(toks[i] | toks[j]), 1))
    return out


def pad_proof(lean_proof: str, n: int = 4) -> str | None:
    """The matched proof with vacuous `have`s spliced in before the real body.

    The erosion channel `reward.py:127-132` names: a `have` chain that mirrors
    the informal argument's shape without doing its work. It must not RAISE the
    score; if it does, the band would pay for padding.
    """
    try:
        _ctx, decl = split_header_and_theorem(lean_proof)
        from lean_interact.utils import split_implementation
        i = split_implementation(decl)
        if i is None:
            return None
        head, body = decl[:i + 2], decl[i + 2:].strip()
        if not body.startswith("by"):
            return None                       # term-mode: nothing to pad into
        inner = body[2:]
        haves = "".join(f"  have pad{k} : ({k}:Nat) = {k} := rfl\n" for k in range(n))
        return f"{head} by\n{haves}{inner}"
    except Exception:
        return None


def run(rows: list[dict], arm: str, out_path: Path, scorer, seed: int) -> list[dict]:
    partner = pair_mismatched(rows, seed) if arm == "mismatched" else {}
    recs, t0 = [], time.time()
    # Line-buffered APPEND, not truncate. Two writers interleaved 11 records into
    # unparseable lines on the first overnight run; O_APPEND makes each write
    # atomic up to PIPE_BUF, so a duplicate writer costs duplicates (which the
    # merge dedups) rather than corruption (which it cannot).
    fh = out_path.open("a", buffering=1)
    for i, r in enumerate(rows):
        proof_nl = (rows[partner[i]]["_proof_nl"] if arm == "mismatched"
                    else r["_proof_nl"])
        lean = pad_proof(r["pred"]) if arm == "padded" else r["pred"]
        if lean is None:                      # padding not applicable
            rec = {"i": r["i"], "arm": arm, "score": None,
                   "error": "not_applicable", "seconds": 0.0}
        else:
            context, _ = split_header_and_theorem(r["gold"])
            res = score_faithfulness(r["_statement"], proof_nl, lean,
                                     scorer=scorer, context=context)
            rec = {"i": r["i"], "arm": arm, "score": res.score, "error": res.error,
                   "n_nl": res.n_nl_blocks, "n_fl": res.n_fl_blocks,
                   "seconds": round(res.seconds, 2),
                   "strategy_match": res.strategy_match,
                   "beq_plus": bool(r.get("beq_plus"))}
        recs.append(rec)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        if (i + 1) % 5 == 0:
            k = sum(x["score"] is not None for x in recs)
            print(f"[faith] {i+1}/{len(rows)} {arm}  scored={k}  "
                  f"{(time.time()-t0)/(i+1):.1f}s/row", flush=True)
    fh.close()
    return recs


def summarise(recs: list[dict]) -> dict:
    out = {}
    for arm in sorted({r["arm"] for r in recs}):
        xs = [r for r in recs if r["arm"] == arm]
        known = [r for r in xs if r["score"] is not None]
        scores = sorted(r["score"] for r in known)
        q = lambda p: scores[int(p * (len(scores) - 1))] if scores else None
        out[arm] = {
            "n": len(xs), "scored": len(known),
            # The validity gate's input. `errors` names the cause so a term-mode
            # population is never mistaken for a broken judge.
            "coverage": len(known) / max(len(xs), 1),
            "errors": dict(Counter(r["error"] for r in xs if r["error"])),
            "mean": sum(scores) / len(scores) if scores else None,
            "p10": q(.10), "p50": q(.50), "p90": q(.90),
            "mean_seconds": sum(r["seconds"] for r in xs) / max(len(xs), 1),
        }
    m, x = out.get("matched"), out.get("mismatched")
    if m and x and m["scored"] and x["scored"]:
        a = [r["score"] for r in recs if r["arm"] == "matched" and r["score"] is not None]
        b = [r["score"] for r in recs if r["arm"] == "mismatched" and r["score"] is not None]
        # AUC = P(matched scores above mismatched), ties counted half. The
        # negative control's acceptance criterion.
        auc = lambda u, v: sum((p > q_) + 0.5 * (p == q_) for p in u for q_ in v) / (len(u) * len(v))
        # Bootstrap CI, because the pass threshold is 0.80 and a point estimate
        # near it decides nothing. If the interval straddles 0.80 the honest
        # report is "not established", not "PASS".
        rng = random.Random(0)
        boots = sorted(auc([rng.choice(a) for _ in a], [rng.choice(b) for _ in b])
                       for _ in range(2000))
        out["negative_control"] = {
            "auc": auc(a, b), "delta_mean": m["mean"] - x["mean"],
            "auc_ci95": [boots[50], boots[1949]], "n_matched": len(a), "n_mismatched": len(b)}
    p = out.get("padded")
    if m and p and p["mean"] is not None and m["mean"] is not None:
        out["padding"] = {"delta_mean": p["mean"] - m["mean"],
                          "raises_score": p["mean"] > m["mean"]}
    return out


def render(s: dict) -> None:
    print(f"\n{'arm':12s} {'n':>4} {'scored':>7} {'coverage':>9} "
          f"{'mean':>7} {'p10':>6} {'p50':>6} {'p90':>6} {'s/row':>7}")
    for arm in [a for a in ARMS if a in s]:
        v = s[arm]
        f = lambda k: f"{v[k]:.3f}" if v[k] is not None else "  --"
        print(f"{arm:12s} {v['n']:>4} {v['scored']:>7} {100*v['coverage']:>8.1f}% "
              f"{f('mean'):>7} {f('p10'):>6} {f('p50'):>6} {f('p90'):>6} "
              f"{v['mean_seconds']:>7.1f}")
        if v["errors"]:
            print(f"             unknown by cause: {v['errors']}")
    if "negative_control" in s:
        nc = s["negative_control"]
        lo, hi = nc["auc_ci95"]
        verdict = ("PASS" if lo >= 0.80 else
                   "NOT ESTABLISHED" if hi >= 0.80 else "FAIL")
        print(f"\nNEGATIVE CONTROL  auc={nc['auc']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
              f"  delta_mean={nc['delta_mean']:+.3f}")
        print(f"                  threshold 0.80 -> {verdict}"
              + ("   (CI straddles it; needs more n)" if lo < 0.80 <= hi else ""))
    if "pool" in s:
        pl = s["pool"]
        print(f"\nPOPULATION  term-mode {100*pl['term_mode_share']:.1f}% of {pl['n_pool']} rows"
              + ("  (scored rows were pre-filtered to tactic-mode)" if pl["tactic_only"] else ""))
    if "padding" in s:
        pd = s["padding"]
        verdict = "FAIL" if pd["raises_score"] else "PASS"
        print(f"PADDING           delta_mean={pd['delta_mean']:+.3f} "
              f"(must not be positive: {verdict})")


def merge(out: Path) -> dict:
    recs, seen, bad = [], set(), 0
    for f in sorted(out.glob("records.*.jsonl")):
        for line in f.open():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1          # an interleaved write, not a verdict
                continue
            key = (r["arm"], r["i"])
            if key in seen:
                continue
            seen.add(key)
            recs.append(r)
    if bad:
        print(f"[faith] skipped {bad} unparseable lines (concurrent writers)", flush=True)
    if not recs:
        raise SystemExit(f"[faith] no records.*.jsonl under {out}")
    s = summarise(recs)
    pools = [json.loads(f.read_text()) for f in out.glob("pool.*.json")]
    if pools:
        # The population-level ceiling, carried into the merged report so a
        # tactic-only run never reads as though coverage were 100%.
        s["pool"] = {"n_pool": max(p["n_pool"] for p in pools),
                     "term_mode_share": max(p["term_mode_share"] for p in pools),
                     "tactic_only": any(p["tactic_only"] for p in pools)}
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, help="a gen_*.jsonl from evaluate_checkpoints.py")
    ap.add_argument("--arm", choices=ARMS, default="matched")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/faithfulness"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--beq-only", action="store_true",
                    help="only rows BEq+ matched -- the population the band would reach")
    ap.add_argument("--tactic-only", action="store_true",
                    help="skip term-mode proofs, which cannot produce FL blocks. The "
                         "term-mode share is still recorded, so coverage stays honest")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    if args.merge:
        render(merge(args.out))
        print(f"\n-> {args.out / 'summary.json'}")
        return
    if args.gen is None:
        raise SystemExit("[faith] --gen is required unless --merge")

    rows = []
    for line in args.gen.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if args.beq_only and not r.get("beq_plus"):
            continue
        rec = recover(r["prompt"])
        if rec is None:
            continue
        r["_statement"], r["_proof_nl"] = rec
        rows.append(r)

    # Recorded BEFORE any filtering: this is the structural ceiling on coverage,
    # and it must not be hidden by the flag that works around it.
    n_pool = len(rows)
    n_term = sum(not is_tactic_mode(r["pred"]) for r in rows)
    if args.tactic_only:
        rows = [r for r in rows if is_tactic_mode(r["pred"])]
    print(f"[faith] pool {n_pool} rows, term-mode {n_term} "
          f"({100*n_term/max(n_pool,1):.1f}%) -- structurally unscorable", flush=True)
    rows = rows[: args.n]
    tag = f"{args.arm}"
    if args.num_shards > 1:
        rows = rows[args.shard::args.num_shards]
        tag += f".shard{args.shard}"
        print(f"[faith] shard {args.shard}/{args.num_shards}: {len(rows)} rows", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    scorer = BEqPlusScorer()
    recs = run(rows, args.arm, args.out / f"records.{tag}.jsonl", scorer, args.seed)
    (args.out / f"pool.{tag}.json").write_text(json.dumps(
        {"n_pool": n_pool, "n_term_mode": n_term,
         "term_mode_share": n_term / max(n_pool, 1),
         "tactic_only": args.tactic_only, "beq_only": args.beq_only}, indent=2))
    render(summarise(recs))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
