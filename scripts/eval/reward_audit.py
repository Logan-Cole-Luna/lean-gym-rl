#!/usr/bin/env python3
"""Does the reward return what it should on inputs whose answer we already know?

Two questions no aggregate over model rollouts can answer:

1. CEILING. Feed the GOLD back in as the prediction. A reward that does not pay
   the top of its ladder for the reference answer has a ceiling below 1, and the
   policy is being told the reference is wrong -- an upper bound on what any
   amount of RL can reach, and invisible in every per-rollout metric.

2. SEPARATION. Feed a graded set of deliberate perturbations of that gold and
   check the reward orders them the way reward.py's table says it should. A
   reward can be perfectly calibrated at the ceiling and still be flat in the
   middle, and under GRPO a flat region contributes exactly zero advantage.

Both run on REAL Lean against the real `compute_score_*` entry points, never a
reimplementation of the ladder -- the wiring between the table and the scorer is
precisely the thing under test, and it is where the two defects this found live.

    python scripts/eval/reward_audit.py --gen results/eval/<arm>/gen_<...>.jsonl \\
        --n 100 --shard 0 --num-shards 5 --out results/reward_audit
    python scripts/eval/reward_audit.py --merge --out results/reward_audit
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reward.reward import OUTCOMES
from reward.reward_fn import (compute_score_gated, compute_score_outcome,
                              compute_score_typecheck)

ARMS = {"outcome": compute_score_outcome, "gated": compute_score_gated,
        "typecheck": compute_score_typecheck}

# Worst to best, for every table and figure this writes.
ORDER = ["empty", "prose", "bad_proof", "trivial_true", "trivial_rfl",
         "gold_sorry", "gold"]

_THEOREM_RE = re.compile(r"^\s*(theorem|lemma)\s", re.M)


def gold_answer(gold_record: str) -> str | None:
    """The theorem+proof the model is asked to produce, minus the given context.

    The prompt supplies the context (`prepare_locolib.py` INSTRUCTION_PROOF), so
    the answer is the last declaration only -- feeding the whole record back
    would be testing a submission no model is asked to make.
    """
    m = list(_THEOREM_RE.finditer(gold_record))
    return gold_record[m[-1].start():].strip() if m else None


def _signature(answer: str) -> str | None:
    """Everything before the top-level `:=`. None if there is no proof body."""
    from lean_interact.utils import split_implementation
    i = split_implementation(answer)
    return answer[:i].rstrip() if i is not None else None


def perturbations(answer: str) -> list[tuple[str, str, str]]:
    """(name, expected outcome, Lean text).

    `expected` is what reward/reward.py's TABLE says should happen -- not what
    was observed. A mismatch is a finding about the reward, or about this
    expectation; both are worth surfacing, neither is assumed.
    """
    out = [("gold", "solved", answer)]
    sig = _signature(answer)
    if sig:
        # Statement intact, proof removed. BEq+ still matches, `proved` false.
        out.append(("gold_sorry", "incomplete_faithful", f"{sig} := by sorry"))
        # Statement intact, proof body Lean REJECTS. Column 2 of the table is
        # "`lean file.lean` exited 0", so this is row 2, 0.05.
        out.append(("bad_proof", "no_elaborate", f"{sig} := by exact?_no_such_tactic"))
    # The two documented hacks, written SELF-CONTAINED rather than by editing the
    # gold's binders -- reusing those produced Lean that failed to elaborate on
    # 3 of 4 golds, which measures the perturbation builder, not the reward.
    # Both compile, both are sorry-free and axiom-clean, and neither formalises
    # anything. reward.py demotes them to `incomplete` via `statement_is_trivial`
    # -- IF that signal is known, which is exactly what this tests.
    out.append(("trivial_rfl", "incomplete",
                "theorem audit_trivial_rfl (n : Nat) : n = n := rfl"))
    out.append(("trivial_true", "incomplete",
                "theorem audit_trivial_true : True := trivial"))
    # Row 2, not row 1: something WAS written and Lean rejects it. Row 1 is an
    # empty submission, which is `empty`.
    out.append(("prose", "no_elaborate", "I believe this follows from the definition."))
    out.append(("empty", "no_answer", ""))
    return out


def score_all(text: str, gold_record: str) -> dict[str, dict]:
    return {name: fn("locolib", text, gold_record) for name, fn in ARMS.items()}


def summarise(recs: list[dict]) -> dict:
    """Per-case rollup. Shared by the live run and `--merge`, so a merged report
    cannot disagree with the run that produced it."""
    by_case: dict[str, list[dict]] = {}
    for x in recs:
        by_case.setdefault(x["case"], []).append(x)
    out = {}
    for case, xs in by_case.items():
        n = len(xs)
        m = lambda k: sum(x[k] for x in xs) / n
        out[case] = {
            "n": n, "expected": xs[0]["expected"],
            "matched_expected": sum(x["outcome"] == x["expected"] for x in xs) / n,
            "mean_outcome": m("score_outcome"), "mean_gated": m("score_gated"),
            "mean_typecheck": m("score_typecheck"),
            "outcome_hist": dict(Counter(x["outcome"] for x in xs)),
            "beq_plus_rate": m("beq_plus"), "proved_rate": m("proved"),
            "typecheck_rate": m("typecheck"), "scorer_error_rate": m("scorer_error"),
        }
    return out


def render(summary: dict) -> None:
    order = [c for c in ORDER if c in summary] + [c for c in summary if c not in ORDER]
    print(f"\n{'case':13s} {'n':>4} {'expected':>20} {'matched':>8} "
          f"{'outcome':>8} {'gated':>7} {'typechk':>8}")
    for case in order:
        v = summary[case]
        print(f"{case:13s} {v['n']:>4} {v['expected']:>20} "
              f"{100*v['matched_expected']:>7.1f}% {v['mean_outcome']:>8.3f} "
              f"{v['mean_gated']:>7.3f} {v['mean_typecheck']:>8.3f}")
    g = summary.get("gold")
    if g:
        print(f"\nCEILING -- the gold fed back as the prediction (n={g['n']}):")
        print(f"  outcome arm   {g['mean_outcome']:.3f} of a possible 0.850")
        print(f"  gated arm     {g['mean_gated']:.3f} of a possible 1.000")
        print(f"  typecheck arm {g['mean_typecheck']:.3f} of a possible 1.000")
        print(f"  BEq+ fires {100*g['beq_plus_rate']:.1f}%   "
              f"`proved` {100*g['proved_rate']:.1f}%   outcomes {g['outcome_hist']}")


def merge(out: Path) -> dict:
    recs = [json.loads(l) for f in sorted(out.glob("records.shard*.jsonl"))
            for l in f.open() if l.strip()]
    if not recs:
        raise SystemExit(f"[audit] no records.shard*.jsonl under {out}")
    summary = summarise(recs)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run(rows: list[dict], out_path: Path, skip_perturbations: bool) -> list[dict]:
    recs, t0 = [], time.time()
    fh = out_path.open("w")
    for i, r in enumerate(rows, 1):
        ans = gold_answer(r["gold"])
        if not ans:
            continue
        cases = [("gold", "solved", ans)] if skip_perturbations else perturbations(ans)
        for name, expect, text in cases:
            s = score_all(text, r["gold"])
            rec = {
                "i": r["i"], "case": name, "expected": expect,
                "outcome": OUTCOMES[int(s["outcome"]["outcome_code"])],
                **{f"score_{k}": v["score"] for k, v in s.items()},
                "beq_plus": bool(s["outcome"]["beq_plus"]),
                "proved": bool(s["outcome"]["proved"]),
                "typecheck": bool(s["outcome"]["typecheck"]),
                "scorer_error": bool(s["outcome"]["scorer_error"]),
            }
            recs.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
        if i % 5 == 0:
            print(f"[audit] {i}/{len(rows)} golds  {len(recs)} scored  "
                  f"{(time.time()-t0)/i:.1f}s/gold", flush=True)
    fh.close()
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, help="a gen_*.jsonl (only its `gold` column is read)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("results/reward_audit"))
    ap.add_argument("--skip-perturbations", action="store_true")
    # Each gold costs ~65s: the two triviality probes elaborate but do NOT match
    # the gold, so they pay the FULL cascade instead of exiting at a rung.
    # Shard it the way beq_calibration.py does.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true",
                    help="roll every records.shard*.jsonl under --out into one "
                         "summary.json and exit; no Lean")
    args = ap.parse_args()

    if args.merge:
        render(merge(args.out))
        print(f"\n-> {args.out / 'summary.json'}")
        return
    if args.gen is None:
        raise SystemExit("[audit] --gen is required unless --merge")

    rows = [json.loads(l) for l in args.gen.open() if l.strip()][: args.n]
    tag = ""
    if args.num_shards > 1:
        rows = rows[args.shard::args.num_shards]
        tag = f".shard{args.shard}"
        print(f"[audit] shard {args.shard}/{args.num_shards}: {len(rows)} golds", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    recs = run(rows, args.out / f"records{tag}.jsonl", args.skip_perturbations)
    summary = summarise(recs)
    (args.out / f"summary{tag}.json").write_text(json.dumps(summary, indent=2))
    render(summary)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
