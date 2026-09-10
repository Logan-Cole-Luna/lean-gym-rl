#!/usr/bin/env python3
"""What changed between two checkpoints' COMPLETIONS, per example.

THE QUESTION THIS ANSWERS. `results/REWARD_ARMS.md` section 5: BEq+ rises ~15pp
between the SFT resume point and step 10, by the SAME amount in all three reward
arms including the exploitable one. An effect identical across three rewards is
not attributable to the reward, and it is larger than everything the rewards go
on to do. Eval settings and truncation were already ruled out. What was never
looked at is the completions themselves.

PAIRED, ON THE ROW INDEX. Both files are the same pinned 760-row val slice in
the same order, so every count here is a transition count for one example, not a
difference of two marginal rates. `converted` (was wrong, now right) and `lost`
are reported separately for the same reason CLAUDE.md gives for gain-rate: arms
differ mostly in how much they DESTROY, and a net delta hides that.

TEXT FEATURES, NOT A METRIC. Nothing here calls Lean. The columns are surface
properties of the emitted string -- did it stop restating the context, did it
stop wrapping in a fence, did it get shorter -- chosen because they are the ways
a completion can start elaborating without the model knowing more mathematics.

    python scripts/eval/jump_diff.py \\
        results/eval/sft3blocolib_proof/gen_sft3blocolib_proof-step76_n760.jsonl \\
        results/eval/rl3b_locolib_proof_lr6_gated/gen_rl3b_locolib_proof_lr6_gated-step10_n760.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Surface features of one completion. Each is a plain predicate on the raw text,
# and each is a way an output can become elaborable without any new mathematics.
FEATURES = {
    # A self-contained preamble the prompt explicitly says not to restate. If it
    # names a `variable` the gold context already binds, Lean rejects the file.
    "restates_context": lambda t: bool(re.search(r"^\s*(import|variable|open|universe)\b", t, re.M)),
    "has_fence": lambda t: "```" in t,
    "prose_before_decl": lambda t: bool(re.match(r"\s*[A-Z][a-z]", t or " ")),
    "has_theorem": lambda t: bool(re.search(r"\b(theorem|lemma)\b", t)),
    "has_proof_body": lambda t: ":=" in t,
    "term_mode": lambda t: ":=" in t and not re.search(r":=\s*by\b", t),
    "uses_sorry": lambda t: bool(re.search(r"\bsorry\b", t)),
    "multi_decl": lambda t: len(re.findall(r"^\s*(theorem|lemma)\b", t, re.M)) > 1,
}


def load(p: Path) -> dict[int, dict]:
    return {r["i"]: r for r in (json.loads(l) for l in p.open() if l.strip())}


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:5.1f}%" if d else "    --"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--key", default="typecheck", choices=("typecheck", "beq_plus"),
                    help="which verdict defines converted/lost")
    ap.add_argument("--examples", type=int, default=3,
                    help="converted rows to print in full")
    args = ap.parse_args()

    a, b = load(args.before), load(args.after)
    both = sorted(set(a) & set(b))
    if not both:
        raise SystemExit("[jump] the two files share no row indices")
    print(f"[jump] {len(both)} paired rows  ({args.before.name} -> {args.after.name})")

    for key in ("typecheck", "beq_plus"):
        conv = [i for i in both if not a[i][key] and b[i][key]]
        lost = [i for i in both if a[i][key] and not b[i][key]]
        r0 = sum(bool(a[i][key]) for i in both)
        r1 = sum(bool(b[i][key]) for i in both)
        print(f"  {key:10s} {pct(r0, len(both))} -> {pct(r1, len(both))}   "
              f"converted {len(conv):4d}   lost {len(lost):4d}")

    conv = [i for i in both if not a[i][args.key] and b[i][args.key]]
    same = [i for i in both if a[i][args.key] == b[i][args.key]]

    # The feature table. `converted` is the population the jump is made of;
    # `unchanged` is the control -- a feature that moves in BOTH is a global
    # style shift, and only a feature that moves in `converted` and NOT in
    # `unchanged` is a candidate explanation.
    print(f"\n[jump] surface features, before -> after, on the {args.key} split")
    print(f"  {'feature':20s} {'converted (n=%d)' % len(conv):>22} "
          f"{'unchanged (n=%d)' % len(same):>22}")
    for name, fn in FEATURES.items():
        def rate(rows, src):
            return sum(fn(src[i]["completion"] or "") for i in rows) / max(len(rows), 1)
        ca, cb = rate(conv, a), rate(conv, b)
        sa, sb = rate(same, a), rate(same, b)
        print(f"  {name:20s} {100*ca:8.1f} -> {100*cb:6.1f}  "
              f"{100*sa:12.1f} -> {100*sb:6.1f}")

    def length(rows, src):
        xs = [len(src[i]["completion"] or "") for i in rows]
        return sum(xs) / max(len(xs), 1)
    print(f"  {'mean chars':20s} {length(conv, a):8.0f} -> {length(conv, b):6.0f}  "
          f"{length(same, a):12.0f} -> {length(same, b):6.0f}")

    for i in conv[:args.examples]:
        print(f"\n[jump] ---- row {i}: {args.key} False -> True ----")
        print(f"  BEFORE: {a[i]['completion']!r}"[:600])
        print(f"  AFTER : {b[i]['completion']!r}"[:600])
        if a[i].get("error"):
            print(f"  before error: {str(a[i]['error'])[:200]}")


if __name__ == "__main__":
    main()
