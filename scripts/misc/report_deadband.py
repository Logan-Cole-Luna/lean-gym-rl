#!/usr/bin/env python3
"""Report what BEq+'s cascade already knows about the dead band.

Reads a `*.scored.jsonl` produced after the cascade instrumentation landed and
answers the Phase 0 question: **how often is the dead-band signal already free?**

    python scripts/misc/report_deadband.py --scored data_3b/rollouts/deadband_sft3b-step93.scored.jsonl

The decision this feeds: a BEq+ AND self-prove ladder is priced in the record at
"~20+ min/step, the dearest arm in the set", on the assumption that the full
5-call `self_prove` ladder has to run on every dead-band rollout. If the cascade
already reaches rung 3 on most of them, `provable` is already paid for and only
the `tauto` triviality split needs a new call.

READ THE CAVEAT IN THE OUTPUT. `provable_alone` is a LOWER BOUND on
`self_prove`'s `provable`: rung 3 proves with tauto + simp_all_arith! (its
`exact? using this` has no `this` in scope), where `self_prove` also tries
`noncomm_ring` and a bare `exact?`. A high rate here is trustworthy; a low one
may just mean the cheaper tactic list is too weak, not that the statements are
false.
"""
import argparse
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: F401  -- repo root + every stage folder

# One definition of the bands, shared with the script that BUILDS this pool.
# This script and make_deadband_subset.py are run as a pair, so a private predicate here
# is a silent disagreement about what the file is supposed to contain.
from make_deadband_subset import band_of


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% CI. Normal approximation is wrong at the extremes and these rates
    may well be near 0 or 1, which is exactly where it misleads."""
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h) * 100, min(1.0, c + h) * 100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.scored) if l.strip()]
    n = len(recs)
    if not n:
        raise SystemExit(f"{args.scored} is empty")
    if "provable_alone" not in recs[0]:
        raise SystemExit(
            f"{args.scored} predates the cascade instrumentation -- no `provable_alone` "
            f"field. Re-score it with the current scripts/pool/score_rollouts.py.")

    print(f"\n=== dead-band cascade instrumentation: {args.scored} ===")
    print(f"{n} rollouts\n")

    # Sanity: this pool is supposed to be dead band only.
    off = [r for r in recs if band_of(r) != "dead"]
    err = [r for r in recs if r.get("error_kind")]
    if off:
        print(f"!! {len(off)} rollouts are NOT dead band.")
        print("   Verdicts can move between scoring passes -- Lean timeouts are not")
        print("   deterministic. Rates below are over the whole file regardless.\n")
    if err:
        kinds = collections.Counter(r["error_kind"] for r in err)
        print(f"!! {len(err)} scorer errors ({len(err)/n*100:.1f}%): {dict(kinds)}")
        print("   These score 0 and bias against correctness; see beq_plus.py.\n")

    print("--- THE QUESTION: is `provable_alone` already free? ---")
    known = [r for r in recs if r.get("provable_alone") is not None]
    lo, hi = wilson(len(known), n)
    print(f"  cascade reached rung 3 (provable_alone computed): "
          f"{len(known):5}/{n}  {len(known)/n*100:5.1f}%  [95% CI {lo:.1f}-{hi:.1f}]")
    if known:
        prov = sum(1 for r in known if r["provable_alone"])
        plo, phi = wilson(prov, len(known))
        print(f"  ... of those, prediction proves on its own: "
              f"{prov:5}/{len(known)}  {prov/len(known)*100:5.1f}%  [95% CI {plo:.1f}-{phi:.1f}]")
        print(f"  ... i.e. {prov/n*100:.1f}% of the dead band would reach the "
              f"'true and non-trivial' rung for free (before the tauto split)")

    print("\n--- why direction 0 stopped ---")
    for k, v in collections.Counter(r.get("stop_reason") for r in recs).most_common():
        label = {None: "ran to completion", "unparseable": "unparseable (no theorem)",
                 "sorry_gate": "sorry gate (would not elaborate beside gold)",
                 "no_rung": "no rung closed (the real dead band)"}.get(k, str(k))
        print(f"  {label:48} {v:5}  {v/n*100:5.1f}%")
    print("  NOTE: 'sorry_gate' is the share where rung 3 is unreachable, so a")
    print("        ladder needs its own prove call for those.")

    print("\n--- cascade rung / convert level (mostly empty on a dead-band pool) ---")
    for k, v in sorted(collections.Counter(r.get("rung") for r in recs).items(),
                       key=lambda kv: (kv[0] is None, kv[0])):
        print(f"  rung {str(k):5} {v:5}  {v/n*100:5.1f}%")
    cl = collections.Counter(r.get("convert_level") for r in recs if r.get("convert_level") is not None)
    print(f"  convert levels: {dict(sorted(cl.items())) if cl else 'none fired'}")
    print(f"  beql: {sum(1 for r in recs if r.get('beql'))}/{n}")

    print("\nCAVEAT: `provable_alone` is a LOWER BOUND on self_prove's `provable`.")
    print("Rung 3 uses tauto + simp_all_arith! (its `exact? using this` has no `this`")
    print("in scope); self_prove also tries noncomm_ring and a bare exact?. It also")
    print("cannot separate `trivial` (tauto alone) -- that needs one more call.\n")


if __name__ == "__main__":
    main()
