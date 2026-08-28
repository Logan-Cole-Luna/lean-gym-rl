#!/usr/bin/env python3
"""pass@k curve from a scored k-sample rollout file.

WHY THIS EXISTS. The Interplay paper (arXiv:2512.07783) argues RL produces a
true CAPABILITY gain only when pass@128 moves -- anything visible at pass@1 but
not at large k is sharpening of a distribution the policy already had, not new
ability. Our arms differ almost entirely in RETENTION (74% of the 0.5B BEq+
effect was protecting answers the policy already produced), which is exactly the
signature of sharpening. We have never measured pass@k above 8, so we cannot
currently claim a capability gain in their sense. This closes that gap.

Uses the standard unbiased estimator (Chen et al. 2021): for a prompt with c
successes out of n samples,
    pass@k = 1 - C(n-c, k) / C(n, k)
which is what the whole-sample rate would be in expectation, not the optimistic
"did any of my n succeed" count.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    # product form avoids overflow and is exact in float for our sizes
    p = 1.0
    for i in range(k):
        p *= (n - c - i) / (n - i)
    return 1.0 - p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--metric", default="beq_plus", choices=["beq_plus", "typecheck"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    groups = collections.defaultdict(list)
    for line in Path(args.scored).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            groups[d["prompt_index"]].append(d)

    # Use the MODAL group size and drop anything short of it, rather than
    # truncating everyone down to the minimum. A run killed mid-write leaves one
    # ragged group, and truncating to that group's size would silently cost the
    # whole top of the curve -- a 7743/7744-line file would report pass@16 as the
    # maximum instead of pass@32.
    sizes = collections.Counter(len(v) for v in groups.values())
    n = sizes.most_common(1)[0][0]
    dropped = sum(c for sz, c in sizes.items() if sz < n)
    if dropped:
        print(f"[passk] group sizes {dict(sorted(sizes.items()))}; "
              f"dropping {dropped} short of n={n}")
    counts = {pi: sum(1 for x in v[:n] if x[args.metric])
              for pi, v in groups.items() if len(v) >= n}
    P = len(counts)

    rows = []
    ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= n]
    for k in ks:
        rate = sum(pass_at_k(n, c, k) for c in counts.values()) / P
        rows.append({"k": k, "rate": rate})

    solved_ever = sum(1 for c in counts.values() if c > 0) / P
    print(f"=== pass@k [{args.label}] metric={args.metric} "
          f"{P} prompts, n={n} samples each ===")
    for r in rows:
        print(f"  pass@{r['k']:<4d} {100*r['rate']:5.1f}%")
    print(f"  ever solved  {100*solved_ever:5.1f}%   (ceiling at this n and temperature)")

    payload = {"label": args.label, "metric": args.metric, "n_prompts": P,
               "n_samples": n, "curve": rows, "ever_solved": solved_ever,
               "success_histogram": dict(sorted(collections.Counter(counts.values()).items()))}
    out = args.out or f"results/passk_{args.label}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
