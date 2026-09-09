#!/usr/bin/env python3
"""Rebuild results/eval/ from re-scored generations, under one named toolchain.

WHY. Not one eval record written before 2026-09-07 carries a `toolchain` field,
and the series were provably not all scored under the same Lean. That produced a
15pp phantom result once (see CLAUDE.md), and it cannot be detected after the
fact from the records themselves. `hpc/rescore_array.slurm` re-derives every
cached generation under v4.23; this turns those into eval records in the layout
every reader in the repo already expects.

THE COMPLETIONS ARE NOT REGENERATED. Only the Lean verdict changes, so a
rebuilt record describes exactly the same model outputs as the record it
replaces. That is what makes the old and new numbers comparable at all, and it
is why `n`, `i` and the per-example ordering are carried across unchanged.

ORIGINALS ARE MOVED, NEVER DELETED. Each series' pre-rebuild records go to
`results/eval/<series>/_superseded_<stamp>/`. They are provenance: several
published numbers were computed from them, and a claim about what changed is
only checkable if both versions survive.

WHAT IS NOT CARRIED OVER. Fields the rescore cannot re-derive because they
describe generation rather than scoring -- `model_dir`, `max_new_tokens`,
`truncated_rate` -- are copied from the old record when present and omitted
otherwise, never invented.

    python scripts/eval/rebuild_evals.py --dry-run     # show the diff first
    python scripts/eval/rebuild_evals.py --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "results" / "eval"
RESCORE = ROOT / "results" / "rescore"


def build_record(old: dict, rs: dict) -> dict:
    """One eval record, rebuilt from a rescore. Same shape readers expect."""
    per = rs["per_example"]
    n = len(per)
    n_tc = sum(e["typecheck"] for e in per)
    n_beq = sum(e["beq_plus"] for e in per)
    n_err = sum(bool(e["error_kind"]) for e in per)
    rec = {
        "n": n,
        "typecheck_rate": n_tc / n if n else 0.0,
        "beq_plus_rate": n_beq / n if n else 0.0,
        "scorer_error_rate": n_err / n if n else 0.0,
        # THE POINT OF THE EXERCISE. Every rebuilt record says which Lean
        # produced it, so this class of bug is self-diagnosing from now on.
        "toolchain": rs.get("toolchain"),
        "rescored_from": rs.get("gen"),
        "rebuilt_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "per_example": per,
    }
    # Derived from THIS rescore, never carried over. `weaker_only_rate` used to
    # be in the copy list below, which put a v4.8-derived rate inside a record
    # whose other rates were v4.23: a single record spanning two toolchains,
    # which is the exact defect this rebuild exists to remove.
    per_ok = per and "gold_implies_pred" in per[0]
    if per_ok:
        rec["weaker_only_rate"] = sum(
            1 for e in per if e["gold_implies_pred"] and not e["beq_plus"]) / n
    elif "weaker_only_rate" in old:
        raise SystemExit(
            "[rebuild] REFUSING: the rescore lacks `gold_implies_pred`, so "
            "`weaker_only_rate` cannot be re-derived, and copying the old value "
            "would mix toolchains inside one record. Re-run the rescore with the "
            "current scripts/eval/rescore_gen.py.")
    # GENERATION-side facts the rescore genuinely cannot know, because they
    # describe decoding rather than scoring. Copied if present, never invented.
    for k in ("model_dir", "max_new_tokens", "truncated_rate"):
        if k in old:
            rec[k] = old[k]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    # MANIFEST-DRIVEN, NOT A DIRECTORY WALK. Scope has to come from the same
    # file that defined what was re-scored, or the two disagree: an rglob over
    # results/eval also finds `archive/` (superseded Lean-Workbook runs that
    # target v4.8) and would rebuild those from v4.23 rescores left behind by an
    # aborted run. Corpus and toolchain travel together; the manifest is where
    # that pairing is recorded.
    ap.add_argument("--manifest", type=Path,
                    default=ROOT / "results" / "manifests" / "rescore_manifest_locolib.txt")
    ap.add_argument("--tag", default="v423", help="rescore tag to rebuild from")
    ap.add_argument("--apply", action="store_true", help="write; default is a dry run")
    # Accepted and explicit, because the docstring advertises it and a flag that
    # errors is worse than one that is redundant. Dry run is already the default;
    # this just lets the documented invocation work.
    ap.add_argument("--dry-run", action="store_true",
                    help="show the diff without writing (the default)")
    ap.add_argument("--only", default="", help="substring filter on the series name")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d")
    if not args.manifest.exists():
        raise SystemExit(f"[rebuild] no manifest at {args.manifest}")
    # The eval records this manifest's generations correspond to, and no others.
    in_scope = set()
    for line in args.manifest.read_text().splitlines():
        g = Path(line.strip())
        if not line.strip():
            continue
        in_scope.add(g.parent / f"eval_{g.name[len('gen_'):-len('.jsonl')]}.json")
    print(f"[rebuild] manifest {args.manifest.name}: {len(in_scope)} records in scope\n")

    rows, missing = [], []
    for ev in sorted(in_scope):
        if not ev.exists() or "_superseded" in str(ev):
            continue
        series = ev.parent.name
        if args.only and args.only not in series:
            continue
        label = ev.stem[len("eval_"):]                 # <label>-stepN_n<n>
        rsf = RESCORE / f"{label}.{args.tag}.json"
        if not rsf.exists():
            missing.append(label)
            continue
        old_doc = json.load(open(ev))
        # Records are keyed by the label inside the json; there is exactly one.
        key = next(iter(old_doc))
        old = old_doc[key]
        rs = json.load(open(rsf))
        new = build_record(old, rs)
        rows.append((ev, key, old, new, rs.get("toolchain")))

    print(f"{'series/label':58s} {'BEq+ old':>9} {'BEq+ new':>9} {'delta':>8}  toolchain")
    for ev, key, old, new, tc in rows:
        o = 100 * old.get("beq_plus_rate", 0.0)
        n = 100 * new["beq_plus_rate"]
        flag = "  <-- CHANGED" if abs(n - o) > 0.5 else ""
        print(f"{ev.parent.name + '/' + key:58s} {o:8.2f}% {n:8.2f}% {n-o:+7.2f}  "
              f"{str(tc):<22}{flag}")
    if missing:
        print(f"\n{len(missing)} eval records have no {args.tag} rescore yet "
              f"(array still running?). First few: {missing[:4]}")

    changed = sum(1 for _, _, o, n, _ in rows
                  if abs(100*n["beq_plus_rate"] - 100*o.get("beq_plus_rate", 0)) > 0.5)
    print(f"\n{len(rows)} records rebuildable, {changed} would change by >0.5pp")
    if args.dry_run and args.apply:
        raise SystemExit("[rebuild] --dry-run and --apply are contradictory")
    if not args.apply:
        print("dry run; pass --apply to write")
        return

    moved = set()
    for ev, key, _old, new, _tc in rows:
        arch = ev.parent / f"_superseded_{stamp}"
        if ev.parent not in moved:
            arch.mkdir(exist_ok=True)
            moved.add(ev.parent)
        shutil.move(str(ev), str(arch / ev.name))
        ev.write_text(json.dumps({key: new}, indent=2))
    print(f"rebuilt {len(rows)} records; originals under _superseded_{stamp}/")


if __name__ == "__main__":
    main()
