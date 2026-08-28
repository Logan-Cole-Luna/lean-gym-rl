#!/usr/bin/env python3
"""Build a MID-TRAINING corpus of Lean 4 theorem statements from Mathlib.

WHAT THIS IS. Continued pre-training data, not SFT data. The Interplay paper
(arXiv:2512.07783) finds that a mid-training stage -- plain next-token LM on a
controlled data mixture, sitting between pre-training and RL -- beats spending
the same compute on RL alone. Our diagnosis points the same way independently:
on prompts the policy NEVER solves, 84.3% still type-check. The model reliably
writes well-formed Lean that means the wrong thing. RL can only re-weight what
the policy already samples, so no reward -- BEq+ included -- can supply missing
Mathlib vocabulary. That is also why 6x the parameters moved SFT BEq+ by +0.6pp.

WHY MATHLIB STATEMENTS. Stripped of their proofs, Mathlib declarations are
almost exactly the output distribution the task asks the policy to produce:
real lemma names, conventional binder style, `Real.sqrt` not `sqrt`, the
difference between `(h : 0 < x)` and `(hx : x > 0)`. 108k of them, on disk.

KNOWN LIMITATION -- READ BEFORE TRUSTING THE CORPUS. Mathlib declarations
routinely depend on `variable` lines and open namespaces, so many are NOT
self-contained: `theorem geom_mean_le_arith_mean_weighted (w z : ι -> R) ...`
leaves `i` and `s` free. Our task's output IS self-contained. Determining
self-containedness needs elaboration, which is 108k Lean calls (~30h) and is not
worth it -- so instead `scripts/data/validate_midtrain_corpus.py` type-checks a
SAMPLE and reports the standalone-valid rate. Run it before spending training
compute. The claim this corpus makes is about vocabulary and idiom, not shape;
SFT teaches shape.

Usage:
    python scripts/data/prepare_midtrain_dataset.py --out-dir data_3b/midtrain
    python scripts/data/validate_midtrain_corpus.py --sample data_3b/midtrain/sample.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DECL_RE = re.compile(r"^(theorem|lemma)\s+([^\s(){\[:]+)")
OPENERS, CLOSERS = "([{⟨⦃", ")]}⟩⦄"


def iter_declarations(path: Path, keep_proof: bool):
    """Yield (name, text) for each top-level theorem/lemma in one .lean file.

    Statement ends at the first `:=` seen at bracket depth 0. Binder defaults
    like `(n : Nat := 3)` sit inside parens, so depth-tracking excludes them
    without a special case. Block comments and `--` lines are skipped because a
    `:=` inside a docstring would otherwise truncate the statement early.
    """
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return

    in_block_comment = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if in_block_comment:
            if "-/" in stripped:
                in_block_comment = False
            i += 1
            continue
        if stripped.startswith("/-"):
            if "-/" not in stripped:
                in_block_comment = True
            i += 1
            continue

        m = DECL_RE.match(line)
        if not m:
            i += 1
            continue

        name = m.group(2)
        buf, depth, end = [], 0, None
        j = i
        while j < len(lines) and j - i < 60:
            cur = lines[j]
            # A new top-level declaration means the previous one never closed --
            # malformed for our purposes, so drop it rather than run them together.
            if j > i and re.match(r"^(theorem|lemma|def|instance|structure|class|namespace|end|@\[)", cur):
                break
            k = 0
            while k < len(cur):
                ch = cur[k]
                if cur.startswith("--", k):
                    cur = cur[:k]
                    break
                if ch in OPENERS:
                    depth += 1
                elif ch in CLOSERS:
                    depth -= 1
                elif depth == 0 and cur.startswith(":=", k):
                    end = (j, k)
                    break
                k += 1
            buf.append(cur)
            if end:
                break
            j += 1

        if end is None:
            i += 1
            continue

        ej, ek = end
        buf[ej - i] = buf[ej - i][:ek]
        text = "\n".join(buf).rstrip()
        if keep_proof:
            text = "\n".join(lines[i : min(ej + 20, len(lines))])
        yield name, text
        i = ej + 1


# Type variables Mathlib declares in `variable` lines and section binders. A
# statement mentioning one of these without binding it is not self-contained.
TYPEVAR_RE = re.compile(r"[({]\s*[^:()\[\]{}]+\s*:\s*("
                        r"[A-Z]|[\u03b1-\u03c9]\u2080?|Type\*?|Sort)\s*[)}]")
INSTANCE_BINDER_RE = re.compile(r"\[[^\]]*\]")


def is_balanced(text: str) -> bool:
    """Reject statements the character cap truncated mid-expression.

    The first corpus emitted things like `IsCompl (A i) (` -- cut inside an open
    paren -- which cannot elaborate and taught the model to produce broken Lean.
    """
    depth = 0
    for ch in text:
        if ch in OPENERS:
            depth += 1
        elif ch in CLOSERS:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def is_concrete(text: str) -> bool:
    """Keep only statements over ground types, matching the task's own targets.

    Measured on 400 validation golds: 3.8% carry an instance binder, 1.2% mention
    `Type*`, 2.2% bind a bare single-capital type variable; 65% are over R and
    25% over N. So this filter is not a compromise -- it selects the part of
    Mathlib that looks like what the policy is asked to write.

    It is also the fix for the dominant failure mode. The first corpus validated
    at 13.0% standalone-valid, and 80.6% of failures were `failed to synthesize
    instance` (Ring R, Semiring R, LE alpha) rather than the free-variable case
    the docstring predicted: Mathlib carries typeclass context in section
    binders, so a lifted statement is missing its algebraic structure, not just a
    name.
    """
    if INSTANCE_BINDER_RE.search(text):
        return False
    if "Type" in text or "Sort" in text:
        return False
    return not TYPEVAR_RE.search(text)


IDENT_RE = re.compile(r"[A-Za-z_\u03b1-\u03c9\u0391-\u03a9][A-Za-z0-9_'\u2080-\u2089.\u03b1-\u03c9]*")
LEAN_KEYWORDS = set("""theorem lemma fun in with at by if then else let have show from
match do return forall exists Prop Type Sort where deriving""".split())


def binder_names(text: str) -> set:
    """Every name the statement itself introduces."""
    names = set()
    for grp in re.findall(r"[({\u2983]\s*([^:()\[\]{}]+?)\s*:", text):
        names |= set(grp.split())
    for grp in re.findall(r"(?:\u2200|\u2203|fun|\u03bb)\s+([^,:]+)[,:]", text):
        names |= {w for w in grp.split() if IDENT_RE.fullmatch(w)}
    for grp in re.findall(r"[\u2211\u220f\u2a06\u2a05]\s*\(?\s*([A-Za-z_][A-Za-z0-9_']*)", text):
        names.add(grp)
    return names


def free_shortnames(text: str) -> set:
    """Short or Greek identifiers used but never bound by the statement itself.

    THIS is the real self-containedness test, and it took two wrong guesses to
    find. Mathlib declares `variable {alpha : Type*} (f : X -> Y) (s : Set alpha)`
    at file or section scope, so a lifted statement references f, X, Y, s, iota,
    F, G with nothing binding them -- Lean reports `function expected at f, term
    has type ?m.63`, i.e. an unsolved metavariable, not a missing name.

    Two earlier filters missed this. Rejecting instance binders and bare
    single-letter binder TYPES caught only the narrow case `(x : alpha)` and left
    `Set alpha`, `F <=> G`, `OrthonormalBasis iota R F` untouched: the corpus went
    13.0% -> 12.0% standalone-valid, i.e. no improvement.

    The discriminator is name SHAPE. Genuine Mathlib globals are long, dotted, or
    multi-word capitalised (`Real.sqrt`, `IsClosed`, `MeasurePreserving`);
    section variables are one or two characters, or Greek. Validated both ways:
    it keeps 366/400 (91.5%) of our own validation golds -- the one rejection has
    a genuinely free `c` and would not elaborate either -- and rejects 18/20 of
    the Mathlib statements Lean had already refused.
    """
    bound = binder_names(text)
    parts = text.split(None, 2)
    body = parts[2] if len(parts) > 2 else text
    out = set()
    for m in IDENT_RE.finditer(body):
        w = m.group(0)
        if w in LEAN_KEYWORDS or "." in w or w in bound:
            continue
        head = w.split("\u2080")[0]
        if head and len(head) <= 2 and not head[0].isdigit():
            out.add(w)
    return out


def is_self_contained(text: str) -> bool:
    return not free_shortnames(text)


def n_binders(text: str) -> int:
    return len(re.findall(r"[(\[{]\s*[^:()\[\]{}]+\s*:", text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mathlib", default=str(PROJECT_ROOT / "repos/mathlib4/Mathlib"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data_3b/midtrain"))
    ap.add_argument("--keep-proof", action="store_true",
                    help="keep the proof body too (much larger, lower density of the target shape)")
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--sample-n", type=int, default=500, help="statements written out for Lean validation")
    ap.add_argument("--min-chars", type=int, default=25)
    ap.add_argument("--max-chars", type=int, default=600,
                    help="max_length is 1024 tokens in run_sft.sh; 600 chars keeps essentially "
                         "everything under that without tokenizing 108k strings")
    ap.add_argument("--allow-abstract", action="store_true",
                    help="keep statements over abstract types / with instance binders. "
                         "OFF by default -- see is_concrete(); the unfiltered corpus "
                         "validated at 13.0%% standalone-valid and hurt type-check.")
    ap.add_argument("--min-binders", type=int, default=1,
                    help="0 keeps binder-free statements; the task's outputs all carry binders")
    # Mirrors the paper's `0.2easy_0.3medium_0.5hard` weighting. Their axis is
    # operation count; ours is binder count. This is the least transferable part
    # of their result, so it is a knob, not a constant.
    ap.add_argument("--mix", default="0.2,0.3,0.5", help="easy,medium,hard sampling weights")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd

    root = Path(args.mathlib)
    files = sorted(root.rglob("*.lean"))
    print(f"[midtrain] {len(files)} .lean files under {root}")

    seen: dict[str, str] = {}
    n_raw = n_trunc = n_abstract = n_free = 0
    for f in files:
        for name, text in iter_declarations(f, args.keep_proof):
            n_raw += 1
            text = text.strip()
            if not (args.min_chars <= len(text) <= args.max_chars):
                continue
            if n_binders(text) < args.min_binders:
                continue
            if not is_balanced(text):
                n_trunc += 1
                continue
            if not args.allow_abstract and not is_concrete(text):
                n_abstract += 1
                continue
            if not args.allow_abstract and not is_self_contained(text):
                n_free += 1
                continue
            seen.setdefault(text, name)
    print(f"[midtrain] {n_raw} declarations parsed")
    print(f"[midtrain]   dropped {n_trunc} truncated mid-expression (unbalanced brackets)")
    print(f"[midtrain]   dropped {n_abstract} abstract "
          f"(instance binders / Type* / bare type variables)"
          f"{' -- FILTER DISABLED' if args.allow_abstract else ''}")
    print(f"[midtrain]   dropped {n_free} referencing unbound section variables")
    print(f"[midtrain] {len(seen)} kept after all filters + dedup")
    if not seen:
        raise SystemExit("no declarations extracted -- check --mathlib path")

    # ---- CONTAMINATION. Assert rather than assume. Lean-Workbook golds are
    # named lean_workbook*, which does not occur in Mathlib -- but the whole
    # point of a check is to fail loudly if that ever stops being true.
    texts = list(seen)
    hits = [t for t in texts if "lean_workbook" in t]
    if hits:
        raise SystemExit(f"[midtrain] FATAL: {len(hits)} declarations mention lean_workbook")
    golds = set()
    for p in ("data_3b/val.parquet", "data_3b/train.parquet"):
        fp = PROJECT_ROOT / p
        if fp.exists():
            golds |= {r["ground_truth"].strip() for r in pd.read_parquet(fp)["reward_model"]}
    overlap = sum(1 for t in texts if t in golds)
    if overlap:
        raise SystemExit(f"[midtrain] FATAL: {overlap} declarations are verbatim split golds")
    print(f"[midtrain] contamination check clean against {len(golds)} split golds")

    # ---- difficulty buckets, EQUAL-SIZED BY CONSTRUCTION.
    # Binder count alone does not work: it is discrete and piles up, so the first
    # version's terciles came out 56,774 / 16,708 / 17,079 and a requested
    # 0.2/0.3/0.5 silently degraded to 0.27/0.36/0.37 -- i.e. the knob the paper
    # cares about was not actually being set. Ranking on (binders, length) keeps
    # binder count as the primary axis while breaking ties smoothly, so splitting
    # at rank thirds gives three equal pools and the mix is exactly honoured.
    ordered = sorted(texts, key=lambda t: (n_binders(t), len(t)))
    third = len(ordered) // 3
    buckets = {"easy": ordered[:third],
               "medium": ordered[third:2 * third],
               "hard": ordered[2 * third:]}
    for k, v in buckets.items():
        b = [n_binders(t) for t in v]
        print(f"[midtrain]   {k:7s} {len(v):6d}  binders {min(b)}-{max(b)}, "
              f"median chars {sorted(len(t) for t in v)[len(v)//2]}")

    w = [float(x) for x in args.mix.split(",")]
    if len(w) != 3 or abs(sum(w) - 1.0) > 1e-6:
        raise SystemExit("--mix needs three comma-separated weights summing to 1")
    rng = random.Random(args.seed)

    # Cap the total at what the SCARCEST bucket can supply at its weight, so the
    # requested mix is delivered exactly rather than quietly distorted.
    feasible = min(int(len(pool) / frac) for pool, frac in zip(buckets.values(), w) if frac > 0)
    total = args.n_train + args.n_val
    if feasible < total:
        print(f"[midtrain] NOTE: mix {args.mix} caps the corpus at {feasible} "
              f"(requested {total}); taking {feasible} so the mix is exact")
        total = feasible

    picked: list[str] = []
    for (k, pool), frac in zip(buckets.items(), w):
        picked += rng.sample(pool, int(round(total * frac)))
    rng.shuffle(picked)
    print(f"[midtrain] sampled {len(picked)} statements at mix {args.mix} (exact)")

    # One assistant-role message == plain LM on the text: verl's
    # MultiTurnSFTDataset sets loss_mask=1 across an assistant turn, so there is
    # no prompt to condition on and every content token is trained -- which is
    # what continued pre-training is. Wrapping it in the chat template rather
    # than feeding raw text is deliberate: it puts the Lean knowledge in the same
    # turn structure SFT and GRPO later use, and reuses the SFT stack instead of
    # standing up a second training path.
    #
    # THE EMPTY SYSTEM MESSAGE IS NOT DECORATION -- measured, do not drop it.
    # verl masks only `loss_mask[:len(generation_prompt)]`, which covers
    # `<|im_start|>system\n` and nothing more. With an assistant message alone,
    # the chat template injects Qwen's DEFAULT system prompt and it lands INSIDE
    # the trained region: 48 of 51 tokens trained, of which 21 were
    # "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." --
    # ~40% of the gradient spent reinforcing boilerplate. Supplying an explicit
    # system message makes verl mask that block (role != assistant -> loss_mask
    # 0), and the trained text becomes exactly the statement. Verified by
    # decoding `input_ids[loss_mask.bool()]` for both variants.
    rows = [{"messages": [{"role": "system", "content": ""},
                          {"role": "assistant", "content": t}]} for t in picked]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_val = min(args.n_val, len(rows) // 10)
    pd.DataFrame(rows[n_val:]).to_parquet(out / "train.parquet")
    pd.DataFrame(rows[:n_val]).to_parquet(out / "val.parquet")

    sample = rng.sample(picked, min(args.sample_n, len(picked)))
    (out / "sample.jsonl").write_text("\n".join(json.dumps({"statement": s}) for s in sample))

    print(f"[midtrain] wrote {out}/train.parquet ({len(rows)-n_val}) "
          f"and val.parquet ({n_val})")
    print(f"[midtrain] wrote {out}/sample.jsonl ({len(sample)}) -- run "
          f"scripts/data/validate_midtrain_corpus.py on it BEFORE training")


if __name__ == "__main__":
    main()
