#!/usr/bin/env python3
"""Build SFT / RL / val splits from LoCoLib (../PartitionAndProve).

WHY A SECOND CORPUS. Lean-Workbook is competition problems; LoCoLib is Mathlib
itself, LLM-informalized. Same task (informal -> formal statement), different
surface context. That makes it *contextually out-of-distribution* for a policy
SFT'd only on Lean-Workbook, which is exactly the axis the Interplay paper
(arXiv:2512.07783) measures.

WHAT IS AND IS NOT USABLE, measured rather than assumed:

  labelled, has gold formal_statement, source=mathlib   18,757   <- the only usable rows
    train_algebraic_structures  10,151
    train_foundations_logic      5,911
    train_number_theory          2,695
  unlabelled, EMPTY formal_statement, source=mizar      18,081   <- no gold, unusable for BEq+
  val_*_unlabelled, EMPTY formal_statement              771      <- no gold, unusable for BEq+
  LoCoLib_distilled                                     18,501   <- 100% uuid SUBSET of the
                                                                    labelled train files

So **LoCoLib ships no usable BEq+ validation set**: the files named `val_*` carry
no gold, and `LoCoLib_distilled` is a subset of train (18,501/18,501 uuids
overlap), so validating on it would be training on the eval set. We carve our own
val slice here, the same way `prepare_dataset.py` does for Lean-Workbook.

THE PREAMBLE IS THE POINT. Every LoCoLib gold ships its own context (imports,
`open`, `variable`, `namespace`) before the theorem. Mid-training on raw Mathlib
previously failed because only ~2.3% of declarations elaborate standalone; these
carry the context they need. We put that context in the PROMPT and ask only for
the theorem, which matches how `reward/beq_plus.py` already works: the header
becomes a cached Lean env, the theorem is what gets scored.

    python scripts/data/prepare_locolib.py --out-dir data_locolib

NOTE this does not verify the statements elaborate under OUR Mathlib
(v4.8.0-rc1; LoCoLib targets 4.23.0). SFT does not need Lean, so it can proceed
either way. RL cannot: check that the golds elaborate before committing to RL.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: F401  -- repo root + every stage folder

# The Lean-Workbook path and the LoCoLib path use the SAME rule for where a
# proof body starts, so they use the same function. prepare_locolib_midtrain.py
# imports it from here, which meant a fix would have reached two of three
# consumers.
from prepare_sft_dataset import strip_proof

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = PROJECT_ROOT.parent / "PartitionAndProve" / "datasets_training" / "LoCoLib"

# Kept as one constant and imported by anything that builds a LoCoLib prompt, so
# the SFT target and the RL prompt cannot drift apart. `prepare_sft_dataset.py`
# learned that lesson on the Lean-Workbook side.
INSTRUCTION = (
    "Formalize the following mathematical statement as a single Lean 4 theorem "
    "declaration (signature only, no proof -- do not include `:=` or a proof term).\n\n"
    "The following Lean context is already in scope; do not restate it:\n"
    "```lean\n{context}\n```\n\n"
    "Statement:\n{informal}\n\nLean 4 theorem:"
)

# The proof-pair variant (`--emit-proof`): asks for a complete theorem AND its
# proof, and additionally feeds the informal proof as input -- LoCoLib's
# labelled rows carry `informal_proof`/`formal_proof` (a real, unstripped
# Mathlib proof), unlike Lean-Workbook. Kept as a SEPARATE constant rather than
# a branch inside INSTRUCTION so neither prompt's wording can drift by editing
# the other -- see the comment above about the SFT target / RL prompt link.
INSTRUCTION_PROOF = (
    "Formalize the following mathematical statement and its proof as a complete "
    "Lean 4 theorem declaration, including the proof.\n\n"
    "The following Lean context is already in scope; do not restate it:\n"
    "```lean\n{context}\n```\n\n"
    "Statement:\n{informal}\n\nProof:\n{informal_proof}\n\nLean 4 theorem and proof:"
)

_THEOREM_RE = re.compile(r"^\s*(theorem|lemma)\s", re.M)


def split_context_and_theorem(formal: str) -> tuple[str, str] | None:
    """(context, theorem) on the LAST theorem/lemma keyword.

    Mirrors `reward.beq_plus.split_header_and_theorem`, but kept separate: that
    one also strips `import` lines because BEq+ supplies its own `import
    Mathlib`. Here we keep the raw split and drop imports in `clean_context`, so
    the two concerns stay visible.
    """
    matches = list(_THEOREM_RE.finditer(formal))
    if not matches:
        return None
    i = matches[-1].start()
    return formal[:i], formal[i:]


def clean_context(context: str) -> str:
    """Drop `import` lines: BEq+ elaborates against its own `import Mathlib`
    base env, so re-importing is redundant and version-fragile."""
    keep = [ln for ln in context.splitlines() if not ln.strip().startswith("import ")]
    return "\n".join(keep).strip()


def norm(text: str) -> str:
    """Whitespace-insensitive key for content-level dedup. Splitting by uuid is
    not enough: the Lean-Workbook split measured 93.4% content overlap across
    distinct ids and a +34.9pp generalisation gap when it was ignored."""
    return " ".join(text.split())


def content_key(r: dict) -> tuple[str, str]:
    """The dedup key. Used both to drop duplicates during load and to verify the
    finished splits are disjoint -- the second is only a check on the first if
    both spell the key the same way, so there is one spelling."""
    return (norm(r["informal"]), norm(r["signature"]))


def load(src: Path) -> tuple[list[dict], Counter]:
    rows, seen_stmt, dropped = [], set(), Counter()
    for f in sorted(src.glob("train_*.jsonl")):
        if "unlabelled" in f.name:
            continue  # no gold formal_statement; nothing for BEq+ to score against
        for line in open(f, errors="replace"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                dropped["unparseable_json"] += 1
                continue
            # `formal_statement` is ALWAYS proof-stripped -- every record ends
            # exactly at `:=` with nothing after, confirmed against source.
            # `formal_proof` carries the real proof; everything BEFORE its `:=`
            # is byte-identical to `formal_statement`, so preferring it changes
            # nothing about the signature/context/dedup key, only recovers the
            # proof body. Measured: assuming they were duplicate fields (they
            # looked identical in a single spot-checked sample, which just
            # hadn't been printed far enough to reach the `:=`) silently built
            # every SFT/RL target in `--emit-proof` mode from truncated text --
            # elaboration failed with "unexpected end of input" on 8/8 sampled
            # failures, at any Mathlib version, because the text had no proof
            # to elaborate at all.
            formal_proof_field = (r.get("formal_proof") or "").strip()
            formal_stmt_field = (r.get("formal_statement") or "").strip()
            formal = formal_proof_field or formal_stmt_field
            informal = (r.get("informal_statement") or "").strip()
            # Only read, never filtered on here: `load()` must return the SAME
            # row population regardless of --emit-proof, so the domain-stratified
            # split in main() stays byte-identical between the signature-only and
            # proof-pair runs (comparable results on identical problems). A row
            # missing informal_proof is dropped later, in main()'s per-item loop,
            # only when --emit-proof actually needs it.
            informal_proof = (r.get("informal_proof") or "").strip()
            if not formal or not informal:
                dropped["missing_field"] += 1
                continue
            parts = split_context_and_theorem(formal)
            if parts is None:
                dropped["no_theorem_keyword"] += 1
                continue
            context, theorem = parts
            theorem = theorem.strip()
            signature = strip_proof(theorem)
            if not signature:
                dropped["empty_signature"] += 1
                continue
            key = content_key({"informal": informal, "signature": signature})
            if key in seen_stmt:
                dropped["duplicate_content"] += 1
                continue
            seen_stmt.add(key)
            rows.append({
                "uuid": r["uuid"],
                "domain": r.get("domain", "unknown"),
                "informal": informal,
                "informal_proof": informal_proof,
                "context": clean_context(context),
                "signature": signature,
                # The theorem HEAD to the end of the record, proof included
                # verbatim (source is `formal_statement`, which for LoCoLib's
                # labelled/mathlib rows is identical to `formal_proof` -- a
                # real, unstripped Mathlib proof, not a placeholder). Unlike
                # `gold_full` this carries no surrounding context/imports, the
                # same convention `signature` already follows.
                "theorem_with_proof": theorem,
                "gold_full": formal,
            })
    return rows, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data_locolib")
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-rl", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    # LENGTH FILTERS. verl's SFT trainer RAISES on an over-long row rather than
    # truncating (multiturn_sft_dataset.py: "sequence_length=... is larger than
    # max_length"), so an unfiltered corpus kills the job mid-epoch. The RL side
    # has its own, smaller cap: run_grpo.sh sets max_prompt_length=768.
    # Measured on this corpus: SFT median 305 tokens, exactly 1 row over 1024.
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-prompt-tokens", type=int, default=768)
    ap.add_argument("--tokenizer", default="/scratch/logan03/ai4math_training_models/qwen2.5-coder-3b-instruct")
    # Builds the theorem+proof-pair variant instead: prompts feed BOTH
    # informal_statement and informal_proof and ask for a complete Lean
    # theorem WITH its proof; targets are `theorem_with_proof` (unstripped)
    # instead of `signature`. Written to *_proof.parquet, alongside (never
    # overwriting) the signature-only files -- see the module docstring.
    ap.add_argument("--emit-proof", action="store_true",
                    help="build the theorem+proof-pair splits (data_locolib/*_proof.parquet) "
                         "instead of the signature-only ones")
    args = ap.parse_args()

    rows, dropped = load(args.src)
    print(f"[locolib] {len(rows)} usable rows from {args.src}")
    if dropped:
        print(f"[locolib] dropped: {dict(dropped)}")
    by_domain = Counter(r["domain"] for r in rows)
    print(f"[locolib] domains: {dict(by_domain)}")

    # STRATIFY BY DOMAIN. The three domains are 54/32/14 percent, so an
    # unstratified shuffle would let a small val slice drift badly on the
    # smallest domain and make SFT/RL/val non-comparable.
    groups = defaultdict(list)
    for r in rows:
        groups[r["domain"]].append(r)
    rng = random.Random(args.seed)
    splits = {"val": [], "rl": [], "sft": []}
    for dom, items in sorted(groups.items()):
        rng.shuffle(items)
        frac = len(items) / len(rows)
        n_val = round(args.n_val * frac)
        n_rl = round(args.n_rl * frac)
        splits["val"] += items[:n_val]
        splits["rl"] += items[n_val:n_val + n_rl]
        splits["sft"] += items[n_val + n_rl:]
    for k in splits:
        rng.shuffle(splits[k])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("[locolib] pandas/pyarrow needed: source hpc/cc_env.sh first")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    # NO SYSTEM TURN. `prepare_dataset.py` and `prepare_sft_dataset.py` both emit
    # a bare user turn, and `evaluate_checkpoints.py` reads `prompt[0]["content"]`.
    # An earlier version of this file added a system turn, so element 0 was the
    # system text and EVERY eval example got the identical prompt "You are an
    # expert in Lean 4 and Mathlib.". The model duly produced one identical
    # theorem for all 999 rows: 100% type-check and 0% BEq+ at step 46, 0% / 0%
    # after. Match the convention; do not reintroduce the system turn here
    # without also fixing every consumer.

    def n_tokens(msgs) -> int:
        return len(tok(tok.apply_chat_template(msgs, tokenize=False)).input_ids)

    def format_prompt(r: dict) -> str:
        if args.emit_proof:
            return INSTRUCTION_PROOF.format(context=r["context"] or "-- (none)",
                                            informal=r["informal"],
                                            informal_proof=r["informal_proof"])
        return INSTRUCTION.format(context=r["context"] or "-- (none)", informal=r["informal"])

    def target_of(r: dict) -> str:
        return r["theorem_with_proof"] if args.emit_proof else r["signature"]

    suffix = "_proof" if args.emit_proof else ""

    for name, items in splits.items():
        kept, over_full, over_prompt, missing_proof = [], 0, 0, 0
        for r in items:
            # Only relevant in --emit-proof mode: a row's split membership is
            # decided above, identically regardless of mode (see load()), so
            # this is the ONLY place a row can drop out because it lacks an
            # informal_proof -- keeping the two runs' splits comparable except
            # for this narrow, counted difference.
            if args.emit_proof and not r["informal_proof"]:
                missing_proof += 1
                continue
            pr = format_prompt(r)
            target = target_of(r)
            full = n_tokens([{"role": "user", "content": pr},
                             {"role": "assistant", "content": target}])
            if full > args.max_tokens:
                over_full += 1
                continue
            # Only the RL/eval splits are generated from, so only they need the
            # prompt to fit on its own. Computing it for sft costs ~11.7k encodes
            # for a number nothing reads.
            if name != "sft" and n_tokens([{"role": "user", "content": pr}]) > args.max_prompt_tokens:
                over_prompt += 1
                continue
            kept.append((r, pr))
        if over_full or over_prompt or missing_proof:
            print(f"[locolib] {name}: dropped {over_full} over {args.max_tokens} tok"
                  + (f", {over_prompt} prompts over {args.max_prompt_tokens}" if over_prompt else "")
                  + (f", {missing_proof} missing informal_proof" if missing_proof else ""))
        # Write the filtered rows BACK. This used to rebind a local, leaving
        # splits[name] holding the pre-filter list -- so the disjointness check
        # at the end was run over rows that never reached the parquet.
        items, prompts = (list(x) for x in zip(*kept)) if kept else ([], [])
        splits[name] = items
        if name == "sft":
            # verl's SFT trainer reads `messages`. An explicit SYSTEM turn is
            # required: verl's loss mask only covers `<|im_start|>system\n`, so a
            # lone user+assistant pair trains on Qwen's injected default system
            # prompt (~40% of tokens).
            df = pd.DataFrame({"messages": [
                [{"role": "user", "content": p},
                 {"role": "assistant", "content": target_of(r)}]
                for p, r in zip(prompts, items)]})
        else:
            df = pd.DataFrame({
                "data_source": ["locolib"] * len(items),
                "prompt": [[{"role": "user", "content": p}] for p in prompts],
                "ability": ["lean4_autoformalization"] * len(items),
                "reward_model": [{"style": "rule", "ground_truth": r["gold_full"]} for r in items],
                "extra_info": [{"split": name, "index": i, "id": r["uuid"],
                                "domain": r["domain"]} for i, r in enumerate(items)],
            })
        path = args.out_dir / f"{name}{suffix}.parquet"
        df.to_parquet(path)
        if name == "val":
            # The val slice has to serve two consumers with different schemas:
            # verl's SFT trainer wants `messages`, the RL/eval path wants
            # prompt + reward_model. Same rows either way, so the SFT loss curve
            # and the BEq+ eval are measured on identical data.
            val_sft_path = args.out_dir / f"val_sft{suffix}.parquet"
            pd.DataFrame({"messages": [
                [{"role": "user", "content": pr},
                 {"role": "assistant", "content": target_of(r)}]
                for pr, r in zip(prompts, items)]}).to_parquet(val_sft_path)
            print(f"[locolib] val_sft {len(items):5} rows -> {val_sft_path}")
        doms = Counter(r["domain"] for r in items)
        print(f"[locolib] {name:4} {len(items):6} rows -> {path}  {dict(doms)}")

    # A held-out val slice is only worth anything if it is genuinely disjoint.
    keys = {k: {content_key(r) for r in v} for k, v in splits.items()}
    for a, b in (("val", "sft"), ("val", "rl"), ("sft", "rl")):
        overlap = len(keys[a] & keys[b])
        print(f"[locolib] content overlap {a} n {b}: {overlap}"
              + ("  <-- LEAK" if overlap else "  ok"))


if __name__ == "__main__":
    main()
