#!/usr/bin/env python3
"""Build an SFT dataset from internlm/Lean-Workbook for verl's SFT trainer.

Why SFT before RL: GRPO computes advantages *within* a rollout group, so it
learns nothing until at least one sample in a group beats its groupmates. From
the raw base model a type-checking sample is rare enough that this is a lottery
(measured: one run's first hit was 1/32 rollouts at step 2 and it bootstrapped to
100%; another drew 480 consecutive misses and never left zero). Winning that
lottery by brute force means large `rollout_n`, which is exactly what makes the
Lean reward expensive -- dozens of concurrent Mathlib-resident REPLs.

SFT sidesteps it. Training directly on (informal -> gold formal) pairs teaches
both halves of the task at once:
  * Lean syntax, so type-check rate starts high instead of at zero, and
  * semantic alignment, since the targets ARE the reference formalizations.
RL then starts from a competent policy and only has to *refine* it, so a small
`rollout_n` suffices and Lean concurrency stays bounded.

Output format is verl's SFT `messages` schema (see
repos/verl/examples/sft/gsm8k/run_qwen2_5_0_5b_fsdp.sh, `data.messages_key`).
The user turn reuses the SAME instruction template as the RL prompt
(scripts/prepare_dataset.py) so there is no train/RL prompt mismatch.

Usage:
    python scripts/prepare_sft_dataset.py --n-train 4000 --n-val 200
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_dataset import INSTRUCTION  # reuse the exact RL prompt template

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def strip_proof(formal_statement: str) -> str:
    """Return just the theorem signature, dropping `:= by sorry` / any proof.

    The RL prompt asks for "signature only, no proof", and BEq+ scores the
    statement, so the SFT target must match that or we would be training the
    model to emit something the reward then penalises.
    """
    s = formal_statement.strip()
    # `:=` at the top level starts the proof/definition body. rfind matches how
    # reward/beq_plus.py's vendored BEq+ code locates the conclusion.
    idx = s.rfind(":=")
    if idx > 0:
        s = s[:idx].strip()
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "sft"))
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    # Drop examples longer than the SFT trainer's max_length instead of letting
    # it truncate them: a truncated theorem statement is a corrupt target, and
    # verl's default truncation=error turns one long example into a crash mid
    # epoch. Measured on Lean-Workbook: mean 221 tokens, p99 485, max 1673, so
    # 1024 keeps 99.9% of the data.
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--tokenizer", default=str(PROJECT_ROOT / "models" / "qwen2.5-coder-0.5b-instruct"))
    args = ap.parse_args()

    import os

    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / ".hf_cache"))
    from datasets import load_dataset

    ds = load_dataset("internlm/Lean-Workbook", split="train")
    recs = []
    for r in ds:
        nl, formal = r.get("natural_language_statement"), r.get("formal_statement")
        if not nl or not formal:
            continue
        target = strip_proof(formal)
        # Skip anything that lost its theorem head during stripping.
        if not re.search(r"\b(theorem|lemma)\b", target):
            continue
        recs.append({"nl": nl, "target": target})
    print(f"loaded {len(recs)} usable (informal, formal-signature) pairs")

    def to_row(rec: dict) -> dict:
        return {
            "messages": [
                {"role": "user", "content": INSTRUCTION.format(informal=rec["nl"])},
                {"role": "assistant", "content": rec["target"]},
            ]
        }

    # EXCLUDE THE RL VALIDATION SET.
    #
    # These datasets were originally built by two independent shuffles, which
    # put 40 of the 80 RL validation examples (50%!) into SFT training. Measured
    # after the fact, SFT scored 32.5% BEq+ on both the seen and unseen halves --
    # identical, so that particular run was not inflated (2 epochs of a 0.5B
    # model over 4k examples does not memorise individual items). But it was
    # luck, not design: any longer/larger SFT run would start memorising, and
    # the headline "SFT beats RL" claim would quietly become train-on-test.
    val_path = PROJECT_ROOT / "data" / "val.parquet"
    if val_path.exists():
        import pandas as _pd

        val_prompts = {r[0]["content"] for r in _pd.read_parquet(val_path)["prompt"]}
        before = len(recs)
        recs = [r for r in recs if INSTRUCTION.format(informal=r["nl"]) not in val_prompts]
        print(f"held-out filter: dropped {before - len(recs)} examples that appear in {val_path.name}")
    else:
        print(f"WARNING: {val_path} not found -- cannot exclude the eval set from SFT training. "
              "Run scripts/prepare_dataset.py first.")

    if args.max_tokens > 0:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        kept = []
        for rec in recs:
            text = tok.apply_chat_template(to_row(rec)["messages"], tokenize=False)
            if len(tok(text)["input_ids"]) <= args.max_tokens:
                kept.append(rec)
        print(f"length filter (<= {args.max_tokens} tokens): kept {len(kept)}/{len(recs)}")
        recs = kept

    random.seed(args.seed)
    random.shuffle(recs)
    train = recs[: args.n_train]
    val = recs[args.n_train : args.n_train + args.n_val]
    print(f"SFT split: {len(train)} train, {len(val)} val")

    import pandas as pd

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([to_row(r) for r in train]).to_parquet(out_dir / "train.parquet")
    pd.DataFrame([to_row(r) for r in val]).to_parquet(out_dir / "val.parquet")
    print(f"wrote {out_dir/'train.parquet'} and {out_dir/'val.parquet'}")
    print("\nexample target:\n  " + train[0]["target"][:160])


if __name__ == "__main__":
    main()
