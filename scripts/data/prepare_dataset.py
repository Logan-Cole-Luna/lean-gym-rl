#!/usr/bin/env python3
"""Convert internlm/Lean-Workbook (HF) into verl's expected prompt/parquet format
for GRPO autoformalization RL.

Lean Workbook: 57K NL<->Lean4 pairs (57,231 in the base split; this HF mirror's
default config has 25,214 rows), built specifically for autoformalization
training, targeting Lean v4.8.0-rc1 -- we build Mathlib4 at the matching
`v4.8.0-rc1` tag in repos/mathlib4 to stay compatible (see reward/beq_plus.py).
Each `formal_statement` is already a bare, self-contained `theorem ... := by
sorry` (no imports/namespace needed -- `import Mathlib` is implicit), so no
header/context splitting is required for this dataset.

Each output row:
  data_source   "lean_workbook"
  prompt        [{"role": "user", "content": <instruction + natural_language_statement>}]
  ability       "lean4_autoformalization"
  reward_model  {"style": "rule", "ground_truth": <formal_statement>}
  extra_info    {"id", "split", "index"}
"""
from __future__ import annotations

import argparse
import random

INSTRUCTION = (
    "Formalize the following mathematical statement as a single Lean 4 theorem "
    "declaration (signature only, no proof -- do not include `:=` or a proof term. "
    "Assume `import Mathlib` is already in scope; do not restate it).\n\n"
    "Statement:\n{informal}\n\nLean 4 theorem:"
)


def to_verl_row(rec: dict, split: str, idx: int) -> dict:
    return {
        "data_source": "lean_workbook",
        "prompt": [{"role": "user", "content": INSTRUCTION.format(informal=rec["natural_language_statement"])}],
        "ability": "lean4_autoformalization",
        "reward_model": {"style": "rule", "ground_truth": rec["formal_statement"]},
        "extra_info": {"split": split, "index": idx, "id": rec["id"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--n-train", type=int, default=400, help="PoC subset size for train")
    ap.add_argument("--n-val", type=int, default=80, help="PoC subset size for val")
    ap.add_argument("--seed", type=int, default=0)
    # RL must not train on prompts SFT already fit. See the block below for the
    # measurement that motivated this; `--no-exclude-sft` restores the old
    # (broken) behaviour only for reproducing historical runs.
    ap.add_argument("--no-exclude-sft", dest="exclude_sft", action="store_false",
                    help="allow RL train prompts that SFT already trained on (NOT recommended)")
    # The SFT set to exclude is NOT always data/sft: a second model scale gets
    # its own split (e.g. data_3b/sft), and excluding the wrong one silently
    # reintroduces the train/test contamination this flag exists to prevent.
    ap.add_argument("--sft-parquet", default="",
                    help="SFT train parquet whose prompts to exclude "
                         "(default: <out-dir>/sft/train.parquet, else data/sft/train.parquet)")
    args = ap.parse_args()

    import os
    from pathlib import Path

    default_hf_home = Path(__file__).resolve().parent.parent / "models" / ".hf_cache"
    os.environ.setdefault("HF_HOME", str(default_hf_home))

    from datasets import load_dataset

    ds = load_dataset("internlm/Lean-Workbook", split="train")
    recs = [r for r in ds if r.get("natural_language_statement") and r.get("formal_statement")]
    print(f"loaded {len(recs)} usable records from internlm/Lean-Workbook")

    random.seed(args.seed)
    random.shuffle(recs)

    # The validation slice is pinned to a FIXED OFFSET so that growing the
    # training set never changes which examples are evaluated. Earlier this was
    # `val = recs[n_train : n_train + n_val]`, which silently swapped the whole
    # eval set the moment n_train changed -- every cached result would have been
    # measured on different problems while looking comparable.
    #
    # With the offset fixed, enlarging --n-val only APPENDS: the first 80 val
    # rows are the same 80 used by every result already in results/, so old
    # per-example records stay valid and comparable.
    VAL_OFFSET = 400
    val_recs = recs[VAL_OFFSET : VAL_OFFSET + args.n_val]
    pool = recs[:VAL_OFFSET] + recs[VAL_OFFSET + args.n_val :]

    # Excluding the val SLICE is not enough. Lean-Workbook contains the same
    # natural_language_statement at multiple indices (400 val rows correspond to
    # 1550 records repo-wide), so slicing by position leaves duplicates of the
    # eval prompts sitting in the training pool. Measured at n_train=400: 20 of
    # the 400 val prompts leaked in; at n_train=4000 that grows toward half the
    # eval set. Filter on prompt CONTENT, which is what the model actually sees.
    val_prompts = {INSTRUCTION.format(informal=r["natural_language_statement"]) for r in val_recs}
    before = len(pool)
    pool = [r for r in pool
            if INSTRUCTION.format(informal=r["natural_language_statement"]) not in val_prompts]
    dropped_val_dupes = before - len(pool)

    # EXCLUDE PROMPTS SFT ALREADY TRAINED ON.
    #
    # This was the single biggest defect in the RL experiments. 93.4% of the RL
    # train prompts had also been in the SFT train set, so GRPO was being asked
    # to improve a policy on problems it had already memorised. Measured on
    # sft-step390 (results/gradient_signal_probe.json, 48 prompts x 8 rollouts):
    #
    #   full-BEq+ rate on RL train prompts  73.7%
    #   full-BEq+ rate on held-out val      38.8%     <- 34.9pp generalisation gap
    #   groups where all 8 rollouts already correct   62.5%
    #   groups able to produce ANY gradient           16.7%
    #
    # GRPO's advantage is r_i - mean(r_group), so a group whose rollouts all
    # score the same teaches nothing. Training on memorised prompts makes 5 of
    # every 6 groups dead by construction, which is why every RL arm drifted
    # instead of learning. Filtering here is what makes the reward comparison a
    # test of the REWARD rather than a test of how much SFT memorised.
    if args.exclude_sft:
        if args.sft_parquet:
            sft_path = Path(args.sft_parquet)
        else:
            cand = Path(args.out_dir) / "sft" / "train.parquet"
            sft_path = cand if cand.exists() else (
                Path(__file__).resolve().parent.parent / "data" / "sft" / "train.parquet")
        if sft_path.exists():
            import pandas as _pd

            sft_prompts = {m[0]["content"] for m in _pd.read_parquet(sft_path)["messages"]}
            before_sft = len(pool)
            pool = [r for r in pool
                    if INSTRUCTION.format(informal=r["natural_language_statement"]) not in sft_prompts]
            print(f"  excluded {before_sft - len(pool)} records whose prompt was in SFT training")
        else:
            print(f"  WARNING: {sft_path} not found -- cannot exclude SFT prompts from RL training.")

    train_recs = pool[: args.n_train]
    if len(train_recs) < args.n_train:
        # Lean-Workbook has 25,214 rows but only 13,297 UNIQUE prompts, and a
        # 20k-row SFT set consumes 11,627 of them. Asking for more RL prompts
        # than remain is silently satisfied with fewer, which would make a
        # "4000-prompt" run secretly reuse a much smaller set across epochs.
        print(f"  WARNING: only {len(train_recs)} records available, {args.n_train} requested. "
              f"Shrink the SFT set (scripts/data/prepare_sft_dataset.py --n-train) to free more.")
    print(f"PoC subset: {len(train_recs)} train, {len(val_recs)} val (of {len(recs)} available)")
    print(f"  val pinned at offset {VAL_OFFSET}; "
          f"dropped {dropped_val_dupes} duplicate-prompt records from the train pool")

    train_rows = [to_verl_row(r, "train", i) for i, r in enumerate(train_recs)]
    val_rows = [to_verl_row(r, "val", i) for i, r in enumerate(val_recs)]

    import pandas as pd
    from pathlib import Path

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(out_dir / "train.parquet")
    pd.DataFrame(val_rows).to_parquet(out_dir / "val.parquet")
    print(f"wrote {out_dir / 'train.parquet'} and {out_dir / 'val.parquet'}")


if __name__ == "__main__":
    main()
