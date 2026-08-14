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
    n_train = min(args.n_train, max(0, len(recs) - args.n_val))
    train_recs = recs[:n_train]
    val_recs = recs[n_train : n_train + args.n_val]
    print(f"PoC subset: {len(train_recs)} train, {len(val_recs)} val (of {len(recs)} available)")

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
