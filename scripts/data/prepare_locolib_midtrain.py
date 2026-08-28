#!/usr/bin/env python3
"""Build a mid-training corpus from LoCoLib.

WHY RETRY MID-TRAINING. The first attempt used raw Mathlib declarations and did
nothing, for a measurable reason: only ~2.3% of Mathlib declarations elaborate
standalone, which left roughly 75k usable tokens. Continued pre-training on 75k
tokens cannot move a 3B model. LoCoLib removes that specific blocker, because
every statement ships the context it needs (imports, `open`, `variable`,
`namespace`).

WHAT GOES IN, AND WHAT IS HELD OUT. Only the rows in the LoCoLib **SFT split**.
The RL and val splits are excluded so that:

  val   never seen at any stage, so it stays a real held-out measurement
  RL    never seen before RL, so GRPO is not re-scoring memorised statements

That matters more here than it did for the Mathlib corpus, which was naturally
disjoint from Lean-Workbook. Here every stage draws on one corpus, so the
holdout has to be enforced rather than assumed.

FORMAT matches `prepare_midtrain_dataset.py`: an empty system turn plus a single
assistant turn holding the text. verl's MultiTurnSFTDataset sets loss_mask=1
across an assistant turn and masks only the generation-prompt prefix, so every
content token is trained and there is no prompt to condition on. That is plain
next-token LM, reusing the SFT stack. The empty system turn is not decorative:
verl's mask only covers `<|im_start|>system\\n`, so without it the model trains
on Qwen's injected default system prompt.

    python scripts/data/prepare_locolib_midtrain.py

By default the text is the statement WITH its context, which is the whole
difference from the corpus that failed. `--include-proofs` adds the proof body
as well (98.9% of rows have one), roughly doubling the token count. It is OFF by
default: the recorded failure was a statements corpus, so keeping this one to
statements isolates corpus quality as the variable rather than changing two
things at once.
"""
import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: F401  -- repo root + every stage folder

from prepare_locolib import (DEFAULT_SRC, load, norm, split_context_and_theorem,
                             strip_proof)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data_locolib",
                    help="where prepare_locolib.py wrote sft/rl/val, used to enforce the holdout")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data_locolib" / "midtrain")
    ap.add_argument("--include-proofs", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="must match the trainer's max_length; verl raises on longer")
    ap.add_argument("--tokenizer", default="/scratch/logan03/ai4math_training_models/qwen2.5-coder-3b-instruct")
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer

    rows, _ = load(args.src)

    # Enforce the holdout by CONTENT, not uuid. The splits were built on
    # (informal, signature) keys, so reuse exactly that key or the holdout leaks.
    held = set()
    for name in ("rl", "val"):
        df = pd.read_parquet(args.splits_dir / f"{name}.parquet")
        for rm in df["reward_model"]:
            parts = split_context_and_theorem(rm["ground_truth"])
            if parts:
                held.add(norm(strip_proof(parts[1])))
    print(f"[midtrain] holding out {len(held)} statements from the rl/val splits")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    kept, skipped_held, skipped_long = [], 0, 0
    for r in rows:
        if norm(r["signature"]) in held:
            skipped_held += 1
            continue
        text = r["gold_full"].strip() if args.include_proofs else (
            (r["context"] + "\n\n" + r["signature"]).strip() if r["context"] else r["signature"])
        msgs = [{"role": "system", "content": ""},
                {"role": "assistant", "content": text}]
        if len(tok(tok.apply_chat_template(msgs, tokenize=False)).input_ids) > args.max_tokens:
            skipped_long += 1
            continue
        kept.append(msgs)

    print(f"[midtrain] {len(rows)} labelled rows")
    print(f"[midtrain]   held out (rl/val)     : {skipped_held}")
    print(f"[midtrain]   over {args.max_tokens} tokens        : {skipped_long}")
    print(f"[midtrain]   kept                  : {len(kept)}")
    if not kept:
        raise SystemExit("[midtrain] nothing left; check the holdout or raise --max-tokens")

    n_tok = sum(len(tok(m[1]["content"]).input_ids) for m in kept)
    print(f"[midtrain] ~{n_tok:,} content tokens "
          f"(the Mathlib attempt that did nothing had ~75,000)")

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    val, train = kept[:args.n_val], kept[args.n_val:]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"messages": train}).to_parquet(args.out_dir / "train.parquet")
    pd.DataFrame({"messages": val}).to_parquet(args.out_dir / "val.parquet")
    with open(args.out_dir / "sample.jsonl", "w") as fh:
        for m in train[:200]:
            fh.write(json.dumps({"statement": m[1]["content"]}) + "\n")
    print(f"[midtrain] train {len(train)} / val {len(val)} -> {args.out_dir}")


if __name__ == "__main__":
    main()
