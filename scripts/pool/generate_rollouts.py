#!/usr/bin/env python3
"""Sample k rollouts per prompt from a checkpoint and dump them unscored.

This is the generation half of the rejection-sampling (RFT / STaR) arm, and it
is deliberately separated from scoring because the two have opposite resource
profiles: generation is GPU-bound and takes minutes, BEq+ scoring is CPU-bound
and takes hours. Splitting them lets scoring run in a parallel process pool on
the CPU while the GPU is free for something else, and makes the expensive half
resumable.

Defaults MATCH the GRPO rollout distribution exactly (k=8, temperature=1.15,
128 new tokens, same train.parquet, same chat template), so the resulting
rollouts are the same population GRPO trains on. That is what makes
"GRPO vs rejection sampling on identical rollouts + identical BEq+ verdicts" a
controlled comparison of the OPTIMIZER rather than of the data.

Usage:
    python scripts/pool/generate_rollouts.py \
        --checkpoint checkpoints/merged/sft-step390 \
        --parquet data/train.parquet --k 8 --out data/rollouts/sft390_k8.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: F401  -- repo root + every stage folder


def _user_turn(row) -> str:
    """The LAST user turn of a chat-format prompt row.

    NOT row[0]: a parquet carrying a system turn puts the system text at index 0
    and every prompt collapses to the same string. That produced a LoCoLib eval
    scoring one identical theorem for all 999 rows.
    """
    turns = [m for m in row if m.get("role") == "user"]
    if not turns:
        raise ValueError(f"no user turn in prompt row: {row!r}")
    return turns[-1]["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="merged HF model dir")
    ap.add_argument("--parquet", default=str(PROJECT_ROOT / "data" / "train.parquet"))
    ap.add_argument("--k", type=int, default=8, help="rollouts per prompt (GRPO rollout_n)")
    ap.add_argument("--temperature", type=float, default=1.15, help="GRPO rollout temperature")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--gpu-frac", type=float, default=0.60)
    ap.add_argument("--limit", type=int, default=0, help="cap unique prompts (0 = all)")
    ap.add_argument("--max-prompt-length", type=int, default=768,
                    help="drop longer prompts, mirroring data.filter_overlong_prompts "
                         "+ data.max_prompt_length in configs/run_grpo.sh")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_parquet(args.parquet)
    # The RL pool has duplicate prompts (1280 rows / 1196 unique); rejection
    # sampling gains nothing from scoring the same prompt twice, and duplicates
    # would silently upweight those prompts in the fine-tuning set.
    seen: dict[str, str] = {}
    for prompt_col, rm in zip(df["prompt"], df["reward_model"]):
        text = _user_turn(prompt_col)
        if text not in seen:
            seen[text] = rm["ground_truth"]
    items = list(seen.items())
    if args.limit:
        items = items[: args.limit]
    print(f"[gen] {len(df)} rows -> {len(items)} unique prompts from {args.parquet}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    # MUST match training-time formatting, or every rollout is off-distribution
    # (see the chat-template note in scripts/eval/evaluate_checkpoints.py).
    texts = [
        tok.apply_chat_template([{"role": "user", "content": p}],
                                tokenize=False, add_generation_prompt=True)
        for p, _ in items
    ]
    # Training drops overlong prompts rather than truncating them
    # (filter_overlong_prompts=True, truncation='error'), so rollouts must drop
    # exactly the same ones -- otherwise this population is not the one GRPO
    # trains on and the comparison is no longer controlled. vLLM would refuse
    # them anyway: max_model_len is prompt+response.
    keep = [i for i, t in enumerate(texts)
            if len(tok(t)["input_ids"]) <= args.max_prompt_length]
    if len(keep) != len(items):
        print(f"[gen] dropped {len(items)-len(keep)} prompts over "
              f"{args.max_prompt_length} tokens (matches training's filter)")
    items = [items[i] for i in keep]
    texts = [texts[i] for i in keep]

    llm = LLM(model=args.checkpoint, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_frac, max_model_len=896, enforce_eager=True,
              seed=args.seed)
    sp = SamplingParams(n=args.k, temperature=args.temperature,
                        max_tokens=args.max_new_tokens, seed=args.seed)
    print(f"[gen] sampling k={args.k} T={args.temperature} ...")
    outs = llm.generate(texts, sp)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as fh:
        for pi, (out, (prompt, gold)) in enumerate(zip(outs, items)):
            for si, cand in enumerate(out.outputs):
                fh.write(json.dumps({
                    "prompt_index": pi, "sample_index": si,
                    "prompt": prompt, "gold": gold, "completion": cand.text,
                }, ensure_ascii=False) + "\n")
                n += 1
    print(f"[gen] wrote {n} rollouts ({len(items)} prompts x {args.k}) -> {out_path}")


if __name__ == "__main__":
    main()
