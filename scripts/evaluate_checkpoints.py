#!/usr/bin/env python3
"""Score trained checkpoints (and the untrained base model) on the SAME metrics,
so the two ablation arms are actually comparable.

Why this exists: verl's `val-core/<data_source>/acc/mean@1` is just the mean of
whatever `custom_reward_function` that run used. The composite arm's validation
number is therefore `0.1*typecheck + 0.9*beq_plus`, while the typecheck-only
arm's is the raw type-check rate -- two different scales. Comparing them
directly would be meaningless, and the composite number alone cannot be
decomposed (a score of 0.0825 is consistent with anything from
"82.5% type-check, 0% BEq+" to "8.25% both").

This script reports BOTH metrics separately for every checkpoint, which is the
comparison the PoC is actually after:

    checkpoint                 typecheck%   beq_plus%
    base (untrained)              ...         ...
    typecheck_only @ step 30      ...         ...
    composite      @ step 30      ...         ...

The paper's hypothesis predicts the typecheck-only arm drives type-check rate up
while BEq+ stays flat (reward hacking: valid Lean that means the wrong thing),
whereas the composite arm moves BEq+ too.

Usage:
    python scripts/evaluate_checkpoints.py \
        --checkpoint base \
        --checkpoint checkpoints/beqplus_rl_poc/qwen25_coder_0_5b_compute_score_composite/global_step_30 \
        --n-eval 80
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MODEL = PROJECT_ROOT / "models" / "qwen2.5-coder-0.5b-instruct"


def resolve_model_path(spec: str) -> tuple[str, str]:
    """Return (label, hf_model_dir) for a checkpoint spec.

    Accepts:
      'base'                      -> the untrained base model
      a merged HF dir             -> used as-is (what `make merge-checkpoints` produces)
      a raw verl checkpoint dir   -> falls back to its actor/huggingface/ subdir
    """
    if spec == "base":
        return "base (untrained)", str(DEFAULT_BASE_MODEL)

    path = Path(spec)
    if not path.exists():
        raise SystemExit(f"No such checkpoint dir: {path}")

    # A merged/plain HF directory already has weights next to its config.
    if has_full_weights(path):
        return path.name, str(path)

    # Otherwise assume verl's checkpoint layout.
    hf_dir = path / "actor" / "huggingface"
    if hf_dir.exists():
        return f"{path.parent.name}@{path.name}", str(hf_dir)

    raise SystemExit(
        f"{path} has neither HF weights nor actor/huggingface/ -- "
        f"run `make merge-checkpoints` first."
    )


def has_full_weights(hf_dir: Path) -> bool:
    """verl saves config/tokenizer into actor/huggingface/ but may keep the
    weights themselves in the sharded model_world_size_*.pt files alongside."""
    return any(hf_dir.glob("*.safetensors")) or any(hf_dir.glob("pytorch_model*.bin"))


def generate(model_dir: str, prompts: list[str], max_new_tokens: int, gpu_frac: float) -> list[str]:
    """Generate completions, applying the model's chat template.

    This MUST match how the prompt is formatted during training, or the model is
    evaluated off-distribution. Both verl's RL rollouts and the SFT trainer apply
    the chat template, so evaluating on raw prompt text silently degrades every
    checkpoint. Observed concretely on the SFT model: with the raw prompt it
    emitted `lean_workbook_plus_11512 : Nat.choose 13 2 = 78` -- no `theorem`
    keyword, so nothing parses and the score is 0/80; with the template applied
    the same model emitted the correct `theorem lean_workbook_plus_11512 : ...`.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.chat_template:
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]

    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_frac,
        max_model_len=896,
        enforce_eager=True,
    )
    # Greedy: we're measuring the policy's mode, not sampling diversity.
    out = llm.generate(prompts, SamplingParams(max_tokens=max_new_tokens, temperature=0.0))
    return [o.outputs[0].text for o in out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", dest="checkpoints", default=[],
                    help="'base' or a verl checkpoint dir (repeatable)")
    ap.add_argument("--val-parquet", default=str(PROJECT_ROOT / "data" / "val.parquet"))
    ap.add_argument("--n-eval", type=int, default=80, help="number of val examples to score")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--gpu-frac", type=float, default=0.35)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "ablation_comparison.json"))
    args = ap.parse_args()

    if not args.checkpoints:
        raise SystemExit("Pass at least one --checkpoint (use 'base' for the untrained model)")

    import pandas as pd

    df = pd.read_parquet(args.val_parquet).head(args.n_eval)
    prompts = [row[0]["content"] for row in df["prompt"]]
    golds = [rm["ground_truth"] for rm in df["reward_model"]]
    print(f"[eval] {len(prompts)} validation examples from {args.val_parquet}")

    # Import the scorer AFTER vllm work, so the Lean REPL isn't held open during
    # generation. (It's CPU-only, but Mathlib's env is memory-hungry.)
    results = {}
    for spec in args.checkpoints:
        label, model_dir = resolve_model_path(spec)
        hf_dir = Path(model_dir)
        if spec != "base" and not has_full_weights(hf_dir):
            print(f"[eval] SKIP {label}: {hf_dir} has no .safetensors/.bin weights "
                  f"(verl kept them sharded; run verl's model merger first, e.g. "
                  f"`python -m verl.model_merger merge --backend fsdp "
                  f"--local_dir {hf_dir.parent} --target_dir <out>`)")
            continue

        print(f"\n[eval] ===== {label} =====")
        print(f"[eval] generating from {model_dir} ...")
        completions = generate(model_dir, prompts, args.max_new_tokens, args.gpu_frac)

        from reward.beq_plus import BEqPlusScorer
        from reward.reward_fn import _clean_solution

        scorer = BEqPlusScorer()
        n_tc = n_beq = 0
        per_example = []
        for i, (comp, gold) in enumerate(zip(completions, golds)):
            pred = _clean_solution(comp)
            r = scorer.score(gold, pred)
            n_tc += bool(r["typecheck"])
            n_beq += bool(r["beq_plus"])
            per_example.append({"i": i, "typecheck": bool(r["typecheck"]),
                                "beq_plus": bool(r["beq_plus"]), "error": r["error"]})
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(completions)}] typecheck={n_tc} beq+={n_beq}", flush=True)

        n = len(completions)
        results[label] = {
            "model_dir": model_dir,
            "n": n,
            "typecheck_rate": n_tc / n if n else 0.0,
            "beq_plus_rate": n_beq / n if n else 0.0,
            "per_example": per_example,
        }
        print(f"[eval] {label}: typecheck {n_tc}/{n} ({100*n_tc/n:.1f}%)  "
              f"beq_plus {n_beq}/{n} ({100*n_beq/n:.1f}%)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n=============== COMPARISON ===============")
    print(f"{'checkpoint':<45} {'typecheck%':>11} {'beq_plus%':>10}")
    for label, r in results.items():
        print(f"{label:<45} {100*r['typecheck_rate']:>10.1f}% {100*r['beq_plus_rate']:>9.1f}%")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
