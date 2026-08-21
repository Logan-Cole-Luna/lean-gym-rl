#!/usr/bin/env python3
"""Measure how much TRAINING SIGNAL each reward function actually delivers.

Motivation
----------
Eval tells you a checkpoint's BEq+ rate. It does not tell you whether the reward
that produced it was ever able to teach anything. Those are different questions,
and the 100-step gated run made the gap obvious: BEq+ fired constantly (mean
train reward 0.573 out of 1.0) yet the policy still drifted from 38.8% to 31.0%
BEq+ while its own training reward did NOT rise (Pearson r = -0.33 vs step).

The reason is GRPO's advantage:

    A_i = r_i - mean(r_group)          (norm_adv_by_std_in_grpo=False)

A group whose rollouts all score the SAME contributes A_i = 0 for every member
-- no gradient at all, regardless of how high or low that shared score is. So
what matters for learning is not the reward's MEAN but its WITHIN-GROUP SPREAD.
A reward can fire on most rollouts and still teach nothing.

This script measures spread directly, and does so as a PAIRED comparison: every
rollout is scored by Lean exactly once, and all reward variants are then derived
from that same scored result. Any difference between rewards is therefore
attributable to the reward's shape alone, not to sampling noise. The expensive
part (typecheck + the bidirectional BEq+ tactic cascade) is paid once per
rollout rather than once per reward.

Reported per reward function:
  * mean reward
  * % of groups that are DEGENERATE (zero spread -> zero gradient), split into
      - starved   : every rollout scored 0 (nothing to reinforce)
      - saturated : every rollout scored the max (nothing left to learn)
  * mean within-group std, which is what the gradient magnitude scales with

Usage:
    python scripts/probe_gradient_signal.py --checkpoint checkpoints/merged/sft-step390
    python scripts/probe_gradient_signal.py --n-prompts 64 --group-size 8
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reward variants to compare. Names must exist in reward/reward_fn.py.
REWARDS = [
    ("typecheck_only", "compute_score_typecheck_only"),  # the "pass@compile" baseline
    ("gated", "compute_score_gated"),                    # semantic-only (BEq+)
    ("shaped", "compute_score_shaped"),
    ("guided", "compute_score_guided"),
    ("composite", "compute_score_composite"),
]


def rollout(model_dir: str, prompts: list[str], group_size: int,
            max_new_tokens: int, temperature: float, gpu_frac: float) -> list[list[str]]:
    """Sample `group_size` completions per prompt, exactly as GRPO rollouts do.

    Temperature and the chat template must match training or the probe measures
    a distribution the trainer never sees. verl's rollout uses the chat template
    and temperature 1.0 (actor_rollout_ref.rollout.temperature), NOT greedy --
    greedy would collapse every group to identical samples and trivially report
    zero spread for every reward.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.chat_template:
        prompts = [
            tok.apply_chat_template([{"role": "user", "content": p}],
                                    tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
    llm = LLM(model=model_dir, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=gpu_frac, max_model_len=896, enforce_eager=True)
    out = llm.generate(prompts, SamplingParams(n=group_size, max_tokens=max_new_tokens,
                                               temperature=temperature, top_p=1.0))
    return [[c.text for c in o.outputs] for o in out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "merged" / "sft-step390"),
                    help="policy to probe -- default is the checkpoint RL actually starts from")
    ap.add_argument("--train-parquet", default=str(PROJECT_ROOT / "data" / "train.parquet"),
                    help="probe the TRAIN distribution: that is what produces gradients")
    ap.add_argument("--n-prompts", type=int, default=48)
    ap.add_argument("--group-size", type=int, default=8, help="must match rollout.n")
    ap.add_argument("--temperature", type=float, default=1.0, help="must match rollout.temperature")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--gpu-frac", type=float, default=0.35)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "gradient_signal_probe.json"))
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_parquet(args.train_parquet).head(args.n_prompts)
    prompts = [row[0]["content"] for row in df["prompt"]]
    golds = [rm["ground_truth"] for rm in df["reward_model"]]
    print(f"[probe] {len(prompts)} prompts x {args.group_size} rollouts "
          f"@ T={args.temperature} from {args.checkpoint}")

    groups = rollout(args.checkpoint, prompts, args.group_size,
                     args.max_new_tokens, args.temperature, args.gpu_frac)

    # Score once, reuse everywhere. Memoising reward_fn._score_pair lets each
    # reward function be called UNMODIFIED -- so the weights and thresholds under
    # test are literally the ones training uses, with no risk of this script
    # drifting out of sync with reward_fn.py.
    import reward.reward_fn as R

    _orig, _cache = R._score_pair, {}

    def _memo(solution_str, ground_truth):
        key = (solution_str, ground_truth)
        if key not in _cache:
            _cache[key] = _orig(solution_str, ground_truth)
        return _cache[key]

    R._score_pair = _memo

    fns = [(name, getattr(R, attr)) for name, attr in REWARDS]
    per_group: list[dict] = []
    n_calls = 0
    for gi, (comps, gold) in enumerate(zip(groups, golds)):
        rec = {"prompt_index": gi, "rewards": {}}
        for name, fn in fns:
            rec["rewards"][name] = [float(fn("lean_workbook", c, gold)["score"]) for c in comps]
        _cache.clear()  # per-prompt cache; different prompts never share a key
        n_calls += len(comps)
        per_group.append(rec)
        print(f"  [{gi+1}/{len(groups)}] scored {n_calls} rollouts", flush=True)

    print("\n" + "=" * 92)
    print(f"{'reward':<16}{'mean':>8}{'degenerate':>12}{'starved':>10}{'saturated':>11}"
          f"{'mean std':>10}{'std|live':>10}")
    print("-" * 92)
    summary = {}
    for name, _ in fns:
        allr = [g["rewards"][name] for g in per_group]
        hi = max(max(r) for r in allr) or 1.0
        stds = [statistics.pstdev(r) for r in allr]
        degen = [s < 1e-12 for s in stds]
        starved = sum(1 for r, d in zip(allr, degen) if d and max(r) < 1e-12)
        satur = sum(1 for r, d in zip(allr, degen) if d and min(r) >= hi - 1e-12)
        live = [s for s in stds if s >= 1e-12]
        n = len(allr)
        summary[name] = {
            "mean_reward": statistics.mean(x for r in allr for x in r),
            "degenerate_frac": sum(degen) / n,
            "starved": starved, "saturated": satur,
            "mean_within_group_std": statistics.mean(stds),
            "mean_std_live_groups": statistics.mean(live) if live else 0.0,
            "n_groups": n,
        }
        s = summary[name]
        print(f"{name:<16}{s['mean_reward']:>8.3f}{100*s['degenerate_frac']:>11.1f}%"
              f"{starved:>10}{satur:>11}{s['mean_within_group_std']:>10.4f}"
              f"{s['mean_std_live_groups']:>10.4f}")
    print("=" * 92)
    print("degenerate = every rollout in the group scored identically -> zero advantage,")
    print("             zero gradient. 'mean std' is what gradient magnitude scales with.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"config": vars(args), "summary": summary, "per_group": per_group}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
