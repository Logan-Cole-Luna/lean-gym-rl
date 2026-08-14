# ai4math_training — BEq+ as an RL Reward for Lean 4 Autoformalization

A proof-of-concept measuring the actual training-time impact of using **BEq+**
(semantic/symbolic statement-equivalence) as an RL reward signal for autoformalization,
per *"Online Reinforcement Learning for Autoformalization"*
([researchgate.net/publication/396825045](https://www.researchgate.net/publication/396825045)).
That paper's core claim: a type-check-only reward is exploitable — the policy learns to
produce syntactically valid but semantically vacuous Lean statements. Composing it with
a BEq+ semantic-equivalence reward against ground truth is meant to close that gap. This
project builds the smallest real pipeline that can measure it: GRPO training with two
swappable reward functions (type-check-only vs. type-check + BEq+), on real data, with a
real Lean toolchain.

## What's here

| Piece | Choice | Why |
|---|---|---|
| RL framework | [verl](https://github.com/volcengine/verl) (`repos/verl`) | Widely-used open RL post-training library for LLMs; native GRPO; pluggable `custom_reward_function`. |
| Algorithm | GRPO, full-parameter (no LoRA) | See *Hardware notes* below for why LoRA was dropped. |
| Base model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | Small enough to actually train on a single 16GB GPU; code-pretrained, so not starting from zero on Lean syntax. |
| Dataset | [`internlm/Lean-Workbook`](https://huggingface.co/datasets/internlm/Lean-Workbook) | 57k well-known, widely-cited NL↔Lean4 pairs (NeurIPS'24), built specifically for autoformalization training. |
| Lean / Mathlib | Mathlib4 @ `v4.8.0-rc1` (`repos/mathlib4`) | Pinned to match Lean-Workbook's target toolchain — built fresh for this project, not shared with other Lean projects on this machine. |
| Semantic reward | **BEq+** (`reward/beq_plus.py`) | Deterministic, CPU-only, LLM-free symbolic equivalence metric (Poiroux et al., EMNLP'25). Algorithm vendored from `augustepoiroux/RLMEval`; wrapped in a persistent Lean REPL server for online per-rollout scoring. |

## Setup

Requires [`uv`](https://astral.sh/uv), a CUDA-capable GPU, and network access for the
one-time downloads.

```bash
make setup          # env + verl + Mathlib4 + model + dataset (~20-30 min, mostly Mathlib)
source .venv/bin/activate
make smoke           # ~1-2 min sanity check: one GRPO step on 8 examples
```

`make help` lists every target. Each setup step is also independently re-runnable
(`make env`, `make verl`, `make mathlib`, `make model`, `make dataset`) if you only need
to redo one piece.

## Running training

```bash
make train-composite    # GRPO with type-check + BEq+ composite reward
make train-typecheck    # GRPO with type-check-only reward (the exploitable baseline)
make train               # both, sequentially — this is a single-GPU PoC, they can't overlap
```

Both read `configs/run_grpo.sh`, which wires the reward function in via
`reward.custom_reward_function.{path,name}` and reads
`REWARD_FN_NAME=compute_score_composite|compute_score_typecheck_only` from the
environment. Every other knob (batch size, rollout group size, LR, step count) is a
shell variable at the top of that script, overridable via env var
(`TRAIN_BATCH_SIZE=32 make train-composite`, etc.) or extra Hydra overrides passed
straight through (`bash configs/run_grpo.sh trainer.total_epochs=5`).

The comparison this whole PoC exists to produce: run both arms for the same number of
steps, and compare `critic/rewards/mean`, `val-core/lean_workbook/acc/mean@1`
(BEq+ pass rate), and the actual generated statements — the type-check-only arm should
show a widening gap between "type-checks" and "means the same thing", while the
composite arm should not.

## Repo layout

```
configs/run_grpo.sh     GRPO launch script (both ablation arms select via REWARD_FN_NAME)
reward/beq_plus.py       BEq+ metric — persistent Lean REPL wrapper (vendored algorithm)
reward/reward_fn.py       verl-facing reward functions (composite / typecheck-only)
scripts/prepare_dataset.py   Lean-Workbook → verl parquet format
scripts/test_lean_interact.py   standalone Lean/Mathlib sanity check
repos/verl/               cloned RL framework (editable install)
repos/mathlib4/           Mathlib4 @ v4.8.0-rc1 (built via `lake exe cache get`)
models/                   downloaded base model + HF cache
data/                     prepared train/val parquet (+ data/smoke/ for the smoke test)
hpc/                      SLURM transfer notes + job template (see hpc/README.md)
```

## Hardware notes (read before changing the model)

This runs on a single RTX 5070 Ti (16GB VRAM). Getting a real GRPO step to fit took
some non-obvious changes from a "normal" verl config — worth knowing before swapping in
a bigger model:

- **`internlm2-math-plus-1.8B` does not fit.** It was the original model choice (same
  family as the paper's autoformalizer, explicitly pretrained on Lean 4 translation),
  but verl's colocated FSDP-actor + vLLM-rollout hybrid engine needs ~100-300MB more
  than this card has during the very first actor→vLLM weight-sync handshake —
  independent of every training-loop tuning knob (batch size, token budgets, LoRA rank,
  KL loss on/off), because none of those apply before that point. The fix that actually
  worked was a smaller model (`Qwen2.5-Coder-0.5B-Instruct`), not more tuning.
- **LoRA doesn't work with this model/verl/vLLM combination.** vLLM's `set_lora` throws
  `IndexError: tuple index out of range` on this model's packed QKV/gate-up layers when
  receiving verl's LoRA weight-transfer buckets — an integration bug, not a memory
  issue. Training here is full-parameter (`lora_rank=0` in `configs/run_grpo.sh`);
  revisit LoRA if that upstream bug gets fixed, or if you move to a bigger GPU where
  full-parameter is the more expensive option instead of the workaround.
- **`flash-attn` isn't installed.** Qwen2.5's default config wants
  `flash_attention_2`; `configs/run_grpo.sh` forces SDPA attention instead
  (`+actor_rollout_ref.model.override_config.attn_implementation=sdpa`,
  `use_remove_padding=False`) rather than building flash-attn from source.
- **`gpu_memory_utilization=0.3`, `max_model_len` capped to prompt+response length.**
  Both matter — vLLM's default `max_model_len` (the model's full context window) makes
  it try to reserve far more KV cache than a tiny-batch smoke test needs.
- A **stray dependency of vLLM (`flashinfer`) ships a broken type annotation**
  (`array.array[int]`, not subscriptable pre-Python-3.12) that crashes engine init the
  moment anything touches its AllReduce fusion pass. `make env` patches it in place
  (`make patch-flashinfer`); this is a venv-local patch, not a real fix, and will need
  reapplying if the venv is rebuilt from scratch with a newer flashinfer release that
  may have already fixed it upstream.

None of this is fundamental — a 24GB+ GPU clears the memory ceiling with room to spare,
at which point re-enabling LoRA and/or a bigger model is a config change, not a rebuild.
See `hpc/README.md` for moving this to a cluster.
