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