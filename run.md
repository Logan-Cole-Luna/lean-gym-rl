# Quick Start: Replicate Current Running Jobs

This documents the commands to replicate the current **proof-pair** RL setup: three reward arms (typecheck, gated, outcome) trained on LoCoLib with BEq+ signal, evaluated at Lean 4.23.0.

## Prerequisites

```bash
# One-time setup
make setup                  # env + verl + Lean/Mathlib + base model

# Verify Lean/Mathlib toolchain
make check-toolchain
```

## Dataset

The LoCoLib proof-pair dataset is pre-prepared in `data_locolib/`:

- `sft_proof.parquet` — SFT training data (proof-pair format)
- `val_sft_proof.parquet` — SFT validation
- `rl_proof.parquet` — RL training pool
- `val_proof.parquet` — RL/eval validation (760 rows, pinned)
- `best_sft_proof.txt` — best SFT checkpoint pointer

No data preparation is needed; the data is already in the repo.

## SFT Baseline

```bash
# Train SFT from scratch (one GPU, ~2h)
source hpc/cc_env.sh
make train-sft

# Merge the checkpoint for vLLM
make merge-sft
```

The merged SFT weights land in `checkpoints/merged/sft-step<N>` and the baseline is pinned in `data_locolib/best_sft_proof.txt`.

## RL Training (Three Arms)

Submit all three arms on SLURM (Narval/Compute Canada):

```bash
# Required Mathlib v4.23 override (not v4.8.0-rc1)
MATHLIB_TAR=/scratch/$USER/mathlib4_v4.23_lake.tar

# Typecheck arm
sbatch --export=ALL,ARM=typecheck,SERIES_TAG=locolib_proof_lr6,\
TRAIN_FILE=data_locolib/rl_proof.parquet,VAL_FILE=data_locolib/val_proof.parquet,\
BEST_SFT=data_locolib/best_sft_proof.txt,\
MATHLIB_TAR=$MATHLIB_TAR,MATHLIB_TAR_FLAT=1,\
LEAN_INTERACT_CACHE_DIR=/scratch/$USER/ai4math_training_lean_interact_cache_v423 \
hpc/grpo.slurm

# Gated arm
sbatch --export=ALL,ARM=gated,SERIES_TAG=locolib_proof_lr6,\
TRAIN_FILE=data_locolib/rl_proof.parquet,VAL_FILE=data_locolib/val_proof.parquet,\
BEST_SFT=data_locolib/best_sft_proof.txt,\
MATHLIB_TAR=$MATHLIB_TAR,MATHLIB_TAR_FLAT=1,\
LEAN_INTERACT_CACHE_DIR=/scratch/$USER/ai4math_training_lean_interact_cache_v423 \
hpc/grpo.slurm

# Outcome arm
sbatch --export=ALL,ARM=outcome,SERIES_TAG=locolib_proof_lr6,\
TRAIN_FILE=data_locolib/rl_proof.parquet,VAL_FILE=data_locolib/val_proof.parquet,\
BEST_SFT=data_locolib/best_sft_proof.txt,\
MATHLIB_TAR=$MATHLIB_TAR,MATHLIB_TAR_FLAT=1,\
LEAN_INTERACT_CACHE_DIR=/scratch/$USER/ai4math_training_lean_interact_cache_v423 \
hpc/grpo.slurm
```

**Or use `hpc/submit.sh` for cleaner naming:**

```bash
bash hpc/submit.sh grpo ARM=typecheck SERIES_TAG=locolib_proof_lr6
bash hpc/submit.sh grpo ARM=gated SERIES_TAG=locolib_proof_lr6
bash hpc/submit.sh grpo ARM=outcome SERIES_TAG=locolib_proof_lr6
```

Default knobs (set in `hpc/job_prelude.sh`, override via `--export`):
- `ACTOR_LR=1e-6` (corrected from 1e-5 that regressed BEq+ performance)
- `TRAIN_BATCH_SIZE=16`
- `ROLLOUT_N=24`
- `TOTAL_STEPS=90` (default; override with `TOTAL_STEPS=...`)

Training logs appear in `results/train/train_metrics/` as each job runs.

## Evaluation

After training completes (each arm takes ~40-50 GPU-hours, logs show `TIMEOUT` at end of each 11h chunk):

```bash
# Evaluate each arm's checkpoint sweep (done automatically, but can re-run)
sbatch --export=ALL,RUN=rl3b_locolib_proof_typecheck,VAL_PARQUET=data_locolib/val_proof.parquet,N_EVAL=760 \
  hpc/grpo_eval.slurm
sbatch --export=ALL,RUN=rl3b_locolib_proof_gated,VAL_PARQUET=data_locolib/val_proof.parquet,N_EVAL=760 \
  hpc/grpo_eval.slurm
sbatch --export=ALL,RUN=rl3b_locolib_proof_outcome,VAL_PARQUET=data_locolib/val_proof.parquet,N_EVAL=760 \
  hpc/grpo_eval.slurm

# Compare arms side-by-side
python scripts/eval/compare_arms.py rl3b_locolib_proof_typecheck rl3b_locolib_proof_gated rl3b_locolib_proof_outcome

# Select best checkpoint by BEq+ + McNemar paired test
python scripts/eval/select_checkpoint.py --baseline sft3blocolib_proof-step76
```

Results land in `results/eval/<model_label>/eval_<label>-step<N>_n760.json`.

## Figures

```bash
# Regenerate all eval + training figures
make plots

# Training curves only
make train-figures

# Just the eval trajectories
make figures
```

Output PNG files: `results/figures/*.png`

## Quick Local Testing

```bash
# Smoke test (1 GRPO step, 8 examples) — proves the whole loop works
source hpc/cc_env.sh
make smoke
```

## Environment (HPC)

```bash
# Always source this before running anything
source hpc/cc_env.sh

# Contains:
# - Module stack (cuda, torch, arrow, etc.)
# - venv on $SCRATCH (torch 2.11, verl, vllm, lean-interact)
# - HF_TOKEN, HF_HOME (model cache)
# - LEAN_INTERACT_CACHE_DIR
# - MATHLIB_TAR defaults (for Lean 4.23.0)
```

See `hpc/NARVAL_NOTES.md` for cluster-specific gotchas (Mathlib staging, ROCR, etc.).
