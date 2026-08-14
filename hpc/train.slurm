#!/usr/bin/env bash
# ============================================================
# GRPO ablation-arm training job (composite BEq+ reward vs. type-check-only).
#
# Mirrors MixtureOfMathExperts/hpc/*.slurm conventions (account, module set via
# hpc/cc_env.sh, log paths). #SBATCH values below are a starting point sized
# for the PoC scale in configs/run_grpo.sh — bump --time/--mem for a longer
# TOTAL_STEPS run, and re-check --account against your own allocation before
# submitting (defaulted here to the same allocation MixtureOfMathExperts uses,
# since this is a sibling project of the same research group).
#
# Submit:
#   sbatch hpc/train.slurm --export=REWARD_FN_NAME=compute_score_composite
#   sbatch hpc/train.slurm --export=REWARD_FN_NAME=compute_score_typecheck_only
#
# Override examples:
#   sbatch hpc/train.slurm --export=REWARD_FN_NAME=compute_score_composite,TOTAL_STEPS=200,TRAIN_BATCH_SIZE=32
# ============================================================

#SBATCH --account=def-vganesh
#SBATCH --job-name=beqplus_grpo
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/beqplus_grpo_%j.out
#SBATCH --error=logs/beqplus_grpo_%j.err

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$SCRATCH/ai4math_training}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
mkdir -p "${PROJECT_ROOT}/logs"

REWARD_FN_NAME="${REWARD_FN_NAME:-compute_score_composite}"
TOTAL_STEPS="${TOTAL_STEPS:-30}"

echo "[train.slurm] reward_fn=${REWARD_FN_NAME} total_steps=${TOTAL_STEPS}"

REWARD_FN_NAME="${REWARD_FN_NAME}" \
  bash configs/run_grpo.sh \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.test_freq=10 \
    trainer.save_freq=10
