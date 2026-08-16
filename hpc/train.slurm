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
#SBATCH --mem=160G   # Lean REPLs are spawned per concurrent request; see README gotcha 1b
#SBATCH --time=04:00:00
#SBATCH --output=logs/beqplus_grpo_%j.out
#SBATCH --error=logs/beqplus_grpo_%j.err

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$SCRATCH/ai4math_training}"
source "${PROJECT_ROOT}/hpc/cc_env.sh"
mkdir -p "${PROJECT_ROOT}/logs"

REWARD_FN_NAME="${REWARD_FN_NAME:-compute_score_gated}"
TOTAL_STEPS="${TOTAL_STEPS:-100}"
# Warm start. RL from the base model does not learn semantics at this scale
# (measured: ~0-4% BEq+ after 30 steps, vs 33.8% for plain SFT), so INIT should
# normally point at a merged SFT checkpoint. Left empty = train from the base
# model, which is only useful for reproducing the from-scratch ablation.
INIT="${INIT:-}"
MODEL_PATH="${INIT:-${MODEL_PATH:-$SCRATCH/ai4math_training_models/qwen2.5-coder-0.5b-instruct}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rl_$(basename "${MODEL_PATH}")_${REWARD_FN_NAME}}"

if [ ! -e "${MODEL_PATH}/config.json" ]; then
  echo "[train.slurm] ERROR: MODEL_PATH=${MODEL_PATH} has no config.json."
  echo "  Pass INIT=<merged HF checkpoint dir> (see 'make merge-sft')."
  exit 1
fi

echo "[train.slurm] reward_fn=${REWARD_FN_NAME} total_steps=${TOTAL_STEPS}"
echo "[train.slurm] init=${MODEL_PATH}"

REWARD_FN_NAME="${REWARD_FN_NAME}" \
MODEL_PATH="${MODEL_PATH}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${PROJECT_ROOT}/results/val_generations/${EXPERIMENT_NAME}}" \
  bash configs/run_grpo.sh \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.test_freq="${TEST_FREQ:-10}" \
    trainer.save_freq="${SAVE_FREQ:-10}"
