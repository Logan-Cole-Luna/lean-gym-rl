#!/usr/bin/env bash
# SFT for Qwen2.5-Coder-3B-Instruct on the data_3b split.
#
# The 3B series is a SEPARATE comparison series with its own baseline. Note
# data_3b/val.parquet's first 400 rows are byte-identical to data/val.parquet,
# so a 3B eval at n=400 is still directly pairable with the 0.5B results.
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=sft_3b
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=logs/sft_3b_%j.out
#SBATCH --error=logs/sft_3b_%j.err

set -uo pipefail
export PROJECT_ROOT=/home/logan03/lean-gym-rl
source "${PROJECT_ROOT}/hpc/cc_env.sh"
unset ROCR_VISIBLE_DEVICES
cd "${PROJECT_ROOT}"
mkdir -p logs

MODEL="${MODEL:-/scratch/logan03/ai4math_training_models/qwen2.5-coder-3b-instruct}"

# TAG scopes every output of this job. Empty = the original Instruct-based
# series. Set TAG=mt when SFT-ing from a mid-trained checkpoint, so the two
# series never share a checkpoint dir, an eval filename, or a best_sft pointer:
#   sbatch --export=ALL,MODEL=<midtrain hf dir>,TAG=mt hpc/sft_3b.slurm
# MODEL=auto:<dir> resolves to the newest global_step_*/huggingface under <dir>
# AT RUNTIME. Needed because this job is chained behind midtrain_3b.slurm, whose
# final step number is not known at submit time.
case "${MODEL}" in
  auto:*)
    _root="${MODEL#auto:}"
    MODEL="$(ls -d "${_root}"/global_step_*/huggingface 2>/dev/null | sort -t_ -k3 -n | tail -1)"
    [ -n "${MODEL}" ] || { echo "[sft_3b] FATAL: no checkpoint under ${_root}"; exit 1; }
    MODEL="$(readlink -f "${MODEL}")"
    echo "[sft_3b] resolved auto -> ${MODEL}" ;;
esac

TAG="${TAG:-}"
SUFFIX="${TAG:+_${TAG}}"
echo "[sft_3b] model=${MODEL}"
echo "[sft_3b] saving to checkpoints/sft_3b${SUFFIX}"

# RESUMABLE: save_contents includes `optimizer` and resume_mode=auto, so a
# walltime or memory kill costs the steps since the last save rather than the
# whole run. The first attempt saved [model,extra,hf_model] only, so when it was
# OOM-killed at step 62 of 93 there was no optimizer state to resume from and
# the third epoch had to be redone from scratch. Costs ~25GB/checkpoint on
# scratch, which is free here.
#
# HOST RAM: 240G, not 120G. param_offload/optimizer_offload (below) fix the GPU
# OOM by moving ~12GB of fp32 params and ~25GB of Adam state to CPU, and each
# `hf_model` checkpoint save gathers another full ~12GB state dict there. At
# 120G the step-62 save was OOM-killed by slurmstepd (MaxRSS 125,827,332K)
# while the GPU sat comfortably at 38.65GB. Narval A100 nodes have ~124G per
# GPU share, so 2 GPUs justifies ~240G.
#
# TWO GPUs, deliberately. At world size 1 FSDP falls back to NO_SHARD -- no
# sharding happens at all -- so a 3B model in verl's default fp32 master weights
# needs ~12GB params + ~12GB grads + ~25GB Adam on a single 40GB card and OOMs
# (measured: 38.25GB allocated before dying). param_offload/optimizer_offload
# are set below but do not rescue NO_SHARD. Sharding across 2 ranks roughly
# halves each term and is the actual fix.

# 8000 examples at verl's default train_batch_size=256 -> 31 optimizer steps per
# epoch (the 0.5B run's "step390" was 20k/256 x 5 epochs). Save once per epoch
# and pick by validation BEq+ with scripts/select_checkpoint.py rather than
# assuming the last epoch is best -- a 3B model overfits 8k faster than 0.5B did.
MODEL_PATH="${MODEL}" \
SAVE_PATH="${PROJECT_ROOT}/checkpoints/sft_3b${SUFFIX}" \
TRAIN_FILE="${PROJECT_ROOT}/data_3b/sft/train.parquet" \
VAL_FILE="${PROJECT_ROOT}/data_3b/sft/val.parquet" \
MICRO_BATCH="${MICRO_BATCH:-1}" \
LR="${LR:-5e-5}" \
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}" \
SFT_SAVE_FREQ="${SFT_SAVE_FREQ:-31}" \
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-6}" \
  NPROC=2 bash scripts/run_sft.sh \
    'checkpoint.save_contents=[model,optimizer,extra,hf_model]' \
    trainer.resume_mode=auto \
    engine.param_offload=True \
    engine.optimizer_offload=True

echo "[sft_3b] done"
ls -d "${PROJECT_ROOT}/checkpoints/sft_3b${SUFFIX}"/global_step_* 2>/dev/null
