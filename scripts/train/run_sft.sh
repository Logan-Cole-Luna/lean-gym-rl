#!/usr/bin/env bash
# SFT stage: teach the base model Lean syntax + semantic alignment BEFORE RL.
#
# Rationale (see scripts/data/prepare_sft_dataset.py for the full argument): GRPO's
# advantage is computed within a rollout group, so RL from the raw base model is
# a lottery -- it learns nothing until some sample in a group beats its peers,
# and from this base model a type-checking sample is rare. Brute-forcing that
# with a large `rollout_n` is precisely what makes the Lean reward expensive
# (dozens of concurrent Mathlib-resident REPLs, ~4.3GB each).
#
# SFT on (informal -> gold formal signature) pairs removes the cold start: the
# policy starts competent, so RL only refines and a SMALL rollout_n suffices.
#
# Usage:
#   bash scripts/train/run_sft.sh                       # 1 epoch, 4k examples
#   TOTAL_EPOCHS=2 MICRO_BATCH=2 bash scripts/train/run_sft.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${VIRTUAL_ENV:-}" ]; then
  for _venv in "$REPO_ROOT/.venv_hpc" "$REPO_ROOT/.venv"; do
    [ -f "$_venv/bin/activate" ] && { source "$_venv/bin/activate"; break; }
  done
fi
python3 -c "import verl" 2>/dev/null || { echo "ERROR: verl not importable; run 'make env'"; exit 1; }

MODEL_PATH=${MODEL_PATH:-$REPO_ROOT/models/qwen2.5-coder-0.5b-instruct}
SAVE_PATH=${SAVE_PATH:-$REPO_ROOT/checkpoints/sft/qwen2.5-coder-0.5b-leanworkbook}
# With use_dynamic_bsz=True (set below) MAX_TOKEN_LEN is the real per-micro-batch
# budget: verl packs sequences up to that many tokens regardless of MICRO_BATCH.
# 2048 == one MAX_LENGTH sequence, which is what fits a 3B on a 40GB A100 (FSDP-
# sharded, host offload on, grad checkpointing on). 8192 packs ~4 and OOMs.
# Raise both on an 80GB card.
MICRO_BATCH=${MICRO_BATCH:-1}
MAX_TOKEN_LEN=${MAX_TOKEN_LEN:-2048}
MAX_LENGTH=${MAX_LENGTH:-1024}
# verl's default is 'error': a single row longer than MAX_LENGTH kills the job
# in the DataLoader worker before step 1. Set TRUNCATION=right to clip the tail
# of the rare over-length row instead (e.g. a corpus kept for instance-match
# where a few gold targets run long).
TRUNCATION=${TRUNCATION:-error}
LR=${LR:-1e-4}
# verl defaults lr_scheduler_type to 'constant'. That leaves every late
# checkpoint taking full-size Adam steps on data the model has already
# memorised, so the eval bounces by several points between adjacent
# checkpoints and the "best" one is partly luck. LR_SCHEDULER=cosine anneals
# it away. Defaults preserve the old behaviour exactly.
LR_SCHEDULER=${LR_SCHEDULER:-constant}
LR_WARMUP_RATIO=${LR_WARMUP_RATIO:-0.0}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.0}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
NPROC=${NPROC:-1}
SAVE_FREQ=${SFT_SAVE_FREQ:-50}
MAX_CKPT_KEEP=${MAX_CKPT_KEEP:-2}
# Recompute activations in the backward pass rather than storing them. On unless
# you have measured headroom to turn it off (GRAD_CKPT=False).
GRAD_CKPT=${GRAD_CKPT:-True}

TRAIN_FILE=${TRAIN_FILE:-$REPO_ROOT/data/sft/train.parquet}
VAL_FILE=${VAL_FILE:-$REPO_ROOT/data/sft/val.parquet}
[ -f "$TRAIN_FILE" ] || { echo "Missing $TRAIN_FILE -- run scripts/data/prepare_sft_dataset.py first"; exit 1; }

export HF_HOME=${HF_HOME:-$REPO_ROOT/models/.hf_cache}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$SAVE_PATH" logs

echo "=============================================================="
echo " SFT: $MODEL_PATH"
echo "   -> $SAVE_PATH   (${TOTAL_EPOCHS} epochs, micro-batch ${MICRO_BATCH})"
echo "=============================================================="

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  -m verl.trainer.sft_trainer \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.messages_key=messages \
    data.micro_batch_size_per_gpu="${MICRO_BATCH}" \
    data.max_token_len_per_gpu="${MAX_TOKEN_LEN}" \
    data.max_length="${MAX_LENGTH}" \
    data.truncation="${TRUNCATION}" \
    optim.lr="${LR}" \
    optim.lr_scheduler_type="${LR_SCHEDULER}" \
    optim.lr_warmup_steps_ratio="${LR_WARMUP_RATIO}" \
    optim.min_lr_ratio="${MIN_LR_RATIO}" \
    engine=fsdp \
    model.path="$MODEL_PATH" \
    model.use_remove_padding=false \
    model.enable_gradient_checkpointing="${GRAD_CKPT}" \
    +model.override_config.attn_implementation=sdpa \
    trainer.default_local_dir="$SAVE_PATH" \
    trainer.project_name=beqplus_sft \
    trainer.experiment_name=qwen25_coder_0_5b_leanworkbook_sft \
    trainer.logger='["console"]' \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.max_ckpt_to_keep="${MAX_CKPT_KEEP}" \
    "$@"

echo ""
echo "SFT complete -> $SAVE_PATH"
echo "Next: RL from the SFT checkpoint, e.g."
echo "  MODEL_PATH=$SAVE_PATH ROLLOUT_N=4 make train-shaped"
