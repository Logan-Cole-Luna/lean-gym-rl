#!/usr/bin/env bash
# SFT stage: teach the base model Lean syntax + semantic alignment BEFORE RL.
#
# Rationale (see scripts/prepare_sft_dataset.py for the full argument): GRPO's
# advantage is computed within a rollout group, so RL from the raw base model is
# a lottery -- it learns nothing until some sample in a group beats its peers,
# and from this base model a type-checking sample is rare. Brute-forcing that
# with a large `rollout_n` is precisely what makes the Lean reward expensive
# (dozens of concurrent Mathlib-resident REPLs, ~4.3GB each).
#
# SFT on (informal -> gold formal signature) pairs removes the cold start: the
# policy starts competent, so RL only refines and a SMALL rollout_n suffices.
#
# Note this stage needs NO Lean at all -- there is no reward function, so none of
# the host-RAM pressure from the RL stages applies here.
#
# Known cosmetic issue: `val/loss` can come out `nan` on a 16GB card. It is
# preceded in the log by a burst of "expandable_segments: memory mapping failed
# with OOM on device 0" while max_memory_allocated sits ~12.3GB -- i.e. the
# validation pass hits GPU memory pressure. Training itself is unaffected (train
# loss stays healthy and the checkpoint evaluates well: 33.8% BEq+), so this is a
# validation-metric artifact, not a bad model. Lower MAX_TOKEN_LEN if you need a
# trustworthy val number; otherwise judge the checkpoint with
# scripts/evaluate_checkpoints.py, which measures what we actually care about.
#
# Usage:
#   bash scripts/run_sft.sh                       # 1 epoch, 4k examples
#   TOTAL_EPOCHS=2 MICRO_BATCH=2 bash scripts/run_sft.sh
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
MICRO_BATCH=${MICRO_BATCH:-1}
# verl's SFT defaults (max_token_len_per_gpu=8192, max_length=1024) pack far more
# tokens per micro-batch than this task needs -- our prompt+target is ~300 tokens
# -- and OOM a 16GB card. Sized to the actual data instead.
MAX_TOKEN_LEN=${MAX_TOKEN_LEN:-2048}
MAX_LENGTH=${MAX_LENGTH:-1024}
LR=${LR:-1e-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
NPROC=${NPROC:-1}
# verl's SFT default is save_freq=-1, i.e. save ONLY at the very last step. That
# is an all-or-nothing bet on a long run -- the RL side already lost 1h48m that
# way to a host OOM kill at step 25/30. Save periodically, but cap retention:
# each checkpoint is ~6.1GB, and a 234-step run saving every 50 steps would
# otherwise leave 30GB behind (the disk has already hit 100% once).
SAVE_FREQ=${SFT_SAVE_FREQ:-50}
MAX_CKPT_KEEP=${MAX_CKPT_KEEP:-2}

TRAIN_FILE=${TRAIN_FILE:-$REPO_ROOT/data/sft/train.parquet}
VAL_FILE=${VAL_FILE:-$REPO_ROOT/data/sft/val.parquet}
[ -f "$TRAIN_FILE" ] || { echo "Missing $TRAIN_FILE -- run scripts/prepare_sft_dataset.py first"; exit 1; }

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
    optim.lr="${LR}" \
    engine=fsdp \
    model.path="$MODEL_PATH" \
    model.use_remove_padding=false \
    model.enable_gradient_checkpointing=True \
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
