#!/usr/bin/env bash
# SFT the base model on the corpus in $DATA_DIR.
#   sbatch --export=ALL hpc/sft.slurm
#   sbatch --export=ALL,TAG=locolib,TRAIN_FILE=$PWD/data_locolib/sft.parquet,\
#     VAL_FILE=$PWD/data_locolib/val_sft.parquet,SFT_SAVE_FREQ=46 hpc/sft.slurm
#
# TAG scopes every output, so two series never share a checkpoint dir, an eval
# filename, or a best-SFT pointer.
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=sft
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh
mkdir -p logs

MODEL="${MODEL:-${BASE_MODEL}}"

# MODEL=auto:<dir> resolves to the newest global_step_*/huggingface under <dir>
# at runtime, for chaining behind a job whose final step number is not known at
# submit time.
case "${MODEL}" in
  auto:*)
    _root="${MODEL#auto:}"
    MODEL="$(ls -d "${_root}"/global_step_*/huggingface 2>/dev/null | sort -t_ -k3 -n | tail -1)"
    [ -n "${MODEL}" ] || { echo "[sft] FATAL: no checkpoint under ${_root}"; exit 1; }
    MODEL="$(readlink -f "${MODEL}")"
    echo "[sft] resolved auto -> ${MODEL}" ;;
esac

TAG="${TAG:-}"
SUFFIX="${TAG:+_${TAG}}"
SAVE_PATH="${PROJECT_ROOT}/checkpoints/${SFT_DIR}${SUFFIX}"
rename_job sft "${TAG}"
echo "[sft] model=${MODEL}"
echo "[sft] saving to ${SAVE_PATH}"

# Two GPUs, deliberately: at world size 1 FSDP falls back to NO_SHARD, so fp32
# master weights, grads and Adam state all sit on one card and OOM. Offloading
# does not rescue NO_SHARD; sharding across ranks does.
#
# 240G host RAM, not 120G: param_offload and optimizer_offload move params and
# Adam state to CPU, and each hf_model save gathers another full state dict
# there.
#
# save_contents includes `optimizer` with resume_mode=auto, so a walltime or
# memory kill costs the steps since the last save rather than the whole run.
#
# SFT_SAVE_FREQ is in optimizer steps, and steps/epoch = rows / train_batch_size
# (256 by default). A different corpus needs its own value. Save once per epoch
# and pick with scripts/eval/select_checkpoint.py rather than assuming the last epoch
# is best.
MODEL_PATH="${MODEL}" \
SAVE_PATH="${SAVE_PATH}" \
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/${DATA_DIR}/sft/train.parquet}" \
VAL_FILE="${VAL_FILE:-${PROJECT_ROOT}/${DATA_DIR}/sft/val.parquet}" \
MICRO_BATCH="${MICRO_BATCH:-1}" \
LR="${LR:-5e-5}" \
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}" \
SFT_SAVE_FREQ="${SFT_SAVE_FREQ:-31}" \
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-6}" \
  NPROC="${NPROC:-2}" bash scripts/train/run_sft.sh \
    'checkpoint.save_contents=[model,optimizer,extra,hf_model]' \
    trainer.resume_mode=auto \
    engine.param_offload=True \
    engine.optimizer_offload=True

echo "[sft] done"
ls -d "${SAVE_PATH}"/global_step_* 2>/dev/null
