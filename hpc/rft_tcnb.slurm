#!/usr/bin/env bash
# Rejection-sampling fine-tuning: BEq+-filtered vs size-matched type-check-filtered.
# Both arms start from sft-step390 and differ ONLY in which rollouts were accepted.
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=rft_tcnb
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/rft_tcnb_%j.out
#SBATCH --error=logs/rft_tcnb_%j.err

set -euo pipefail
export PROJECT_ROOT=/home/logan03/lean-gym-rl
source "${PROJECT_ROOT}/hpc/cc_env.sh"

# Narval's SLURM exports ROCR_VISIBLE_DEVICES (AMD) alongside CUDA_VISIBLE_DEVICES;
# verl's Worker._setup_env_cuda_visible_devices hard-errors when both are set.
unset ROCR_VISIBLE_DEVICES
cd "${PROJECT_ROOT}"
mkdir -p logs checkpoints/merged

MERGED="${PROJECT_ROOT}/checkpoints/merged/sft-step390"

# ── 1. Merge the SFT FSDP checkpoint into HF format (needs only the model shard,
#       not the optimizer state) ────────────────────────────────────────────────
if [ ! -f "${MERGED}/model.safetensors" ] && [ ! -f "${MERGED}/pytorch_model.bin" ]; then
  echo "[rft] merging global_step_390 -> ${MERGED}"
  python3 -m verl.model_merger merge --backend fsdp \
    --local_dir "${PROJECT_ROOT}/global_step_390" \
    --target_dir "${MERGED}"
else
  echo "[rft] merged checkpoint already present, skipping merge"
fi
ls -la "${MERGED}"

# ── 2. Train both arms, identical hyperparameters ─────────────────────────────
# Checkpoints go to /scratch via the checkpoints symlink -- /home's 50GB quota
# cannot hold 6 of them and a truncated optimizer write killed the first run.
# 'hf_model' saves ready-to-load HF weights (no model_merger step at eval time);
# the optimizer state is dropped because RFT never resumes.
# 1143 train examples. train_batch_size 64 -> ~17 optimizer steps/epoch;
# 3 epochs -> ~51 steps, saving once per epoch so all three are evaluable.
run_arm () {
  local tag="$1" data="$2"
  echo "=============================================================="
  echo "[rft] arm=${tag} data=${data}"
  echo "=============================================================="
  MODEL_PATH="${MERGED}" \
  SAVE_PATH="${PROJECT_ROOT}/checkpoints/rft/${tag}" \
  TRAIN_FILE="${PROJECT_ROOT}/${data}/train.parquet" \
  VAL_FILE="${PROJECT_ROOT}/${data}/val.parquet" \
  LR=1e-5 TOTAL_EPOCHS=3 SFT_SAVE_FREQ=17 MAX_CKPT_KEEP=5 \
  MICRO_BATCH=4 \
    bash scripts/run_sft.sh data.train_batch_size=64 \
      'checkpoint.save_contents=[model,extra,hf_model]' \
      trainer.resume_mode=disable
}

run_arm tcnb data/rft_tcnb

echo "[rft] both arms complete"
ls -R "${PROJECT_ROOT}/checkpoints/rft" | head -40
