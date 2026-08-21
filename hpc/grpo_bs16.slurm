#!/usr/bin/env bash
# GRPO at TRAIN_BATCH_SIZE=16 on an A100 40GB -- the batch the 16GB card could not
# afford. The placebo control showed update noise alone costs 14.3pp of BEq+ in 50
# steps, because at batch 4 with ~31-37% informative groups each Adam update was
# driven by ~1.5 prompts. 16 x 8 = 128 sequences/step puts that near ~5.
#   sbatch --export=ALL,ARM=gated  grpo_bs16.slurm
#   sbatch --export=ALL,ARM=placebo grpo_bs16.slurm
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=grpo_bs16
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=11:00:00
#SBATCH --output=logs/grpo_bs16_%j.out
#SBATCH --error=logs/grpo_bs16_%j.err

set -uo pipefail
export PROJECT_ROOT=/scratch/logan03/lean-gym-rl-rl_reward
REAL=/home/logan03/lean-gym-rl
source "${REAL}/hpc/cc_env.sh"

# Narval's SLURM exports ROCR_VISIBLE_DEVICES alongside CUDA_VISIBLE_DEVICES;
# verl's Worker._setup_env_cuda_visible_devices hard-errors when both are set.
unset ROCR_VISIBLE_DEVICES
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Mathlib is 36,700 files; read cold off Lustre, `import Mathlib` blew the old
# hardcoded 600s lean_interact timeout on compute nodes as well as login nodes.
# Extract the prebuilt tar to node-local NVMe instead (41s vs 1984s for cp -a).
if [ -n "${SLURM_TMPDIR:-}" ] && [ -f /scratch/logan03/mathlib4_v4.8.0-rc1.tar ]; then
  echo "[grpo] staging mathlib4 + REPL cache to ${SLURM_TMPDIR} ..."
  t0=$(date +%s)
  tar -xf /scratch/logan03/mathlib4_v4.8.0-rc1.tar -C "${SLURM_TMPDIR}"
  cp -a "${LEAN_INTERACT_CACHE_DIR}" "${SLURM_TMPDIR}/lean_interact_cache"
  export MATHLIB_ROOT="${SLURM_TMPDIR}/mathlib4"
  export LEAN_INTERACT_CACHE_DIR="${SLURM_TMPDIR}/lean_interact_cache"
  echo "[grpo] staging took $(( $(date +%s) - t0 ))s"
else
  export MATHLIB_ROOT="${REAL}/repos/mathlib4"
fi
export BEQ_ENV_TIMEOUT="${BEQ_ENV_TIMEOUT:-2400}"

ARM="${ARM:?set ARM=gated|placebo via --export=ALL,ARM=...}"
case "$ARM" in
  gated)   REWARD_FN_NAME=compute_score_gated ;;
  placebo) REWARD_FN_NAME=compute_score_placebo ;;
  *) echo "ARM must be gated|placebo"; exit 1 ;;
esac

# Each AgentLoopWorker builds its OWN Lean REPL (~4.3GB resident). The default of
# 2 was tuned for a 59GB box; here Lean is the throughput bottleneck (128
# rollouts/step to score) and we have the RAM, so widen it. 8 x ~4.3GB = ~34GB.
export AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-8}"

# 1280 disjoint RL prompts / batch 16 = exactly 80 steps = one clean epoch.
MODEL_PATH="${REAL}/checkpoints/merged/sft-step390" \
REWARD_FN_NAME="${REWARD_FN_NAME}" \
TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 ROLLOUT_N=8 \
SAVE_FREQ=10 MAX_CKPT_KEEP=12 \
EXPERIMENT_NAME="rl_${ARM}_bs16" \
VALIDATION_DATA_DIR="${REAL}/results/val_generations_${ARM}_bs16" \
  bash configs/run_grpo.sh \
    trainer.total_training_steps=80 \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    'actor_rollout_ref.actor.checkpoint.save_contents=[model,extra,hf_model]'

echo "[grpo_bs16] arm=${ARM} done"
