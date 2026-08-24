#!/usr/bin/env bash
# GRPO at batch 16 on the 3B policy.  ARM=gated|placebo|gated_filtered
#   sbatch --export=ALL,ARM=gated hpc/grpo_3b.slurm
#
# 11h CHUNKS -- the longest that still sits in the good priority band here.
# 3h chunks were measured to be a false economy: four of them took ~22h wall for
# 12h of compute, including a single 9h gap between chunk 1 ending and chunk 2
# starting. At ~10.9 min/step an 11h chunk covers ~60 steps, so step 20 -> 150
# is 3 chunks rather than 9. Chain them with --dependency=afterany: (a walltime
# kill is the expected end of a chunk, not a failure); resume_mode=auto picks up
# from the last checkpoint. Lean scoring is ~2.1s/rollout at 8-12 workers, so 150 steps
# x 128 rollouts is ~11h -- but CC gives short jobs much better backfill
# priority, and verl's resume_mode=auto continues toward total_training_steps
# from the last checkpoint. So submit several of these chained with
# `--dependency=afterany:` (afterany, NOT afterok: a walltime kill is the
# expected way each chunk ends, and it is not a failure).
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=grpo3b_bb
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=11:00:00
#SBATCH --output=logs/grpo3b_bb_%j.out
#SBATCH --error=logs/grpo3b_bb_%j.err

set -uo pipefail
export PROJECT_ROOT=/home/logan03/lean-gym-rl
REAL="${PROJECT_ROOT}"
source "${REAL}/hpc/cc_env.sh"
unset ROCR_VISIBLE_DEVICES
cd "${REAL}"
export PYTHONPATH="${REAL}:${PYTHONPATH:-}"

if [ -n "${SLURM_TMPDIR:-}" ] && [ -f /scratch/logan03/mathlib4_v4.8.0-rc1.tar ]; then
  t0=$(date +%s)
  tar -xf /scratch/logan03/mathlib4_v4.8.0-rc1.tar -C "${SLURM_TMPDIR}"
  cp -a "${LEAN_INTERACT_CACHE_DIR}" "${SLURM_TMPDIR}/lean_interact_cache"
  export MATHLIB_ROOT="${SLURM_TMPDIR}/mathlib4"
  export LEAN_INTERACT_CACHE_DIR="${SLURM_TMPDIR}/lean_interact_cache"
  echo "[grpo3b] staging took $(( $(date +%s) - t0 ))s"
fi
export BEQ_ENV_TIMEOUT="${BEQ_ENV_TIMEOUT:-2400}"

MULTITURN="${MULTITURN:-1}"
[ "$MULTITURN" = "1" ] && MULTITURN_ON=1 || MULTITURN_ON=""
ARM="${ARM:?set ARM=gated|placebo|gated_filtered}"
CKPT=$(cat data_3b/best_sft.txt 2>/dev/null)
[ -n "${CKPT}" ] || { echo "[grpo3b] FATAL: data_3b/best_sft.txt missing"; exit 1; }
case "${CKPT}" in /*) ;; *) CKPT="$(cd "$(dirname "${CKPT}")" && pwd)/$(basename "${CKPT}")" ;; esac
[ -d "${CKPT}" ] || { echo "[grpo3b] FATAL: checkpoint dir not found: ${CKPT}"; exit 1; }

EXTRA=()
case "$ARM" in
  gated)   REWARD_FN_NAME=compute_score_gated ;;
  gated_filtered)
    REWARD_FN_NAME=compute_score_gated
    # DAPO-style: discard groups that cannot produce an advantage and refill.
    # ~3x generation cost, but it attacks the dead-batch problem head-on.
    export FILTER_GROUPS=1 FILTER_GROUPS_METRIC=semantic_signal ;;
  placebo)
    REWARD_FN_NAME=compute_score_placebo
    # The control MUST be fitted to this policy's geometry, not 0.5B's.
    [ -f data_3b/placebo_constants.sh ] || {
      echo "[grpo3b] FATAL: data_3b/placebo_constants.sh missing -- run rollouts_3b first."
      echo "[grpo3b] An uncalibrated placebo is not a control."; exit 1; }
    . data_3b/placebo_constants.sh
    echo "[grpo3b] placebo fitted to 3B: GROUP_P=${BEQ_PLACEBO_GROUP_P} ROLLOUT_P=${BEQ_PLACEBO_ROLLOUT_P}" ;;
  *) echo "bad ARM"; exit 1 ;;
esac

# BATCH 256 -- project-numina/kimina-prover-rl's production Lean-RL setting
# (train_batch_size=256, ppo_mini_batch_size=32, kl_loss_coef=0.0). Independent
# corroboration of what our own placebo showed: at batch 4 noise damage compounds
# without bound; at 16 it equilibrates. This asks whether more still helps.
# COST: 256 x 8 = 2048 rollouts/step at ~2.6s each over 24 Lean workers is
# ~87 min/step, so an 11h job covers ~8 steps. Target is 40 steps, not 150.
# We keep kl_loss_coef at our 0.01 rather than their 0.0 -- changing batch AND
# removing the KL anchor at once would confound the two.
#
# (batch-16 note) 4300 RL prompts / batch 16 = 268 steps per epoch, so 150 steps stays well
# inside one pass -- no prompt is ever seen twice.
MODEL_PATH="${CKPT}" \
REWARD_FN_NAME="${REWARD_FN_NAME}" \
TRAIN_BATCH_SIZE=256 PPO_MINI_BATCH_SIZE=32 ROLLOUT_N=8 \
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.30}" \
AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-24}" \
NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}" \
SAVE_FREQ=10 MAX_CKPT_KEEP=20 \
EXPERIMENT_NAME="rl3b_bb_${ARM}" \
  bash configs/run_grpo.sh \
    ${MULTITURN_ON:+actor_rollout_ref.rollout.multi_turn.enable=True} \
    ${MULTITURN_ON:+actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS:-2}} \
    ${MULTITURN_ON:+actor_rollout_ref.rollout.multi_turn.function_tool_path=${REAL}/reward/lean_tool.py} \
    data.train_files="${REAL}/data_3b/train.parquet" \
    data.val_files="${REAL}/data_3b/val.parquet" \
    trainer.total_training_steps="${TOTAL_STEPS:-40}" \
    trainer.resume_mode=auto \
    trainer.test_freq="${TEST_FREQ:-50}" \
    trainer.save_freq=10 \
    'actor_rollout_ref.actor.checkpoint.save_contents=[model,extra,hf_model]'

echo "[grpo3b] arm=${ARM} done"
