#!/usr/bin/env bash
# One GRPO arm per job.
#   sbatch --export=ALL,ARM=gated hpc/grpo.slurm
#   ARM=gated|typecheck|outcome|gated_edge
#
# Runs in 11h chunks chained with --dependency=afterany: a walltime kill is the
# expected end of a chunk, not a failure, and resume_mode=auto continues from the
# last checkpoint. Shorter chunks queue faster but lose more to gaps between them.
#
# Corpus and naming come from DATA_DIR / RUN_PREFIX / PROJECT_NAME in
# job_prelude.sh; override those to run the same arms on another corpus.
#SBATCH --account=def-vganesh
#SBATCH --job-name=grpo
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=11:00:00
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh

stage_mathlib "[grpo]"

ARM="${ARM:?set ARM=gated|typecheck|outcome|gated_edge}"
# Which SFT series the arms train from. Arms resumed from different baselines
# are not comparable to each other.
BEST_SFT="${BEST_SFT:-${DATA_DIR}/best_sft.txt}"
SERIES_TAG="${SERIES_TAG:-}"
SERIES_SUFFIX="${SERIES_TAG:+_${SERIES_TAG}}"
RUN_NAME="${RUN_PREFIX}${SERIES_SUFFIX}_${ARM}"

CKPT=$(cat "${BEST_SFT}" 2>/dev/null)
[ -n "${CKPT}" ] || { echo "[grpo] FATAL: ${BEST_SFT} missing"; exit 1; }
case "${CKPT}" in /*) ;; *) CKPT="$(cd "$(dirname "${CKPT}")" && pwd)/$(basename "${CKPT}")" ;; esac
[ -d "${CKPT}" ] || { echo "[grpo] FATAL: checkpoint dir not found: ${CKPT}"; exit 1; }

EXTRA=()
TRAIN_FILE="${TRAIN_FILE:-${REAL}/${DATA_DIR}/train.parquet}"
# The val slice must follow the corpus: BEq+ needs a gold, so validating against
# another corpus's golds measures transfer, not fit.
VAL_FILE="${VAL_FILE:-${REAL}/${DATA_DIR}/val.parquet}"

case "$ARM" in
  gated)
    # The default: BEq+ semantic signal only, flat floor everywhere else.
    REWARD_FN_NAME=compute_score_gated ;;
  typecheck)
    # Ablation baseline. Exploitable by construction, since the statement is the
    # model's output and `theorem x : 1 = 1` satisfies it. Its BEq+ is expected
    # to fall; the success criterion is that the reward rises, which separates
    # "BEq+ does not teach" from "this stack cannot optimise anything".
    REWARD_FN_NAME=compute_score_typecheck_only ;;
  gated_edge)
    # Same reward as `gated`; only the prompt pool differs, restricted to
    # prompts scoring 1-7/8 at k=8 from this SFT checkpoint. Only those groups
    # produce a nonzero advantage, so the informative fraction rises from ~46%
    # to ~100% at no extra Lean cost. A static filter on a moving target:
    # prompts leave the band as the policy improves.
    REWARD_FN_NAME=compute_score_gated
    TRAIN_FILE="${REAL}/${DATA_DIR}/train_edge.parquet"
    [ -f "${TRAIN_FILE}" ] || {
      echo "[grpo] FATAL: ${TRAIN_FILE} missing."
      echo "[grpo] Run hpc/score_pool.slurm then hpc/build_edge_pool.slurm."; exit 1; } ;;
  outcome)
    # The six-outcome ladder from reward/reward.py: 0.00 / 0.05 / 0.15 / 0.50.
    # Unlike gated it pays a graded reward below BEq+, so groups with no semantic
    # signal still carry a within-group gradient. Run it against gated rather
    # than in place of it.
    REWARD_FN_NAME=compute_score_outcome ;;
  *) echo "[grpo] bad ARM: ${ARM}"; exit 1 ;;
esac

# SERIES_TAG in the name too: two series can run the same ARM (e.g. a plain
# edge-pool arm vs a SERIES_TAG=proof variant), and without it they are
# indistinguishable in squeue.
rename_job grpo "${SERIES_TAG:+${SERIES_TAG}_}${ARM}"

# Arms separate and then flatten well before 90 steps, and they are compared at
# matched steps, so the useful range is capped by the shortest arm anyway.
TOTAL_STEPS="${TOTAL_STEPS:-90}"

# verl derives load_contents from save_contents, so the first chunk that turns
# the optimizer on tries to load an optimizer out of a checkpoint written while
# it was off and dies on the missing optim shard. Detect that one-time case and
# load model+extra only for this chunk; later chunks resume Adam normally.
LOAD_CONTENTS=""
CKPT_ROOT="${REAL}/checkpoints/${PROJECT_NAME}/${RUN_NAME}"
LAST_CKPT="$(ls -d "${CKPT_ROOT}"/global_step_* 2>/dev/null \
             | sed 's/.*global_step_//' | sort -n | tail -1)"
if [ -n "${LAST_CKPT}" ] \
   && ! ls "${CKPT_ROOT}/global_step_${LAST_CKPT}/actor"/optim_world_size_*.pt >/dev/null 2>&1; then
  LOAD_CONTENTS="model,extra"
  echo "[grpo] step_${LAST_CKPT} has no optimizer shard; loading model,extra for"
  echo "[grpo] this chunk only. Adam cold-starts once here."
fi

# Do not put a comment inside the continued command below: bash joins the
# continuation and the `#` silently drops every remaining argument.
MODEL_PATH="${CKPT}" \
REWARD_FN_NAME="${REWARD_FN_NAME}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}" \
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}" \
ROLLOUT_N="${ROLLOUT_N:-8}" \
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.30}" \
AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-24}" \
NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}" \
SAVE_FREQ="${SAVE_FREQ:-10}" MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-20}" \
EXPERIMENT_NAME="${RUN_NAME}" \
  bash configs/run_grpo.sh \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.resume_mode=auto \
    trainer.test_freq="${TEST_FREQ:-50}" \
    trainer.save_freq="${SAVE_FREQ:-10}" \
    'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]' \
    ${LOAD_CONTENTS:+"actor_rollout_ref.actor.checkpoint.load_contents=[${LOAD_CONTENTS}]"} \
    ${EXTRA[@]+"${EXTRA[@]}"}
STATUS=$?

# job_prelude.sh deliberately omits `set -e` (some jobs continue past a soft
# failure), so a crash here -- Ray's node ever failed to come up on a bad node,
# main_ppo raising before the first step -- would otherwise fall through to
# the "done" line below and exit 0. That is a silent no-op: the chunk trains
# nothing, the dependency chain still fires, and eval "completes" against
# whatever checkpoint (or none) was already there. Fail the job instead.
if [ "${STATUS}" -ne 0 ]; then
  echo "[grpo] arm=${ARM} run=${RUN_NAME} FAILED (exit ${STATUS})"
  exit "${STATUS}"
fi

echo "[grpo] arm=${ARM} run=${RUN_NAME} done"
