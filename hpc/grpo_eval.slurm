#!/usr/bin/env bash
# Evaluate every checkpoint of one GRPO run against the pinned validation slice.
#   sbatch --export=ALL,RUN=<experiment name> hpc/grpo_eval.slurm
#
# One job rather than chunks: results are cached per checkpoint, so a walltime
# kill loses nothing and a resubmit picks up where it stopped. Output names
# (eval_<RUN>-step<N>_n<n>.json) are what scripts/eval/compare_arms.py pairs on.
#SBATCH --account=def-vganesh
#SBATCH --job-name=grpo_eval
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=09:00:00
#SBATCH --output=logs/grpo_eval_%j.out
#SBATCH --error=logs/grpo_eval_%j.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh
stage_mathlib "[eval]"
# Fall back to the shared checkout only if staging did not happen.
export MATHLIB_ROOT="${MATHLIB_ROOT:-${REAL}/repos/mathlib4}"

RUN="${RUN:?set RUN=<experiment name> via --export=ALL,RUN=...}"

# OUT_LABEL names the results files and the merged symlink; RUN names the
# checkpoint directory. They differ only when cross-evaluating one run on
# another corpus's val slice, which must NOT write records under the home run's
# label: select_checkpoint.py and compare_arms.py merge by the label inside the
# json, so an OOD record filed under the in-domain label silently averages a
# transfer number into a fit number. miniF2F evals therefore go to
# OUT_LABEL=minif2f_<arm>, matching the existing results/eval/minif2f_* dirs.
LABEL="${OUT_LABEL:-${RUN}}"

# n=1000 gives an MDE of ~3.5pp against ~5.6pp at 400. Most effects here are
# 1-3pp, so a smaller n lands them directionally right and statistically mute.
N_EVAL="${N_EVAL:-1000}"
VAL_PARQUET="${VAL_PARQUET:-${DATA_DIR}/val.parquet}"
# Passed EXPLICITLY, never left to the script's default, because the default
# changed (128 -> 512) and this job caches per checkpoint: a series half-scored
# under each budget would silently mix two generation budgets inside one
# trajectory. Every eval json now records `max_new_tokens`, so check it before
# pairing an old series against a new one -- the locolib_proof_lr6 arms were all
# scored at 128 and are NOT directly comparable to a 512-budget run.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

rename_job eval "${LABEL#${RUN_PREFIX}_}"

RUNDIR="${REAL}/checkpoints/${PROJECT_NAME}/${RUN}"
mkdir -p "${REAL}/checkpoints/merged"

steps=$(ls -d "${RUNDIR}"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n)
echo "[eval] run=${RUN} label=${LABEL} val=${VAL_PARQUET} steps: ${steps:-NONE}"

EVAL_DIR="${REAL}/results/eval/${LABEL}"
mkdir -p "${EVAL_DIR}"
for step in ${steps}; do
  out="${EVAL_DIR}/eval_${LABEL}-step${step}_n${N_EVAL}.json"
  [ -f "$out" ] && { echo "[eval] cached: ${LABEL}-step${step} (n=${N_EVAL})"; continue; }
  # REUSE_CACHED=1 (default) keeps a result already scored at another n rather
  # than re-scoring: cross-arm comparisons are computed on the shared prefix
  # anyway. Set REUSE_CACHED=0 when the extra rows are actually wanted.
  if [ "${REUSE_CACHED:-1}" = "1" ]; then
    other=$(ls "${EVAL_DIR}/eval_${LABEL}-step${step}_n"*.json 2>/dev/null | head -1)
    [ -n "$other" ] && { echo "[eval] cached at another n -> reusing $(basename "$other")"; continue; }
  fi
  src="${RUNDIR}/global_step_${step}/actor/huggingface"
  ls "${src}"/*.safetensors >/dev/null 2>&1 || { echo "[eval] SKIP step${step}: no hf weights"; continue; }
  # evaluate_checkpoints.py labels records by directory name, so the symlink
  # name must be unique per checkpoint.
  model="${REAL}/checkpoints/merged/${LABEL}-step${step}"
  ln -sfn "${src}" "${model}"
  echo "[eval] ===== ${LABEL}-step${step} ====="
  python3 scripts/eval/evaluate_checkpoints.py \
    --checkpoint "${model}" --val-parquet "${REAL}/${VAL_PARQUET}" \
    --n-eval "${N_EVAL}" --workers "${BEQ_WORKERS:-8}" --out "${out}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    ${STANDALONE:+--standalone} \
    || echo "[eval] EVAL FAILED ${LABEL}-step${step}"
done
echo "[eval] run=${RUN} label=${LABEL} done"
