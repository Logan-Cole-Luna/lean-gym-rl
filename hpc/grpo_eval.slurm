#!/usr/bin/env bash
# Evaluate every checkpoint of one GRPO run against the pinned validation slice.
#   sbatch --export=ALL,RUN=<experiment name> hpc/grpo_eval.slurm
#
# One job rather than chunks: results are cached per checkpoint, so a walltime
# kill loses nothing and a resubmit picks up where it stopped. Output names
# (eval_<RUN>-step<N>_n<n>.json) are what scripts/eval/compare_arms.py pairs on.
#SBATCH --account=def-vganesh_gpu
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

# n=1000 gives an MDE of ~3.5pp against ~5.6pp at 400. Most effects here are
# 1-3pp, so a smaller n lands them directionally right and statistically mute.
N_EVAL="${N_EVAL:-1000}"
VAL_PARQUET="${VAL_PARQUET:-${DATA_DIR}/val.parquet}"

rename_job eval "${RUN#${RUN_PREFIX}_}"

RUNDIR="${REAL}/checkpoints/${PROJECT_NAME}/${RUN}"
mkdir -p "${REAL}/checkpoints/merged"

steps=$(ls -d "${RUNDIR}"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n)
echo "[eval] run=${RUN} val=${VAL_PARQUET} steps: ${steps:-NONE}"

for step in ${steps}; do
  out="${REAL}/results/eval_${RUN}-step${step}_n${N_EVAL}.json"
  [ -f "$out" ] && { echo "[eval] cached: ${RUN}-step${step} (n=${N_EVAL})"; continue; }
  # REUSE_CACHED=1 (default) keeps a result already scored at another n rather
  # than re-scoring: cross-arm comparisons are computed on the shared prefix
  # anyway. Set REUSE_CACHED=0 when the extra rows are actually wanted.
  if [ "${REUSE_CACHED:-1}" = "1" ]; then
    other=$(ls "${REAL}/results/eval_${RUN}-step${step}_n"*.json 2>/dev/null | head -1)
    [ -n "$other" ] && { echo "[eval] cached at another n -> reusing $(basename "$other")"; continue; }
  fi
  src="${RUNDIR}/global_step_${step}/actor/huggingface"
  ls "${src}"/*.safetensors >/dev/null 2>&1 || { echo "[eval] SKIP step${step}: no hf weights"; continue; }
  # evaluate_checkpoints.py labels records by directory name, so the symlink
  # name must be unique per checkpoint.
  model="${REAL}/checkpoints/merged/${RUN}-step${step}"
  ln -sfn "${src}" "${model}"
  echo "[eval] ===== ${RUN}-step${step} ====="
  python3 scripts/eval/evaluate_checkpoints.py \
    --checkpoint "${model}" --val-parquet "${REAL}/${VAL_PARQUET}" \
    --n-eval "${N_EVAL}" --workers "${BEQ_WORKERS:-8}" --out "${out}" \
    || echo "[eval] EVAL FAILED ${RUN}-step${step}"
done
echo "[eval] run=${RUN} done"
