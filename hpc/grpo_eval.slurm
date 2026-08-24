#!/usr/bin/env bash
# Evaluate a GRPO run's checkpoints on the fixed 400-example validation slice.
#   sbatch --export=ALL,RUN=rl_placebo_bs16 grpo_eval.slurm
# ONE job, not 3h chunks. A 15-checkpoint sweep at n=1000 is ~7h; splitting it
# for backfill priority costs a fresh queue wait per chunk, which on this
# cluster outweighs the priority gain. Results are still cached per checkpoint,
# so a walltime kill loses nothing and a resubmit picks up where it stopped.
# Output names (eval_<RUN>-step<N>_n400.json) are what scripts/compare_arms.py
# pairs on, so e.g. rl_placebo_bs16 vs rl_placebo_clean works directly.
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=grpo_eval
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=09:00:00
#SBATCH --output=logs/grpo_eval_%j.out
#SBATCH --error=logs/grpo_eval_%j.err

# THIS JOB USED TO RUN FROM A SECOND CHECKOUT.
# It set PROJECT_ROOT=/scratch/logan03/lean-gym-rl-rl_reward and put that on
# PYTHONPATH, so every eval imported scripts and reward/ from a tree last touched
# 2026-08-20 while training ran from /home. Audited before changing it: the
# scratch tree is a strict SUBSET (85 substantive lines only in home, zero only
# in scratch) and `score`, `typecheck_ex` and `check_theorem_equivalence` are
# identical there, so no cached eval number is affected. But fixes to
# scripts/evaluate_checkpoints.py silently did not reach eval jobs, which is not
# a property worth keeping. One tree now.
source /home/logan03/lean-gym-rl/hpc/job_prelude.sh
stage_mathlib "[eval]"
# Fall back to the shared checkout only if staging did not happen.
export MATHLIB_ROOT="${MATHLIB_ROOT:-${REAL}/repos/mathlib4}"

# Defaults are the 0.5B series (n=400 on data/val.parquet). The 3B series uses
# n=1000 on data_3b/val.parquet -- whose first 400 rows are byte-identical, so
# both remain paired-comparable. Override per run:
#   sbatch --export=ALL,RUN=rl3b_placebo,N_EVAL=1000,VAL_PARQUET=data_3b/val.parquet
# n=1000, not 400. The 3B val split has 1000 rows and the MDE at 1000 is ~3.5pp
# against ~5.6pp at 400 -- most real effects here are 1-3pp, so the smaller n
# lands them directionally right and statistically mute. The old 400 default
# already bit once: rl3b_v2_placebo, the control the headline result is measured
# against, was evaluated at 400 while every other arm used 1000.
# The first 400 rows are shared with data/val.parquet, so n=400 stays available
# for cross-scale paired tests against the 0.5B series.
N_EVAL="${N_EVAL:-1000}"
# Default follows the RUN's series. data/val.parquet is the 400-row 0.5B set;
# rl3b_* arms belong to the 1000-row data_3b split. The first 400 rows of the two
# are byte-identical (verified: prompts and golds match in order), so the old
# blanket default happened to be harmless at n=400 -- but it silently capped
# N_EVAL=1000 runs at 400 rows while still naming the file _n1000.
case "${RUN:-}" in
  rl3b*|*3b*) _default_val=data_3b/val.parquet ;;
  *)          _default_val=data/val.parquet ;;
esac
VAL_PARQUET="${VAL_PARQUET:-${_default_val}}"
RUN="${RUN:?set RUN=rl_gated_bs16|rl_placebo_bs16 via --export=ALL,RUN=...}"
RUNDIR="${REAL}/checkpoints/beqplus_rl_poc/${RUN}"
mkdir -p "${REAL}/checkpoints/merged"

# Ascending: the early steps are where the bs4 arms had already diverged.
steps=$(ls -d "${RUNDIR}"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n)
echo "[eval] run=${RUN} steps: ${steps:-NONE}"

for step in ${steps}; do
  out="${REAL}/results/eval_${RUN}-step${step}_n${N_EVAL}.json"
  # REUSE_CACHED=1 (default): if this step was already scored at ANY n, keep it
  # rather than re-scoring at the requested n. A 15-checkpoint arm at n=1000 is
  # ~6h of Lean for power we only use on rows 400-1000, and every cross-arm
  # comparison is computed on the shared 400-row prefix anyway. Set REUSE_CACHED=0
  # to force the requested n when you actually want the extra rows.
  [ -f "$out" ] && { echo "[eval] cached: ${RUN}-step${step} (n=${N_EVAL})"; continue; }
  if [ "${REUSE_CACHED:-1}" = "1" ]; then
    other=$(ls "${REAL}/results/eval_${RUN}-step${step}_n"*.json 2>/dev/null | head -1)
    [ -n "$other" ] && { echo "[eval] cached at another n -> reusing $(basename "$other")"; continue; }
  fi
  src="${RUNDIR}/global_step_${step}/actor/huggingface"
  ls "${src}"/*.safetensors >/dev/null 2>&1 || { echo "[eval] SKIP step${step}: no hf weights"; continue; }
  # Unique dir name: evaluate_checkpoints.py labels records by directory name.
  model="${REAL}/checkpoints/merged/${RUN}-step${step}"
  ln -sfn "${src}" "${model}"
  echo "[eval] ===== ${RUN}-step${step} ====="
  python3 scripts/evaluate_checkpoints.py \
    --checkpoint "${model}" --val-parquet "${REAL}/${VAL_PARQUET}" \
    --n-eval ${N_EVAL} --workers "${BEQ_WORKERS:-8}" --out "${out}" \
    || echo "[eval] EVAL FAILED ${RUN}-step${step}"
done
echo "[eval] run=${RUN} done"
