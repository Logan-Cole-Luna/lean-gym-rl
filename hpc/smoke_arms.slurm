#!/usr/bin/env bash
# Pre-flight for the c512 reward arms. CPU only, minutes, no GPU.
#
#   bash hpc/submit.sh hpc/smoke_arms.slurm GEN=<gen_*.jsonl> N=24
#
# Run this BEFORE a training arm, not after. A reward that raises loses the JOB,
# not the rollout, and `@_never_raises` converts that into a silent stream of
# all-zero dicts instead -- which trains the policy on noise for a full walltime.
# See scripts/eval/smoke_new_arms.py for what it asserts.
#SBATCH --account=def-vganesh
#SBATCH --job-name=smoke_arms
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_arms_%j.out
#SBATCH --error=logs/smoke_arms_%j.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh
stage_mathlib "[smoke]"
export MATHLIB_ROOT="${MATHLIB_ROOT:-${REAL}/repos/mathlib4}"

GEN="${GEN:-${REAL}/results/eval/rl3b_locolib_proof_lr6_outcome/gen_rl3b_locolib_proof_lr6_outcome-step90_n760.jsonl}"
[ -f "${GEN}" ] || { echo "[smoke] FATAL: no such gen file: ${GEN}"; exit 1; }

rename_job smoke "arms"
python3 "${REAL}/scripts/eval/smoke_new_arms.py" \
  --gen "${GEN}" --n "${N:-24}" \
  --out "${REAL}/results/smoke_new_arms.json"
