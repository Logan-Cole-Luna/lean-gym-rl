#!/usr/bin/env bash
# Type-check a mid-training corpus sample. Short job -- it exists separately so
# the corpus can be judged BEFORE committing GPU-hours to training on it.
#   sbatch --export=ALL,SAMPLE=data_3b/midtrain_v2/sample.jsonl,OUT=results/midtrain_v2_validation.json hpc/validate_corpus.slurm
#SBATCH --account=def-vganesh_gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --job-name=validate_corpus
#SBATCH --output=logs/validate_corpus_%j.out
#SBATCH --error=logs/validate_corpus_%j.err

set -uo pipefail
export PROJECT_ROOT=/home/logan03/lean-gym-rl
source "${PROJECT_ROOT}/hpc/cc_env.sh"
unset ROCR_VISIBLE_DEVICES
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Mathlib must be on node-local NVMe or `import Mathlib` blows the Lean timeout
# on compute nodes too, not just login nodes.
if [ -n "${SLURM_TMPDIR:-}" ] && [ -f /scratch/logan03/mathlib4_v4.8.0-rc1.tar ]; then
  t0=$(date +%s)
  tar -xf /scratch/logan03/mathlib4_v4.8.0-rc1.tar -C "${SLURM_TMPDIR}"
  cp -a "${LEAN_INTERACT_CACHE_DIR}" "${SLURM_TMPDIR}/lean_interact_cache"
  export MATHLIB_ROOT="${SLURM_TMPDIR}/mathlib4"
  export LEAN_INTERACT_CACHE_DIR="${SLURM_TMPDIR}/lean_interact_cache"
  echo "[validate] staging took $(( $(date +%s) - t0 ))s"
fi
export BEQ_ENV_TIMEOUT="${BEQ_ENV_TIMEOUT:-2400}"

python3 scripts/validate_midtrain_corpus.py \
  --sample "${SAMPLE:?set SAMPLE=<sample.jsonl>}" \
  --out "${OUT:-results/midtrain_corpus_validation.json}" \
  --workers "${WORKERS:-8}"
