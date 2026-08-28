#!/usr/bin/env bash
# Does LoCoLib's gold elaborate under OUR pinned Mathlib (v4.8.0-rc1 vs
# LoCoLib's native 4.23.0)? CPU-only (no GPU, no policy) -- just Lean.
#
#   sbatch hpc/probe_locolib_elab.slurm                       # signature-only gate
#   sbatch --export=ALL,CHECK_PROOF=1 hpc/probe_locolib_elab.slurm   # proof-pair gate
#
# See scripts/misc/probe_locolib_elab.py's docstring for what context_ok /
# gold_ok / proved mean and why the gap between them is the actionable number.
#SBATCH --account=def-vganesh
#SBATCH --job-name=probe_locolib_elab
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --time=01:00:00
#SBATCH --output=logs/probe_locolib_elab_%j.out
#SBATCH --error=logs/probe_locolib_elab_%j.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh

stage_mathlib "[elab]"

echo "[elab] mathlib=${MATHLIB_ROOT} toolchain=$(cat "${MATHLIB_ROOT}/lean-toolchain" 2>/dev/null)"

N="${N:-400}"
WORKERS="${WORKERS:-8}"
PARQUET="${PARQUET:-data_locolib/rl.parquet}"

ARGS=(--parquet "${PARQUET}" --n "${N}" --workers "${WORKERS}")
if [ "${CHECK_PROOF:-0}" = "1" ]; then
  ARGS+=(--check-proof)
  rename_job elab proof
else
  rename_job elab sig
fi

python3 scripts/misc/probe_locolib_elab.py "${ARGS[@]}"
