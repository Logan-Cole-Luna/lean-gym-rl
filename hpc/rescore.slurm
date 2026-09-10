#!/usr/bin/env bash
# Re-score already-generated completions under ONE named Lean environment.
#
#   bash hpc/submit.sh hpc/rescore.slurm GEN=<gen_*.jsonl> TAG=v423 \
#     MATHLIB_TAR=/scratch/logan03/mathlib4_v4.23_lake.tar MATHLIB_TAR_FLAT=1 \
#     LEAN_INTERACT_CACHE_DIR=/scratch/logan03/ai4math_training_lean_interact_cache_v423
#
# CPU ONLY. The completions already exist; this is Lean, not generation, so it
# needs no GPU and queues far faster than an eval.
#
# THE EXPERIMENT. Run this twice on the SAME gen file, once per Mathlib tar. The
# text is identical by construction, so any rate difference is the environment.
# See scripts/eval/rescore_gen.py for why that question is open.
#SBATCH --account=def-vganesh
#SBATCH --job-name=rescore
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/rescore_%j.out
#SBATCH --error=logs/rescore_%j.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh
stage_mathlib "[rescore]"
export MATHLIB_ROOT="${MATHLIB_ROOT:-${REAL}/repos/mathlib4}"

GEN="${GEN:?set GEN=<gen_*.jsonl>}"
[ -f "${GEN}" ] || { echo "[rescore] FATAL: no such file: ${GEN}"; exit 1; }
# TAG names the environment, not the model -- the whole point is that one gen
# file is scored under several, so the model is already in the filename.
TAG="${TAG:?set TAG=<environment name, e.g. v423 or v48>}"

LABEL="$(basename "${GEN}" .jsonl)"; LABEL="${LABEL#gen_}"
OUT="${REAL}/results/rescore/${LABEL}.${TAG}.json"
mkdir -p "$(dirname "${OUT}")"

rename_job rescore "${TAG}"
echo "[rescore] gen=${GEN}"
echo "[rescore] MATHLIB_TAR=${MATHLIB_TAR} FLAT=${MATHLIB_TAR_FLAT:-0}"
echo "[rescore] LEAN_INTERACT_CACHE_DIR=${LEAN_INTERACT_CACHE_DIR}"
[ -f "${MATHLIB_ROOT}/lean-toolchain" ] \
  && echo "[rescore] toolchain: $(cat "${MATHLIB_ROOT}/lean-toolchain")"

python3 "${REAL}/scripts/eval/rescore_gen.py" \
  --gen "${GEN}" --n "${N:-0}" --workers "${BEQ_WORKERS:-8}" --out "${OUT}"
