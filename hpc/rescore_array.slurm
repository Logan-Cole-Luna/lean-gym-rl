#!/usr/bin/env bash
# Re-score cached generations under ONE verified Lean toolchain, in CHUNKS.
#
#   sbatch --array=0-7 hpc/rescore_array.slurm
#
# ONE TASK PER CHUNK, NOT PER FILE. Each worker holds a ~4.3GB Mathlib REPL that
# takes minutes to build, and every task also stages a 5.4GB tar to node-local
# disk. Paying that per FILE meant 318 stagings and ~2500 REPL builds to score a
# median file of ~400 rows: the overhead dwarfed the work and saturated the
# queue. A chunk hands many files to one long-lived pool, so the fixed cost is
# paid 8 times instead of 318.
#
# CPU ONLY. The completions already exist; this is Lean, not generation.
#
# THE MANIFEST EXCLUDES results/eval/archive. Those are superseded
# Lean-Workbook-era runs that target v4.8.0-rc1; re-scoring them under v4.23
# would be the same category error this exercise exists to correct.
#SBATCH --account=def-vganesh
#SBATCH --job-name=rescore_chunk
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --output=logs/rescore_chunk_%A_%a.out
#SBATCH --error=logs/rescore_chunk_%A_%a.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh
stage_mathlib "[rescore]"

MANIFEST="${MANIFEST:-${REAL}/results/manifests/rescore_manifest.txt}"
TAG="${TAG:-v423}"
REQUIRE="${REQUIRE_TOOLCHAIN:-leanprover/lean4:v4.23.0}"
NCHUNKS="${NCHUNKS:-8}"

# NO FALLBACK TO THE SHARED CHECKOUT. `stage_mathlib` returns 0 even when it did
# nothing, and the usual `MATHLIB_ROOT:-${REAL}/repos/mathlib4` fallback then
# points at v4.8 -- which silently produced three outputs named `.v423.json`
# computed under v4.8 on the first attempt. A job whose entire purpose is to pin
# the toolchain must fail instead. `--require-toolchain` re-checks in Python.
if [ -z "${MATHLIB_ROOT:-}" ] || [ ! -f "${MATHLIB_ROOT}/lean-toolchain" ]; then
  echo "[rescore] FATAL: Mathlib was not staged (MATHLIB_ROOT=${MATHLIB_ROOT:-unset})."
  echo "[rescore] Refusing to fall back to the shared checkout, which is the wrong"
  echo "[rescore] toolchain and would mislabel every output as ${TAG}."
  exit 1
fi
echo "[rescore] chunk ${SLURM_ARRAY_TASK_ID}/${NCHUNKS} toolchain=$(cat "${MATHLIB_ROOT}/lean-toolchain")"

# STANDALONE=1 for a manifest whose predictions carry their own imports. The
# mode must match the ORIGINAL eval: scoring a self-contained snippet through
# the shared-header path strips its header and collapses the rate to ~0%, which
# reads as a catastrophic regression rather than as a scoring-mode mismatch.
# `scripts/eval/rebuild_evals.py` refuses to rebuild a record whose rescore was
# produced in the other mode.
STANDALONE_FLAG=""
[ "${STANDALONE:-0}" = "1" ] && STANDALONE_FLAG="--standalone"
echo "[rescore] scoring mode: ${STANDALONE_FLAG:-shared-header}"

python3 "${REAL}/scripts/eval/rescore_gen.py" \
  --manifest "${MANIFEST}" ${STANDALONE_FLAG} \
  --chunk "${SLURM_ARRAY_TASK_ID}" --num-chunks "${NCHUNKS}" \
  --out-dir "${REAL}/results/rescore" --tag "${TAG}" \
  --require-toolchain "${REQUIRE}" \
  --workers "${BEQ_WORKERS:-8}" --result-timeout "${RESULT_TIMEOUT:-900}"
