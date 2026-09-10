#!/usr/bin/env bash
# FL <-> FL proof-structure agreement against the GOLD proof, per checkpoint.
#
#   bash hpc/submit.sh hpc/alignment.slurm GEN=<gen_*.jsonl> N=120
#
# The OTHER faithfulness axis. `scripts/eval/faithfulness_eval.py` scores the
# Lean proof against the ENGLISH argument (NL -> FL, judged by a model);
# this scores it against the reference LEAN proof's intermediate goal states
# (FL <-> FL, no judge, no English). They answer different questions and neither
# substitutes for the other.
#
# NOT RUNNABLE ON miniF2F. miniF2F ships statements only -- every gold ends at
# `:= by` with an empty body -- so there is no reference proof to align against
# and `proof_alignment.py` exits 2 rather than scoring every row 0.0, which
# would read as a catastrophic policy failure instead of an undefined metric.
# Run this on a proof-carrying corpus (data_locolib/val_proof*.parquet).
#
# CPU ONLY, and SLOW: one pair is an n_pred x n_gold matrix of full BEq+
# cascades. Shards are separate processes because the scorer holds one Lean REPL
# with BEQ_MAX_CONCURRENT=1, so there is no in-process parallelism to have.
#SBATCH --account=def-vganesh
#SBATCH --job-name=alignment
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/alignment_%j.out
#SBATCH --error=logs/alignment_%j.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh
stage_mathlib "[align]"
if [ -z "${MATHLIB_ROOT:-}" ] || [ ! -f "${MATHLIB_ROOT}/lean-toolchain" ]; then
  echo "[align] FATAL: Mathlib was not staged; refusing to score against the"
  echo "[align] shared checkout, which is a different toolchain."
  exit 1
fi
echo "[align] toolchain=$(cat "${MATHLIB_ROOT}/lean-toolchain")"

# GENS takes a whitespace-separated LIST, scored in turn; GEN is the single-file
# spelling. Staging Mathlib is 90s and one alignment run is hours, so batching
# files into one job mostly buys queue slots rather than time.
# GENS_FILE is one path per line; see hpc/faith_eval.slurm for why a list goes
# through a file rather than through --export.
if [ -n "${GENS_FILE:-}" ]; then
  [ -f "${GENS_FILE}" ] || { echo "[align] FATAL: no such GENS_FILE: ${GENS_FILE}"; exit 1; }
  GENS="$(grep -v '^[[:space:]]*$' "${GENS_FILE}" | tr '\n' ' ')"
fi
GENS="${GENS:-${GEN:?set GEN=<gen_*.jsonl>, GENS=<list>, or GENS_FILE=<file>}}"
for g in ${GENS}; do
  [ -f "${g}" ] || { echo "[align] FATAL: no such gen file: ${g}"; exit 1; }
done

N="${N:-120}"
SEED="${SEED:-0}"
SHARDS="${SHARDS:-6}"

_first="$(basename "$(echo ${GENS} | awk '{print $1}')" .jsonl)"; _first="${_first#gen_}"
rename_job align "${_first}"
echo "[align] n=${N} seed=${SEED} shards=${SHARDS}"
echo "[align] $(echo ${GENS} | wc -w) gen file(s) to score"

# --n with a fixed --seed samples the SAME rows for every checkpoint, so two
# alignment numbers are paired rather than merely both being averages of 120
# arbitrary proofs.
for GEN in ${GENS}; do
  LABEL="$(basename "${GEN}" .jsonl)"; LABEL="${LABEL#gen_}"
  OUT="${REAL}/results/alignment/${LABEL}"
  mkdir -p "${OUT}"
  if [ -f "${OUT}/summary.json" ]; then
    echo "[align] cached: ${LABEL}"; continue
  fi
  echo "[align] ##### ${LABEL} -> ${OUT}"
  pids=()
  for sh in $(seq 0 $((SHARDS - 1))); do
    python3 "${REAL}/scripts/eval/proof_alignment.py" \
      --gen "${GEN}" --n "${N}" --seed "${SEED}" \
      --shard "${sh}" --num-shards "${SHARDS}" --out "${OUT}" &
    pids+=($!)
  done
  wait "${pids[@]}"
  python3 "${REAL}/scripts/eval/proof_alignment.py" --merge --out "${OUT}"
  echo "[align] ${LABEL} done -> ${OUT}/summary.json"
done
echo "[align] all gen files done"
