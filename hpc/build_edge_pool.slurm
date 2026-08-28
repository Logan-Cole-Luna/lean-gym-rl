#!/usr/bin/env bash
# Build the edge-of-competence RL pool from every k=8 scoring pass available.
# Runs after hpc/score_pool.slurm; gates the gated_edge arm.
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=build_edge
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/build_edge_%j.out
#SBATCH --error=logs/build_edge_%j.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh

EDGE_TAGS="${EDGE_TAGS:-pool_s1 pool_s2}"
SRC_PARQUET="${SRC_PARQUET:-${DATA_DIR}/train.parquet}"
OUT_PARQUET="${OUT_PARQUET:-${DATA_DIR}/train_edge.parquet}"
MIN_POOL="${MIN_POOL:-700}"

rename_job edge

PAIRS=()
for t in ${EDGE_TAGS}; do
  S="${DATA_DIR}/rollouts/${t}.scored.jsonl"; R="${DATA_DIR}/rollouts/${t}.jsonl"
  if [ -f "$S" ] && [ -f "$R" ]; then PAIRS+=(--pair "${S}:${R}"); else echo "[edge] skipping ${t} (missing)"; fi
done
[ ${#PAIRS[@]} -gt 0 ] || { echo "[edge] FATAL: no scored files"; exit 1; }

python3 scripts/pool/make_difficulty_subset.py "${PAIRS[@]}" \
  --parquet "${SRC_PARQUET}" --out "${OUT_PARQUET}" \
  --min-wins 1 --max-wins 7 --batch-size "${BATCH_SIZE:-16}" \
  --total-steps "${EDGE_STEPS:-150}" || exit 1

# The pool must be large enough that the run is not many epochs over a handful
# of prompts, or memorisation confounds the comparison. Fail here rather than
# train a run whose result cannot be interpreted.
OUT_PARQUET="${OUT_PARQUET}" MIN_POOL="${MIN_POOL}" python3 - <<'PY' || exit 1
import os, sys, pandas as pd
n = len(pd.read_parquet(os.environ["OUT_PARQUET"]))
mn = int(os.environ["MIN_POOL"])
print(f"[edge] pool size {n}")
if n < mn:
    print(f"[edge] FATAL: {n} prompts is under the {mn} minimum.")
    print("[edge] Score another slice, or cut EDGE_STEPS.")
    sys.exit(1)
PY
echo "[edge] built ${OUT_PARQUET}"
