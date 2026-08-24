#!/usr/bin/env bash
# Build the edge-of-competence RL pool from every k=8 scoring pass available.
# Runs after hpc/score_pool_3b.slurm slices 1 and 2; gates the gated_edge arm.
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=build_edge
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/build_edge_%j.out
#SBATCH --error=logs/build_edge_%j.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh

PAIRS=()
for t in sft3b_k8 pool_s1 pool_s2; do
  S="data_3b/rollouts/${t}.scored.jsonl"; R="data_3b/rollouts/${t}.jsonl"
  if [ -f "$S" ] && [ -f "$R" ]; then PAIRS+=(--pair "${S}:${R}"); else echo "[edge] skipping ${t} (missing)"; fi
done
[ ${#PAIRS[@]} -gt 0 ] || { echo "[edge] FATAL: no scored files"; exit 1; }

python3 scripts/make_difficulty_subset.py "${PAIRS[@]}" \
  --parquet data_3b/train.parquet --out data_3b/train_edge.parquet \
  --min-wins 1 --max-wins 7 --batch-size 16 --total-steps 150 || exit 1

# GUARD. The whole point is a pool big enough that 150 steps is not many epochs
# over a handful of prompts. Under ~700 prompts that is >3.4 epochs and
# memorisation confounds the comparison -- fail here rather than train a run
# whose result cannot be interpreted.
python3 - <<'PY' || exit 1
import sys, pandas as pd
n = len(pd.read_parquet("data_3b/train_edge.parquet"))
print(f"[edge] pool size {n}")
if n < 700:
    print("[edge] FATAL: pool too small for 150 steps at batch 16 (>3.4 epochs).")
    print("[edge] Score another slice, or cut TOTAL_STEPS, before running gated_edge.")
    sys.exit(1)
PY
echo "[edge] built data_3b/train_edge.parquet"
