#!/usr/bin/env bash
# Does raising k rescue the prompts the policy never solves?
#
# 47% of the RL pool produced 0/8 BEq+. If those are beyond the model's reach it
# is a CAPABILITY limit and scaling is the answer; if they are merely rarely
# reached it is an EXPLORATION limit and rollout_n/temperature is the answer.
# Extrapolation cannot decide this (a binomial fit assigns p=0 to every 0/8
# prompt, assuming the conclusion), so sample more on exactly those prompts.
#   sbatch hpc/starved_k32.slurm
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=starved_k32
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=07:00:00
#SBATCH --output=logs/starved_k32_%j.out
#SBATCH --error=logs/starved_k32_%j.err

set -uo pipefail
export PROJECT_ROOT=/scratch/logan03/lean-gym-rl-rl_reward
REAL=/home/logan03/lean-gym-rl
source "${REAL}/hpc/cc_env.sh"
unset ROCR_VISIBLE_DEVICES
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [ -n "${SLURM_TMPDIR:-}" ] && [ -f /scratch/logan03/mathlib4_v4.8.0-rc1.tar ]; then
  t0=$(date +%s)
  tar -xf /scratch/logan03/mathlib4_v4.8.0-rc1.tar -C "${SLURM_TMPDIR}"
  cp -a "${LEAN_INTERACT_CACHE_DIR}" "${SLURM_TMPDIR}/lean_interact_cache"
  export MATHLIB_ROOT="${SLURM_TMPDIR}/mathlib4"
  export LEAN_INTERACT_CACHE_DIR="${SLURM_TMPDIR}/lean_interact_cache"
  echo "[k32] staging took $(( $(date +%s) - t0 ))s"
fi
export BEQ_ENV_TIMEOUT="${BEQ_ENV_TIMEOUT:-2400}"

SUBSET="${SUBSET:-${REAL}/data/starved250.parquet}"
# The subset MUST be built by scripts/make_starved_subset.py, which joins on
# prompt TEXT. generate_rollouts.py dedupes and drops overlong prompts then
# re-indexes, so prompt_index is NOT a row number in data/train.parquet -- a
# positional join silently yields prompts that were never starved (observed:
# "starved" prompts scoring 32/32, impossible for something that went 0/8).
K="${K:-32}"
TAG="starved250_k${K}"
OUT="${REAL}/data/rollouts/${TAG}.jsonl"

# 1. Generate. Same temperature as training/the k=8 pass, or the comparison is
#    against a different sampling distribution and means nothing.
if [ ! -f "${OUT}" ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 scripts/generate_rollouts.py \
    --checkpoint "${REAL}/checkpoints/merged/sft-step390" \
    --parquet "${SUBSET}" --k "${K}" --temperature 1.15 --out "${OUT}" || exit 1
else
  echo "[k32] rollouts already present, skipping generation"
fi

# 2. Score. Append-only and resumable, so a walltime kill loses nothing.
python3 scripts/score_rollouts.py \
  --rollouts "${OUT}" --out "${REAL}/data/rollouts/${TAG}.scored.jsonl" --workers 8

# 3. The answer: what fraction of never-solved prompts now succeed at least once?
python3 - "${REAL}/data/rollouts/${TAG}.scored.jsonl" "${K}" <<'PY'
import json, sys, collections
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
K=int(sys.argv[2])
by=collections.defaultdict(list)
for r in rows: by[r["prompt_index"]].append(r)
solved=sum(1 for g in by.values() if any(x["beq_plus"] for x in g))
tc=sum(1 for g in by.values() if any(x["typecheck"] for x in g))
n=len(by)
print(f"\n=== k={K} on prompts that were 0/8 at k=8 ===")
print(f"prompts: {n}   rollouts: {len(rows)}")
print(f"now solved at least once (BEq+): {solved}/{n} = {100*solved/n:.1f}%")
print(f"  (type-check at least once:     {tc}/{n} = {100*tc/n:.1f}%)")
print("near 0% => CAPABILITY limit: scale the model.")
print("clearly >0% => EXPLORATION limit: raise rollout_n / temperature first.")
PY
