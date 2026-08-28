#!/usr/bin/env bash
# Score one slice of the RL pool at k=8, so build_edge_pool has enough prompts
# to filter without heavy repetition during training.
#   sbatch --export=ALL,SLICE=1 hpc/score_pool.slurm
#
# Scoring the whole pool at k=8 fits no walltime, hence slices. They are disjoint
# by construction, so run them in parallel rather than chained.
#SBATCH --account=def-vganesh
#SBATCH --job-name=score_pool
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/score_pool_%j.out
#SBATCH --error=logs/score_pool_%j.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh

stage_mathlib "[pool]"

SLICE="${SLICE:?set SLICE=1|2}"
PARQ="${POOL_PARQUET:-${DATA_DIR}/pool_slice${SLICE}.parquet}"
[ -f "${PARQ}" ] || { echo "[pool] FATAL: ${PARQ} missing"; exit 1; }

CKPT=$(cat "${BEST_SFT:-${DATA_DIR}/best_sft.txt}" 2>/dev/null)
case "${CKPT}" in /*) ;; *) CKPT="$(cd "$(dirname "${CKPT}")" && pwd)/$(basename "${CKPT}")" ;; esac
[ -d "${CKPT}" ] || { echo "[pool] FATAL: checkpoint dir not found: ${CKPT}"; exit 1; }

rename_job pool "s${SLICE}"

TAG="${POOL_TAG:-pool_s${SLICE}}"
OUT="${DATA_DIR}/rollouts/${TAG}.jsonl"
SCORED="${DATA_DIR}/rollouts/${TAG}.scored.jsonl"
mkdir -p "${DATA_DIR}/rollouts"

# Must match the training sampler's temperature: this pass exists to predict
# which groups GRPO will find informative, and a different temperature samples a
# different distribution.
if [ ! -f "${OUT}" ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 scripts/pool/generate_rollouts.py --checkpoint "${CKPT}" \
    --parquet "${PARQ}" --k "${K:-8}" --temperature "${TEMPERATURE:-1.15}" --out "${OUT}" || exit 1
fi
python3 scripts/pool/score_rollouts.py --rollouts "${OUT}" --out "${SCORED}" --workers 8 || exit 1
echo "[pool] slice ${SLICE} done -> ${SCORED}"
