#!/usr/bin/env bash
# Score one slice of the 3B RL pool at k=8 so the difficulty filter has enough
# prompts to train on without heavy repetition.
#   sbatch --export=ALL,SLICE=1 hpc/score_pool_3b.slurm
#
# WHY SLICES. The geometry pass (rollouts_3b) scored 585 prompts, of which 267
# land in the 1-7/8 informative band. At batch 16 that is 16.7 steps/epoch, so a
# 150-step run would be 9 EPOCHS over 267 prompts -- memorisation would confound
# any gain the curation produced. Two 900-prompt slices bring the scored pool to
# ~2,385 and the edge pool to ~1,090, i.e. ~2.2 epochs. Acceptable.
#
# The full 4,300 pool at k=8 is ~34,400 BEq+ calls (~20h) and fits no walltime;
# 900 prompts is 7,200 calls at ~2.08s = ~4.2h. Slices are disjoint by
# construction (built by prompt text, excluding everything already scored) so
# the two jobs run in PARALLEL rather than chained.
#SBATCH --account=def-vganesh_gpu
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
PARQ="data_3b/pool_slice${SLICE}.parquet"
[ -f "${PARQ}" ] || { echo "[pool] FATAL: ${PARQ} missing"; exit 1; }

CKPT=$(cat data_3b/best_sft.txt 2>/dev/null)
case "${CKPT}" in /*) ;; *) CKPT="$(cd "$(dirname "${CKPT}")" && pwd)/$(basename "${CKPT}")" ;; esac
[ -d "${CKPT}" ] || { echo "[pool] FATAL: checkpoint dir not found: ${CKPT}"; exit 1; }

TAG="pool_s${SLICE}"
OUT="data_3b/rollouts/${TAG}.jsonl"
SCORED="data_3b/rollouts/${TAG}.scored.jsonl"
mkdir -p data_3b/rollouts

# temperature 1.15 == the training sampler. A different temperature would
# measure a different distribution than the one GRPO actually sees, and the
# whole point of this pass is to predict which groups will be informative.
if [ ! -f "${OUT}" ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 scripts/generate_rollouts.py --checkpoint "${CKPT}" \
    --parquet "${PARQ}" --k 8 --temperature 1.15 --out "${OUT}" || exit 1
fi
python3 scripts/score_rollouts.py --rollouts "${OUT}" --out "${SCORED}" --workers 8 || exit 1
echo "[pool] slice ${SLICE} done -> ${SCORED}"
