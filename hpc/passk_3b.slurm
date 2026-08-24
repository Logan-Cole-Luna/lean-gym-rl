#!/usr/bin/env bash
# pass@32 on a PINNED 250-prompt validation subset.
#   sbatch --export=ALL,CKPT_DIR=/abs/path,LABEL=sft3b-step93 hpc/passk_3b.slurm
# CKPT_DIR defaults to the best SFT checkpoint.
#
# WHY. The Interplay paper (arXiv:2512.07783) treats a pass@128 move as the only
# evidence of a true capability gain; anything visible at pass@1 but not at large
# k is sharpening of a distribution the policy already had. Our arms differ
# almost entirely in RETENTION -- 74% of the 0.5B BEq+ effect was protecting
# answers the policy already produced -- which is exactly that signature. We have
# never measured above k=8, so we cannot presently claim BEq+ produces a
# capability gain in their sense. This job is what would let us.
#
# k=32 not 128: 250 x 32 = 8,000 BEq+ calls at ~2.08s = ~4.6h, which fits. 128
# would be ~18h and fits nothing. k=32 still separates sharpening from capability
# for effects of the size we see.
#
# THE SUBSET IS PINNED (data_3b/passk_subset.parquet, seed 7, 250 of the 1,000
# val rows). Every model measured this way must use those exact rows or the
# curves are not pairable. Do not regenerate it.
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=passk_3b
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/passk_3b_%j.out
#SBATCH --error=logs/passk_3b_%j.err

set -uo pipefail
export PROJECT_ROOT=/home/logan03/lean-gym-rl
REAL="${PROJECT_ROOT}"
source "${REAL}/hpc/cc_env.sh"
unset ROCR_VISIBLE_DEVICES
cd "${REAL}"
export PYTHONPATH="${REAL}:${PYTHONPATH:-}"

if [ -n "${SLURM_TMPDIR:-}" ] && [ -f /scratch/logan03/mathlib4_v4.8.0-rc1.tar ]; then
  t0=$(date +%s)
  tar -xf /scratch/logan03/mathlib4_v4.8.0-rc1.tar -C "${SLURM_TMPDIR}"
  cp -a "${LEAN_INTERACT_CACHE_DIR}" "${SLURM_TMPDIR}/lean_interact_cache"
  export MATHLIB_ROOT="${SLURM_TMPDIR}/mathlib4"
  export LEAN_INTERACT_CACHE_DIR="${SLURM_TMPDIR}/lean_interact_cache"
  echo "[passk] staging took $(( $(date +%s) - t0 ))s"
fi
export BEQ_ENV_TIMEOUT="${BEQ_ENV_TIMEOUT:-2400}"

# BEST_SFT names the pointer file written by eval_3b_sft.slurm. Read at RUNTIME
# so this job can be chained behind the eval that creates it.
BEST_SFT="${BEST_SFT:-data_3b/best_sft.txt}"
CKPT_DIR="${CKPT_DIR:-$(cat "${BEST_SFT}" 2>/dev/null)}"
[ -n "${CKPT_DIR}" ] || { echo "[passk] FATAL: ${BEST_SFT} missing or empty"; exit 1; }
case "${CKPT_DIR}" in /*) ;; *) CKPT_DIR="$(cd "$(dirname "${CKPT_DIR}")" && pwd)/$(basename "${CKPT_DIR}")" ;; esac
[ -d "${CKPT_DIR}" ] || { echo "[passk] FATAL: checkpoint dir not found: ${CKPT_DIR}"; exit 1; }
# LABEL names every output of this job. It must NOT default to
# basename(CKPT_DIR): every checkpoint path ends in /huggingface, so the default
# was literally "huggingface" for every model. Two runs would then collide on
# data_3b/rollouts/passk_huggingface_k32.jsonl, and because generation is skipped
# when that file exists, the second model would silently be scored on the FIRST
# model's rollouts and reported as its own pass@k. Same class of trap as the
# eval-label collapse documented in CLAUDE.md.
if [ -z "${LABEL:-}" ]; then
  # <run>/global_step_<N>/huggingface -> <run>-step<N>
  _step=$(basename "$(dirname "${CKPT_DIR}")")          # global_step_93
  _run=$(basename "$(dirname "$(dirname "${CKPT_DIR}")")")
  case "${_step}" in
    global_step_*) LABEL="${_run}-step${_step##global_step_}" ;;
    *)             LABEL="${_run}" ;;
  esac
fi
case "${LABEL}" in
  huggingface|""|global_step_*)
    echo "[passk] FATAL: refusing to run with LABEL='${LABEL}' -- it would collide"
    echo "[passk] across models. Pass LABEL=<something unique> explicitly."; exit 1 ;;
esac
echo "[passk] label=${LABEL}  ckpt=${CKPT_DIR}"
K="${K:-32}"

PARQ=data_3b/passk_subset.parquet
[ -f "${PARQ}" ] || { echo "[passk] FATAL: ${PARQ} missing"; exit 1; }

OUT="data_3b/rollouts/passk_${LABEL}_k${K}.jsonl"
SCORED="data_3b/rollouts/passk_${LABEL}_k${K}.scored.jsonl"
mkdir -p data_3b/rollouts

# Temperature 1.15 matches training and every other rollout pass here. pass@k is
# defined against a sampling distribution, so a different temperature produces a
# different curve that cannot be compared with the rest.
if [ ! -f "${OUT}" ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 scripts/generate_rollouts.py --checkpoint "${CKPT_DIR}" \
    --parquet "${PARQ}" --k "${K}" --temperature 1.15 --out "${OUT}" || exit 1
fi
python3 scripts/score_rollouts.py --rollouts "${OUT}" --out "${SCORED}" --workers 8 || exit 1
python3 scripts/passk_report.py --scored "${SCORED}" --label "${LABEL}" \
  --out "results/passk_${LABEL}_k${K}.json" || exit 1
echo "[passk] ${LABEL} done"
