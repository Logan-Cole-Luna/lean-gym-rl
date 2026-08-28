#!/usr/bin/env bash
# pass@k for one checkpoint, on a pinned validation subset.
#   sbatch --export=ALL,CKPT_DIR=/abs/path,LABEL=<unique> hpc/passk.slurm
# CKPT_DIR defaults to the best SFT checkpoint.
#
# pass@1 rising while pass@k is flat is sharpening of a distribution the policy
# already had, not a capability gain, so arms are checked at both. k=32 is the
# largest that fits a 6h walltime and still separates the two for effects of the
# size seen here.
#
# The subset is PINNED. Every model measured this way must use the same rows or
# the curves are not pairable. Do not regenerate it.
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=passk
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/passk_%j.out
#SBATCH --error=logs/passk_%j.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh

stage_mathlib "[passk]"

# Read at runtime so this job can be chained behind the eval that writes it.
BEST_SFT="${BEST_SFT:-${DATA_DIR}/best_sft.txt}"
CKPT_DIR="${CKPT_DIR:-$(cat "${BEST_SFT}" 2>/dev/null)}"
[ -n "${CKPT_DIR}" ] || { echo "[passk] FATAL: ${BEST_SFT} missing or empty"; exit 1; }
case "${CKPT_DIR}" in /*) ;; *) CKPT_DIR="$(cd "$(dirname "${CKPT_DIR}")" && pwd)/$(basename "${CKPT_DIR}")" ;; esac
[ -d "${CKPT_DIR}" ] || { echo "[passk] FATAL: checkpoint dir not found: ${CKPT_DIR}"; exit 1; }

# LABEL names every output. It must not fall back to basename(CKPT_DIR): every
# checkpoint path ends in /huggingface, so two runs would share a rollout file,
# and generation is skipped when that file exists -- the second model would be
# scored on the first model's rollouts.
if [ -z "${LABEL:-}" ]; then
  # SFT writes <run>/global_step_<N>/huggingface, verl writes an extra actor/.
  _p="${CKPT_DIR%/}"
  _p="${_p%/huggingface}"
  _p="${_p%/actor}"
  _step=$(basename "${_p}")
  _run=$(basename "$(dirname "${_p}")")
  case "${_step}" in
    global_step_*) LABEL="${_run}-step${_step##global_step_}" ;;
    *)             LABEL="${_run}" ;;
  esac
fi
case "${LABEL}" in
  huggingface|""|global_step_*)
    echo "[passk] FATAL: LABEL='${LABEL}' would collide across models."
    echo "[passk] Pass LABEL=<something unique> explicitly."; exit 1 ;;
esac
echo "[passk] label=${LABEL}  ckpt=${CKPT_DIR}"

rename_job passk "${LABEL}"

K="${K:-32}"
PARQ="${PASSK_PARQUET:-${DATA_DIR}/passk_subset.parquet}"
[ -f "${PARQ}" ] || { echo "[passk] FATAL: ${PARQ} missing"; exit 1; }

OUT="${DATA_DIR}/rollouts/passk_${LABEL}_k${K}.jsonl"
SCORED="${DATA_DIR}/rollouts/passk_${LABEL}_k${K}.scored.jsonl"
mkdir -p "${DATA_DIR}/rollouts"

# pass@k is defined against a sampling distribution, so this must match the
# temperature every other rollout pass uses or the curves are not comparable.
if [ ! -f "${OUT}" ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 scripts/pool/generate_rollouts.py --checkpoint "${CKPT_DIR}" \
    --parquet "${PARQ}" --k "${K}" --temperature "${TEMPERATURE:-1.15}" --out "${OUT}" || exit 1
fi
python3 scripts/pool/score_rollouts.py --rollouts "${OUT}" --out "${SCORED}" --workers "${BEQ_WORKERS:-8}" || exit 1

# Both verdicts from the same rollouts. The compiling curve is free here, since
# score_rollouts already wrote `typecheck` beside `beq_plus`. The _tc suffix
# keeps it off the beq_plus record; evalio.load_passk filters on the `metric`
# field inside the file.
python3 scripts/eval/passk_report.py --scored "${SCORED}" --label "${LABEL}" \
  --out "results/passk_${LABEL}_k${K}.json" || exit 1
python3 scripts/eval/passk_report.py --scored "${SCORED}" --label "${LABEL}" \
  --metric typecheck --out "results/passk_${LABEL}_k${K}_tc.json" || exit 1
echo "[passk] ${LABEL} done"
