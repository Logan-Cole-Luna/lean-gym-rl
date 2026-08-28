#!/usr/bin/env bash
# sbatch wrapper that names the job before it queues.
#
#   bash hpc/submit.sh hpc/grpo.slurm ARM=outcome DATA_DIR=data_locolib
#   bash hpc/submit.sh --dependency=afterany:123 hpc/grpo.slurm ARM=gated
#
# `#SBATCH --job-name` is fixed when the file is written and cannot see the
# knobs, and a job that renames itself at runtime keeps the placeholder name for
# its whole time in the queue -- which is exactly when you are reading squeue.
# So the name is computed here and passed to sbatch.
#
# Everything before the .slurm path is forwarded to sbatch verbatim; everything
# after it is KEY=VALUE and goes into --export=ALL,...
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SBATCH_ARGS=()
SCRIPT=""
EXPORTS=()
for arg in "$@"; do
  if [ -z "${SCRIPT}" ] && [ "${arg}" != "${arg%.slurm}" ]; then
    SCRIPT="${arg}"
  elif [ -z "${SCRIPT}" ]; then
    SBATCH_ARGS+=("${arg}")
  else
    EXPORTS+=("${arg}")
  fi
done
[ -n "${SCRIPT}" ] || { echo "usage: $0 [sbatch args] hpc/<job>.slurm [KEY=VALUE ...]"; exit 1; }

# Read the knobs out of the KEY=VALUE list so the name matches what the job will
# actually see, without sourcing job_prelude.sh (which would need a cluster).
get() { local k="$1" d="${2:-}"; for e in ${EXPORTS[@]+"${EXPORTS[@]}"}; do
          [ "${e%%=*}" = "$k" ] && { echo "${e#*=}"; return; }; done; echo "$d"; }

DATA_DIR="$(get DATA_DIR data_3b)"
RUN_PREFIX="$(get RUN_PREFIX rl3b)"
DATA_TAG="$(get DATA_TAG "${DATA_DIR#data_}")"
MODEL_TAG="$(get MODEL_TAG "${RUN_PREFIX#rl}")"

TASK="$(basename "${SCRIPT}" .slurm)"
case "${TASK}" in
  grpo)       ARM="$(get ARM)" ;;
  grpo_eval)  ARM="$(get RUN)"; ARM="${ARM#${RUN_PREFIX}_}"; TASK=eval ;;
  eval_sft)   ARM="$(get TAG)"; TASK=evalsft ;;
  sft|midtrain) ARM="$(get TAG)" ;;
  score_pool) ARM="s$(get SLICE)"; TASK=pool ;;
  build_edge_pool) ARM=""; TASK=edge ;;
  passk)      ARM="$(get LABEL)" ;;
  *)          ARM="" ;;
esac

# Run names already carry the series, and the series usually names the corpus,
# so an arm taken from RUN/LABEL/TAG can repeat what the dataset half will say.
# Drop the duplication: "rl3b_loco_outcome" -> "outcome", not "loco_outcome".
ARM="${ARM#${RUN_PREFIX}_}"
ARM_HEAD="${ARM%%_*}"
if [ -n "${ARM_HEAD}" ] && [ "${DATA_TAG#${ARM_HEAD}}" != "${DATA_TAG}" ]; then
  ARM="${ARM#${ARM_HEAD}_}"
fi
[ "${ARM}" = "${DATA_TAG}" ] && ARM=""

# The default corpus is named after the model, so the tags collide there and the
# dataset half would say nothing twice.
SUFFIX=""
[ "${DATA_TAG}" != "${MODEL_TAG}" ] && SUFFIX="_${DATA_TAG}"
NAME="${TASK}_${MODEL_TAG}${ARM:+-${ARM}}${SUFFIX}"

EXPORT_ARG="ALL"
for e in ${EXPORTS[@]+"${EXPORTS[@]}"}; do EXPORT_ARG="${EXPORT_ARG},${e}"; done

echo "[submit] ${NAME}  <- ${SCRIPT}" >&2
exec sbatch --job-name="${NAME}" ${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"} \
     --export="${EXPORT_ARG}" "${SCRIPT}"
