#!/usr/bin/env bash
# Common prelude for every SLURM job. Source it first, after the #SBATCH block:
#
#     source /home/logan03/lean-gym-rl/hpc/job_prelude.sh    # then use $REAL
#     stage_mathlib "[tag]"                                  # only if you need Lean
#
# ROCR_VISIBLE_DEVICES must be unset: Narval sets it alongside
# CUDA_VISIBLE_DEVICES and verl's worker init rejects it. PYTHONPATH must point
# at the repo root because reward/ and scripts/ are imported by package path
# from Ray workers, whose cwd is elsewhere.

set -uo pipefail
export PROJECT_ROOT=/home/logan03/lean-gym-rl
REAL="${PROJECT_ROOT}"
source "${REAL}/hpc/cc_env.sh"
unset ROCR_VISIBLE_DEVICES
cd "${REAL}" || exit 1
export PYTHONPATH="${REAL}:${PYTHONPATH:-}"

# Corpus and run naming. Nothing below hardcodes a model or a size; override
# these to point the same jobs at another corpus or another series. Defaults
# point at LoCoLib's proof-pair task -- the only task variant this repo trains
# now (the signature-only Lean-Workbook line, and LoCoLib's own signature-only
# variant, were removed: a model that never writes a proof isn't doing
# autoformalization, just statement translation).
: "${DATA_DIR:=data_locolib}"           # splits, rollouts and pool parquets
: "${SFT_DIR:=sft_3b_locolib_proof}"    # checkpoints/<SFT_DIR><TAG>
: "${SFT_LABEL:=sft3blocolib_proof}"    # eval label and merged-checkpoint prefix
: "${RUN_PREFIX:=rl3b}"           # GRPO experiment name: <RUN_PREFIX>_<ARM>
: "${PROJECT_NAME:=beqplus_rl_poc}"
# The model SFT starts from. The only place a specific model is named.
: "${BASE_MODEL:=/scratch/logan03/ai4math_training_models/qwen2.5-coder-3b-instruct}"
# Short tags for job names. Derived from the knobs above so they follow a
# corpus or model switch automatically; override either directly.
: "${DATA_TAG:=${DATA_DIR#data_}}"     # data_locolib -> locolib
: "${MODEL_TAG:=${RUN_PREFIX#rl}}"     # rl3b         -> 3b
export DATA_DIR SFT_DIR SFT_LABEL RUN_PREFIX PROJECT_NAME BASE_MODEL DATA_TAG MODEL_TAG


# Fallback naming for a job submitted with plain sbatch. This runs at job START,
# so the job carries its placeholder name for its whole time in the queue --
# which is when you are reading squeue. Prefer hpc/submit.sh, which sets the
# name at submit time; this only catches direct sbatch calls.
rename_job() {
  local task="$1" arm="${2:-}"
  # The default corpus is named after the model, so the two tags collide there
  # and "sft_3b_3b" says nothing twice. Emit the dataset only when it differs.
  local suffix=""
  [ "${DATA_TAG}" != "${MODEL_TAG}" ] && suffix="_${DATA_TAG}"
  local name="${task}_${MODEL_TAG}${arm:+-${arm}}${suffix}"
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    scontrol update JobId="${SLURM_JOB_ID}" JobName="${name}" 2>/dev/null \
      && echo "[job] renamed to ${name}"
  fi
}

MATHLIB_TAR="${MATHLIB_TAR:-/scratch/logan03/mathlib4_v4.8.0-rc1.tar}"

# Copy Mathlib to node-local NVMe: 41s against 1984s off shared storage, without
# which `import Mathlib` exceeds BEQ_ENV_TIMEOUT on compute nodes as well.
#
# MATHLIB_TAR_FLAT: tars are not all laid out the same. The default
# mathlib4_v4.8.0-rc1.tar wraps everything in a `mathlib4/` prefix (built via
# `tar -cf ... mathlib4/` from its parent), so extracting into SLURM_TMPDIR
# directly lands MATHLIB_ROOT correctly. mathlib4_v4.23_lake.tar (a second
# toolchain, staged for the LoCoLib theorem+proof-pair task) was built FROM
# INSIDE its directory and has no such prefix -- extracting it the same way
# leaves MATHLIB_ROOT pointing at an empty dir, which lean_interact reports as
# "Unable to determine Lean version" (no lean-toolchain found), not as a
# missing-directory error. Set MATHLIB_TAR_FLAT=1 for a tar shaped that way;
# default 0 preserves the original, extensively-relied-upon behaviour exactly.
stage_mathlib() {
  local tag="${1:-stage}"
  if [ -z "${SLURM_TMPDIR:-}" ] || [ ! -f "${MATHLIB_TAR}" ]; then
    echo "${tag} WARNING: no SLURM_TMPDIR or no ${MATHLIB_TAR}; Lean will run off"
    echo "${tag} shared storage and will very likely exceed BEQ_ENV_TIMEOUT."
    return 0
  fi
  local t0; t0=$(date +%s)
  if [ "${MATHLIB_TAR_FLAT:-0}" = "1" ]; then
    mkdir -p "${SLURM_TMPDIR}/mathlib4"
    tar -xf "${MATHLIB_TAR}" -C "${SLURM_TMPDIR}/mathlib4" || return 1
  else
    tar -xf "${MATHLIB_TAR}" -C "${SLURM_TMPDIR}" || return 1
  fi
  cp -a "${LEAN_INTERACT_CACHE_DIR}" "${SLURM_TMPDIR}/lean_interact_cache" || return 1
  export MATHLIB_ROOT="${SLURM_TMPDIR}/mathlib4"
  export LEAN_INTERACT_CACHE_DIR="${SLURM_TMPDIR}/lean_interact_cache"
  echo "${tag} staging took $(( $(date +%s) - t0 ))s"
}

export BEQ_ENV_TIMEOUT="${BEQ_ENV_TIMEOUT:-2400}"
