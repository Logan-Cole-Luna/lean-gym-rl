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

# DEFAULTS TO v4.23, MATCHING DATA_DIR's LoCoLib default above. It used to
# default to the v4.8.0-rc1 tar, left over from the deleted Lean-Workbook line,
# and every job that forgot to override it scored LoCoLib against a Mathlib
# missing the modules LoCoLib's own golds import. That is not a hypothetical:
# `hpc/eval_sft.slurm` was run that way and `hpc/grpo_eval.slurm` was not, which
# manufactured an entire 15pp "RL gain" out of a toolchain mismatch -- see
# CLAUDE.md's trap list. The three settings move TOGETHER; overriding one alone
# is what the FLAT note below is about.
MATHLIB_TAR="${MATHLIB_TAR:-/scratch/logan03/mathlib4_v4.23_lake.tar}"
MATHLIB_TAR_FLAT="${MATHLIB_TAR_FLAT:-1}"
export LEAN_INTERACT_CACHE_DIR="${LEAN_INTERACT_CACHE_DIR:-/scratch/logan03/ai4math_training_lean_interact_cache_v423}"

# Copy Mathlib to node-local NVMe: 41s against 1984s off shared storage, without
# which `import Mathlib` exceeds BEQ_ENV_TIMEOUT on compute nodes as well.
#
# MATHLIB_TAR_FLAT: tars are not all laid out the same, and the flag must be set
# to match the tar or the failure is silent-ish and misleading.
# mathlib4_v4.23_lake.tar (the default, for LoCoLib) was built FROM INSIDE its
# directory and carries no `mathlib4/` prefix, so it needs FLAT=1 -- extracting
# it without leaves MATHLIB_ROOT pointing at an empty dir, which lean_interact
# reports as "Unable to determine Lean version" (no lean-toolchain found) rather
# than as a missing directory. mathlib4_v4.8.0-rc1.tar (the old Lean-Workbook
# toolchain) DOES wrap everything in `mathlib4/` and needs FLAT=0. So the two
# always move together, which is why they are defaulted together above:
#   v4.23  -> MATHLIB_TAR_FLAT=1, LEAN_INTERACT_CACHE_DIR=..._v423
#   v4.8   -> MATHLIB_TAR_FLAT=0, LEAN_INTERACT_CACHE_DIR=...cache
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
  # STAGE THE REPL CACHE WITH tar, NOT `cp -a`. `cp -a` implies --preserve=all,
  # which copies extended attributes; from Lustre to node-local disk that
  # intermittently fails with "preserving permissions ... Numerical result out
  # of range" (ERANGE on the xattr copy) even though every byte landed, and cp
  # still exits 1. Two rescore chunks died that way on 2026-09-09: stage_mathlib
  # returned before exporting MATHLIB_ROOT, and 14 of 46 files went unscored.
  # GNU tar does not store xattrs unless asked, and it preserves the modes the
  # REPL build actually needs.
  local cache_dst="${SLURM_TMPDIR}/lean_interact_cache"
  rm -rf "${cache_dst}"; mkdir -p "${cache_dst}"
  if ! tar -cf - -C "${LEAN_INTERACT_CACHE_DIR}" . \
       | tar -xf - -C "${cache_dst}"; then
    echo "${tag} FATAL: could not stage ${LEAN_INTERACT_CACHE_DIR}"
    return 1
  fi
  # Verify the copy rather than trusting an exit status, since the failure this
  # replaces was a nonzero status over a complete tree. A short copy would hand
  # Lean a half-built REPL, which fails much later and much less legibly.
  local n_src n_dst
  n_src=$(find "${LEAN_INTERACT_CACHE_DIR}" -type f | wc -l)
  n_dst=$(find "${cache_dst}" -type f | wc -l)
  if [ "${n_src}" -ne "${n_dst}" ]; then
    echo "${tag} FATAL: staged REPL cache is short: ${n_dst}/${n_src} files"
    return 1
  fi
  export MATHLIB_ROOT="${SLURM_TMPDIR}/mathlib4"
  export LEAN_INTERACT_CACHE_DIR="${SLURM_TMPDIR}/lean_interact_cache"
  # PRINT THE TOOLCHAIN, not just the seconds. Two jobs scoring the same corpus
  # against different Mathlib versions is invisible in every other line of every
  # log, and it silently produced a 15pp phantom result once already. The only
  # tell in the old logs was the staging duration tracking the tar's size.
  echo "${tag} staged $(basename "${MATHLIB_TAR}") in $(( $(date +%s) - t0 ))s" \
       "-- toolchain $(cat "${MATHLIB_ROOT}/lean-toolchain" 2>/dev/null || echo UNKNOWN)"
}

export BEQ_ENV_TIMEOUT="${BEQ_ENV_TIMEOUT:-2400}"
