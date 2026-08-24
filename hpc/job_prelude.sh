#!/usr/bin/env bash
# Common prelude for every SLURM job in this repo.  Source it as the first thing
# after the #SBATCH block:
#
#     source /home/logan03/lean-gym-rl/hpc/job_prelude.sh    # then use $REAL
#     stage_mathlib "[tag]"                                  # only if you need Lean
#
# Each line below is load-bearing and each one records a failure:
#
#   unset ROCR_VISIBLE_DEVICES   Narval sets it alongside CUDA_VISIBLE_DEVICES and
#                                verl's worker init hard-errors on it.
#   cc_env.sh                    module stack + the venv on $SCRATCH. pyarrow comes
#                                from the arrow/23.0.1 MODULE, not pip, so anything
#                                run without this wrongly reports it missing.
#   PYTHONPATH=$REAL             reward/ and scripts/ are imported by absolute
#                                package path from inside Ray workers, whose cwd is
#                                not this directory.
#
# stage_mathlib() copies Mathlib to node-local NVMe: 41s vs 1984s off Lustre.
# Without it `import Mathlib` exceeds the Lean timeout on COMPUTE nodes too, not
# just login nodes -- this is not a login-node-only problem.

set -uo pipefail
export PROJECT_ROOT=/home/logan03/lean-gym-rl
REAL="${PROJECT_ROOT}"
source "${REAL}/hpc/cc_env.sh"
unset ROCR_VISIBLE_DEVICES
cd "${REAL}" || exit 1
export PYTHONPATH="${REAL}:${PYTHONPATH:-}"

MATHLIB_TAR="${MATHLIB_TAR:-/scratch/logan03/mathlib4_v4.8.0-rc1.tar}"

stage_mathlib() {
  local tag="${1:-stage}"
  if [ -z "${SLURM_TMPDIR:-}" ] || [ ! -f "${MATHLIB_TAR}" ]; then
    echo "${tag} WARNING: no SLURM_TMPDIR or no ${MATHLIB_TAR}; Lean will run off"
    echo "${tag} shared storage and will very likely exceed BEQ_ENV_TIMEOUT."
    return 0
  fi
  local t0; t0=$(date +%s)
  tar -xf "${MATHLIB_TAR}" -C "${SLURM_TMPDIR}" || return 1
  cp -a "${LEAN_INTERACT_CACHE_DIR}" "${SLURM_TMPDIR}/lean_interact_cache" || return 1
  export MATHLIB_ROOT="${SLURM_TMPDIR}/mathlib4"
  export LEAN_INTERACT_CACHE_DIR="${SLURM_TMPDIR}/lean_interact_cache"
  echo "${tag} staging took $(( $(date +%s) - t0 ))s"
}

export BEQ_ENV_TIMEOUT="${BEQ_ENV_TIMEOUT:-2400}"
