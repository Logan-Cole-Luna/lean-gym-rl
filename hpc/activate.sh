#!/bin/bash
# ============================================================
# ai4math_training — activate the right Python environment for THIS machine.
#
# SOURCE this, do not execute it:
#     source hpc/activate.sh
#
# Why it exists: the Makefile's recipes used to hardcode
# `source $(PROJECT)/.venv/bin/activate`, which only exists on a local dev box.
# On Compute Canada the venv lives on $SCRATCH (project /project quota is
# file-count limited) and needs the module stack loaded first, so every one of
# those targets failed there with "No such file or directory" and then, more
# confusingly, "No module named 'verl'" from the system python.
#
# This is the module set from hpc/cc_env.sh, minus the things only a SLURM job
# needs (offline HF flags, Lean cache paths, cd into PROJECT_ROOT). Batch jobs
# should keep sourcing hpc/cc_env.sh; interactive `make` targets source this.
# ============================================================

if [ -d /cvmfs/soft.computecanada.ca ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module --force purge >/dev/null 2>&1 || true
    module load StdEnv/2023 >/dev/null 2>&1
    module load cuda/12.2 cudnn/9.2.1.18 >/dev/null 2>&1
    module load python/3.11.5 >/dev/null 2>&1
    module load gcc arrow/23.0.1 opencv/4.13.0 >/dev/null 2>&1
    # shellcheck disable=SC1091
    source "${SCRATCH}/ai4math_training_venv_hpc/bin/activate"
    export PATH="$HOME/.elan/bin:$PATH"
    export LEAN_INTERACT_CACHE_DIR="${LEAN_INTERACT_CACHE_DIR:-$(pwd)/.lean_interact_cache}"
else
    _repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    for _venv in "$_repo/.venv_hpc" "$_repo/.venv"; do
        if [ -f "$_venv/bin/activate" ]; then
            # shellcheck disable=SC1091
            source "$_venv/bin/activate"
            break
        fi
    done
    unset _repo _venv
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
