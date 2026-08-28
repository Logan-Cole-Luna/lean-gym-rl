#!/bin/bash
# ============================================================
# ai4math_training — Python wrapper for login-node make targets on CC.
#
# Mirrors MixtureOfMathExperts/hpc/python_login.sh: loads the required CC
# modules BEFORE activating the venv, then exec's python with any arguments
# passed in. Used by the Makefile so that login-node targets like
# `make model` / `make dataset` work without manually loading modules first.
#
# Usage (via Makefile):
#   bash hpc/python_login.sh scripts/data/prepare_dataset.py
#   bash hpc/python_login.sh -c "import torch; print(torch.__version__)"
# ============================================================

set -euo pipefail

# Load CC modules only when the CVMFS software stack is present (i.e. we're
# actually on a Compute Canada / DRAC node, not some other machine).
if [ -d /cvmfs/soft.computecanada.ca ]; then
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module --force purge 2>/dev/null || true
    module load StdEnv/2023
    module load python/3.11.5
    module load gcc arrow/23.0.1 opencv/4.13.0
fi

source "${SCRATCH}/ai4math_training_venv_hpc/bin/activate"

exec python "$@"
