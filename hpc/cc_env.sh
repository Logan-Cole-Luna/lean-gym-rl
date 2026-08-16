#!/bin/bash
# ============================================================
# ai4math_training — Compute Canada shared environment for SLURM jobs.
# Source this at the top of every job script:
#   source /path/to/ai4math_training/hpc/cc_env.sh
#
# Mirrors MixtureOfMathExperts/hpc/cc_env.sh's pattern/module set. Prerequisites:
#   1. Run  bash hpc/setup_cc.sh          once on the login node
#   2. Run  python hpc/prefetch_models.py once on the login node
#   3. Run  bash hpc/lean_cache_and_build.sh  once on the login node
#   4. Ensure ~/.hf_token exists (for gated/rate-limited HF access)
# ============================================================

# --- Modules ---
# GPU generation on the allocated node determines the right CUDA module — check
# `nvidia-smi` on a GPU node and adjust if this doesn't match (this project's local
# dev used CUDA 13.x for a Blackwell consumer card; CC GPU nodes are typically
# V100/A100/H100, for which the StdEnv/2023 cuda/12.2 module below is a reasonable
# starting point, same as MixtureOfMathExperts uses).
module --force purge
module load StdEnv/2023
module load cuda/12.2 cudnn/9.2.1.18
module load python/3.11.5
module load gcc arrow/23.0.1 opencv/4.13.0

# --- Project root ---
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$SCRATCH/ai4math_training}}"

# --- Activate the persistent CC venv ---
source "${SCRATCH}/ai4math_training_venv_hpc/bin/activate"

# --- HF offline mode (compute nodes have no internet) ---
DEFAULT_SHARED_HF_HOME="${HOME}/scratch/.cache/huggingface"
if [[ -d "${DEFAULT_SHARED_HF_HOME}/hub" ]]; then
	export HF_HOME="${HF_HOME:-${DEFAULT_SHARED_HF_HOME}}"
	export HF_HUB_CACHE="${HF_HUB_CACHE:-${DEFAULT_SHARED_HF_HOME}/hub}"
	export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${DEFAULT_SHARED_HF_HOME}/datasets}"
else
	export HF_HOME="${HF_HOME:-${SCRATCH}/ai4math_training_models/.hf_cache}"
	export HF_HUB_CACHE="${HF_HUB_CACHE:-${SCRATCH}/ai4math_training_models/.hf_cache/hub}"
	export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH}/ai4math_training_models/.hf_cache/datasets}"
fi
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export HF_TOKEN=$(cat "$HOME/.hf_token" 2>/dev/null || true)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# --- lean-interact REPL cache (built once on the login node by
#     hpc/lean_cache_and_build.sh; must not need to rebuild on a compute node,
#     which typically has no internet to fetch/build a REPL binary from scratch).
export LEAN_INTERACT_CACHE_DIR="${LEAN_INTERACT_CACHE_DIR:-${PROJECT_ROOT}/.lean_interact_cache}"

# --- Elan / lake on PATH ---
export PATH="$HOME/.elan/bin:$PATH"

# --- CUDA allocator fix used throughout local dev (see README's Hardware notes) ---
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- Move into project directory ---
cd "${PROJECT_ROOT}"

echo "[cc_env] Project root: ${PROJECT_ROOT}"
echo "[cc_env] HF_HUB_CACHE: ${HF_HUB_CACHE}"
echo "[cc_env] Python: $(which python) ($(python --version 2>&1))"
echo "[cc_env] CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'unknown')"
