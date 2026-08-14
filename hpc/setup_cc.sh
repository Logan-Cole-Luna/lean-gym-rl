#!/bin/bash
# ============================================================
# ai4math_training — One-time Compute Canada setup (run on login node)
#
# Mirrors MixtureOfMathExperts/hpc/setup_cc.sh's structure/phases, adapted for
# this project's stack (verl + vLLM + lean-interact instead of xlora/trl).
#
# Usage:
#   cd $SCRATCH/ai4math_training
#   bash hpc/setup_cc.sh
#
# What it does:
#   1. Loads CC modules (same as SLURM jobs will use, via hpc/cc_env.sh)
#   2. Creates a persistent venv at .venv_hpc/
#   3. Installs CC-wheelhouse packages (--no-index, fast, no internet needed)
#   4. Clones verl (if not already present) and installs verl[vllm] + the
#      remaining PyPI-only deps (needs internet — login node only)
#   5. Installs lean-interact + patches the flashinfer bug (see README)
#   6. Creates HF cache directory
#   7. Verifies key imports
#
# After this, run (still on the login node):
#   bash hpc/lean_cache_and_build.sh   # build Mathlib4
#   python hpc/prefetch_models.py      # download the model + dataset
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv_hpc"

echo "=========================================="
echo "  ai4math_training — Compute Canada Setup"
echo "  Project: ${PROJECT_ROOT}"
echo "  Venv:    ${VENV_DIR}"
echo "=========================================="

# --- Load CC modules (must match hpc/cc_env.sh) ---
echo "[setup] Loading CC modules..."
module --force purge
module load StdEnv/2023
module load cuda/12.2 cudnn/9.2.1.18
module load python/3.11.5
module load gcc

echo "[setup] Python: $(which python3) ($(python3 --version))"

# --- Create virtual environment ---
if [[ -d "${VENV_DIR}" ]]; then
    echo "[setup] Venv already exists at ${VENV_DIR}, reusing."
else
    echo "[setup] Creating virtual environment (with system site-packages)..."
    python3 -m venv --system-site-packages "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

echo "[setup] Upgrading pip..."
pip install --no-index --upgrade pip 2>/dev/null || pip install --upgrade pip

# =============================================================
# Phase 1: CC wheelhouse packages (fast, no internet needed)
# =============================================================
echo ""
echo "[setup] Installing CC wheelhouse packages (--no-index)..."
pip install --no-index \
    numpy \
    scipy \
    pandas \
    torch \
    transformers \
    datasets \
    peft \
    accelerate \
    huggingface_hub \
    safetensors \
    tokenizers \
    sentencepiece \
    loguru \
    tqdm \
    pyyaml \
    rich \
    toml \
    filelock \
    gitpython \
    python-dotenv \
    wandb \
    scikit-learn \
    psutil \
    networkx \
    pyarrow \
    fastapi \
    uvicorn \
    || echo "[setup] Note: some wheelhouse packages unavailable, will fall back to PyPI below."

echo "[setup] CC wheelhouse packages installed (where available)."

# =============================================================
# Phase 2: verl + vLLM + PyPI-only packages (needs internet — login node only)
# =============================================================
echo ""
echo "[setup] Cloning verl (if not already present)..."
if [[ ! -d "${PROJECT_ROOT}/repos/verl" ]]; then
    git clone --depth 1 https://github.com/volcengine/verl.git "${PROJECT_ROOT}/repos/verl"
fi

echo "[setup] Installing verl[vllm] from ${PROJECT_ROOT}/repos/verl (editable)..."
(cd "${PROJECT_ROOT}/repos/verl" && pip install -e ".[vllm]")
pip install "TransferQueue==0.1.8"

echo "[setup] Installing lean-interact..."
pip install "lean-interact==0.11.5"

# =============================================================
# Phase 3: patch the flashinfer bug (see README's Hardware notes)
# =============================================================
FLASHINFER_FILE="${VENV_DIR}/lib/python3.11/site-packages/flashinfer/comm/fd_exchange.py"
if [[ -f "${FLASHINFER_FILE}" ]] && grep -q 'array.array\[int\]' "${FLASHINFER_FILE}"; then
    sed -i 's/-> tuple\[tuple\[int, int, array.array\[int\]\]\]:/-> "tuple[tuple[int, int, array.array]]":/' "${FLASHINFER_FILE}"
    echo "[setup] patched flashinfer's broken type annotation"
fi

# =============================================================
# Phase 4: Create HF cache directory
# =============================================================
HF_CACHE="${PROJECT_ROOT}/models/.hf_cache"
mkdir -p "${HF_CACHE}"
mkdir -p "${PROJECT_ROOT}/logs"
echo "[setup] HF cache directory: ${HF_CACHE}"

# =============================================================
# Phase 5: Verify key imports
# =============================================================
echo ""
echo "[setup] Verifying imports..."
python -c "
import torch
print(f'  torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
import transformers
print(f'  transformers {transformers.__version__}')
import datasets
print(f'  datasets {datasets.__version__}')
import verl
print(f'  verl OK')
import vllm
print(f'  vllm {vllm.__version__}')
import lean_interact
print(f'  lean_interact OK')
print()
print('All imports OK!')
"

echo ""
echo "=========================================="
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. bash hpc/lean_cache_and_build.sh   # build Mathlib4"
echo "    2. python hpc/prefetch_models.py      # download model + dataset"
echo "    3. sbatch hpc/train.slurm --export=REWARD_FN_NAME=compute_score_composite"
echo "=========================================="
