#!/bin/bash
# Downloads Mathlib4's prebuilt oleans cache on the login node, then builds.
# Mirrors MixtureOfMathExperts/hpc/lean_cache_and_build.sh's first two steps
# (this project doesn't need the separate lean_mathlib_env rsync step MoME
# does, since reward/beq_plus.py points lean-interact at repos/mathlib4
# directly rather than a wrapper Lake project).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${PROJECT_ROOT}/logs/lean_cache_login.log"
mkdir -p "${PROJECT_ROOT}/logs"

MATHLIB4_TAG="v4.8.0-rc1"   # keep in sync with Makefile's MATHLIB4_TAG

exec > >(tee "${LOG}") 2>&1

echo "=== Mathlib4 cache + build ==="
echo "Started: $(date)"

if command -v elan &>/dev/null; then
    echo "elan already installed: $(elan --version)"
else
    curl -sSf https://elan.lean-lang.org/elan-init.sh | sh -s -- -y
fi
export PATH="$HOME/.elan/bin:$PATH"

if [[ ! -d "${PROJECT_ROOT}/repos/mathlib4" ]]; then
    git clone --branch "${MATHLIB4_TAG}" --depth 1 \
      https://github.com/leanprover-community/mathlib4.git "${PROJECT_ROOT}/repos/mathlib4"
fi

cd "${PROJECT_ROOT}/repos/mathlib4"
echo ""
echo "=== Step 1: Download Mathlib4 oleans cache ==="
lake exe cache get || echo "WARNING: cache download failed (will build from source, much slower)"

echo ""
echo "=== Step 2: Build ==="
lake build

echo ""
echo "=== Step 3: Warm the lean-interact REPL cache ==="
# So the first SLURM job doesn't spend its (possibly offline) compute-node time
# building the REPL binary — do it here, on the login node, with internet.
cd "${PROJECT_ROOT}"
source .venv_hpc/bin/activate 2>/dev/null || true
export LEAN_INTERACT_CACHE_DIR="${PROJECT_ROOT}/.lean_interact_cache"
python3 scripts/test_lean_interact.py || echo "WARNING: lean-interact smoke test failed — check .venv_hpc setup"

echo ""
echo "=== Done ==="
echo "Finished: $(date)"
