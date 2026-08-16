#!/usr/bin/env bash
# Two-phase curriculum, trained FROM SCRATCH.
#
#   Phase 1 ("pass@"):  base model  --[type-check reward]-->  learns to emit Lean
#                       that actually elaborates. Dense, cheap, easy signal.
#   Phase 2 ("BEq+"):   phase-1 ckpt --[semantic reward]-->   learns to emit Lean
#                       that means the right thing.
#
# Why two runs instead of switching the reward mid-run: verl does not pass the
# global step to `custom_reward_function` (extra_info carries only dataset fields
# + num_turns), so an in-run switch would need a call-counting hack that
# validation passes would corrupt. Chaining via a checkpoint handoff is both
# robust and the standard way curricula are implemented.
#
# The handoff point matters. Phase 1 is deliberately stopped at the MIDPOINT
# rather than run to convergence: by step ~26 the type-check-only objective
# saturates at 100% and the policy collapses onto trivially-true boilerplate
# (see results/typecheck_only_sample_generations.md) with entropy ~0.62. Handing
# off from a collapsed policy would give phase 2 nothing to explore with. Stopping
# at the midpoint keeps entropy and output diversity alive.
#
# Usage:
#   bash scripts/run_curriculum.sh                       # 15 + 15 steps, shaped phase 2
#   TOTAL_STEPS=40 PHASE2_REWARD=compute_score_composite bash scripts/run_curriculum.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate the project venv unless one is already active (on a cluster,
# hpc/cc_env.sh will have activated .venv_hpc before this runs). Without this,
# invoking the script directly rather than via `make` silently falls back to the
# system python and dies with "No module named 'verl'".
if [ -z "${VIRTUAL_ENV:-}" ]; then
  for _venv in "$REPO_ROOT/.venv_hpc" "$REPO_ROOT/.venv"; do
    if [ -f "$_venv/bin/activate" ]; then
      # shellcheck disable=SC1091
      source "$_venv/bin/activate"
      break
    fi
  done
fi
python3 -c "import verl" 2>/dev/null || {
  echo "ERROR: 'verl' is not importable with $(command -v python3)."
  echo "       Run 'make env' (or 'make env-hpc'), or activate the venv first."
  exit 1
}

TOTAL_STEPS=${TOTAL_STEPS:-30}
SWITCH_AT=${SWITCH_AT:-$((TOTAL_STEPS / 2))}   # steps of phase 1
PHASE2_STEPS=$((TOTAL_STEPS - SWITCH_AT))
PHASE2_REWARD=${PHASE2_REWARD:-compute_score_shaped}

P1_EXP=${P1_EXP:-curriculum_p1_typecheck}
P2_EXP=${P2_EXP:-curriculum_p2_${PHASE2_REWARD}}
PROJECT_NAME=${PROJECT_NAME:-beqplus_rl_poc}

CKPT_ROOT="$REPO_ROOT/checkpoints/$PROJECT_NAME"
P1_CKPT="$CKPT_ROOT/$P1_EXP/global_step_${SWITCH_AT}"
P1_MERGED="$REPO_ROOT/checkpoints/merged/${P1_EXP}-step${SWITCH_AT}"

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-$REPO_ROOT/models/.hf_cache}

echo "=============================================================="
echo " CURRICULUM (from scratch)"
echo "   phase 1: $SWITCH_AT steps   reward=compute_score_typecheck_only"
echo "   phase 2: $PHASE2_STEPS steps   reward=$PHASE2_REWARD"
echo "=============================================================="

# ---------------- Phase 1: pass@ / type-check ----------------
if [ -d "$P1_CKPT" ]; then
  echo "[phase1] $P1_CKPT already exists -- skipping phase 1."
else
  echo "[phase1] training from the base model with the type-check reward ..."
  REWARD_FN_NAME=compute_score_typecheck_only \
  EXPERIMENT_NAME="$P1_EXP" \
  PROJECT_NAME="$PROJECT_NAME" \
    bash configs/run_grpo.sh \
      trainer.total_training_steps="$SWITCH_AT" \
      trainer.save_freq="$SWITCH_AT" \
      trainer.test_freq=5
fi

test -d "$P1_CKPT" || { echo "ERROR: phase 1 produced no checkpoint at $P1_CKPT"; exit 1; }

# ---------------- Handoff: merge FSDP shards -> HF ----------------
if [ -d "$P1_MERGED" ]; then
  echo "[handoff] $P1_MERGED already exists -- skipping merge."
else
  echo "[handoff] merging phase-1 weights -> $P1_MERGED"
  python3 -m verl.model_merger merge --backend fsdp \
    --local_dir "$P1_CKPT/actor" --target_dir "$P1_MERGED"
fi

# ---------------- Phase 2: BEq+ semantics ----------------
echo "[phase2] warm-starting from phase 1, reward=$PHASE2_REWARD ..."
MODEL_PATH="$P1_MERGED" \
REWARD_FN_NAME="$PHASE2_REWARD" \
EXPERIMENT_NAME="$P2_EXP" \
PROJECT_NAME="$PROJECT_NAME" \
  bash configs/run_grpo.sh \
    trainer.total_training_steps="$PHASE2_STEPS" \
    trainer.save_freq="$PHASE2_STEPS" \
    trainer.test_freq=5

echo ""
echo "Curriculum complete."
echo "  phase 1 ckpt: $P1_CKPT"
echo "  phase 2 ckpt: $CKPT_ROOT/$P2_EXP/global_step_${PHASE2_STEPS}"
echo "Next: make evaluate  (or scripts/evaluate_checkpoints.py with those paths)"
