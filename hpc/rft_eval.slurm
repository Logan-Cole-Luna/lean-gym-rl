#!/usr/bin/env bash
# Evaluate one RFT arm's checkpoints on the fixed 400-example validation slice.
# Reports type-check% and BEq+% with per-example records, so results pair against
# the cached gated/placebo/SFT evals under McNemar.
#
# One arm per job: BEq+ scoring is serial inside evaluate_checkpoints.py (~40 min
# per 400-example checkpoint), so all nine in one job would be ~7h of wall-clock.
#   sbatch --export=ALL,ARM=beq rft_eval.slurm
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=rft_eval
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=logs/rft_eval_%j.out
#SBATCH --error=logs/rft_eval_%j.err

set -uo pipefail
# Run against the staged rl_reward branch: main's reward/beq_plus.py lacks the
# probe_stronger arg and the gold_implies_pred/semantic_signal fields that
# evaluate_checkpoints.py records.
export PROJECT_ROOT=/scratch/logan03/lean-gym-rl-rl_reward
REAL=/home/logan03/lean-gym-rl
source "${REAL}/hpc/cc_env.sh"

# Narval's SLURM exports ROCR_VISIBLE_DEVICES (AMD) alongside CUDA_VISIBLE_DEVICES;
# verl's Worker._setup_env_cuda_visible_devices hard-errors when both are set.
unset ROCR_VISIBLE_DEVICES
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
# Mathlib's .lake holds ~4200 olean files (~2GB). Reading them cold off Lustre
# blows lean_interact's `import Mathlib` timeout -- it failed at 600s on BOTH a
# login node and a compute node. Stage to node-local NVMe, where many-small-file
# reads are fast, and keep a raised timeout as a safety net.
if [ -n "${SLURM_TMPDIR:-}" ]; then
  echo "[eval] staging mathlib4 + REPL cache to ${SLURM_TMPDIR} ..."
  t0=$(date +%s)
  # mathlib4 is 36,700 files. Copying them individually off Lustre took 1984s
  # (~18 files/s -- metadata-bound). Extracting one sequential tar is far
  # faster; the archive is built once by hand on /scratch.
  if [ -f /scratch/logan03/mathlib4_v4.8.0-rc1.tar ]; then
    tar -xf /scratch/logan03/mathlib4_v4.8.0-rc1.tar -C "${SLURM_TMPDIR}"
  else
    cp -a "${REAL}/repos/mathlib4" "${SLURM_TMPDIR}/mathlib4"
  fi
  cp -a "${LEAN_INTERACT_CACHE_DIR}" "${SLURM_TMPDIR}/lean_interact_cache"
  echo "[eval] staging took $(( $(date +%s) - t0 ))s"
  export MATHLIB_ROOT="${SLURM_TMPDIR}/mathlib4"
  export LEAN_INTERACT_CACHE_DIR="${SLURM_TMPDIR}/lean_interact_cache"
else
  export MATHLIB_ROOT="${REAL}/repos/mathlib4"
fi
export BEQ_ENV_TIMEOUT="${BEQ_ENV_TIMEOUT:-2400}"
mkdir -p "${REAL}/logs" "${REAL}/checkpoints/merged"

N_EVAL=400
ARM="${ARM:?set ARM=beq|tc|tcnb via --export=ALL,ARM=...}"

# Highest step first: the 3-epoch checkpoint is the headline number.
steps=$(ls -d "${REAL}/checkpoints/rft/${ARM}"/global_step_* 2>/dev/null \
        | sed 's/.*global_step_//' | sort -rn)
echo "[eval] arm=${ARM} steps: ${steps:-NONE}"

for step in ${steps}; do
  src="${REAL}/checkpoints/rft/${ARM}/global_step_${step}"
  out="${REAL}/results/eval_rft_${ARM}-step${step}_n${N_EVAL}.json"
  if [ -f "$out" ]; then echo "[eval] cached: rft_${ARM}-step${step}"; continue; fi

  # evaluate_checkpoints.py labels a record by its DIRECTORY NAME, and
  # select_checkpoint.py merges records across files by that label -- pointing
  # straight at <ckpt>/huggingface would label every checkpoint "huggingface"
  # and collapse them into one. Give each a unique name.
  model="${REAL}/checkpoints/merged/rft_${ARM}-step${step}"
  if ls "${src}/huggingface"/*.safetensors >/dev/null 2>&1; then
    ln -sfn "${src}/huggingface" "${model}"
    echo "[eval] hf_model weights via ${model}"
  elif ! ls "${model}"/*.safetensors >/dev/null 2>&1; then
    # Fallback for checkpoints written before save_contents included hf_model.
    [ -e "${src}/huggingface" ] || ln -sfn "${REAL}/global_step_390/huggingface" "${src}/huggingface"
    echo "[eval] merging ${src} -> ${model}"
    python3 -m verl.model_merger merge --backend fsdp \
      --local_dir "${src}" --target_dir "${model}" \
      || { echo "[eval] MERGE FAILED rft_${ARM}-step${step}"; continue; }
  fi

  echo "[eval] ===== rft_${ARM}-step${step} ====="
  python3 scripts/evaluate_checkpoints.py \
    --checkpoint "${model}" \
    --val-parquet "${REAL}/data/val.parquet" \
    --n-eval ${N_EVAL} \
    --workers "${BEQ_WORKERS:-8}" \
    --out "${out}" || echo "[eval] EVAL FAILED rft_${ARM}-step${step}"
done

echo "[eval] arm=${ARM} done"
ls -la "${REAL}/results"/eval_rft_${ARM}-* 2>/dev/null || true
