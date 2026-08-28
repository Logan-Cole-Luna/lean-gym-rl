#!/usr/bin/env bash
# Write BEST_FILE = the newest global_step_*/huggingface under CKPT_DIR.
# No epoch-selection eval (skips the usual "pick by validation BEq+, not last
# epoch" step from hpc/eval_sft.slurm) -- used to auto-chain GRPO behind an SFT
# job via --dependency=afterany, when the final step number isn't known at
# submit time and a full eval sweep isn't warranted for a short SFT run.
#   sbatch --dependency=afterany:<sft_job_id> --export=ALL,CKPT_DIR=...,BEST_FILE=... hpc/resolve_best_sft.slurm
#SBATCH --account=def-vganesh
#SBATCH --job-name=resolve_best_sft
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/resolve_best_sft_%j.out
#SBATCH --error=logs/resolve_best_sft_%j.err

set -euo pipefail
cd /home/logan03/lean-gym-rl

CKPT_DIR="${CKPT_DIR:?set CKPT_DIR=checkpoints/<sft dir>}"
BEST_FILE="${BEST_FILE:?set BEST_FILE=<path to write>}"

# NOT `sort -t_ -k3`: CKPT_DIR itself typically has underscores (e.g.
# sft_3b_locolib_proof), which throws off fixed-field-number sorting on the
# FULL path -- measured picking global_step_76 over global_step_114 out of
# {38,76,114}. Sort on the extracted step NUMBER instead.
LATEST=$(ls -d "${CKPT_DIR}"/global_step_*/huggingface 2>/dev/null \
         | awk -F'global_step_' '{print $2, $0}' | sort -k1 -n | tail -1 | cut -d' ' -f2-)
[ -n "${LATEST}" ] || { echo "[resolve] FATAL: no checkpoint under ${CKPT_DIR}"; exit 1; }
LATEST="$(readlink -f "${LATEST}")"
mkdir -p "$(dirname "${BEST_FILE}")"
echo "${LATEST}" > "${BEST_FILE}"
echo "[resolve] ${BEST_FILE} -> ${LATEST}"
