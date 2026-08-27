#!/usr/bin/env bash
# Does the FULL self-prove ladder fire where the cascade's cheap one did not?
#   sbatch hpc/probe_selfprove.slurm
#
# Phase 0 found provable_alone available on 99.7% of the dead band but TRUE on
# 0/598. Rung 3 only tries tauto + simp_all_arith!; `self_prove` also tries
# noncomm_ring and a bare `exact?`. This runs both the strong ladder on the same
# 600 predictions AND the same ladder on the GOLDS -- the control that decides
# whether the 0.15 rung is mismeasured or structurally empty for Lean-Workbook.
#
# No GPU: pure Lean. Bare def-vganesh auto-assigns the cpu sub-account.
#SBATCH --account=def-vganesh
#SBATCH --job-name=probe_selfprove
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/probe_selfprove_%j.out
#SBATCH --error=logs/probe_selfprove_%j.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh
stage_mathlib "[selfprove]"

POOL="${POOL:-deadband_sft3b-step93}"
IN="data_3b/rollouts/${POOL}.jsonl"
[ -f "${IN}" ] || { echo "[selfprove] FATAL: ${IN} missing"; exit 1; }

python3 scripts/probe_selfprove.py --rollouts "${IN}" \
  --out "data_3b/rollouts/${POOL}.selfprove.jsonl" --workers "${WORKERS:-12}" || exit 1

# The control: same ladder, reference statements.
python3 scripts/probe_selfprove.py --rollouts "${IN}" --golds \
  --out "data_3b/rollouts/${POOL}.selfprove_gold.jsonl" --workers "${WORKERS:-12}" || exit 1
