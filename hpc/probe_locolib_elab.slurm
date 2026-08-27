#!/usr/bin/env bash
# GATE FOR LOCOLIB RL: do its golds elaborate under our Mathlib v4.8.0-rc1?
# LoCoLib targets Lean 4.23.0. If the golds do not elaborate here, BEq+ cannot
# score against them and RL on this corpus is impossible. SFT is unaffected.
#   sbatch hpc/probe_locolib_elab.slurm
# No GPU: pure Lean. Bare def-vganesh auto-assigns the cpu sub-account.
#SBATCH --account=def-vganesh
#SBATCH --job-name=locolib_elab
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/locolib_elab_%j.out
#SBATCH --error=logs/locolib_elab_%j.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh
stage_mathlib "[elab]"

python3 scripts/probe_locolib_elab.py \
  --parquet "${PARQUET:-data_locolib/rl.parquet}" \
  --out "results/locolib_elab.jsonl" \
  --n "${N:-400}" --workers "${WORKERS:-12}"
