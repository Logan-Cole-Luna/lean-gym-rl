#!/usr/bin/env bash
# PHASE 0 MEASUREMENT: how much process-level signal is BEq+ already computing
# and throwing away on the dead band?
#
#   sbatch hpc/score_deadband.slurm
#   sbatch --export=ALL,POOL=deadband_sft3b-step93,WORKERS=12 hpc/score_deadband.slurm
#
# THE QUESTION. `compute_score_gated` scores ~32% of rollouts 0.0 -- statements
# that type-check but where no direction of BEq+ proves. Inside the cascade,
# rung 3 already runs `provable_without_have`: whether the PREDICTION proves on
# its own, with no reference to the gold. That is (a weaker form of) exactly the
# signal a dead-band ladder wants, and `score()` discarded it until now.
#
# It is only computed when rungs 1 and 2 both failed AND the `sorry` gate passed.
# How often that holds on the dead band decides whether a BEq+ AND self-prove
# ladder costs ~nothing extra or needs its own prove pass -- the record currently
# assumes the latter and prices the arm at "~20+ min/step, the dearest in the
# set". This job settles it before we spend an arm on the assumption.
#
# NO GPU. Nothing here generates; the rollouts already exist and scoring is pure
# Lean, i.e. CPU. Requesting no GPU means it does not queue behind or consume the
# allocation the training arms need. Use the bare `def-vganesh` account -- it
# auto-assigns to the _cpu / _gpu sub-account based on what the job requests, so
# naming the sub-account directly is unnecessary and gets it wrong when the
# request changes.
#SBATCH --account=def-vganesh
#SBATCH --job-name=score_deadband
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/score_deadband_%j.out
#SBATCH --error=logs/score_deadband_%j.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh

stage_mathlib "[deadband]"

POOL="${POOL:-deadband_sft3b-step93}"
IN="data_3b/rollouts/${POOL}.jsonl"
OUT="data_3b/rollouts/${POOL}.scored.jsonl"
[ -f "${IN}" ] || { echo "[deadband] FATAL: ${IN} missing -- run scripts/make_deadband_subset.py"; exit 1; }

# 12 workers x BEQ_MEMORY_LIMIT_MB (8GB default, ~4.3GB of it resident Mathlib)
# is ~96GB against --mem=120G. Do not raise workers without raising mem.
#
# --result-timeout is not optional: multiprocessing.Pool hangs forever if a
# worker dies holding a task, which has cost this project ~9 GPU-hours and
# cancelled every afterok behind it. The dead band is the slow class (full
# cascade to exhaustion, up to ~18 Lean calls at 30s), so the timeout is
# generous.
python3 scripts/score_rollouts.py \
  --rollouts "${IN}" --out "${OUT}" \
  --workers "${WORKERS:-12}" --result-timeout "${RESULT_TIMEOUT:-900}" || exit 1

echo "[deadband] scored -> ${OUT}"
python3 scripts/report_deadband.py --scored "${OUT}"
