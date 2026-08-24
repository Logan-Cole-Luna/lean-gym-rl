#!/usr/bin/env bash
# Type-check a mid-training corpus sample. Short job -- it exists separately so
# the corpus can be judged BEFORE committing GPU-hours to training on it.
#   sbatch --export=ALL,SAMPLE=data_3b/midtrain_v2/sample.jsonl,OUT=results/midtrain_v2_validation.json hpc/validate_corpus.slurm
#SBATCH --account=def-vganesh_gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --job-name=validate_corpus
#SBATCH --output=logs/validate_corpus_%j.out
#SBATCH --error=logs/validate_corpus_%j.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh

# Mathlib must be on node-local NVMe or `import Mathlib` blows the Lean timeout
# on compute nodes too, not just login nodes.
stage_mathlib "[validate]"

python3 scripts/validate_midtrain_corpus.py \
  --sample "${SAMPLE:?set SAMPLE=<sample.jsonl>}" \
  --out "${OUT:-results/midtrain_corpus_validation.json}" \
  --workers "${WORKERS:-8}"
