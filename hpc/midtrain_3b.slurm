#!/usr/bin/env bash
# MID-TRAINING: continued LM pre-training on Mathlib theorem statements, inserted
# between the pretrained model and our task SFT.
#   sbatch hpc/midtrain_3b.slurm
#
# WHY (Interplay paper, arXiv:2512.07783): a mid-training stage beats spending
# the same compute on RL alone. Our own diagnosis agrees independently -- on
# prompts the policy NEVER solves, 84.3% still type-check, i.e. the model writes
# well-formed Lean that means the wrong thing. RL can only re-weight what the
# policy already samples, so no reward (BEq+ included) can supply missing Mathlib
# vocabulary. Consistent with 6x the parameters moving SFT BEq+ by only +0.6pp,
# and with SFT scaling being the only intervention that ever raised BEq+ at 0.5B.
#
# STARTS FROM THE INSTRUCT MODEL, DELIBERATELY. The design argument for using the
# BASE model is real (raw LM training degrades instruction following), but this
# job is an EXPERIMENT: to attribute anything to mid-training, everything else
# must be held fixed against the existing 39.4% baseline. Swapping the starting
# checkpoint at the same time would change two things at once. If the gate below
# passes, re-run the chain from Qwen2.5-Coder-3B (base) as the follow-up.
#
# THIS INVALIDATES THE CURRENT 3B BASELINE. Everything downstream of it is a new
# comparison series, exactly as the 0.5B -> 3B move was. Do not start it until
# the arm set now running has been read out.
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=midtrain_3b
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=11:00:00
#SBATCH --output=logs/midtrain_3b_%j.out
#SBATCH --error=logs/midtrain_3b_%j.err

set -uo pipefail
export PROJECT_ROOT=/home/logan03/lean-gym-rl
REAL="${PROJECT_ROOT}"
source "${REAL}/hpc/cc_env.sh"
unset ROCR_VISIBLE_DEVICES
cd "${REAL}"
export PYTHONPATH="${REAL}:${PYTHONPATH:-}"
mkdir -p logs

TRAIN="${REAL}/data_3b/midtrain/train.parquet"
[ -f "${TRAIN}" ] || { echo "[midtrain] FATAL: ${TRAIN} missing -- run scripts/prepare_midtrain_dataset.py"; exit 1; }

# ---- CORPUS QUALITY GATE ---------------------------------------------------
# Mathlib declarations lean on `variable` lines and open namespaces, so many are
# NOT self-contained once lifted out of their file -- and our task's outputs are.
# Elaborating all 108k is ~30h; a 500-statement sample bounds the proportion to
# about +/-4.4%. This needs Mathlib staged, which is why it runs here rather than
# on a login node. It is a REPORT, not a hard gate: a low rate does not make the
# corpus useless (it still teaches vocabulary), it makes the claim about
# statement SHAPE unsupportable, which is a thing we should know before reading
# the downstream numbers.
if [ ! -f results/midtrain_corpus_validation.json ]; then
  if [ -n "${SLURM_TMPDIR:-}" ] && [ -f /scratch/logan03/mathlib4_v4.8.0-rc1.tar ]; then
    t0=$(date +%s)
    tar -xf /scratch/logan03/mathlib4_v4.8.0-rc1.tar -C "${SLURM_TMPDIR}"
    cp -a "${LEAN_INTERACT_CACHE_DIR}" "${SLURM_TMPDIR}/lean_interact_cache"
    export MATHLIB_ROOT="${SLURM_TMPDIR}/mathlib4"
    export LEAN_INTERACT_CACHE_DIR="${SLURM_TMPDIR}/lean_interact_cache"
    echo "[midtrain] staging took $(( $(date +%s) - t0 ))s"
  fi
  BEQ_ENV_TIMEOUT=2400 python3 scripts/validate_midtrain_corpus.py \
    --sample data_3b/midtrain/sample.jsonl --workers 8 \
    || echo "[midtrain] corpus validation failed -- continuing, but read the log"
fi

# ---- the stage itself ------------------------------------------------------
# Plain next-token LM. Each row is a SINGLE assistant-role message holding one
# Mathlib statement: verl's MultiTurnSFTDataset sets loss_mask=1 across an
# assistant turn (masking only the generation-prompt prefix), so there is no
# prompt to condition on and every content token is trained. That is continued
# pre-training, reusing the SFT stack rather than standing up a second training
# path -- and it places the Lean knowledge inside the same turn structure that
# SFT and GRPO later use.
#
# LR 2e-5, below SFT's 5e-5: this stage should move the representation, not
# overwrite it. One epoch over ~59k statements.
#
# MAX_LENGTH 512 (not SFT's 1024) because statements are capped at 600 chars,
# and MAX_TOKEN_LEN 4096 because short sequences pack more per micro-batch.
#
# 2 GPUs and 240G for the same reasons as hpc/sft_3b.slurm -- at world size 1
# FSDP falls back to NO_SHARD and a 3B model OOMs a 40GB card; at 120G the
# hf_model checkpoint gather was OOM-killed by slurmstepd. Do not shrink either.
MODEL_PATH="${MODEL:-/scratch/logan03/ai4math_training_models/qwen2.5-coder-3b-instruct}" \
SAVE_PATH="${REAL}/checkpoints/midtrain_3b" \
TRAIN_FILE="${TRAIN}" \
VAL_FILE="${REAL}/data_3b/midtrain/val.parquet" \
MICRO_BATCH="${MICRO_BATCH:-2}" \
MAX_TOKEN_LEN="${MAX_TOKEN_LEN:-4096}" \
MAX_LENGTH="${MAX_LENGTH:-512}" \
LR="${LR:-2e-5}" \
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}" \
SFT_SAVE_FREQ="${SFT_SAVE_FREQ:-60}" \
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-4}" \
  NPROC=2 bash scripts/run_sft.sh \
    'checkpoint.save_contents=[model,optimizer,extra,hf_model]' \
    trainer.resume_mode=auto \
    trainer.experiment_name=qwen25_coder_3b_mathlib_midtrain \
    engine.param_offload=True \
    engine.optimizer_offload=True

echo "[midtrain] done"
ls -d "${REAL}/checkpoints/midtrain_3b"/global_step_* 2>/dev/null
echo "[midtrain] next: SFT from the last checkpoint, e.g."
echo "  L=\$(ls -d checkpoints/midtrain_3b/global_step_* | sort -t_ -k3 -n | tail -1)/huggingface"
echo "  sbatch --export=ALL,MODEL=\$(readlink -f \$L),TAG=mt hpc/sft_3b.slurm"
