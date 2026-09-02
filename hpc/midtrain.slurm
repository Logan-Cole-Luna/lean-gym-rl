#!/usr/bin/env bash
# Mid-training: continued LM pre-training on Lean theorem statements, between the
# pretrained model and the task SFT.
#   sbatch hpc/midtrain.slurm
#   sbatch --export=ALL,TAG=locolib,MIDTRAIN_DIR=data_locolib/midtrain,SFT_SAVE_FREQ=21 hpc/midtrain.slurm
#
# RL can only re-weight what the policy already samples, so no reward can supply
# missing Mathlib vocabulary. This stage is the intervention that can.
#
# Starting from the instruct model rather than the base model is deliberate: it
# holds everything else fixed against the existing SFT baseline. Swapping the
# starting checkpoint at the same time would move two variables at once.
#
# A run of this invalidates the current baseline, so everything downstream of it
# is a new comparison series.
#SBATCH --account=def-vganesh
#SBATCH --job-name=midtrain
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=11:00:00
#SBATCH --output=logs/midtrain_%j.out
#SBATCH --error=logs/midtrain_%j.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh
mkdir -p logs

# TAG scopes the save dir, the experiment name and the validation report, so a
# second corpus cannot overwrite the first.
TAG="${TAG:-}"
SUFFIX="${TAG:+_${TAG}}"
MIDTRAIN_DIR="${MIDTRAIN_DIR:-${DATA_DIR}/midtrain}"
SAVE_PATH="${REAL}/checkpoints/${MIDTRAIN_CKPT_DIR:-midtrain}${SUFFIX}"
TRAIN="${REAL}/${MIDTRAIN_DIR}/train.parquet"
[ -f "${TRAIN}" ] || {
  echo "[midtrain] FATAL: ${TRAIN} missing."
  echo "[midtrain] Build it with scripts/data/prepare_midtrain_dataset.py."; exit 1; }

# Corpus gate. Many Mathlib declarations lean on `variable` lines and open
# namespaces, so they do not elaborate once lifted out of their file, while this
# task's outputs must. A low rate does not make the corpus useless -- it still
# teaches vocabulary -- but it does make any claim about statement SHAPE
# unsupportable, which is worth knowing before reading the downstream numbers.
# Needs Mathlib staged, hence a compute node.
REPORT="results/midtrain_corpus_validation${SUFFIX}.json"
if [ ! -f "${REPORT}" ]; then
  stage_mathlib "[midtrain]"
  python3 scripts/data/validate_midtrain_corpus.py \
    --sample "${MIDTRAIN_DIR}/sample.jsonl" --out "${REPORT}" --workers "${BEQ_WORKERS:-8}" \
    || echo "[midtrain] corpus validation failed -- continuing, but read the log"
fi

rename_job midtrain "${TAG}"
echo "[midtrain] model=${MODEL:-${BASE_MODEL}}"
echo "[midtrain] corpus=${MIDTRAIN_DIR}  saving to ${SAVE_PATH}"

# Plain next-token LM. Each row is a single assistant-role message holding one
# statement: verl's MultiTurnSFTDataset masks only the generation-prompt prefix,
# so every content token is trained. That reuses the SFT stack for continued
# pre-training, and puts the Lean knowledge inside the turn structure SFT and
# GRPO later use.
#
# LR 2e-5 sits below SFT's 5e-5: this stage should move the representation, not
# overwrite it. MAX_LENGTH 512 because statements are capped well under that, and
# short sequences pack more per micro-batch at MAX_TOKEN_LEN 4096.
#
# Two GPUs and 240G for the same reasons as hpc/sft.slurm: FSDP is NO_SHARD at
# world size 1, and each hf_model save gathers a full state dict on the host.
#
# Hardware profile: 2x A100-40GB. On an 80GB card raise MICRO_BATCH and
# MAX_TOKEN_LEN; see README "Hardware".
MODEL_PATH="${MODEL:-${BASE_MODEL}}" \
SAVE_PATH="${SAVE_PATH}" \
TRAIN_FILE="${TRAIN}" \
VAL_FILE="${REAL}/${MIDTRAIN_DIR}/val.parquet" \
MICRO_BATCH="${MICRO_BATCH:-2}" \
MAX_TOKEN_LEN="${MAX_TOKEN_LEN:-4096}" \
MAX_LENGTH="${MAX_LENGTH:-512}" \
GRAD_CKPT="${GRAD_CKPT:-True}" \
LR="${LR:-2e-5}" \
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}" \
SFT_SAVE_FREQ="${SFT_SAVE_FREQ:-60}" \
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-4}" \
  NPROC="${NPROC:-2}" bash scripts/train/run_sft.sh \
    'checkpoint.save_contents=[model,optimizer,extra,hf_model]' \
    trainer.resume_mode=auto \
    trainer.experiment_name="midtrain${SUFFIX}" \
    engine.param_offload=True \
    engine.optimizer_offload=True

echo "[midtrain] done"
ls -d "${SAVE_PATH}"/global_step_* 2>/dev/null
echo "[midtrain] next: SFT from the last checkpoint --"
echo "  sbatch --export=ALL,MODEL=auto:${SAVE_PATH},TAG=mt${SUFFIX} hpc/sft.slurm"
