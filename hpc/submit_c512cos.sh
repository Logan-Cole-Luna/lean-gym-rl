#!/usr/bin/env bash
# The c512cos series: typecheck / gated / outcome at max_response_length=512 with
# a cosine LR decay. Everything else matches the c512 outcome arm byte for byte,
# so c512cos_outcome vs c512_outcome isolates the schedule and the other two
# pair against their locolib_proof_lr6 twins at the same reward.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

COMMON=(
  SERIES_TAG=locolib_proof_c512cos
  TRAIN_FILE=data_locolib/rl_proof_nl.parquet
  VAL_FILE=data_locolib/val_proof_nl.parquet
  BEST_SFT=data_locolib/best_sft_proof.txt
  ACTOR_LR=1e-6
  LR_WARMUP_RATIO=0.05
  LR_SCHEDULER_TYPE=cosine
  MIN_LR_RATIO=0.1
  MATHLIB_TAR=/scratch/logan03/mathlib4_v4.23_lake.tar
  MATHLIB_TAR_FLAT=1
  LEAN_INTERACT_CACHE_DIR=/scratch/logan03/ai4math_training_lean_interact_cache_v423
)

for ARM in typecheck gated outcome; do
  # Chunk 1 free-running, chunk 2 on afterany: a walltime kill is the expected
  # end of a chunk, and resume_mode=auto picks the schedule back up mid-curve.
  id1=$(bash /home/logan03/lean-gym-rl/hpc/submit.sh /home/logan03/lean-gym-rl/hpc/grpo.slurm \
          ARM="${ARM}" "${COMMON[@]}" | awk '{print $NF}')
  id2=$(bash /home/logan03/lean-gym-rl/hpc/submit.sh --dependency=afterany:"${id1}" \
          /home/logan03/lean-gym-rl/hpc/grpo.slurm ARM="${ARM}" "${COMMON[@]}" | awk '{print $NF}')
  echo "[c512cos] ${ARM}: chunk1=${id1} chunk2=${id2}"
done
