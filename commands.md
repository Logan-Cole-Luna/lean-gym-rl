### Train
make train-sft                                    # SFT from base
make train-sft INIT=checkpoints/merged/foo        # SFT warm-started
make train-rl                                     # RL from base
make train-rl INIT=checkpoints/merged/sft-step30  # RL warm-started

make train-sft                                    # SFT from base

make train-rl INIT=checkpoints/merged/sft-step30  # RL warm-started

## Eval

make eval-ckpt CKPT=checkpoints/merged/sft-step30 N_EVAL=400
make eval-ckpt CKPT=checkpoints/beqplus_rl_poc/rl_from_sft-step30_compute_score_guided N_EVAL=400
make compare



# 0. Reclaim disk — safe to run right now, it only touches raw non-latest steps
make prune-checkpoints            # dry run: shows ~55GB
make prune-checkpoints CONFIRM=1

# 1. Scaled SFT — 20,000 examples x 3 epochs = 234 steps (~40 min)
make train-sft TOTAL_EPOCHS=3 LR=5e-5 \
  SFT_EXTRA="optim.lr_scheduler_type=cosine optim.lr_warmup_steps_ratio=0.03" \
  2>&1 | tee logs/sft_scaled_$(date +%Y%m%d_%H%M%S).log
make merge-sft                    # -> checkpoints/merged/sft-step234

# 2. RL from that checkpoint — 100 steps x batch 8 = 800 prompts (~5h at 187s/step)
make train-rl INIT=checkpoints/merged/sft-step234 \
  REWARD_FN_NAME=compute_score_guided \
  TOTAL_STEPS=200 SAVE_FREQ=25 TEST_FREQ=25 \
  2>&1 | tee logs/rl_scaled_$(date +%Y%m%d_%H%M%S).log

# 3. Evaluate both on the clean 400-example val set
make eval-ckpt CKPT=checkpoints/merged/sft-step234 N_EVAL=400
make eval-ckpt CKPT=checkpoints/beqplus_rl_poc/rl_from_sft-step234_compute_score_guided N_EVAL=400
make compare




make train-sft TOTAL_EPOCHS=5 LR=5e-5 \
  SFT_EXTRA="optim.lr_scheduler_type=cosine optim.lr_warmup_steps_ratio=0.03" \
  2>&1 | tee logs/sft_scaled_$(date +%Y%m%d_%H%M%S).log && \
make merge-sft && \
SFT_STEP=$(cat checkpoints/sft/scratch/latest_checkpointed_iteration.txt) && \
make train-rl INIT=checkpoints/merged/sft-step${SFT_STEP} \
  REWARD_FN_NAME=compute_score_guided \
  TOTAL_STEPS=200 SAVE_FREQ=25 TEST_FREQ=25 \
  2>&1 | tee logs/rl_scaled_$(date +%Y%m%d_%H%M%S).log && \
make eval-ckpt CKPT=checkpoints/merged/sft-step${SFT_STEP} N_EVAL=400 && \
make eval-ckpt CKPT=checkpoints/beqplus_rl_poc/rl_from_sft-step${SFT_STEP}_compute_score_guided N_EVAL=400 && \
make compare



make train-rl INIT=checkpoints/merged/sft-step${SFT_STEP} \
  REWARD_FN_NAME=compute_score_guided \
  TOTAL_STEPS=50 SAVE_FREQ=25 TEST_FREQ=25 \
  2>&1 | tee logs/rl_scaled_$(date +%Y%m%d_%H%M%S).log && \
make eval-ckpt CKPT=checkpoints/beqplus_rl_poc/rl_from_sft-step${SFT_STEP}_compute_score_guided N_EVAL=400 && \
make compare


# Current Results:

cached result files used (2):
  - eval_rl_from_sft-step30_compute_score_guided-step30_n400.json
  - eval_sft-step30_n400.json
excluded as stale (pre chat-template fix): ablation_comparison.json

======================================================================================
checkpoint                                          type-check%    BEq+%      n  description
--------------------------------------------------------------------------------------
sft-step30                                            76.5%    34.0%   400  SFT on (informal -> gold formal)
rl_from_sft-step30_compute_score_guided-step30        84.2%    29.0%   400  
======================================================================================