#!/usr/bin/env bash
# GRPO (full-parameter) | vLLM rollout | FSDP training | single RTX 5070 Ti (16GB)
# BEq+ RL PoC for Lean 4 autoformalization (Qwen2.5-Coder-0.5B-Instruct, Lean-Workbook).
#
# Select the reward function (the whole point of the PoC -- compare these two):
#   REWARD_FN_NAME=compute_score_typecheck_only  (ablation baseline: type-check-only, exploitable)
#   REWARD_FN_NAME=compute_score_composite        (default: type-check + BEq+ semantic reward)
#
# Adapted from repos/verl/examples/tuning/lora/run_qwen3_8b_fsdp.sh, scaled down to a
# single 16GB GPU. Two things this hardware could NOT run, ruled out empirically (see
# the plan doc / session transcript for the full debugging trail):
#   - internlm2-math-plus-1.8B: actor<->vLLM initial weight-sync handshake OOMs by
#     ~100-300MB on this 16GB card, independent of every training-loop tuning knob
#     (batch size, token budgets, LoRA rank, KL on/off) -- root cause is fixed
#     per-process overhead scaling with model size, resolved by using a smaller model.
#   - LoRA (any model): vLLM's set_lora crashes with `IndexError: tuple index out of
#     range` on this verl/vLLM version's packed QKV/gate-up layer weight transfer --
#     an unrelated integration bug, sidestepped by full-parameter training instead
#     (affordable once the smaller model freed up enough memory).
# LORA_RANK=0 (default below) means full-parameter; set it >0 to try LoRA again once
# that transfer bug is fixed upstream.

set -xeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate the project venv unless one is already active (hpc/cc_env.sh activates
# .venv_hpc before this runs on a cluster). Without this, invoking the script
# directly rather than via `make` silently falls back to the system python and
# dies with "No module named 'verl'".
if [ -z "${VIRTUAL_ENV:-}" ]; then
  for _venv in "$REPO_ROOT/.venv_hpc" "$REPO_ROOT/.venv"; do
    if [ -f "$_venv/bin/activate" ]; then
      # shellcheck disable=SC1091
      source "$_venv/bin/activate"
      break
    fi
  done
fi
python3 -c "import verl" 2>/dev/null || {
  echo "ERROR: 'verl' is not importable with $(command -v python3). Run 'make env'."
  exit 1
}

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-$REPO_ROOT/models/qwen2.5-coder-0.5b-instruct}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}

train_batch_size=${TRAIN_BATCH_SIZE:-16}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
max_prompt_length=${MAX_PROMPT_LENGTH:-768}
max_response_length=${MAX_RESPONSE_LENGTH:-128}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-1024}

actor_lr=${ACTOR_LR:-1e-5}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

lora_rank=${LORA_RANK:-0}
lora_alpha=${LORA_ALPHA:-16}

rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.3}
rollout_n=${ROLLOUT_N:-4}

total_epochs=${TOTAL_EPOCHS:-3}
save_freq=${SAVE_FREQ:--1}
test_freq=${TEST_FREQ:-5}

reward_fn_name=${REWARD_FN_NAME:-compute_score_composite}
reward_num_workers=${REWARD_NUM_WORKERS:-2}

# THE memory knob for this project. The reward runs inside each AgentLoopWorker,
# and each of those processes builds its own BEqPlusScorer -> its own Lean REPL
# with Mathlib resident (~4.3GB each). verl's default of 8 therefore costs ~34GB
# of host RAM in Lean alone, which is what exhausted a 59GB box (measured: 30+
# Lean processes, 56/59GB used). Note that `reward.num_workers` does NOT control
# this -- tuning that knob has no effect on the number of Lean servers.
# Generation is bottlenecked on the single vLLM server anyway, so extra
# agent-loop workers buy little here while costing a full Mathlib each.
agent_loop_workers=${AGENT_LOOP_WORKERS:-2}

project_name=${PROJECT_NAME:-beqplus_rl_poc}
experiment_name=${EXPERIMENT_NAME:-qwen25_coder_0_5b_${reward_fn_name}}
# ---- end user-adjustable ----
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files=$REPO_ROOT/data/train.parquet
    data.val_files=$REPO_ROOT/data/val.parquet
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.trust_remote_code=True
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.lora_rank=${lora_rank}
    actor_rollout_ref.model.lora_alpha=${lora_alpha}
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length))
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.agent.num_workers=${agent_loop_workers}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

REWARD=(
    reward.custom_reward_function.path=$REPO_ROOT/reward/reward_fn.py
    reward.custom_reward_function.name=${reward_fn_name}
    reward.num_workers=${reward_num_workers}
)

TRAINER=(
    # A rollout group whose samples all fail (Lean scoring error, generation
    # error, ...) leaves the sync replay buffer with nothing to materialise and
    # aborts the run with "no materializable trajectories". That is a real risk
    # here because the reward depends on an external Lean process. Refilling
    # replaces such groups instead of crashing.
    trainer.v1.sampler.sync_refill_failed_groups=True
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=1
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    # Each checkpoint is ~6.1GB (FSDP shards + optimizer state), so a 100-step
    # run saving every 10 steps writes 61GB. An earlier set of runs filled the
    # disk to 100% this way. Keep only the most recent few -- the merged HF
    # weights under checkpoints/merged/ (954MB) are what evaluation reads, and
    # those are produced explicitly by `make eval-ckpt`/`merge-sft`.
    trainer.max_actor_ckpt_to_keep=${MAX_CKPT_KEEP:-2}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

EXTRA=(
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
