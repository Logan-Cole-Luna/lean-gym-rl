#!/usr/bin/env bash
# GRPO (full-parameter) | vLLM rollout | FSDP training | single RTX 5070 Ti (16GB)
# BEq+ RL PoC for Lean 4 autoformalization (Qwen2.5-Coder-0.5B-Instruct, Lean-Workbook).
#
# Select the reward function (see reward/reward_fn.py for the full argument):
#   REWARD_FN_NAME=compute_score_outcome         (DEFAULT: graded six-outcome ladder)
#   REWARD_FN_NAME=compute_score_gated           (BEq+ semantic signal only)
#   REWARD_FN_NAME=compute_score_typecheck  (ablation baseline: exploitable)

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
# DRY_RUN=1 prints the fully-resolved launch command and exits without touching
# the GPU. Worth using before every real submission: a Hydra typo otherwise
# surfaces minutes into a job, after Ray and vLLM have already started.
DRY_RUN=${DRY_RUN:-0}

python3 -c "import verl" 2>/dev/null || {
  if [ "$DRY_RUN" = "1" ]; then
    echo "WARNING: 'verl' is not importable here; continuing because DRY_RUN=1."
  else
    echo "ERROR: 'verl' is not importable with $(command -v python3). Run 'make env'."
    exit 1
  fi
}

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-$REPO_ROOT/models/qwen2.5-coder-0.5b-instruct}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}

train_batch_size=${TRAIN_BATCH_SIZE:-16}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
max_prompt_length=${MAX_PROMPT_LENGTH:-768}
max_response_length=${MAX_RESPONSE_LENGTH:-128}
# Tokens per training micro-batch under use_dynamic_bsz, for BOTH the actor's
# backward pass (train_batch) and compute_log_prob (actor + ref, forward-only).
#
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-896}
log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-896}
# Fused log-prob kernels (actor_rollout_ref.model.use_fused_kernels). OFF, and
# the default should stay OFF -- MEASURED, not assumed:
use_fused_kernels=${USE_FUSED_KERNELS:-False}
fused_kernels_backend=${FUSED_KERNELS_BACKEND:-torch}

# TODO hyperparam search
actor_lr=${ACTOR_LR:-1e-6}
lr_warmup_ratio=${LR_WARMUP_RATIO:-0.0}

# ---- the anti-drift settings ----
kl_loss_coef=${KL_LOSS_COEF:-0.01}
entropy_coeff=${ENTROPY_COEFF:-0}
entropy_chunking=${ENTROPY_CHUNKING:-True}
entropy_checkpointing=${ENTROPY_CHECKPOINTING:-True}
# GRPO advantage normalisation.
norm_adv_by_std=${NORM_ADV_BY_STD:-False}

lora_rank=${LORA_RANK:-0}
lora_alpha=${LORA_ALPHA:-16}

rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.3}
# Was 4.
rollout_n=${ROLLOUT_N:-8}
# THE exploration lever 
rollout_temperature=${ROLLOUT_TEMPERATURE:-1.15}

total_epochs=${TOTAL_EPOCHS:-3}
save_freq=${SAVE_FREQ:--1}
test_freq=${TEST_FREQ:-5}

reward_fn_name=${REWARD_FN_NAME:-compute_score_outcome}
reward_num_workers=${REWARD_NUM_WORKERS:-2}

filter_groups=${FILTER_GROUPS:-0}
filter_groups_metric=${FILTER_GROUPS_METRIC:-semantic_signal}
validation_data_dir=${VALIDATION_DATA_DIR:-}

agent_loop_workers=${AGENT_LOOP_WORKERS:-2}

export VERL_FILE_LOGGER_ROOT="${VERL_FILE_LOGGER_ROOT:-${REPO_ROOT}/results/train/train_metrics}"
project_name=${PROJECT_NAME:-beqplus_rl_poc}
experiment_name=${EXPERIMENT_NAME:-qwen25_coder_0_5b_${reward_fn_name}}
mkdir -p "${VERL_FILE_LOGGER_ROOT}/${project_name}"
export VERL_FILE_LOGGER_PATH="${VERL_FILE_LOGGER_PATH:-${VERL_FILE_LOGGER_ROOT}/${project_name}/${experiment_name}.${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}.jsonl}"
echo "[run_grpo] step metrics -> ${VERL_FILE_LOGGER_PATH}"
# ---- end user-adjustable ----

# Fail fast, before Ray/vLLM spin up
max_possible_seq_len=$((max_prompt_length + max_response_length))
if (( ppo_max_token_len_per_gpu < max_possible_seq_len )); then
  echo "ERROR: ppo_max_token_len_per_gpu (${ppo_max_token_len_per_gpu}) must be >= max_prompt_length + max_response_length (${max_possible_seq_len}), or the actor's backward pass will crash on any sequence longer than the budget." >&2
  exit 1
fi
if (( log_prob_max_token_len_per_gpu < max_possible_seq_len )); then
  echo "ERROR: log_prob_max_token_len_per_gpu (${log_prob_max_token_len_per_gpu}) must be >= max_prompt_length + max_response_length (${max_possible_seq_len}), or compute_log_prob will crash on any sequence longer than the budget." >&2
  exit 1
fi
if [ "${use_fused_kernels}" = "True" ] && [ "${fused_kernels_backend}" = "triton" ]; then
  echo "ERROR: FUSED_KERNELS_BACKEND=triton requires use_remove_padding=True (which needs flash-attn, not installed here); it returns 1-D log_probs that crash prepare_model_outputs. Use FUSED_KERNELS_BACKEND=torch." >&2
  exit 1
fi
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std}
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
    actor_rollout_ref.model.use_fused_kernels=${use_fused_kernels}
    actor_rollout_ref.model.fused_kernel_options.impl_backend=${fused_kernels_backend}
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${lr_warmup_ratio}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=${entropy_chunking}
    actor_rollout_ref.actor.entropy_checkpointing=${entropy_checkpointing}
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
    actor_rollout_ref.rollout.temperature=${rollout_temperature}
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu}
    actor_rollout_ref.rollout.agent.num_workers=${agent_loop_workers}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

REWARD=(
    reward.custom_reward_function.path=$REPO_ROOT/reward/reward_fn.py
    reward.custom_reward_function.name=${reward_fn_name}
    reward.num_workers=${reward_num_workers}
)

TRAINER=(
    trainer.v1.sampler.sync_refill_failed_groups=True
    trainer.balance_batch=True
    trainer.logger='["console","file"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=1
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.max_actor_ckpt_to_keep=${MAX_CKPT_KEEP:-2}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

EXTRA=(
)

if [ "${filter_groups}" != "0" ]; then
  EXTRA+=(
    algorithm.filter_groups.enable=True
    algorithm.filter_groups.metric=${filter_groups_metric}
    algorithm.filter_groups.max_inflight_gen_batches=${FILTER_MAX_INFLIGHT:-2}
  )
fi

if [ -n "${validation_data_dir}" ]; then
  mkdir -p "${validation_data_dir}"
  EXTRA+=( trainer.validation_data_dir="${validation_data_dir}" )
fi

########################### launch ###########################
if [ "$DRY_RUN" = "1" ]; then
  set +x
  echo ""
  echo "=== DRY RUN: reward=${reward_fn_name} rollout_n=${rollout_n} "\
"batch=${train_batch_size} kl=${kl_loss_coef} entropy=${entropy_coeff} "\
"norm_adv_by_std=${norm_adv_by_std} filter_groups=${filter_groups} ==="
  printf '%s\n' python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" "${MODEL[@]}" "${ACTOR[@]}" "${ROLLOUT[@]}" \
    "${REF[@]}" "${REWARD[@]}" "${TRAINER[@]}" "${EXTRA[@]}" "$@"
  exit 0
fi

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
