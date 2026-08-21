#!/usr/bin/env bash
# GRPO (full-parameter) | vLLM rollout | FSDP training | single RTX 5070 Ti (16GB)
# BEq+ RL PoC for Lean 4 autoformalization (Qwen2.5-Coder-0.5B-Instruct, Lean-Workbook).
#
# Select the reward function (see reward/reward_fn.py for the full argument):
#   REWARD_FN_NAME=compute_score_gated           (DEFAULT: semantic signal only)
#   REWARD_FN_NAME=compute_score_guided          (similarity-shaped; the arm that regressed)
#   REWARD_FN_NAME=compute_score_shaped          (graded BEq+ ladder, no similarity)
#   REWARD_FN_NAME=compute_score_composite       (the paper's 0.1*tc + 0.9*BEq+)
#   REWARD_FN_NAME=compute_score_typecheck_only  (ablation baseline: exploitable)
#
# WHY THESE DEFAULTS (all measured -- results/compare.txt, 400 val examples)
# --------------------------------------------------------------------------
# 200 GRPO steps from the SFT policy with the guided reward moved BEq+ from
# 38.8% DOWN to 29.0% while type-check went 76.2% UP to 84.2% (McNemar p<1e-4).
# Four things in this file were responsible, and all four defaults changed:
#
#   1. The reward paid type-check + similarity, which are the only terms that can
#      rank rollouts inside a group where nothing proves equivalent -- i.e. most
#      groups. Fixed in reward/reward_fn.py (compute_score_gated is now default);
#      FILTER_GROUPS=1 here is the stronger, more expensive version.
#   2. norm_adv_by_std_in_grpo renormalised near-tie groups back to full gradient
#      scale, amplifying scorer noise.        -> NORM_ADV_BY_STD=False
#   3. entropy fell to 0.015 with rollout_n=4, so groups held near-duplicates and
#      there was nothing to learn from.  -> ROLLOUT_N=8, ROLLOUT_TEMPERATURE=1.15
#      (NOT an entropy bonus -- that OOM'd this 16GB card twice; see below)
#   4. kl_loss_coef=0.001 let the policy drift to KL 0.5 from the SFT reference,
#      which is the best BEq+ policy available. -> KL_LOSS_COEF=0.01
#
# Cost note: rollout_n 4 -> 8 roughly doubles wall-clock per step (~100s -> ~200s
# at train_batch_size=8), because Lean scoring is serialised and dominates.
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
# THIS IS NOT A FREE MEMORY KNOB -- it has a HARD FLOOR.
# verl/utils/seqlen_balancing.py:384 (rearrange_micro_batches) asserts
# max_token_len >= max_seq_len, because a single sequence can never be split
# across micro-batches. So this can never legally go below
# max_prompt_length + max_response_length (= 896 here), no matter how tight
# GPU memory is. 896 is a true cap, not a guess: filter_overlong_prompts +
# truncation='error' bound the prompt at 768, and rollout.max_model_len is
# set to the same 896 sum below.
#
# An earlier attempt to fix an OOM by dropping this to 512 was therefore never
# valid -- it just traded the allocator error for an assertion error, firing
# first in compute_log_prob (max_seq_len=577 > 512) and then, once log-probs
# had their own budget, at the identical assertion in the actor backward.
#
# 896 (the exact floor) is chosen over a rounder 1024 deliberately: it is the
# smallest legal value, so it is also the cheapest legal one (~12% less than
# 1024 on every tensor that scales with tokens). Raise it only if you also
# raise the seqlen budget, and expect ~2GB per extra 896 tokens.
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-896}
log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-896}
# Fused log-prob kernels (actor_rollout_ref.model.use_fused_kernels). OFF, and
# the default should stay OFF -- MEASURED, not assumed:
#
#   896 tokens, vocab 151936, this card:  plain 1.94GB  vs  fused/torch 1.90GB
#   1024 tokens:                          plain 2.21GB  vs  fused/torch 2.17GB
#
# i.e. ~2%. Chunking the logits does NOT help here because the peak is not the
# logits tensor -- it is dominated by the vocab-sized lm_head weight gradient
# ([151936 x 896]) plus the fp32 softmax temporaries, and the fused path
# allocates those per chunk anyway. It is not worth taking a
# less-travelled code path for 2%:
#
#   - backend=triton is outright BROKEN in our configuration. Its
#     LinearCrossEntropy flattens to (bsz*seqlen,) and never restores the
#     batch dim (verl/utils/kernel/linear_cross_entropy.py:70-83); it is only
#     consumed by the use_remove_padding=True branch, which wants exactly that
#     flat layout. Our use_remove_padding=False branch indexes
#     output.log_probs.shape[1] (workers/engine/fsdp/transformer_impl.py:1424)
#     and dies with `IndexError: tuple index out of range`.
#   - backend=torch does return the right 2-D shape and does run, so set
#     USE_FUSED_KERNELS=True FUSED_KERNELS_BACKEND=torch if you want it -- but
#     see the 2% above before bothering.
#
# use_remove_padding must stay False regardless: that path needs flash-attn
# varlen and flash_attn is not installed in this venv (nor readily buildable
# for this card's sm_120) -- hence attn_implementation=sdpa in MODEL below.
#
# The real reason the historic runs OOM'd in loss.backward() was the entropy
# bonus, NOT the token budget -- confirmed from the log this file already
# cites: logs/train_gated_20260816_131510.log ran with entropy_coeff=0.005
# AND ppo_max_token_len_per_gpu=1024, and died with 13.15GB already allocated
# by PyTorch while trying to add 670MB. The entropy term adds several more
# vocab-sized tensors on top of the ~2.2GB measured above. With
# entropy_coeff=0 (the default below) the logits path costs ~1.9GB at 896,
# which fits alongside vLLM's 0.3 share. That is why the fix for the OOM is
# "leave the entropy bonus off", not "starve the token budget".
use_fused_kernels=${USE_FUSED_KERNELS:-False}
fused_kernels_backend=${FUSED_KERNELS_BACKEND:-torch}

actor_lr=${ACTOR_LR:-1e-5}

# ---- the anti-drift settings (see "Why these defaults" below) ----
# KL anchor. Was 0.001, which let measured actor/kl_loss reach 0.5 over 200
# steps -- i.e. the policy wandered a long way from the SFT reference, which is
# the best BEq+ policy we have. 0.01 is chosen to hold measured KL around or
# below 0.05; WATCH `actor/kl_loss` in the log and raise this if it climbs.
kl_loss_coef=${KL_LOSS_COEF:-0.01}
# Exploration. The problem being solved: entropy fell monotonically to 0.015 by
# step 100 of the guided run, at which point the rollouts in a group are
# near-duplicates and GRPO has nothing to compare.
#
# THE ENTROPY BONUS IS OFF, DELIBERATELY -- it cost two runs to 16GB OOMs.
# A nonzero coefficient makes the training backward hold entropy intermediates
# alongside the full-vocab logits gradient, and on this card the margin is only
# tens of MB. Both crashes were in `loss.backward()` at the step right after
# validation (logs/train_gated_20260816_131510.log), and verl's own memory
# mitigations below (chunked + recomputed entropy) were applied and still not
# enough. Chasing that margin is not worth another lost run.
#
# Exploration comes from ROLLOUT_TEMPERATURE instead, which costs ZERO training
# memory: it only widens the sampling distribution, and verl divides logits by
# the same temperature when computing log-probs, so the policy gradient stays
# consistent. Combined with rollout_n=8 that is enough to keep rollout groups
# from collapsing into duplicates.
#
# The one thing temperature does NOT do is actively resist the policy sharpening
# over training -- that is what an entropy bonus buys. KL anchoring to the SFT
# reference (kl_loss_coef above) is doing that job here. If you move to a card
# with real headroom, ENTROPY_COEFF=0.005 plus the two flags below is the
# stronger setup; watch `actor/entropy` and keep it off the floor either way.
entropy_coeff=${ENTROPY_COEFF:-0}
entropy_chunking=${ENTROPY_CHUNKING:-True}
entropy_checkpointing=${ENTROPY_CHECKPOINTING:-True}
# GRPO advantage normalisation. verl's default divides each group's advantage by
# that group's reward std, so a group whose rewards differ by 0.02 produces the
# SAME gradient magnitude as one whose rewards differ by 0.5. With a noisy
# external scorer that turns near-ties (often just Lean timeout jitter) into
# full-scale updates. Turning it off is the Dr. GRPO correction and is what makes
# the reward re-weighting in reward/reward_fn.py actually bite.
norm_adv_by_std=${NORM_ADV_BY_STD:-False}

lora_rank=${LORA_RANK:-0}
lora_alpha=${LORA_ALPHA:-16}

rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.3}
# Was 4. Raising this is the single most effective exploration lever here: a
# group only produces a gradient if its samples DIFFER in reward, and for a
# prompt the policy solves ~15% of the time, P(mixed group) goes 0.48 -> 0.73
# from n=4 to n=8. It is also the main cost knob -- Lean scoring is serialised,
# so wall-clock per step scales roughly linearly with train_batch_size*n.
rollout_n=${ROLLOUT_N:-8}
# THE exploration lever for this project, now that the entropy bonus is off (see
# entropy_coeff above). verl divides logits by this when computing log-probs, so
# sampling above 1.0 stays consistent with the policy gradient -- and unlike an
# entropy bonus it costs no training memory at all. 1.15 is a deliberately mild
# widening; raise toward 1.3 if `actor/entropy` still trends to the floor, lower
# to 1.0 if generations start degrading.
rollout_temperature=${ROLLOUT_TEMPERATURE:-1.15}

total_epochs=${TOTAL_EPOCHS:-3}
save_freq=${SAVE_FREQ:--1}
test_freq=${TEST_FREQ:-5}

reward_fn_name=${REWARD_FN_NAME:-compute_score_gated}
reward_num_workers=${REWARD_NUM_WORKERS:-2}

# DAPO-style group filtering: DISCARD rollout groups whose samples all share the
# same `semantic_signal` (all failed, or all fully equivalent) and generate
# replacements, so every training group is one the policy can actually learn
# from. This is the strongest form of "train on the learnable window only".
#
# It is OFF by default purely on cost: replacements are generated one prompt at a
# time until the train batch refills, so generation cost scales as 1/keep_rate,
# measured/estimated at ~3x here. `compute_score_gated` already zeroes the
# advantage for those same groups at no extra cost -- it just doesn't reclaim
# their slot in the batch. Turn this on when compute allows; it strictly
# dominates the reward-side gate.
filter_groups=${FILTER_GROUPS:-0}
filter_groups_metric=${FILTER_GROUPS_METRIC:-semantic_signal}
# Where verl dumps validation generations (prompt, output, gold, and every
# reward_extra_info field). Off unless set -- these are what you read to see
# HOW a checkpoint changed, not just that its score moved.
validation_data_dir=${VALIDATION_DATA_DIR:-}

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

# Fail fast, before Ray/vLLM spin up, if either token-budget knob can't
# possibly hold the longest sequence the run can produce. Without this the job
# gets N steps in (past whichever early batches happen to have short
# sequences) before dying in rearrange_micro_batches -- see the postmortem
# above ppo_max_token_len_per_gpu.
max_possible_seq_len=$((max_prompt_length + max_response_length))
if (( ppo_max_token_len_per_gpu < max_possible_seq_len )); then
  echo "ERROR: ppo_max_token_len_per_gpu (${ppo_max_token_len_per_gpu}) must be >= max_prompt_length + max_response_length (${max_possible_seq_len}), or the actor's backward pass will crash on any sequence longer than the budget." >&2
  exit 1
fi
if (( log_prob_max_token_len_per_gpu < max_possible_seq_len )); then
  echo "ERROR: log_prob_max_token_len_per_gpu (${log_prob_max_token_len_per_gpu}) must be >= max_prompt_length + max_response_length (${max_possible_seq_len}), or compute_log_prob will crash on any sequence longer than the budget." >&2
  exit 1
fi
# The triton fused backend only emits the flat (total_nnz,) log_probs that the
# use_remove_padding=True branch expects; with remove_padding off it crashes in
# prepare_model_outputs with "IndexError: tuple index out of range". Guarded
# rather than merely documented because the failure lands ~10 steps in, deep in
# a Ray traceback. See the fused_kernels_backend comment above.
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

# Group filtering needs the metric to exist in reward_extra_info at SAMPLING
# time, which is why every reward function in reward/reward_fn.py returns a dict
# rather than a bare float. verl raises at startup if the key is ever missing, so
# a typo here fails loudly instead of silently training on unfiltered batches.
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
