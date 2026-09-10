#!/usr/bin/env bash
# One GRPO arm per job.
#   sbatch --export=ALL,ARM=gated hpc/grpo.slurm
#   ARM=gated|typecheck|outcome|outcome_brevity|outcome_faith|outcome_v2|gated_edge
#
# The two faithfulness arms need a THIRD GPU for the judge server and will
# refuse to start without one:
#   bash hpc/submit.sh --gpus-per-node=a100:3 hpc/grpo.slurm ARM=outcome_faith ...
#
# Runs in 11h chunks chained with --dependency=afterany: a walltime kill is the
# expected end of a chunk, not a failure, and resume_mode=auto continues from the
# last checkpoint. Shorter chunks queue faster but lose more to gaps between them.
#
# Corpus and naming come from DATA_DIR / RUN_PREFIX / PROJECT_NAME in
# job_prelude.sh; override those to run the same arms on another corpus.
#SBATCH --account=def-vganesh
#SBATCH --job-name=grpo
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=11:00:00
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh

stage_mathlib "[grpo]"

ARM="${ARM:?set ARM=gated|typecheck|outcome|outcome_brevity|outcome_faith|outcome_v2|gated_edge}"
# Which SFT series the arms train from. Arms resumed from different baselines
# are not comparable to each other.
BEST_SFT="${BEST_SFT:-${DATA_DIR}/best_sft.txt}"
SERIES_TAG="${SERIES_TAG:-}"
SERIES_SUFFIX="${SERIES_TAG:+_${SERIES_TAG}}"
RUN_NAME="${RUN_PREFIX}${SERIES_SUFFIX}_${ARM}"

CKPT=$(cat "${BEST_SFT}" 2>/dev/null)
[ -n "${CKPT}" ] || { echo "[grpo] FATAL: ${BEST_SFT} missing"; exit 1; }
case "${CKPT}" in /*) ;; *) CKPT="$(cd "$(dirname "${CKPT}")" && pwd)/$(basename "${CKPT}")" ;; esac
[ -d "${CKPT}" ] || { echo "[grpo] FATAL: checkpoint dir not found: ${CKPT}"; exit 1; }

EXTRA=()
TRAIN_FILE="${TRAIN_FILE:-${REAL}/${DATA_DIR}/train.parquet}"
# The val slice must follow the corpus: BEq+ needs a gold, so validating against
# another corpus's golds measures transfer, not fit.
VAL_FILE="${VAL_FILE:-${REAL}/${DATA_DIR}/val.parquet}"

case "$ARM" in
  gated)
    # The default: BEq+ semantic signal only, flat floor everywhere else.
    REWARD_FN_NAME=compute_score_gated ;;
  typecheck)
    # Ablation baseline. Exploitable by construction, since the statement is the
    # model's output and `theorem x : 1 = 1` satisfies it. Its BEq+ is expected
    # to fall; the success criterion is that the reward rises, which separates
    # "BEq+ does not teach" from "this stack cannot optimise anything".
    REWARD_FN_NAME=compute_score_typecheck ;;
  gated_edge)
    # Same reward as `gated`; only the prompt pool differs, restricted to
    # prompts scoring 1-7/8 at k=8 from this SFT checkpoint. Only those groups
    # produce a nonzero advantage, so the informative fraction rises from ~46%
    # to ~100% at no extra Lean cost. A static filter on a moving target:
    # prompts leave the band as the policy improves.
    REWARD_FN_NAME=compute_score_gated
    TRAIN_FILE="${REAL}/${DATA_DIR}/train_edge.parquet"
    [ -f "${TRAIN_FILE}" ] || {
      echo "[grpo] FATAL: ${TRAIN_FILE} missing."
      echo "[grpo] Run hpc/score_pool.slurm then hpc/build_edge_pool.slurm."; exit 1; } ;;
  outcome)
    # The six-outcome ladder from reward/reward.py: 0.00 / 0.05 / 0.15 / 0.50.
    # Unlike gated it pays a graded reward below BEq+, so groups with no semantic
    # signal still carry a within-group gradient. Run it against gated rather
    # than in place of it.
    #
    # ALSO THE CONTROL for the three arms below, which are this reward plus one
    # term each. A new-term arm must be paired against an `outcome` run at the
    # SAME settings, not against the locolib_proof_lr6 one: that series ran at
    # max_response_length=128 and this one does not.
    REWARD_FN_NAME=compute_score_outcome ;;
  outcome_brevity)
    # outcome + the proof-LENGTH deduction on the SOLVED row.
    REWARD_FN_NAME=compute_score_outcome_brevity ;;
  outcome_faith)
    # outcome + the NL -> FL faithfulness band on the SOLVED row.
    REWARD_FN_NAME=compute_score_outcome_faith
    NEEDS_JUDGE=1 ;;
  outcome_v2)
    # outcome + BOTH new terms. "Outcome v2".
    REWARD_FN_NAME=compute_score_outcome_v2
    NEEDS_JUDGE=1 ;;
  *) echo "[grpo] bad ARM: ${ARM}"; exit 1 ;;
esac

# The faithfulness band reads `informal` / `informal_proof` off each row's
# extra_info, and the stock LoCoLib parquets do not carry them -- the reward
# would score every rollout `no_informal` (unknown), pay nothing, and quietly
# behave exactly like plain `outcome`. Fail instead of running a null arm.
# `scripts/data/add_informal_to_extra_info.py` writes the *_nl.parquet pair.
if [ "${NEEDS_JUDGE:-0}" = "1" ]; then
  if ! python3 - "${TRAIN_FILE}" <<'PYCHK'
import sys, pandas as pd
info = pd.read_parquet(sys.argv[1])["extra_info"]
missing = sum("informal_proof" not in (dict(r) if r is not None else {}) for r in info)
if missing:
    print(f"[grpo] FATAL: {missing}/{len(info)} rows of {sys.argv[1]} carry no "
          "`informal_proof` in extra_info. Run "
          "scripts/data/add_informal_to_extra_info.py and pass the *_nl.parquet.")
    sys.exit(1)
print(f"[grpo] corpus check: {len(info)}/{len(info)} rows carry informal text")
PYCHK
  then exit 1; fi
fi

# SERIES_TAG in the name too: two series can run the same ARM (e.g. a plain
# edge-pool arm vs a SERIES_TAG=proof variant), and without it they are
# indistinguishable in squeue.
rename_job grpo "${SERIES_TAG:+${SERIES_TAG}_}${ARM}"

# Arms separate and then flatten well before 90 steps, and they are compared at
# matched steps, so the useful range is capped by the shortest arm anyway.
TOTAL_STEPS="${TOTAL_STEPS:-90}"

# verl derives load_contents from save_contents, so the first chunk that turns
# the optimizer on tries to load an optimizer out of a checkpoint written while
# it was off and dies on the missing optim shard. Detect that one-time case and
# load model+extra only for this chunk; later chunks resume Adam normally.
LOAD_CONTENTS=""
CKPT_ROOT="${REAL}/checkpoints/${PROJECT_NAME}/${RUN_NAME}"
LAST_CKPT="$(ls -d "${CKPT_ROOT}"/global_step_* 2>/dev/null \
             | sed 's/.*global_step_//' | sort -n | tail -1)"
if [ -n "${LAST_CKPT}" ] \
   && ! ls "${CKPT_ROOT}/global_step_${LAST_CKPT}/actor"/optim_world_size_*.pt >/dev/null 2>&1; then
  LOAD_CONTENTS="model,extra"
  echo "[grpo] step_${LAST_CKPT} has no optimizer shard; loading model,extra for"
  echo "[grpo] this chunk only. Adam cold-starts once here."
fi

# ---------------------------------------------------------------------------
# The faithfulness judge, when this arm needs one.
#
# ITS OWN GPU, NOT A SHARE OF THE TRAINER'S. The measured peak on a training
# card is 37.8GB reserved of 40 (FSDP gather + activations + vLLM at 0.30), so
# there is no room for a 15GB judge beside it. The job therefore asks for one
# more card than it trains on, and CUDA_VISIBLE_DEVICES is split explicitly:
# without the split, Ray would claim all three and the judge would land on a
# card verl is already using.
#
# ONE PROCESS, NOT ONE PER WORKER. 24 agent-loop workers each constructing a 7B
# judge is 360GB of weights. They all talk HTTP to this one, which batches
# concurrent requests internally (utils/hf_judge.py).
#
# A DEAD JUDGE IS NOT A SOFT FAILURE. Every faithfulness verdict would come back
# `no_judge` -> unknown -> a flat 0.85 on every SOLVED row, which is numerically
# identical to running plain `outcome`. The arm would complete, produce
# plausible metrics, and mean nothing. So this block fails the job instead.
# ---------------------------------------------------------------------------
if [ "${NEEDS_JUDGE:-0}" = "1" ]; then
  # Qwen2.5-7B-Instruct trains, FormalRx-8b evaluates, and they must not be
  # swapped or shared: scoring a policy with the judge it was trained against
  # measures how well it learned to please that judge. Both are resident in the
  # offline HF cache, which compute nodes require (no internet).
  JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
  JUDGE_PORT="${JUDGE_PORT:-8732}"

  IFS=',' read -ra _GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
  if [ "${#_GPUS[@]}" -lt 3 ]; then
    echo "[grpo] FATAL: ARM=${ARM} needs a judge GPU on top of the ${NGPUS_PER_NODE:-2}"
    echo "[grpo] training GPUs, but CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "[grpo] has ${#_GPUS[@]}. Resubmit with --gpus-per-node=a100:3."
    exit 1
  fi
  JUDGE_GPU="${_GPUS[-1]}"
  TRAIN_GPUS="$(IFS=,; echo "${_GPUS[*]:0:${#_GPUS[@]}-1}")"

  JUDGE_LOG="${REAL}/logs/faith_judge_${SLURM_JOB_ID:-manual}.log"
  echo "[grpo] judge ${JUDGE_MODEL} on GPU ${JUDGE_GPU}, trainer on ${TRAIN_GPUS}"
  echo "[grpo] judge log -> ${JUDGE_LOG}"
  CUDA_VISIBLE_DEVICES="${JUDGE_GPU}" \
    python3 "${REAL}/scripts/eval/faithfulness_judge_server.py" \
      --model "${JUDGE_MODEL}" --port "${JUDGE_PORT}" \
      > "${JUDGE_LOG}" 2>&1 &
  JUDGE_PID=$!
  # The judge outlives the trainer otherwise and holds the GPU until walltime.
  trap 'kill "${JUDGE_PID}" 2>/dev/null' EXIT

  # Loading 15GB off shared storage and warming vLLM takes minutes, so the wait
  # is generous; what it must not do is proceed on a half-started server.
  if ! python3 - "${JUDGE_PORT}" "${JUDGE_MODEL}" "${JUDGE_PID}" <<'PYWAIT'
import json, os, signal, sys, time, urllib.request
port, want, pid = sys.argv[1], sys.argv[2], int(sys.argv[3])
deadline = time.time() + float(os.environ.get("JUDGE_WAIT_S", "1800"))
while time.time() < deadline:
    try:
        os.kill(pid, 0)                     # server died -> stop waiting on it
    except OSError:
        print("[grpo] judge process exited before becoming ready")
        sys.exit(1)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            h = json.loads(r.read())
        if h.get("ok"):
            # Assert WHICH judge answered. A stale server left on this port from
            # another job would otherwise be used silently, and if it were the
            # eval judge the whole train/eval separation would be gone.
            if h.get("model") != want:
                print(f"[grpo] FATAL: port {port} is serving {h.get('model')!r}, "
                      f"not {want!r}. Another judge is already on this port.")
                sys.exit(1)
            print(f"[grpo] judge ready: {h.get('model')}")
            sys.exit(0)
    except Exception:
        pass
    time.sleep(10)
print("[grpo] judge did not become ready in time")
sys.exit(1)
PYWAIT
  then
    echo "[grpo] FATAL: judge server never came up; see ${JUDGE_LOG}."
    echo "[grpo] Refusing to train ARM=${ARM} without it -- every faithfulness"
    echo "[grpo] verdict would be unknown and this arm would silently become"
    echo "[grpo] plain \`outcome\`."
    kill "${JUDGE_PID}" 2>/dev/null
    exit 1
  fi
  export FAITH_JUDGE_URL="http://127.0.0.1:${JUDGE_PORT}/generate"
  export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
fi

# Hardware profile: Narval a100:2 == 2x A100-40GB. Full-parameter 3B GRPO plus a
# frozen ref model does not fit resident on 40GB, so FSDP_OFFLOAD keeps params
# and Adam state on the host; the token budgets pack ~4 sequences per
# micro-batch; vLLM takes 30% of VRAM (the rest is the FSDP gather plus
# activations); CUDA graphs stay off. On an 80GB card (A100-80GB / H100) override
# at submit time:
#   FSDP_OFFLOAD=0 PPO_MAX_TOKEN_LEN_PER_GPU=16384 \
#   LOG_PROB_MAX_TOKEN_LEN_PER_GPU=16384 ROLLOUT_GPU_MEM_UTIL=0.6 ENFORCE_EAGER=False
# AGENT_LOOP_WORKERS is Lean/host-RAM parallelism, not VRAM -- see README "Hardware".
#
# Do not put a comment inside the continued command below: bash joins the
# continuation and the `#` silently drops every remaining argument.
MODEL_PATH="${CKPT}" \
REWARD_FN_NAME="${REWARD_FN_NAME}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}" \
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}" \
ROLLOUT_N="${ROLLOUT_N:-8}" \
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.30}" \
FSDP_OFFLOAD="${FSDP_OFFLOAD:-1}" \
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}" \
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-4096}" \
ENFORCE_EAGER="${ENFORCE_EAGER:-True}" \
LAYERED_SUMMON="${LAYERED_SUMMON:-True}" \
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}" \
AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-24}" \
NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}" \
SAVE_FREQ="${SAVE_FREQ:-10}" MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-20}" \
EXPERIMENT_NAME="${RUN_NAME}" \
  bash configs/run_grpo.sh \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.resume_mode=auto \
    trainer.test_freq="${TEST_FREQ:-50}" \
    trainer.save_freq="${SAVE_FREQ:-10}" \
    'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]' \
    ${LOAD_CONTENTS:+"actor_rollout_ref.actor.checkpoint.load_contents=[${LOAD_CONTENTS}]"} \
    ${EXTRA[@]+"${EXTRA[@]}"}
STATUS=$?

# job_prelude.sh deliberately omits `set -e` (some jobs continue past a soft
# failure), so a crash here -- Ray's node ever failed to come up on a bad node,
# main_ppo raising before the first step -- would otherwise fall through to
# the "done" line below and exit 0. That is a silent no-op: the chunk trains
# nothing, the dependency chain still fires, and eval "completes" against
# whatever checkpoint (or none) was already there. Fail the job instead.
if [ "${STATUS}" -ne 0 ]; then
  echo "[grpo] arm=${ARM} run=${RUN_NAME} FAILED (exit ${STATUS})"
  exit "${STATUS}"
fi

echo "[grpo] arm=${ARM} run=${RUN_NAME} done"
