#!/usr/bin/env bash
# GRPO at batch 16 on the 3B policy.
#   ARM=gated|placebo|typecheck|selfprove|guided|gated_edge|gated_filtered
#   sbatch --export=ALL,ARM=gated hpc/grpo_3b.slurm
#
# 11h CHUNKS -- the longest that still sits in the good priority band here.
# 3h chunks were measured to be a false economy: four of them took ~22h wall for
# 12h of compute, including a single 9h gap between chunk 1 ending and chunk 2
# starting. At ~10.9 min/step an 11h chunk covers ~60 steps, so step 20 -> 150
# is 3 chunks rather than 9. Chain them with --dependency=afterany: (a walltime
# kill is the expected end of a chunk, not a failure); resume_mode=auto picks up
# from the last checkpoint. Lean scoring is ~2.1s/rollout at 8-12 workers, so 150 steps
# x 128 rollouts is ~11h -- but CC gives short jobs much better backfill
# priority, and verl's resume_mode=auto continues toward total_training_steps
# from the last checkpoint. So submit several of these chained with
# `--dependency=afterany:` (afterany, NOT afterok: a walltime kill is the
# expected way each chunk ends, and it is not a failure).
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=grpo_3b
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=11:00:00
#SBATCH --output=logs/grpo_3b_%j.out
#SBATCH --error=logs/grpo_3b_%j.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh

stage_mathlib "[grpo3b]"

# MULTI-TURN IS OFF BY DEFAULT, AND WAS A NO-OP WHEN IT DEFAULTED TO ON.
#
# The intent was to give the policy a Lean type-check tool mid-rollout
# (reward/lean_tool.py) so it sees the compiler's real diagnostics before the
# reward lands. Setting multi_turn.enable=True and function_tool_path DOES NOT
# DO THAT. verl picks the agent loop from a per-row `agent_name` field, falling
# back to rollout.default_agent_loop, which is "single_turn_agent"
# (repos/verl/verl/workers/config/rollout.py:74) and which nothing here
# overrides. Our parquet rows carry only data_source/prompt/ability/
# reward_model/extra_info -- no agent_name, no tools_kwargs -- so
# SingleTurnAgentLoop applies the chat template with NO `tools=` argument and
# does one generate call. verl's own e2e test for this feature injects
# `agent_name = "tool_agent"` per row; see
# repos/verl/tests/experimental/agent_loop/test_basic_agent_loop.py.
#
# The tool schema does reach RLHFDataset's prompt-length filter, so leaving this
# on is not free: it perturbs which rows survive filtering while buying nothing.
# It also cannot work at the current budget -- max_response_length is 128 tokens
# TOTAL across turns, which does not fit a tool call plus a tool response plus a
# revised theorem.
#
# TO ACTUALLY ENABLE IT: set actor_rollout_ref.rollout.agent.default_agent_loop=
# tool_agent (or add agent_name to the parquet in scripts/prepare_dataset.py),
# AND raise MAX_RESPONSE_LENGTH well past 128. Then re-run the WHOLE comparison
# set -- it changes the rollout distribution, so arms run with and without it
# are not comparable.
MULTITURN="${MULTITURN:-0}"
[ "$MULTITURN" = "1" ] && MULTITURN_ON=1 || MULTITURN_ON=""
ARM="${ARM:?set ARM=gated|placebo|typecheck|selfprove|guided|gated_edge|gated_filtered}"
# BEST_SFT selects which SFT series the RL arms train from. Default is the
# original Instruct-based one; BEST_SFT=data_3b/best_sft_mt.txt runs the same
# arms on top of the mid-trained model. Arms from different series are NOT
# comparable -- they have different baselines.
BEST_SFT="${BEST_SFT:-data_3b/best_sft.txt}"
SERIES_TAG="${SERIES_TAG:-}"
SERIES_SUFFIX="${SERIES_TAG:+_${SERIES_TAG}}"
CKPT=$(cat "${BEST_SFT}" 2>/dev/null)
[ -n "${CKPT}" ] || { echo "[grpo3b] FATAL: ${BEST_SFT} missing"; exit 1; }
case "${CKPT}" in /*) ;; *) CKPT="$(cd "$(dirname "${CKPT}")" && pwd)/$(basename "${CKPT}")" ;; esac
[ -d "${CKPT}" ] || { echo "[grpo3b] FATAL: checkpoint dir not found: ${CKPT}"; exit 1; }

EXTRA=()
TRAIN_FILE="${TRAIN_FILE:-${REAL}/data_3b/train.parquet}"
case "$ARM" in
  gated)   REWARD_FN_NAME=compute_score_gated ;;
  selfprove)
    # THE DECISIVE GOLD-FREE ARM. Strongest reward obtainable without ever
    # consulting the gold: type-check + provable + not closed by `tauto` alone.
    # Modelled on kimina-prover-rl's compile reward, with the anti-triviality
    # guard their task does not need (their statement is an input; ours is the
    # output, so `theorem x : 1 = 1` would otherwise score full marks).
    # Paired against `gated` this asks the project's core question directly:
    # can compiler-only signals substitute for gold-referenced semantics?
    #
    # FEWER AGENT-LOOP WORKERS THAN EVERY OTHER ARM, on memory. This is the only
    # arm that runs `exact?` -- a whole-Mathlib search -- on EVERY rollout that
    # type-checks, via _prove_with. Those searches grow a REPL toward the 8 GB
    # BEQ_MEMORY_LIMIT_MB cap, and 24 x 8 GB is 192 GB against this job's 120 G.
    # The first attempt (1586327) had its REPLs recycled out from under in-flight
    # calls and wedged the step. beq_plus.py now survives that, but a recycle
    # still costs a rollout, and a node OOM would kill Ray workers outright --
    # which no reward-side guard can recover from. 16 x 8 GB worst case, ~69 GB
    # typical, leaves room for vLLM's host allocations.
    AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-16}"
    REWARD_FN_NAME=compute_score_selfprove ;;
  typecheck)
    # LOOP-HEALTH PROBE -- not a baseline we want to "succeed".
    #
    # Success criterion is that the REWARD rises, nothing else. Its BEq+ is
    # EXPECTED to fall, and that is the point, not a malfunction: at 0.5B this
    # reward went 0.000 -> 1.000 in 30 steps while BEq+ collapsed to 3.8%.
    #
    # Why it cannot be made honest here: in autoformalization the statement is
    # the model's OUTPUT, so "it type-checks" is satisfied by `theorem x : 1 = 1`.
    # Contrast project-numina/kimina-prover-rl, whose compile reward is sound
    # only because their statement is an INPUT -- reward/proof_utils.py does
    # extract_proof_from_text(output, formal_statement) with formal_statement
    # taken from extra_info, so the model supplies only the proof body and
    # cannot change the goal. That asymmetry is why they need no semantic reward
    # and we do. Any sound reward for THIS task must compare against the gold,
    # which is what BEq+ does -- so there is no non-exploitable non-BEq+
    # baseline to run.
    #
    # What this arm buys: it separates "BEq+ does not teach" from "this 3B stack
    # cannot optimise anything". If the reward does NOT rise, the loop is broken
    # and no conclusion about BEq+ is admissible. It also re-tests the paper's
    # exploitability claim on a 3B policy.
    REWARD_FN_NAME=compute_score_typecheck_only ;;
  guided)
    # PROCESS-LEVEL REWARD (Interplay paper, arXiv:2512.07783 (d)).
    #
    # Why this arm and not a new reward: 32.3% of 3B-baseline rollouts TYPE-CHECK
    # but earn no direction, and `compute_score_gated` pays them 0.00 -- exactly
    # what it pays unparseable garbage. That dead band is a third of all mass, so
    # for a third of prompts the group reward is CONSTANT and GRPO's advantage is
    # identically zero there however the group is composed. It is the structural
    # reason the gain channel has been pinned at 4.9-7.3% across every arm.
    #
    # compute_score_guided splits that band with gold-referenced similarity:
    #   no type-check                 0.00
    #   type-checks                   0.10 + 0.10*sim   (0.10 - 0.20)
    #   + one direction              +0.15              (0.25 - 0.35)
    #   + both directions            +0.65              (0.90 - 1.00)
    # Measured feasibility: token-level Jaccard against the gold separates BEq+
    # from type-check-only rollouts at AUC 0.706, at zero Lean cost.
    #
    # norm_adv_by_std_in_grpo=False matters MOST here: a group that is uniformly
    # "type-checks only" has a reward spread of ~0.01, and verl's default
    # renormalises that back to full scale, handing a near-noise similarity
    # difference the same gradient as proving equivalence -- the exact failure in
    # this reward's header (policy copied gold tokens into invalid Lean, entropy
    # collapsed to 0.127).
    #
    # It is set REDUNDANTLY below. configs/run_grpo.sh:189 already defaults
    # `norm_adv_by_std=${NORM_ADV_BY_STD:-False}`, so EVERY arm -- gated, placebo
    # and the rest -- has always run with it off. Restating it here is a guard
    # against that default changing, NOT something that distinguishes this arm.
    # Do not read a gated-vs-guided difference as coming from this setting.
    #
    # Similarity CANNOT see semantic inversion: smoke-tested, flipping `<` to `>`
    # scores 0.970 while a correctly-renamed variant scores 0.820. It guides the
    # search; only BEq+ certifies. That is why the sim weight (0.10) is strictly
    # below the one-direction step (0.15).
    REWARD_FN_NAME=compute_score_guided
    EXTRA+=(algorithm.norm_adv_by_std_in_grpo=False) ;;
  gated_edge)
    # EDGE-OF-COMPETENCE POOL (Interplay paper (b)). Identical reward to `gated`;
    # the ONLY difference is the prompt pool, restricted to prompts that scored
    # 1-7/8 at k=8 from this same SFT checkpoint. Those are the only groups that
    # can produce a nonzero advantage, so informative goes 46% -> ~100% by
    # construction, at zero extra Lean cost per step (contrast FILTER_GROUPS=1,
    # which buys the same thing online at ~8x because refill is one prompt at a
    # time).
    #
    # It is a STATIC filter on a MOVING target: prompts migrate
    # starved -> informative -> saturated as the policy improves, so the pool
    # decays in usefulness over the run. Paired against `gated` on the full pool,
    # that decay is the measurement -- if difficulty curation helps at all it
    # should show early and fade.
    REWARD_FN_NAME=compute_score_gated
    TRAIN_FILE="${REAL}/data_3b/train_edge.parquet"
    [ -f "${TRAIN_FILE}" ] || { echo "[grpo3b] FATAL: ${TRAIN_FILE} missing -- run hpc/score_pool_3b.slurm then make_difficulty_subset.py"; exit 1; } ;;
  gated_filtered)
    REWARD_FN_NAME=compute_score_gated
    # DAPO-style: discard groups that cannot produce an advantage and refill.
    # ~3x generation cost, but it attacks the dead-batch problem head-on.
    export FILTER_GROUPS=1 FILTER_GROUPS_METRIC=semantic_signal ;;
  placebo)
    # RETIRED 2026-08-24. The control answered its question -- paired
    # gated-vs-placebo is +18.2pp, p=7e-16 -- and that result stands as recorded
    # in results/. DO NOT run new placebo arms; new arms are paired against SFT
    # and against their own twin at the previous setting, which is a tighter
    # control anyway (one variable moves instead of the whole reward).
    #
    # Its most useful legacy is a negative result about the OPTIMISER, not any
    # reward: a zero-information reward should do approximately nothing under
    # correct regularisation, and instead it took BEq+ 39.4% -> 13.0%. That is
    # what put ACTOR_LR (1e-5, i.e. 10x the GRPO standard) under suspicion.
    #
    # Set PLACEBO_ANYWAY=1 to run it regardless, e.g. to reproduce the recorded
    # result. Everything below still works.
    [ "${PLACEBO_ANYWAY:-0}" = "1" ] || {
      echo "[grpo3b] REFUSING: the placebo arm is retired (see the comment in this file)."
      echo "[grpo3b] Pair new arms against SFT and their own 1e-5 twin instead."
      echo "[grpo3b] PLACEBO_ANYWAY=1 overrides."; exit 1; }
    REWARD_FN_NAME=compute_score_placebo
    # The control MUST be fitted to this policy's geometry, not 0.5B's.
    # Per-series: the fit is to a specific policy's geometry, so a mid-trained
    # run must use its own constants, never the Instruct series'.
    PLACEBO_CONSTS="data_3b/placebo_constants${SERIES_SUFFIX}.sh"
    [ -f "${PLACEBO_CONSTS}" ] || {
      echo "[grpo3b] FATAL: ${PLACEBO_CONSTS} missing -- run rollouts_3b first."
      echo "[grpo3b] An uncalibrated placebo is not a control."; exit 1; }
    . "${PLACEBO_CONSTS}"
    echo "[grpo3b] placebo fitted to 3B: GROUP_P=${BEQ_PLACEBO_GROUP_P} ROLLOUT_P=${BEQ_PLACEBO_ROLLOUT_P}" ;;
  *) echo "bad ARM"; exit 1 ;;
esac

# 4300 RL prompts / batch 16 = 268 steps per epoch, so even 150 steps stays
# inside one pass -- no prompt is ever seen twice.
#
# DEFAULT IS 90, DOWN FROM 150. Every arm that got that far had already separated
# and then flattened: gated sits 34-40% from step 30 to 90, the placebo is on its
# floor by 30, and typecheck's collapse is visible by 50. The extra 60 steps cost
# ~16 GPU-hours per BEq+ arm (16 min/step) and changed no conclusion. Arms are
# also compared at matched steps, so the useful range is capped by the SHORTEST
# arm regardless. Figures cap at 90 to match (scripts/make_figures.py --max-step).
MODEL_PATH="${CKPT}" \
REWARD_FN_NAME="${REWARD_FN_NAME}" \
TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 ROLLOUT_N=8 \
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.30}" \
AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-24}" \
NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}" \
SAVE_FREQ=10 MAX_CKPT_KEEP=20 \
EXPERIMENT_NAME="rl3b${SERIES_SUFFIX}_${ARM}" \
  bash configs/run_grpo.sh \
    ${MULTITURN_ON:+actor_rollout_ref.rollout.multi_turn.enable=True} \
    ${MULTITURN_ON:+actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS:-2}} \
    ${MULTITURN_ON:+actor_rollout_ref.rollout.multi_turn.function_tool_path=${REAL}/reward/lean_tool.py} \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${REAL}/data_3b/val.parquet" \
    trainer.total_training_steps="${TOTAL_STEPS:-90}" \
    trainer.resume_mode=auto \
    trainer.test_freq="${TEST_FREQ:-50}" \
    trainer.save_freq=10 \
    # SAVE THE OPTIMIZER. Without it a resumed chunk cold-starts Adam's moment
    # estimates, and arms here run as chained afterany chunks, so every walltime
    # boundary was a silent optimiser reset. It already happened once mid-run on
    # rl3b_lr6_gated_edge (chunk 2 resumed from step 50) with no visible shock,
    # but a reset is indistinguishable from a real trend change in the metrics,
    # which matters most on exactly the long runs where resumes are frequent.
    # Costs disk: checkpoints roughly double. MAX_CKPT_KEEP bounds it.
    'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]' \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "[grpo3b] arm=${ARM} done"
