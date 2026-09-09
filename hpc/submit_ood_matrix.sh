#!/usr/bin/env bash
# The OOD + faithfulness matrix for the c512 / c512cos arms.
#
#   bash hpc/submit_ood_matrix.sh
#
# THREE MEASUREMENTS PER MODEL, and they answer different questions:
#
#   miniF2F BEq+/type-check   does it TRANSFER off LoCoLib? (competition problems,
#                             a header the model never saw, statements only)
#   NL -> FL faithfulness     does the Lean proof carry out the ENGLISH argument?
#                             judged by microsoft/phi-4, the EVAL judge, which is
#                             deliberately not the Qwen2.5-7B the faith arms train
#                             against.
#   FL <-> FL alignment       does it walk the same road as the GOLD proof?
#                             No judge, no English: BEq+ over intermediate goal
#                             states.
#
# FL <-> FL IS NOT RUN ON miniF2F. Every miniF2F gold ends at `:= by` with an
# empty body (0/244 carry a proof), so there is no reference trajectory to align
# against. It runs on the LoCoLib val slice, where 760/760 golds carry proofs.
#
# Arms still training get their evals queued behind their last chunk with
# --dependency=afterany, so nothing here needs babysitting.
#
# miniF2F IS OPT-IN: pass WITH_MINIF2F=1. It is off by default because the OOD
# question was deferred in favour of finishing the in-domain series first, and a
# half-run OOD sweep is worse than none (see
# results/eval/_partial_minif2f_20260909/README.md).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
R="$(pwd)"

MINI="data_minif2f/val_proof.parquet"; MINI_N=244
LOCO="data_locolib/val_proof_nl.parquet"; LOCO_N=760
LEAN=(MATHLIB_TAR=/scratch/logan03/mathlib4_v4.23_lake.tar MATHLIB_TAR_FLAT=1
      LEAN_INTERACT_CACHE_DIR=/scratch/logan03/ai4math_training_lean_interact_cache_v423)

# run -> the training chunk its evals must wait for ("" = already finished)
DONE_ARMS=(rl3b_locolib_proof_c512_outcome
           rl3b_locolib_proof_c512_outcome_brevity
           rl3b_locolib_proof_c512cos_typecheck)
RUNNING_ARMS=(rl3b_locolib_proof_c512cos_gated:2709592
              rl3b_locolib_proof_c512cos_outcome:2709594
              rl3b_locolib_proof_c512_outcome_faith:2639458
              rl3b_locolib_proof_c512_outcome_v2:2639462)

sub() { bash "${R}/hpc/submit.sh" "$@" | awk '{print $NF}'; }

mini_label() { echo "minif2f_${1#rl3b_locolib_proof_}"; }

W1=()   # eval jobs whose gens exist soon
W2=()   # eval jobs gated behind training

if [ "${WITH_MINIF2F:-0}" = "1" ]; then
  echo "=== miniF2F, arms already at step 90 ==="
  for run in "${DONE_ARMS[@]}"; do
    id=$(sub "${R}/hpc/grpo_eval.slurm" RUN="${run}" OUT_LABEL="$(mini_label "${run}")" \
             VAL_PARQUET="${MINI}" N_EVAL="${MINI_N}" "${LEAN[@]}")
    echo "  $(mini_label "${run}") -> ${id}"; W1+=("${id}")
  done

  echo "=== miniF2F, the SFT baseline (re-run at 512 under v4.23) ==="
  # The existing results/eval/minif2f_capped records predate both the cap change
  # and the toolchain fix: they record neither `max_new_tokens` nor `toolchain`,
  # so they were scored at 128 under v4.8 and cannot anchor a 512/v4.23 arm.
  # BEST_FILE redirected on purpose. eval_sft.slurm ends by pinning the best
  # checkpoint by BEq+, and its default target is data_locolib/best_sft.txt --
  # which every arm resumes from. Letting an OOD transfer score re-pin the SFT
  # baseline is a category error, so it writes to a file nothing reads.
  id=$(sub "${R}/hpc/eval_sft.slurm" OUT_LABEL=minif2f_sft_proof \
           BEST_FILE="${R}/data_minif2f/best_sft_proof_ood.txt" \
           VAL_PARQUET="${MINI}" N_EVAL="${MINI_N}" "${LEAN[@]}")
  echo "  minif2f_sft_proof -> ${id}"; W1+=("${id}")
fi

echo "=== in-domain evals, arms still training ==="
for pair in "${RUNNING_ARMS[@]}"; do
  run="${pair%%:*}"; dep="${pair##*:}"
  a=$(sub --dependency=afterany:"${dep}" "${R}/hpc/grpo_eval.slurm" \
        RUN="${run}" VAL_PARQUET="${LOCO}" N_EVAL="${LOCO_N}" "${LEAN[@]}")
  echo "  ${run}: locolib -> ${a}  (after ${dep})"
  W2+=("${a}")
  if [ "${WITH_MINIF2F:-0}" = "1" ]; then
    b=$(sub --dependency=afterany:"${dep}" "${R}/hpc/grpo_eval.slurm" \
          RUN="${run}" OUT_LABEL="$(mini_label "${run}")" \
          VAL_PARQUET="${MINI}" N_EVAL="${MINI_N}" "${LEAN[@]}")
    echo "  ${run}: minif2f -> ${b}  (after ${dep})"
    W2+=("${b}")
  fi
done

# ---------------------------------------------------------------------------
# Faithfulness, both axes, at step 90 only.
#
# ONE NUMBER PER ARM PER METRIC, not a trajectory. A judged metric costs a 22GB
# model load plus minutes per proof, and an FL <-> FL pair is an n x m matrix of
# BEq+ cascades. The step-90 row is what the table needs; a per-step trend is a
# separate, much larger job and is not assumed here.
# ---------------------------------------------------------------------------
join_dep() { local IFS=:; echo "$*"; }

mini_gens() { for run in "$@"; do
    l="$(mini_label "${run}")"
    echo "${R}/results/eval/${l}/gen_${l}-step90_n${MINI_N}.jsonl"
  done; }
loco_gens() { for run in "$@"; do
    echo "${R}/results/eval/${run}/gen_${run}-step90_n${LOCO_N}.jsonl"
  done; }

ALL_RUNS=("${DONE_ARMS[@]}")
for pair in "${RUNNING_ARMS[@]}"; do ALL_RUNS+=("${pair%%:*}"); done

echo "=== NL -> FL faithfulness (judge microsoft/phi-4, matched arm only) ==="
DEP_ALL="$(join_dep "${W1[@]}" "${W2[@]}")"
mkdir -p "${R}/results/manifests/genlists"
loco_gens "${ALL_RUNS[@]}" > "${R}/results/manifests/genlists/locolib_step90.txt"

f2=$(sub --dependency=afterany:"${DEP_ALL}" "${R}/hpc/faith_eval.slurm" \
       GENS_FILE="${R}/results/manifests/genlists/locolib_step90.txt" \
       ARMS=matched N=240 JUDGE_MODEL=microsoft/phi-4 "${LEAN[@]}")
echo "  locolib -> ${f2}"
if [ "${WITH_MINIF2F:-0}" = "1" ]; then
  mini_gens "${ALL_RUNS[@]}" > "${R}/results/manifests/genlists/minif2f_step90.txt"
  f1=$(sub --dependency=afterany:"${DEP_ALL}" "${R}/hpc/faith_eval.slurm" \
         GENS_FILE="${R}/results/manifests/genlists/minif2f_step90.txt" \
         ARMS=matched N=240 JUDGE_MODEL=microsoft/phi-4 "${LEAN[@]}")
  echo "  minif2f -> ${f1}"
fi

echo "=== FL <-> FL alignment (LoCoLib only; miniF2F has no gold proofs) ==="
a1=$(sub --dependency=afterany:"${DEP_ALL}" "${R}/hpc/alignment.slurm" \
       GENS_FILE="${R}/results/manifests/genlists/locolib_step90.txt" N=100 SEED=0 "${LEAN[@]}")
echo "  locolib -> ${a1}"
