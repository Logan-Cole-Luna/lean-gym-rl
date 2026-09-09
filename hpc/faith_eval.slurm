#!/usr/bin/env bash
# The NL -> FL faithfulness VALIDITY GATE. Run this before any arm trains on it.
#
#   bash hpc/submit.sh hpc/faith_eval.slurm GEN=<gen_*.jsonl> N=120
#
# reward/reward.py:362-369 makes three things non-optional before a JUDGED metric
# may back a reward band, and this job produces all three:
#
#   * unknown returns None, never 0            -> `unknown_rate`, broken out by cause
#   * a validity gate on the unknown rate      -> `coverage`
#   * a NEGATIVE CONTROL, because "unlike BEq+ this will be judged rather than
#     derived, and is therefore gameable by exactly the policy being trained"
#                                              -> the `mismatched` arm, AUC >= 0.80
#
# Plus the PADDING control: vacuous `have`s spliced ahead of the real proof must
# not RAISE the score, or the band pays for a proof-shaped imitation of the
# English argument -- the erosion channel reward.py:127-132 names by hand.
#
# TWO MODES. The default runs all three arms, which is the GATE. Pass
# ARMS=matched to spend the budget on scoring one model's own proofs instead,
# once the gate has already been passed for that judge.
#
# THE JUDGE HERE IS THE EVAL JUDGE, and it is deliberately NOT the one the
# training loop uses. Scoring a policy with the judge it was trained against
# measures how well it learned to please that judge; the two must stay disjoint.
#   training loop  Qwen/Qwen2.5-7B-Instruct   (hpc/grpo.slurm)
#   evaluation     microsoft/phi-4            (here)
#
# It defaulted to LARK-Lab/FormalRx-8b, which FAILED this very gate: AUC 0.488
# against the mismatched control (chance) and padding RAISED its score
# 0.310 -> 0.670. Four judges were gated on the same 120 rows and the Lean
# SPECIALIST came last; see CLAUDE.md's judge table. Do not restore a judge here
# from reputation.
#SBATCH --account=def-vganesh
#SBATCH --job-name=faith_eval
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/faith_eval_%j.out
#SBATCH --error=logs/faith_eval_%j.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh
stage_mathlib "[faith]"
export MATHLIB_ROOT="${MATHLIB_ROOT:-${REAL}/repos/mathlib4}"

# The policy's OWN proofs, not the golds: the reward would see these, and the
# term-mode share differs between the two populations (48.7% gold, 43.8% policy).
#
# GENS takes a whitespace-separated LIST and scores each in turn inside ONE
# judge's lifetime. Loading phi-4 is 22GB off shared storage plus a vLLM warm-up,
# so paying it per checkpoint would cost more than the scoring does. GEN stays
# as the single-file spelling.
# GENS_FILE is one path per line, and is how a batch of files should actually
# be passed: hpc/submit.sh folds every knob into a single --export=ALL,... list
# that sbatch splits on commas, so a whitespace-separated GENS survives only by
# luck. A file has no such problem.
if [ -n "${GENS_FILE:-}" ]; then
  [ -f "${GENS_FILE}" ] || { echo "[faith] FATAL: no such GENS_FILE: ${GENS_FILE}"; exit 1; }
  GENS="$(grep -v '^[[:space:]]*$' "${GENS_FILE}" | tr '\n' ' ')"
fi
GENS="${GENS:-${GEN:-${REAL}/results/eval/rl3b_locolib_proof_lr6_gated/gen_rl3b_locolib_proof_lr6_gated-step90_n760.jsonl}}"
for g in ${GENS}; do
  [ -f "${g}" ] || { echo "[faith] FATAL: no such gen file: ${g}"; exit 1; }
done

JUDGE_MODEL="${JUDGE_MODEL:-microsoft/phi-4}"
JUDGE_PORT="${JUDGE_PORT:-8733}"
N="${N:-120}"
# All three arms, unless told otherwise. `ARMS=matched` scores one model rather
# than gating a judge: the negative and padding controls test the JUDGE, and
# re-running them per checkpoint would pay for the same answer many times.
ARMS="${ARMS:-matched mismatched padded}"
# Lean is the serial part (one REPL per shard, BEQ_MAX_CONCURRENT=1 inside each),
# so shards buy real throughput; the judge batches their concurrent requests into
# single vLLM calls, which is why one judge serves all of them.
SHARDS="${SHARDS:-4}"
_first="$(basename "$(echo ${GENS} | awk '{print $1}')" .jsonl)"; _first="${_first#gen_}"
rename_job faith "${_first}"
echo "[faith] judge=${JUDGE_MODEL} n=${N} shards=${SHARDS} arms=${ARMS}"
echo "[faith] $(echo ${GENS} | wc -w) gen file(s) to score"

JUDGE_LOG="${REAL}/logs/faith_judge_${SLURM_JOB_ID:-manual}.log"
python3 "${REAL}/scripts/eval/faithfulness_judge_server.py" \
  --model "${JUDGE_MODEL}" --port "${JUDGE_PORT}" > "${JUDGE_LOG}" 2>&1 &
JUDGE_PID=$!
trap 'kill "${JUDGE_PID}" 2>/dev/null' EXIT

if ! python3 - "${JUDGE_PORT}" "${JUDGE_MODEL}" "${JUDGE_PID}" <<'PYWAIT'
import json, os, sys, time, urllib.request
port, want, pid = sys.argv[1], sys.argv[2], int(sys.argv[3])
deadline = time.time() + float(os.environ.get("JUDGE_WAIT_S", "1800"))
while time.time() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        print("[faith] judge process exited before becoming ready"); sys.exit(1)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            h = json.loads(r.read())
        if h.get("ok"):
            # Assert WHICH judge answered: a stale server on this port would
            # otherwise be used silently, and if it were the TRAINING judge the
            # whole point of this job would be gone.
            if h.get("model") != want:
                print(f"[faith] FATAL: port {port} serves {h.get('model')!r}, not {want!r}")
                sys.exit(1)
            print(f"[faith] judge ready: {h.get('model')}"); sys.exit(0)
    except Exception:
        pass
    time.sleep(10)
print("[faith] judge did not become ready in time"); sys.exit(1)
PYWAIT
then
  echo "[faith] FATAL: judge never came up; see ${JUDGE_LOG}"; exit 1
fi

export FAITH_JUDGE_URL="http://127.0.0.1:${JUDGE_PORT}/generate"
# Turn an adapter bug into a visible line rather than an anonymous unknown --
# every failure path in reward/faithfulness.py returns None, so without this an
# AttributeError in the glue looks exactly like Lean timing out.
export FAITH_DEBUG="${FAITH_DEBUG:-1}"

# --tactic-only spends the judge budget on the population that CAN be scored;
# the term-mode share is still recorded from the unfiltered pool, so coverage
# stays honest about the structural ceiling.
# WAIT ON THE SHARD PIDS, NEVER A BARE `wait`. The judge server above is also a
# background child of this shell and it never exits, so a bare `wait` blocks on
# it forever: the first arm completes, the loop hangs, and the job sits burning
# walltime with a healthy judge and no workers. That is exactly what happened on
# job 2628759 -- matched finished in 12 minutes, then 80 minutes of nothing.
for GEN in ${GENS}; do
  LABEL="$(basename "${GEN}" .jsonl)"; LABEL="${LABEL#gen_}"
  OUT="${REAL}/results/faithfulness/${LABEL}"
  mkdir -p "${OUT}"
  # Skip what is already scored, so a walltime kill costs the current file and
  # nothing before it. Same contract as grpo_eval.slurm's per-checkpoint cache.
  if [ -f "${OUT}/summary.json" ]; then
    echo "[faith] cached: ${LABEL}"; continue
  fi
  echo "[faith] ##### ${LABEL} -> ${OUT}"
  for arm in ${ARMS}; do
    echo "[faith] ===== ${LABEL} / ${arm} ====="
    pids=()
    for sh in $(seq 0 $((SHARDS - 1))); do
      python3 "${REAL}/scripts/eval/faithfulness_eval.py" \
        --gen "${GEN}" --arm "${arm}" --n "${N}" --tactic-only \
        --shard "${sh}" --num-shards "${SHARDS}" --out "${OUT}" &
      pids+=($!)
    done
    wait "${pids[@]}"
    echo "[faith] ${LABEL} / ${arm} complete"
  done
  python3 "${REAL}/scripts/eval/faithfulness_eval.py" --merge --out "${OUT}"
  echo "[faith] ${LABEL} done -> ${OUT}/summary.json"
done
echo "[faith] all gen files done"
