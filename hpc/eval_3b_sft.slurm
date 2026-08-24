#!/usr/bin/env bash
# Evaluate every 3B SFT epoch checkpoint and record which is best by BEq+.
# The whole 3B series is built on this checkpoint, so picking the last epoch by
# assumption would bake an avoidable error into everything downstream --
# select_checkpoint.py exists precisely because that mistake was made before.
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=eval_3b_sft
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/eval_3b_sft_%j.out
#SBATCH --error=logs/eval_3b_sft_%j.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh

stage_mathlib "[eval3b]"

N_EVAL="${N_EVAL:-1000}"
# TAG must match the one given to hpc/sft_3b.slurm. It scopes the checkpoint dir,
# the eval label (and therefore every results/ filename), and the best_sft
# pointer, so the mid-trained series cannot silently overwrite the original.
TAG="${TAG:-}"
SUFFIX="${TAG:+_${TAG}}"
LABEL="sft3b${TAG:+${TAG}}"
BEST_FILE="data_3b/best_sft${SUFFIX}.txt"
echo "[eval3b] series=${LABEL} ckpts=checkpoints/sft_3b${SUFFIX} pointer=${BEST_FILE}"
mkdir -p checkpoints/merged

for d in $(ls -d checkpoints/sft_3b${SUFFIX}/global_step_* 2>/dev/null | sort -t_ -k3 -n); do
  step=${d##*global_step_}
  out="results/eval_${LABEL}-step${step}_n${N_EVAL}.json"
  [ -f "$out" ] && { echo "[eval3b] cached step${step}"; continue; }
  ls "${d}/huggingface"/*.safetensors >/dev/null 2>&1 || { echo "[eval3b] SKIP step${step}"; continue; }
  # Unique dir name: evaluate_checkpoints.py labels records by directory name.
  model="checkpoints/merged/${LABEL}-step${step}"
  ln -sfn "${REAL}/${d}/huggingface" "${model}"
  echo "[eval3b] ===== ${LABEL}-step${step} ====="
  python3 scripts/evaluate_checkpoints.py --checkpoint "${model}" \
    --val-parquet "${REAL}/data_3b/val.parquet" \
    --n-eval "${N_EVAL}" --workers "${BEQ_WORKERS:-8}" --out "${out}" \
    || echo "[eval3b] FAILED step${step}"
done

# Record the winner so the rest of the chain trains from it rather than guessing.
LABEL="${LABEL}" SUFFIX="${SUFFIX}" BEST_FILE="${BEST_FILE}" N_EVAL="${N_EVAL}" python3 - <<'PY'
import json, glob, re, pathlib
best=None
import os
LABEL, SUFFIX, BEST = os.environ["LABEL"], os.environ["SUFFIX"], os.environ["BEST_FILE"]
# Restrict to the n this job actually ran. Globbing _n*.json would let a step
# evaluated at n=400 and another at n=1000 compete on rates measured over
# different example sets, and pick a "best" checkpoint on that basis.
N = os.environ["N_EVAL"]
for f in glob.glob(f"results/eval_{LABEL}-step*_n{N}.json"):
    step=int(re.search(r"-step(\d+)_", f).group(1))
    r=next(iter(json.load(open(f)).values()))
    if best is None or r["beq_plus_rate"] > best[1]:
        best=(step, r["beq_plus_rate"], r["typecheck_rate"])
if best is None:
    raise SystemExit(f"no {LABEL} evals found")
step, beq, tc = best
# ABSOLUTE: this path is handed to verl as actor_rollout_ref.model.path and to
# vLLM inside Ray workers, whose working directory is not this one. A relative
# path resolves fine in the shell here and then fails inside the workers.
p=str(pathlib.Path(f"checkpoints/sft_3b{SUFFIX}/global_step_{step}/huggingface").resolve())
pathlib.Path(BEST).write_text(p + "\n")
print(f"\n[eval3b] BEST {LABEL}: step {step} -- BEq+ {100*beq:.1f}%  type-check {100*tc:.1f}%")
print(f"[eval3b] wrote {BEST} -> {p}")
print("[eval3b] NOTE: n=1000 here; the first 400 rows match data/val.parquet")
print("[eval3b]       exactly, so this is still paired-comparable with 0.5B.")
PY
