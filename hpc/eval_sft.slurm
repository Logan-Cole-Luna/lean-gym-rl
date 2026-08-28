#!/usr/bin/env bash
# Evaluate every SFT epoch checkpoint and record which is best by BEq+.
#   sbatch --export=ALL hpc/eval_sft.slurm
#   sbatch --export=ALL,TAG=locolib,VAL_PARQUET=data_locolib/val.parquet,N_EVAL=999 hpc/eval_sft.slurm
#
# Everything downstream trains from the checkpoint this picks, so it is measured
# rather than assumed to be the last epoch.
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=eval_sft
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/eval_sft_%j.out
#SBATCH --error=logs/eval_sft_%j.err

source /home/logan03/lean-gym-rl/hpc/job_prelude.sh

stage_mathlib "[evalsft]"

N_EVAL="${N_EVAL:-1000}"
# A corpus must be evaluated against the val slice it was built from: BEq+ needs
# a gold, so scoring against another corpus's golds measures transfer, not fit.
VAL_PARQUET="${VAL_PARQUET:-${DATA_DIR}/val.parquet}"

# TAG must match the one given to hpc/sft.slurm; it scopes the checkpoint dir,
# every results/ filename, and the best-SFT pointer.
TAG="${TAG:-}"
SUFFIX="${TAG:+_${TAG}}"
SERIES="${SFT_LABEL}${TAG}"
# OUT_LABEL names the results files and merged symlink; SERIES names the
# checkpoint directory. They differ only when cross-evaluating one series on
# another corpus's val slice, which must not write records under the home
# series' label -- select_checkpoint.py merges by the label inside the json.
LABEL="${OUT_LABEL:-${SERIES}}"
BEST_FILE="${BEST_FILE:-${DATA_DIR}/best_sft${SUFFIX}.txt}"
CKPT_DIR="checkpoints/${SFT_DIR}${SUFFIX}"
echo "[evalsft] series=${SERIES} label=${LABEL} ckpts=${CKPT_DIR} val=${VAL_PARQUET}"
rename_job evalsft "${TAG}"
mkdir -p checkpoints/merged

for d in $(ls -d ${CKPT_DIR}/global_step_* 2>/dev/null | sort -t_ -k3 -n); do
  step=${d##*global_step_}
  out="results/eval_${LABEL}-step${step}_n${N_EVAL}.json"
  [ -f "$out" ] && { echo "[evalsft] cached step${step}"; continue; }
  ls "${d}/huggingface"/*.safetensors >/dev/null 2>&1 || { echo "[evalsft] SKIP step${step}"; continue; }
  # evaluate_checkpoints.py labels records by directory name, so the symlink
  # name must be unique per checkpoint.
  model="checkpoints/merged/${LABEL}-step${step}"
  ln -sfn "${REAL}/${d}/huggingface" "${model}"
  echo "[evalsft] ===== ${LABEL}-step${step} ====="
  python3 scripts/eval/evaluate_checkpoints.py --checkpoint "${model}" \
    --val-parquet "${REAL}/${VAL_PARQUET}" \
    --n-eval "${N_EVAL}" --workers "${BEQ_WORKERS:-8}" --out "${out}" \
    || echo "[evalsft] FAILED step${step}"
done

LABEL="${LABEL}" CKPT_DIR="${CKPT_DIR}" BEST_FILE="${BEST_FILE}" N_EVAL="${N_EVAL}" python3 - <<'PY'
import glob, json, os, pathlib, re

LABEL = os.environ["LABEL"]
CKPT_DIR = os.environ["CKPT_DIR"]
BEST = os.environ["BEST_FILE"]
N = os.environ["N_EVAL"]

# Restrict to the n this job ran: comparing a step evaluated at n=400 against one
# at n=1000 would pick a winner on rates measured over different example sets.
best = None
for f in glob.glob(f"results/eval_{LABEL}-step*_n{N}.json"):
    step = int(re.search(r"-step(\d+)_", f).group(1))
    r = next(iter(json.load(open(f)).values()))
    if best is None or r["beq_plus_rate"] > best[1]:
        best = (step, r["beq_plus_rate"], r["typecheck_rate"])
if best is None:
    raise SystemExit(f"no {LABEL} evals found")

step, beq, tc = best
# Absolute: this path is handed to verl and to vLLM inside Ray workers, whose
# working directory is not this one.
p = str(pathlib.Path(f"{CKPT_DIR}/global_step_{step}/huggingface").resolve())
pathlib.Path(BEST).write_text(p + "\n")
print(f"\n[evalsft] BEST {LABEL}: step {step} -- BEq+ {100*beq:.1f}%  type-check {100*tc:.1f}%")
print(f"[evalsft] wrote {BEST} -> {p}")
PY
