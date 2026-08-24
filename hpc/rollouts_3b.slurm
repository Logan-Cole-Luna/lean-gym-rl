#!/usr/bin/env bash
# Generate + BEq+-score k=8 rollouts from the best 3B SFT checkpoint, then
# measure the group geometry and fit the placebo control to it.
#
# This job is the GO/NO-GO for the whole 3B RL series. "Informative" is a
# hump-shaped function of model strength -- 0/8 and 8/8 both give zero advantage
# -- so a stronger model can have FEWER learnable groups, not more. If that has
# happened, the dependent GRPO jobs must not run.
#SBATCH --account=def-vganesh_gpu
#SBATCH --job-name=rollouts_3b
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/rollouts_3b_%j.out
#SBATCH --error=logs/rollouts_3b_%j.err

set -uo pipefail
export PROJECT_ROOT=/home/logan03/lean-gym-rl
REAL="${PROJECT_ROOT}"
source "${REAL}/hpc/cc_env.sh"
unset ROCR_VISIBLE_DEVICES
cd "${REAL}"
export PYTHONPATH="${REAL}:${PYTHONPATH:-}"

if [ -n "${SLURM_TMPDIR:-}" ] && [ -f /scratch/logan03/mathlib4_v4.8.0-rc1.tar ]; then
  t0=$(date +%s)
  tar -xf /scratch/logan03/mathlib4_v4.8.0-rc1.tar -C "${SLURM_TMPDIR}"
  cp -a "${LEAN_INTERACT_CACHE_DIR}" "${SLURM_TMPDIR}/lean_interact_cache"
  export MATHLIB_ROOT="${SLURM_TMPDIR}/mathlib4"
  export LEAN_INTERACT_CACHE_DIR="${SLURM_TMPDIR}/lean_interact_cache"
  echo "[roll3b] staging took $(( $(date +%s) - t0 ))s"
fi
export BEQ_ENV_TIMEOUT="${BEQ_ENV_TIMEOUT:-2400}"

# BEST_SFT / TAG select the series. The geometry gate is one of the two
# measurements that judge mid-training (the other is pass@32), so it must be
# runnable against the mid-trained checkpoint without clobbering the original
# stats, placebo constants, or rollout files.
BEST_SFT="${BEST_SFT:-data_3b/best_sft.txt}"
TAG="${TAG:-}"
SUFFIX="${TAG:+_${TAG}}"
CKPT=$(cat "${BEST_SFT}" 2>/dev/null)
[ -n "${CKPT}" ] || { echo "[roll3b] FATAL: ${BEST_SFT} missing"; exit 1; }
case "${CKPT}" in /*) ;; *) CKPT="$(cd "$(dirname "${CKPT}")" && pwd)/$(basename "${CKPT}")" ;; esac
[ -d "${CKPT}" ] || { echo "[roll3b] FATAL: checkpoint dir not found: ${CKPT}"; exit 1; }
echo "[roll3b] rolling out from ${CKPT}"

# GEOMETRY SUBSAMPLE. This job exists to measure the group geometry and fit the
# placebo -- both are single proportions, so a sample is as good as the census.
# The full 4,300-prompt pool at k=8 is 34,400 BEq+ calls = ~20h at 8 workers,
# which does not fit any sane walltime; 600 prompts is 2.8h and pins the
# informative fraction to about +/-3.7% at 95%, far tighter than the gate needs.
# GRPO still TRAINS on all 4,300 -- only this measurement is subsampled.
N_GEOM="${N_GEOM:-600}"
mkdir -p data_3b/rollouts
python3 - "$N_GEOM" "$SUFFIX" <<'PYX'
import sys, pandas as pd
n=int(sys.argv[1]); suffix=sys.argv[2]; df=pd.read_parquet("data_3b/train.parquet")
df.sample(n=min(n,len(df)), random_state=0).reset_index(drop=True).to_parquet(f"data_3b/geom_subset{suffix}.parquet")
print(f"[roll3b] geometry subsample: {min(n,len(df))} of {len(df)} RL prompts -> data_3b/geom_subset{suffix}.parquet")
PYX
RTAG="sft3b${SUFFIX}_k8"
OUT="data_3b/rollouts/${RTAG}.jsonl"
SCORED="data_3b/rollouts/${RTAG}.scored.jsonl"

# Same temperature as training, or the measured geometry describes a different
# sampling distribution than the one GRPO will actually see.
if [ ! -f "${OUT}" ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 scripts/generate_rollouts.py --checkpoint "${CKPT}" \
    --parquet "data_3b/geom_subset${SUFFIX}.parquet" --k 8 --temperature 1.15 --out "${OUT}" || exit 1
fi
python3 scripts/score_rollouts.py --rollouts "${OUT}" --out "${SCORED}" --workers 8 || exit 1

python3 scripts/build_rft_dataset.py --scored "${SCORED}" --rollouts "${OUT}" \
  --filter beq_plus --out-dir "data_3b/rft_beq${SUFFIX}" --stats-out "results/rollout_stats_3b${SUFFIX}.json"

# ---- GO/NO-GO --------------------------------------------------------------
python3 - "${SCORED}" <<'PY' || exit 1
import json, sys, collections
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by=collections.defaultdict(list)
for r in rows: by[r["prompt_index"]].append(r)
k=max(len(v) for v in by.values())
g=[v for v in by.values() if len(v)==k]
wins=[sum(bool(x["beq_plus"]) for x in v) for v in g]; P=len(g)
starved=sum(1 for c in wins if c==0); sat=sum(1 for c in wins if c==k)
inf=P-starved-sat
pass1=sum(wins)/(P*k)
print(f"\n=== 3B group geometry ({P} groups, k={k}) ===")
print(f"pass@1 {100*pass1:.1f}%   starved {100*starved/P:.1f}%   "
      f"saturated {100*sat/P:.1f}%   INFORMATIVE {100*inf/P:.1f}%")
print(f"(0.5B reference: pass@1 38.9%, starved 47.0%, saturated 22.1%, informative 30.9%)")
if inf/P < 0.25 or sat/P > 0.45:
    print("\nGATE FAILED: too few learnable groups -- the scale-up moved the policy")
    print("past the informative peak (or not far enough). Dependent GRPO jobs will")
    print("NOT run. Reconsider before spending GPU-days.")
    sys.exit(1)
if inf/P <= 0.309:
    print("\nGATE PASSED but informative did NOT improve on 0.5B's 30.9%.")
    print("GRPO gains no extra signal from the scale-up; weigh that when reading")
    print("the results rather than treating 3B as strictly better.")
else:
    print(f"\nGATE PASSED: informative {100*inf/P:.1f}% > 30.9% at 0.5B.")
PY

# ---- fit the control to THIS policy ----------------------------------------
python3 scripts/calibrate_placebo.py --scored "${SCORED}" \
  --out "data_3b/placebo_constants${SUFFIX}.sh" || exit 1
echo "[roll3b] done"
