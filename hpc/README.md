# Moving this PoC to Compute Canada / DRAC

Mirrors `MixtureOfMathExperts/hpc/`'s pattern (same `setup_cc.sh` / `cc_env.sh` /
`python_login.sh` split, same `def-vganesh` allocation default) — adapted for this
project's stack (verl + vLLM + lean-interact instead of PyPantograph/xlora/trl). None
of this has been run on an actual cluster yet; treat it as a strong starting point, not
a tested pipeline, and work through the checklist below before your first submission.

## Files

| File | Runs where | Purpose |
|---|---|---|
| `setup_cc.sh` | login node, once | Creates `$SCRATCH/ai4math_training_venv_hpc`, installs CC-wheelhouse packages, then verl/vLLM/lean-interact from PyPI (needs internet) |
| `cc_env.sh` | sourced by every job | Loads modules, activates the `$SCRATCH` venv, sets HF offline-mode env vars |
| `python_login.sh` | login node | Module-loading `python` wrapper, used by the Makefile's `$(PYTHON)` on CC |
| `lean_cache_and_build.sh` | login node, once | Downloads Mathlib4's prebuilt oleans, builds, warms the lean-interact REPL cache |
| `prefetch_models.py` | login node, once | Downloads the base model + Lean-Workbook dataset for offline compute-node access |
| `train.slurm` | `sbatch`'d | The actual training job — sources `cc_env.sh`, runs `configs/run_grpo.sh` |

## One-time setup

The project itself (this git checkout) can live anywhere, including `$HOME` — it's
small and git-tracked. `make setup-hpc` redirects everything that isn't (the venv,
`repos/verl` + `repos/mathlib4`'s build cache, model weights, the lean-interact REPL
cache) to flat `$SCRATCH/ai4math_training_*` directories, via `env`-vars for the venv/
models/cache and a `repos -> $SCRATCH/ai4math_training_repos` symlink for `repos/` (so
every script that references `repos/verl` / `repos/mathlib4` as a relative path keeps
working unchanged). This matters because `$HOME` on CC has a tight *file-count* quota
(not just space) that `repos/mathlib4`'s build cache alone (tens of thousands of files)
can blow through.

```bash
cd <wherever you cloned this repo>   # $HOME is fine
make setup-hpc      # = env-hpc + lean_cache_and_build.sh + prefetch_models.py + dataset prep
```

`make setup` also auto-detects Compute Canada and delegates to `setup-hpc`, so running
the local (uv, `$HOME` venv) path by habit isn't a trap anymore.

Equivalently, step by step:

```bash
bash hpc/setup_cc.sh              # .venv_hpc + verl/vllm/lean-interact
bash hpc/lean_cache_and_build.sh  # Mathlib4 @ v4.8.0-rc1
bash hpc/python_login.sh hpc/prefetch_models.py   # model + dataset
make dataset                       # build data/train.parquet, data/val.parquet
```

## Submitting a job

```bash
make submit-composite   # sbatch hpc/train.slurm --export=REWARD_FN_NAME=compute_score_composite
make submit-typecheck   # sbatch hpc/train.slurm --export=REWARD_FN_NAME=compute_score_typecheck_only
# or directly, with more control:
TOTAL_STEPS=200 sbatch hpc/train.slurm --export=ALL,REWARD_FN_NAME=compute_score_composite,TOTAL_STEPS=200
```

## Before your first real submission, check

1. **`--account`.** `train.slurm` defaults to `def-vganesh` (same as
   `MixtureOfMathExperts`, since this is a sibling project) — confirm that's still the
   right allocation, or change it.

2. **GPU / CUDA module vs. torch build.** `setup_cc.sh` installs torch from the CC
   wheelhouse (`--no-index`), which is built against whatever CUDA module is loaded
   (`cuda/12.2` by default in `cc_env.sh`/`setup_cc.sh`). Check the actual GPU node
   type (`nvidia-smi` after `salloc --gpus-per-node=1`) and bump the CUDA module if the
   wheelhouse's torch build doesn't support it. (Local dev on this project's RTX 5070
   Ti needed CUDA 13.x — that's very unlikely to be the situation on a CC GPU node,
   which are typically V100/A100/H100.)

3. **verl/vLLM/TransferQueue/lean-interact are NOT in the CC wheelhouse.**
   `setup_cc.sh` clones verl and installs these from PyPI/git in Phase 2 — that phase
   needs internet and must run on the login node, never inside a job.

4. **Compute nodes are offline.** `cc_env.sh` sets `HF_HUB_OFFLINE=1` /
   `TRANSFORMERS_OFFLINE=1` / `HF_DATASETS_OFFLINE=1` and points `HF_HOME` at whatever
   was populated by `prefetch_models.py` on the login node. If a job errors trying to
   reach the network, something wasn't prefetched — rerun `prefetch_models.py`, not the
   job.

5. **`~/.elan` visibility.** `lean_cache_and_build.sh` installs `elan` to `~/.elan` and
   builds Mathlib4 under `repos/mathlib4`. Confirm `~/.elan` is visible from compute
   nodes (some clusters separate home/scratch visibility) — `cc_env.sh` already adds
   `~/.elan/bin` to `PATH`.

6. **`#SBATCH` sizing.** `train.slurm`'s `--time=04:00:00`/`--mem=64G`/
   `--cpus-per-task=8` are sized for the PoC-scale `TOTAL_STEPS=30` default in
   `configs/run_grpo.sh`. Raise them together with `TOTAL_STEPS` for a longer real run.

7. **Scaling up.** On a cluster GPU (A100 40/80GB+) the memory ceiling that forced this
   PoC onto a 0.5B model / full-parameter training (see root `README.md`'s *Hardware
   notes*) goes away — a bigger base model and/or re-enabling LoRA (once its vLLM
   weight-transfer bug is worked around) are both just config changes at that point,
   not new engineering.
