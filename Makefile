# ============================================================
# ai4math_training — BEq+ RL PoC for Lean 4 Autoformalization
# ============================================================
# make setup             — full local setup (env + verl + Lean/Mathlib + model + data)
# make setup-hpc         — Compute Canada / DRAC cluster setup (see hpc/README.md)
# make env               — create the CUDA venv and install verl+vllm+lean-interact
# make env-hpc           — CC venv via hpc/setup_cc.sh (wheelhouse + verl/vllm from PyPI)
# make verl              — clone verl
# make mathlib           — build Mathlib4 at the pinned tag (matches the dataset)
# make model             — download the base model from Hugging Face
# make dataset           — prepare the Lean-Workbook train/val parquet files
# make smoke             — tiny (8/4-example) end-to-end GRPO smoke test
# make train-composite   — real GRPO run, BEq+ composite reward (local/interactive)
# make train-typecheck   — real GRPO run, type-check-only reward (ablation baseline)
# make train             — both of the above, sequentially (single GPU)
# make submit-composite  — sbatch hpc/train.slurm with the composite reward
# make submit-typecheck  — sbatch hpc/train.slurm with the typecheck-only reward
# make check-toolchain   — verify elan/Mathlib4/lean-interact all agree on the Lean version
# make clean             — remove Python cache artifacts
# make clean-lean        — remove Mathlib4 build artifacts
# ============================================================

SHELL       := /bin/bash
PROJECT     := $(shell pwd)
VENV        := $(PROJECT)/.venv
VENV_HPC    := $(PROJECT)/.venv_hpc

# Pinned to match internlm/Lean-Workbook's target toolchain (see reward/beq_plus.py).
# If you swap datasets, re-pin both of these together and re-run `make mathlib`.
LEAN_TOOLCHAIN := leanprover/lean4:v4.8.0-rc1
MATHLIB4_TAG   := v4.8.0-rc1

VERL_REPO   := https://github.com/volcengine/verl.git
MODEL_ID    := Qwen/Qwen2.5-Coder-0.5B-Instruct
MODEL_DIR   := models/qwen2.5-coder-0.5b-instruct

# CUDA build of torch to install — must match the driver's CUDA version
# (`nvidia-smi` header). Override with `make env TORCH_INDEX=...` if different.
TORCH_INDEX := https://download.pytorch.org/whl/cu130

# Same CVMFS-presence check MixtureOfMathExperts/Makefile uses to detect
# Compute Canada / DRAC nodes (more reliable than "is sbatch on PATH", which is
# also true on a plain login node without the software stack loaded).
ON_CC := $(shell [ -d /cvmfs/soft.computecanada.ca ] && echo "1" || echo "0")
ifeq ($(ON_CC),1)
  PYTHON := bash $(PROJECT)/hpc/python_login.sh
else
  PYTHON := $(VENV)/bin/python3
endif

# ── Setup ─────────────────────────────────────────────────────────────────────

.PHONY: setup
setup: env verl mathlib model dataset
	@echo ""
	@echo "Setup complete. Activate your environment:"
	@echo "  source $(VENV)/bin/activate"
	@echo "Then try:  make smoke"

.PHONY: setup-hpc
setup-hpc: env-hpc
	@bash hpc/lean_cache_and_build.sh
	@bash hpc/python_login.sh hpc/prefetch_models.py
	@$(MAKE) --no-print-directory dataset
	@echo ""
	@echo "HPC setup complete. See hpc/README.md — adapt module loads / #SBATCH"
	@echo "values to your cluster before submitting hpc/train.slurm."

# ── Python environment ───────────────────────────────────────────────────────
# Uses `uv` (https://astral.sh/uv) for venv + package management. Falls back to
# a clear error if uv isn't on PATH rather than silently doing the wrong thing.

UV := $(shell command -v uv 2>/dev/null)

.PHONY: env
env:
	@if [ -z "$(UV)" ]; then echo "uv not found — install from https://astral.sh/uv"; exit 1; fi
	@if [ ! -d "$(VENV)" ]; then uv venv --python 3.11 "$(VENV)"; fi
	@source "$(VENV)/bin/activate" && uv pip install torch --index-url $(TORCH_INDEX)
	@$(MAKE) --no-print-directory _install-deps VENVDIR=$(VENV)
	@echo "Python environment ready: $(VENV)"

## env-hpc does NOT use uv — Compute Canada / DRAC nodes don't reliably have it,
## and their wheelhouse (--no-index) packages need a plain venv against the
## loaded module's system site-packages. Delegates entirely to hpc/setup_cc.sh,
## which mirrors MixtureOfMathExperts/hpc/setup_cc.sh's phased approach.
.PHONY: env-hpc
env-hpc:
	@bash hpc/setup_cc.sh

# Shared by env and env-hpc — installs verl[vllm] (once repos/verl exists), plus
# lean-interact and the flashinfer patch this specific environment needs.
.PHONY: _install-deps
_install-deps:
	@source "$(VENVDIR)/bin/activate" && \
	  if [ -d repos/verl ]; then \
	    (cd repos/verl && uv pip install -e ".[vllm]"); \
	    uv pip install "TransferQueue==0.1.8"; \
	  fi; \
	  uv pip install "lean-interact==0.11.5" huggingface_hub datasets pandas pyarrow
	@$(MAKE) --no-print-directory patch-flashinfer VENVDIR=$(VENVDIR)

# vLLM's flashinfer dependency ships a broken type annotation
# (`array.array[int]`, not subscriptable pre-3.12) in one file, which crashes
# any vLLM engine init that touches the AllReduce fusion pass. Idempotent.
.PHONY: patch-flashinfer
patch-flashinfer:
	@f="$(VENVDIR)/lib/python3.11/site-packages/flashinfer/comm/fd_exchange.py"; \
	if [ -f "$$f" ] && grep -q 'array.array\[int\]' "$$f"; then \
	  sed -i 's/-> tuple\[tuple\[int, int, array.array\[int\]\]\]:/-> "tuple[tuple[int, int, array.array]]":/' "$$f"; \
	  echo "[patch-flashinfer] patched $$f"; \
	else \
	  echo "[patch-flashinfer] nothing to do"; \
	fi

# ── verl ──────────────────────────────────────────────────────────────────────

.PHONY: verl
verl: repos/verl

repos/verl:
	git clone --depth 1 $(VERL_REPO) repos/verl
	@$(MAKE) --no-print-directory _install-deps VENVDIR=$(VENV)

# ── Lean / Mathlib4 ───────────────────────────────────────────────────────────
# NOT reused from ../MixtureOfMathExperts — that project is pinned to Lean
# 4.23.0; this one is pinned to whatever the training dataset targets
# (currently v4.8.0-rc1, for internlm/Lean-Workbook). Own checkout, own build.

.PHONY: mathlib
mathlib: elan-install repos/mathlib4
	@echo "[mathlib] fetching prebuilt oleans (falls back to source build)..."
	@cd repos/mathlib4 && ~/.elan/bin/lake exe cache get
	@cd repos/mathlib4 && ~/.elan/bin/lake build
	@echo "Mathlib4 @ $(MATHLIB4_TAG) built"

repos/mathlib4:
	git clone --branch $(MATHLIB4_TAG) --depth 1 \
	  https://github.com/leanprover-community/mathlib4.git repos/mathlib4

.PHONY: elan-install
elan-install:
	@if command -v elan &>/dev/null; then \
	  echo "elan already installed: $$(elan --version)"; \
	else \
	  curl -sSf https://elan.lean-lang.org/elan-init.sh | sh -s -- -y; \
	fi

# Verifies Mathlib4's pinned toolchain matches LEAN_TOOLCHAIN (reward/beq_plus.py
# points lean-interact at repos/mathlib4 directly, so these must agree).
.PHONY: check-toolchain
check-toolchain:
	@want=$$(echo "$(LEAN_TOOLCHAIN)" | sed 's|.*lean4:v\{0,1\}||'); \
	have=$$(sed 's|.*lean4:v\{0,1\}||' repos/mathlib4/lean-toolchain 2>/dev/null); \
	echo "  LEAN_TOOLCHAIN=$$want  repos/mathlib4=$${have:-?}"; \
	if [ "$$have" != "$$want" ]; then \
	  echo "ERROR: repos/mathlib4 is pinned to a different Lean version."; \
	  exit 1; \
	fi; \
	echo "Toolchains agree"

# ── Model + dataset ───────────────────────────────────────────────────────────

.PHONY: model
model: $(MODEL_DIR)/config.json

$(MODEL_DIR)/config.json:
	@mkdir -p models/.hf_cache
	@HF_HOME=$(PROJECT)/models/.hf_cache $(PYTHON) -c \
	  "from huggingface_hub import snapshot_download; snapshot_download('$(MODEL_ID)', local_dir='$(MODEL_DIR)')"

# Dataset sizes are STAMPED, not plain file targets. A file target compares
# mtimes only, so `make dataset RL_N_TRAIN=4000` would see an up-to-date
# data/train.parquet and silently keep the 400-row one -- you would then run a
# "longer" training that quietly re-walked the same tiny set.
# Regenerating with the same sizes is a no-op in content: prepare_dataset.py
# seeds its shuffle and pins the val slice to a fixed offset, so val.parquet is
# byte-stable and already-cached eval results stay comparable.
# Defaults MUST match what is on disk, or `make train-sft` (which depends on
# these) would see a stamp mismatch and quietly rebuild a smaller dataset.
RL_N_TRAIN ?= 4000
RL_N_VAL   ?= 400

.PHONY: dataset
dataset:
	@want="n_train=$(RL_N_TRAIN) n_val=$(RL_N_VAL)"; \
	 if [ -f data/train.parquet ] && [ "$$(cat data/.stamp 2>/dev/null)" = "$$want" ]; then \
	   echo "[dataset] up to date ($$want)"; \
	 else \
	   HF_HOME=$(PROJECT)/models/.hf_cache $(PYTHON) scripts/prepare_dataset.py \
	     --n-train $(RL_N_TRAIN) --n-val $(RL_N_VAL) && echo "$$want" > data/.stamp; \
	 fi

# ── Run ───────────────────────────────────────────────────────────────────────

.PHONY: smoke
smoke:
	@source "$(VENV)/bin/activate" && $(PYTHON) scripts/prepare_dataset.py --out-dir data/smoke --n-train 8 --n-val 4
	@source "$(VENV)/bin/activate" && \
	 TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 ROLLOUT_N=2 TOTAL_EPOCHS=1 \
	 PROJECT_NAME=beqplus_smoke EXPERIMENT_NAME=smoke \
	 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	 HF_HOME=$(PROJECT)/models/.hf_cache \
	 bash configs/run_grpo.sh \
	   data.train_files=$(PROJECT)/data/smoke/train.parquet \
	   data.val_files=$(PROJECT)/data/smoke/val.parquet \
	   trainer.total_training_steps=1

.PHONY: train-composite
train-composite:
	@source "$(VENV)/bin/activate" && \
	 REWARD_FN_NAME=compute_score_composite \
	 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	 HF_HOME=$(PROJECT)/models/.hf_cache \
	 bash configs/run_grpo.sh trainer.total_training_steps=30 trainer.test_freq=10 trainer.save_freq=10

.PHONY: train-typecheck
train-typecheck:
	@source "$(VENV)/bin/activate" && \
	 REWARD_FN_NAME=compute_score_typecheck_only \
	 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	 HF_HOME=$(PROJECT)/models/.hf_cache \
	 bash configs/run_grpo.sh trainer.total_training_steps=30 trainer.test_freq=10 trainer.save_freq=10

# Single GPU: the two ablation arms cannot run concurrently.
.PHONY: train
train: train-composite train-typecheck

# ── Curriculum ────────────────────────────────────────────────────────────────
# Trains FROM SCRATCH in two phases: first half on the cheap dense type-check
# ("pass@") reward to learn Lean syntax, second half on BEq+ for semantics.
# Implemented as a checkpoint handoff because verl does not expose the global
# step to reward functions -- see scripts/run_curriculum.sh for the full
# rationale, including why phase 1 stops at the midpoint rather than converging.
#
#   TOTAL_STEPS=30                          total across both phases (default)
#   SWITCH_AT=15                            steps of phase 1 (default TOTAL/2)
#   PHASE2_REWARD=compute_score_shaped      graded ladder (default)
#   PHASE2_REWARD=compute_score_composite   strict 0.1*typecheck + 0.9*BEq+
.PHONY: train-curriculum
train-curriculum:
	@source "$(VENV)/bin/activate" && bash scripts/run_curriculum.sh

# ── SFT → RL (the pipeline that actually works) ───────────────────────────────
# Measured: SFT alone reaches 33.8% BEq+ vs 3.8% for the best from-scratch RL arm.
# BEq+ as a *reward* only informs the policy when it fires (almost never, early);
# the same signal as a *supervised target* is dense. SFT then gives RL a policy
# where BEq+ fires ~1/3 of the time, which removes the starvation that made
# from-scratch RL fail and lets RL run with a small rollout_n.
# NOTE: MERGED is defined further down (Evaluation section) and `:=` is
# simply-expanded, so referencing it here would silently yield an empty prefix.
# Derive from $(PROJECT) instead.
# SFT_RUN names the checkpoint directory. It defaults to the warm-start tag, so
# a from-scratch SFT lands in checkpoints/sft/scratch and `make merge-sft` finds
# it without being told where -- previously SFT_CKPT was hardcoded to one run's
# name while train-sft wrote to another, so merge-sft looked in the wrong place.
# Recursive (`?=`/`=`) assignment, not `:=`, because _INIT_TAG is defined below.
SFT_RUN    ?= $(_INIT_TAG)
SFT_CKPT   ?= $(PROJECT)/checkpoints/sft/$(SFT_RUN)
SFT_MERGED ?= $(PROJECT)/checkpoints/merged/sft-step30

# Stamped for the same reason as `dataset` above. Depends on `dataset` because
# prepare_sft_dataset.py reads data/val.parquet to exclude the eval set from SFT
# training -- without it the filter silently no-ops and SFT trains on the test set.
SFT_N_TRAIN ?= 20000
SFT_N_VAL   ?= 500

.PHONY: sft-dataset
sft-dataset: dataset
	@want="n_train=$(SFT_N_TRAIN) n_val=$(SFT_N_VAL)"; \
	 if [ -f data/sft/train.parquet ] && [ "$$(cat data/sft/.stamp 2>/dev/null)" = "$$want" ]; then \
	   echo "[sft-dataset] up to date ($$want)"; \
	 else \
	   source "$(VENV)/bin/activate" && HF_HOME=$(PROJECT)/models/.hf_cache \
	     python3 scripts/prepare_sft_dataset.py \
	       --n-train $(SFT_N_TRAIN) --n-val $(SFT_N_VAL) && \
	   echo "$$want" > data/sft/.stamp; \
	 fi

# ── Stage entry points: train-sft / train-rl ─────────────────────────────────
# Both take an OPTIONAL starting checkpoint via INIT=<merged HF dir>.
# With no INIT they train from the base model (from scratch).
#
#   make train-sft                                  # SFT from the base model
#   make train-sft INIT=checkpoints/merged/foo      # continue SFT from a ckpt
#   make train-rl                                   # RL from the base model
#   make train-rl INIT=checkpoints/merged/sft-step30
#   make train-rl INIT=... REWARD_FN_NAME=compute_score_guided TOTAL_STEPS=30
#
# INIT must be a MERGED HF directory (one containing model.safetensors), not a
# raw verl checkpoint -- verl saves FSDP shards, so run `make merge-sft` (or
# verl.model_merger) first. The recipe checks and fails loudly rather than
# letting vLLM load a weightless config, which is how a whole SFT eval once came
# back as 0%.
# Defaults chosen from failures, not taste:
#   SAVE_FREQ=10  -- an earlier version saved only at TOTAL_STEPS, so a crash at
#                    step 25/30 (Ray GCS killed by the host OOM killer) threw away
#                    1h48m of compute with nothing on disk. Checkpoint often.
#   TRAIN_BATCH_SIZE=8 -- run_grpo.sh defaults to 16, which doubles concurrent
#                    rollouts and therefore concurrent Lean work; the 16-wide run
#                    peaked at 45.8GB host RAM and was OOM-killed, while every
#                    completed run on this box used 8.
INIT ?=
_INIT_MODEL = $(if $(INIT),$(INIT),$(PROJECT)/$(MODEL_DIR))
# Tag runs so a from-scratch run and a warm-started one never share a checkpoint
# directory (they are different experiments and must not overwrite each other).
_INIT_TAG = $(if $(INIT),from_$(notdir $(INIT)),scratch)

.PHONY: _check-init
_check-init:
	@if [ -n "$(INIT)" ]; then \
	  test -d "$(INIT)" || { echo "INIT=$(INIT) does not exist"; exit 1; }; \
	  ls "$(INIT)"/*.safetensors >/dev/null 2>&1 || ls "$(INIT)"/pytorch_model*.bin >/dev/null 2>&1 || { \
	    echo "INIT=$(INIT) has no model weights (*.safetensors / pytorch_model*.bin)."; \
	    echo "It is probably a raw verl checkpoint -- merge it first, e.g.:"; \
	    echo "  make merge-sft      # for SFT checkpoints"; \
	    exit 1; }; \
	  echo "[init] warm-starting from $(INIT)"; \
	else \
	  echo "[init] no INIT given -- training from the base model $(MODEL_DIR)"; \
	fi

.PHONY: train-sft
train-sft: sft-dataset _check-init
	@MODEL_PATH=$(_INIT_MODEL) \
	 SAVE_PATH=$${SAVE_PATH:-$(SFT_CKPT)} \
	 bash scripts/run_sft.sh $(SFT_EXTRA)

.PHONY: train-rl
train-rl: _check-init
	@source "$(VENV)/bin/activate" && \
	 MODEL_PATH=$(_INIT_MODEL) \
	 REWARD_FN_NAME=$${REWARD_FN_NAME:-compute_score_guided} \
	 EXPERIMENT_NAME=$${EXPERIMENT_NAME:-rl_$(_INIT_TAG)_$${REWARD_FN_NAME:-compute_score_guided}} \
	 ROLLOUT_N=$${ROLLOUT_N:-4} AGENT_LOOP_WORKERS=$${AGENT_LOOP_WORKERS:-1} \
	 TRAIN_BATCH_SIZE=$${TRAIN_BATCH_SIZE:-8} PPO_MINI_BATCH_SIZE=$${PPO_MINI_BATCH_SIZE:-8} \
	 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	 HF_HOME=$(PROJECT)/models/.hf_cache \
	 bash configs/run_grpo.sh \
	   trainer.total_training_steps=$${TOTAL_STEPS:-30} \
	   trainer.test_freq=$${TEST_FREQ:-10} trainer.save_freq=$${SAVE_FREQ:-10}

# Back-compat alias; `make train-sft` is the current entry point.
.PHONY: sft
sft: train-sft

# SFT checkpoints are FSDP-sharded like RL ones; merge before vLLM can load them.
.PHONY: merge-sft
merge-sft:
	@source "$(VENV)/bin/activate" && \
	 step=$$(cat $(SFT_CKPT)/latest_checkpointed_iteration.txt 2>/dev/null); \
	 test -n "$$step" || { echo "No SFT checkpoint -- run 'make sft' first."; exit 1; }; \
	 python3 -m verl.model_merger merge --backend fsdp \
	   --local_dir $(SFT_CKPT)/global_step_$$step \
	   --target_dir $(PROJECT)/checkpoints/merged/sft-step$$step

# Thin wrappers over `train-rl` for the specific reward functions. Each accepts
# INIT= exactly like train-rl, e.g.
#   make train-guided INIT=checkpoints/merged/sft-step30
.PHONY: train-from-sft
train-from-sft:
	@$(MAKE) --no-print-directory train-rl INIT=$(SFT_MERGED)

.PHONY: train-guided
train-guided:
	@$(MAKE) --no-print-directory train-rl REWARD_FN_NAME=compute_score_guided

.PHONY: train-shaped
train-shaped:
	@$(MAKE) --no-print-directory train-rl REWARD_FN_NAME=compute_score_shaped

# ── Evaluation ────────────────────────────────────────────────────────────────
# The head-to-head comparison the PoC exists to produce. NOTE: verl's own
# `val-core/.../acc/mean@1` is just the mean of that run's reward function, so
# the two arms' validation numbers are on different scales and are NOT
# comparable to each other. `make evaluate` re-scores both final checkpoints
# with BOTH metrics (type-check rate and BEq+ rate) so they can be compared.

STEP        ?= 30
CKPT_ROOT   := $(PROJECT)/checkpoints/beqplus_rl_poc
CKPT_COMP   := $(CKPT_ROOT)/qwen25_coder_0_5b_compute_score_composite/global_step_$(STEP)
CKPT_TC     := $(CKPT_ROOT)/qwen25_coder_0_5b_compute_score_typecheck_only/global_step_$(STEP)
MERGED      := $(PROJECT)/checkpoints/merged

# verl keeps actor weights in FSDP-sharded .pt files; actor/huggingface/ holds
# only config+tokenizer. vLLM needs real HF weights, so merge them first.
.PHONY: merge-checkpoints
merge-checkpoints:
	@source "$(VENV)/bin/activate" && \
	 for arm in composite typecheck_only; do \
	   src=$(CKPT_ROOT)/qwen25_coder_0_5b_compute_score_$$arm/global_step_$(STEP)/actor; \
	   dst=$(MERGED)/$$arm-step$(STEP); \
	   if [ -d "$$src" ] && [ ! -d "$$dst" ]; then \
	     echo "[merge] $$arm step $(STEP) -> $$dst"; \
	     python3 -m verl.model_merger merge --backend fsdp --local_dir "$$src" --target_dir "$$dst" || \
	       echo "[merge] FAILED for $$arm (checkpoint may not exist yet)"; \
	   else \
	     echo "[merge] skip $$arm (missing src or dst already exists)"; \
	   fi; \
	 done

# Parse verl's per-step console metrics out of logs/ into curves + a quantified
# per-reward impact table (dead steps split into starved vs saturated).
.PHONY: plots
plots:
	@source "$(VENV)/bin/activate" && python3 scripts/plot_results.py

# Evaluate ONE checkpoint dir (merging it first if needed) and write its own
# result JSON, so the expensive BEq+ scoring is never repeated for models that
# have already been measured. CKPT may be a raw verl run dir (latest step is
# picked automatically) or an already-merged HF dir.
#   make eval-ckpt CKPT=checkpoints/beqplus_rl_poc/rl_from_sft-step30_compute_score_guided
.PHONY: eval-ckpt
eval-ckpt:
	@test -n "$(CKPT)" || { echo "Usage: make eval-ckpt CKPT=<verl run dir | merged dir>"; exit 1; }
	@source "$(VENV)/bin/activate" && set -e; \
	 src="$(CKPT)"; name=$$(basename "$$src"); \
	 if ls "$$src"/*.safetensors >/dev/null 2>&1; then \
	   merged="$$src"; \
	 else \
	   step=$$(cat "$$src/latest_checkpointed_iteration.txt" 2>/dev/null); \
	   test -n "$$step" || { echo "No latest_checkpointed_iteration.txt in $$src"; exit 1; }; \
	   merged="$(PROJECT)/checkpoints/merged/$${name}-step$${step}"; \
	   if [ ! -d "$$merged" ]; then \
	     echo "[merge] $$src/global_step_$$step -> $$merged"; \
	     python3 -m verl.model_merger merge --backend fsdp \
	       --local_dir "$$src/global_step_$$step/actor" --target_dir "$$merged"; \
	   fi; \
	 fi; \
	 out="$(PROJECT)/results/eval_$$(basename $$merged)_n$${N_EVAL:-80}.json"; \
	 echo "[eval] $$merged -> $$out"; \
	 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$(PROJECT)/models/.hf_cache \
	 python3 scripts/evaluate_checkpoints.py --checkpoint "$$merged" \
	   --n-eval $${N_EVAL:-80} --out "$$out"

# Aggregate every cached result JSON into one table (no re-scoring).
.PHONY: compare
compare:
	@source "$(VENV)/bin/activate" && python3 scripts/compare_results.py

.PHONY: evaluate
evaluate: merge-checkpoints
	@source "$(VENV)/bin/activate" && \
	 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	 HF_HOME=$(PROJECT)/models/.hf_cache \
	 python3 scripts/evaluate_checkpoints.py \
	   --checkpoint base \
	   --checkpoint $(MERGED)/typecheck_only-step$(STEP) \
	   --checkpoint $(MERGED)/composite-step$(STEP) \
	   --n-eval $${N_EVAL:-80}

# ── HPC submission (SLURM) ────────────────────────────────────────────────────
# train-composite/train-typecheck above are for local/interactive runs (they
# always activate $(VENV), not $(VENV_HPC)). On a cluster, submit hpc/train.slurm
# instead — these targets are just a one-line convenience wrapper for it.

.PHONY: submit-composite
submit-composite:
	sbatch hpc/train.slurm --export=REWARD_FN_NAME=compute_score_composite,TOTAL_STEPS=$${TOTAL_STEPS:-200}

.PHONY: submit-typecheck
submit-typecheck:
	sbatch hpc/train.slurm --export=REWARD_FN_NAME=compute_score_typecheck_only,TOTAL_STEPS=$${TOTAL_STEPS:-200}

# ── Clean ─────────────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	find $(PROJECT) -type d -name __pycache__ -not -path "*/repos/*" -not -path "*/.venv*" -exec rm -rf {} + 2>/dev/null; true

# Kill every process a training run can leave behind. Run this between runs --
# vLLM's engine workers are spawned children that do NOT match `ray::`, so a
# pattern that only catches Ray actors leaves a multi-GB GPU+RAM squatter alive.
# A leftover worker is what caused Ray's OOM killer to take out the reward
# workers mid-run (symptom: "running: N, finished: 0" forever, no Lean process).
.PHONY: kill-stale
kill-stale:
	@for p in $$(ps -eo pid,args | grep -E "ray::|main_ppo|run_curriculum|raylet|gcs_server|VLLM::|vllm" | grep -v grep | awk '{print $$1}'); do \
	  kill -9 $$p 2>/dev/null || true; \
	done; sleep 5; \
	echo "remaining: $$(ps -eo args | grep -E 'ray::|main_ppo|VLLM::' | grep -v grep | wc -l)"; \
	echo "gpu: $$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || echo n/a)"; \
	free -h 2>/dev/null | sed -n '2p' || true

# Raw verl checkpoints are ~6.1GB each (FSDP shards + optimizer state); the
# merged HF weights that evaluation actually loads are 954MB. Six finished runs
# filled this disk to 100%. This prunes RAW checkpoints that are NOT the latest
# step of their run -- they are only useful for resuming a run mid-flight, which
# is meaningless once the run has finished. Merged dirs are never touched.
#
# Prints what it would remove; pass CONFIRM=1 to actually delete.
#   make prune-checkpoints            # dry run
#   make prune-checkpoints CONFIRM=1
.PHONY: prune-checkpoints
prune-checkpoints:
	@total=0; \
	 for run in checkpoints/*/*/; do \
	   latest=$$(cat "$$run/latest_checkpointed_iteration.txt" 2>/dev/null) || continue; \
	   [ -n "$$latest" ] || continue; \
	   for step in "$$run"global_step_*; do \
	     [ -d "$$step" ] || continue; \
	     [ "$$(basename $$step)" = "global_step_$$latest" ] && continue; \
	     sz=$$(du -sm "$$step" | cut -f1); total=$$((total+sz)); \
	     if [ -n "$(CONFIRM)" ]; then echo "[prune] rm $$step ($${sz}MB)"; rm -rf "$$step"; \
	     else echo "[dry-run] would rm $$step ($${sz}MB)"; fi; \
	   done; \
	 done; \
	 echo "---"; \
	 if [ -n "$(CONFIRM)" ]; then echo "freed $${total}MB"; \
	 else echo "would free $${total}MB   (re-run with CONFIRM=1 to delete)"; fi; \
	 df -h . | tail -1

.PHONY: clean-lean
clean-lean:
	rm -rf repos/mathlib4/.lake/build || true

.PHONY: clean-all
clean-all: clean clean-lean

# ── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo ""
	@echo "ai4math_training — BEq+ RL PoC targets"
	@echo "  make setup             Full local setup (env + verl + Lean/Mathlib + model + data)"
	@echo "  make setup-hpc         Compute Canada / DRAC setup (see hpc/README.md first)"
	@echo "  make env               Create CUDA venv, install verl[vllm] + lean-interact"
	@echo "  make env-hpc           CC venv via hpc/setup_cc.sh (wheelhouse + verl/vllm)"
	@echo "  make verl              Clone verl"
	@echo "  make mathlib           Build Mathlib4 at the pinned tag"
	@echo "  make model             Download the base model"
	@echo "  make dataset           Prepare Lean-Workbook train/val parquet"
	@echo "  make smoke             Tiny end-to-end GRPO smoke test (~1 step, 8 examples)"
	@echo "  make train-composite   Real GRPO run: type-check + BEq+ composite reward (local)"
	@echo "  make train-typecheck   Real GRPO run: type-check-only reward (local, ablation)"
	@echo "  make train             Both ablation arms, sequentially (local)"
	@echo "  make train-shaped      GRPO with the graded/process-level BEq+ reward"
	@echo "  make train-curriculum  Two-phase curriculum from scratch (pass@ then BEq+)"
	@echo "  make train-sft         SFT; INIT=<merged dir> to warm-start, else from scratch"
	@echo "  make train-rl          GRPO; INIT=<merged dir> to warm-start, else from scratch"
	@echo "                           REWARD_FN_NAME=compute_score_{guided,shaped,composite,typecheck_only}"
	@echo "  make merge-sft         Merge the SFT checkpoint to HF format"
	@echo "  make train-from-sft    RL starting from the SFT policy (small rollout_n)"
	@echo "  make evaluate          Score checkpoints on BOTH metrics (the valid comparison)"
	@echo "  make plots             Training curves + per-reward impact quantification"
	@echo "  make submit-composite  sbatch hpc/train.slurm with the composite reward"
	@echo "  make submit-typecheck  sbatch hpc/train.slurm with the typecheck-only reward"
	@echo "  make check-toolchain   Verify Lean/Mathlib4 toolchain pin"
	@echo "  make kill-stale        Kill leftover ray/vLLM processes between runs"
	@echo "  make clean             Remove Python cache files"
	@echo "  make clean-lean        Remove Mathlib4 build artifacts"
	@echo ""

.DEFAULT_GOAL := help
