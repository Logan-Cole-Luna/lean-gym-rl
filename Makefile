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

# Pinned to match internlm/Lean-Workbook's target toolchain (see reward/beq_plus.py).
# If you swap datasets, re-pin both of these together and re-run `make mathlib`.
LEAN_TOOLCHAIN := leanprover/lean4:v4.8.0-rc1
MATHLIB4_TAG   := v4.8.0-rc1

VERL_REPO   := https://github.com/volcengine/verl.git
MODEL_ID    := Qwen/Qwen2.5-Coder-0.5B-Instruct

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

# Model weights + HF cache are large, easily re-downloaded, and not the thing
# that needs backing up — on CC they go under $SCRATCH instead of $(PROJECT),
# which sits on a group-shared /project filesystem with a tight file-count quota.
ifeq ($(ON_CC),1)
  MODELS_ROOT := $(shell echo $$SCRATCH)/ai4math_training_models
else
  MODELS_ROOT := $(PROJECT)/models
endif
MODEL_DIR := $(MODELS_ROOT)/qwen2.5-coder-0.5b-instruct

# The CC venv (verl+vllm+torch+ray, ~150 packages) is tens of thousands of
# small files — also goes under $SCRATCH for the same file-count-quota reason.
ifeq ($(ON_CC),1)
  VENV_HPC := $(shell echo $$SCRATCH)/ai4math_training_venv_hpc
else
  VENV_HPC := $(PROJECT)/.venv_hpc
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
	@mkdir -p $(MODELS_ROOT)/.hf_cache
	@HF_HOME=$(MODELS_ROOT)/.hf_cache $(PYTHON) -c \
	  "from huggingface_hub import snapshot_download; snapshot_download('$(MODEL_ID)', local_dir='$(MODEL_DIR)')"

.PHONY: dataset
dataset: data/train.parquet

data/train.parquet: scripts/prepare_dataset.py
	@HF_HOME=$(MODELS_ROOT)/.hf_cache $(PYTHON) scripts/prepare_dataset.py

# ── Run ───────────────────────────────────────────────────────────────────────

.PHONY: smoke
smoke:
	@source "$(VENV)/bin/activate" && $(PYTHON) scripts/prepare_dataset.py --out-dir data/smoke --n-train 8 --n-val 4
	@source "$(VENV)/bin/activate" && \
	 TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 ROLLOUT_N=2 TOTAL_EPOCHS=1 \
	 PROJECT_NAME=beqplus_smoke EXPERIMENT_NAME=smoke \
	 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	 HF_HOME=$(MODELS_ROOT)/.hf_cache \
	 bash configs/run_grpo.sh \
	   data.train_files=$(PROJECT)/data/smoke/train.parquet \
	   data.val_files=$(PROJECT)/data/smoke/val.parquet \
	   trainer.total_training_steps=1

.PHONY: train-composite
train-composite:
	@source "$(VENV)/bin/activate" && \
	 REWARD_FN_NAME=compute_score_composite \
	 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	 HF_HOME=$(MODELS_ROOT)/.hf_cache \
	 bash configs/run_grpo.sh trainer.total_training_steps=30 trainer.test_freq=10 trainer.save_freq=10

.PHONY: train-typecheck
train-typecheck:
	@source "$(VENV)/bin/activate" && \
	 REWARD_FN_NAME=compute_score_typecheck_only \
	 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	 HF_HOME=$(MODELS_ROOT)/.hf_cache \
	 bash configs/run_grpo.sh trainer.total_training_steps=30 trainer.test_freq=10 trainer.save_freq=10

# Single GPU: the two ablation arms cannot run concurrently.
.PHONY: train
train: train-composite train-typecheck

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
	@echo "  make submit-composite  sbatch hpc/train.slurm with the composite reward"
	@echo "  make submit-typecheck  sbatch hpc/train.slurm with the typecheck-only reward"
	@echo "  make check-toolchain   Verify Lean/Mathlib4 toolchain pin"
	@echo "  make clean             Remove Python cache files"
	@echo "  make clean-lean        Remove Mathlib4 build artifacts"
	@echo ""

.DEFAULT_GOAL := help
