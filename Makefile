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
# make smoke             — tiny (8/4-example) end-to-end GRPO smoke test
# make train             — both of the above, sequentially (single GPU)
# make submit ARM=…      — sbatch hpc/grpo.slurm for one arm
# make check-toolchain   — verify elan/Mathlib4/lean-interact all agree on the Lean version
# make clean             — remove Python cache artifacts
# make clean-lean        — remove Mathlib4 build artifacts
# ============================================================

SHELL       := /bin/bash
PROJECT     := $(shell pwd)
VENV        := $(PROJECT)/.venv

# Pinned to match LoCoLib's target toolchain (see CLAUDE.md, reward/beq_plus.py).
# If you swap datasets, re-pin both of these together and re-run `make mathlib`.
LEAN_TOOLCHAIN := leanprover/lean4:v4.23.0
MATHLIB4_TAG   := v4.23.0

# verl's main branch has since moved to verl==0.10.0.dev0, which hard-pins its
# [vllm] extra to vllm==0.24.0 and pulls in a much heavier dependency tree
# (qwen_vl_utils's audio/video codecs, Ascend NPU's TransferQueue, etc.) that
# doesn't build here (pyav vs. Cython 3 incompatibility). v0.9.0 is the last
# tag with the original vllm>=0.18.0 range this project was built against, and
# skips those extras entirely (they're separate extras, not part of [vllm]).
# Keep in sync with hpc/setup_cc.sh's VERL_TAG.
VERL_TAG    := v0.9.0
VERL_REPO   := https://github.com/volcengine/verl.git
# Override to change model scale, e.g.
#   make model MODEL_ID=Qwen/Qwen2.5-Coder-3B-Instruct
# MODEL_DIR below is DERIVED from this, so the two can never drift apart.
MODEL_ID    ?= Qwen/Qwen2.5-Coder-0.5B-Instruct

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

# Every recipe below activates through this rather than hardcoding
# $(VENV)/bin/activate, which does not exist on Compute Canada (the venv lives
# on $SCRATCH and needs the module stack loaded first). See hpc/activate.sh.
ACTIVATE := source $(PROJECT)/hpc/activate.sh

# Model weights + HF cache are large, easily re-downloaded, and not the thing
# that needs backing up — on CC they go under $SCRATCH instead of $(PROJECT),
# which sits on a group-shared /project filesystem with a tight file-count quota.
ifeq ($(ON_CC),1)
  MODELS_ROOT := $(shell echo $$SCRATCH)/ai4math_training_models
else
  MODELS_ROOT := $(PROJECT)/models
endif
# Derived from MODEL_ID (basename, lowercased) rather than hardcoded, so
# `make model MODEL_ID=...` lands somewhere that matches what was asked for
# instead of silently overwriting the 0.5B checkout.
MODEL_SLUG := $(shell echo '$(MODEL_ID)' | sed 's|.*/||' | tr 'A-Z' 'a-z')
MODEL_DIR  := $(MODELS_ROOT)/$(MODEL_SLUG)

# The CC venv (verl+vllm+torch+ray, ~150 packages) is tens of thousands of
# small files — also goes under $SCRATCH for the same file-count-quota reason.
ifeq ($(ON_CC),1)
  SCRATCH_DIR := $(shell echo $$SCRATCH)
  VENV_HPC := $(SCRATCH_DIR)/ai4math_training_venv_hpc
else
  VENV_HPC := $(PROJECT)/.venv_hpc
endif

# ── Setup ─────────────────────────────────────────────────────────────────────

# On CC, `make setup` (uv, a $HOME venv) is the wrong path — it filled the
# $HOME file-count quota (uv's cache alone is 80k+ small files) the first time
# this got run there. Detect the cluster and hand off to `setup-hpc` instead of
# making the user remember which target to type.
.PHONY: setup
ifeq ($(ON_CC),1)
setup:
	@echo "[setup] Compute Canada / DRAC node detected — this is 'make setup-hpc', not 'make setup'."
	@echo "[setup] (uv + a \$$HOME venv will blow the file-count quota here; see hpc/README.md.)"
	@$(MAKE) --no-print-directory setup-hpc
else
setup: env verl mathlib model
	@echo ""
	@echo "Setup complete. Activate your environment:"
	@echo "  source $(VENV)/bin/activate"
	@echo ""
	@echo "Data: LoCoLib proof-pair dataset is in data_locolib/ (pre-prepared)."
	@echo "Then try:  make smoke"
endif

# repos/verl and repos/mathlib4's build cache are tens of thousands of files
# between them -- left under this project's own $(PROJECT)/repos (a $HOME
# checkout) they blow the file-count quota regardless of MODELS_ROOT/VENV_HPC
# already being on $SCRATCH. The project itself stays in $HOME (small, git-
# tracked, backed up); only `repos/` gets redirected, via a symlink so every
# script that references `repos/verl` / `repos/mathlib4` as a relative path
# keeps working unchanged. Idempotent, and preserves an existing repos/ clone
# on first run instead of discarding it.
.PHONY: _link-repos-hpc
_link-repos-hpc:
	@target="$(SCRATCH_DIR)/ai4math_training_repos"; \
	mkdir -p "$$target"; \
	if [ -L repos ]; then \
	  true; \
	elif [ -d repos ]; then \
	  echo "[repos] moving existing repos/ to $$target ..."; \
	  mv repos/* "$$target"/ 2>/dev/null || true; \
	  rmdir repos; \
	  ln -s "$$target" repos; \
	else \
	  ln -s "$$target" repos; \
	fi; \
	echo "[repos] repos -> $$target"

.PHONY: setup-hpc
setup-hpc: env-hpc
	@bash hpc/lean_cache_and_build.sh
	@bash hpc/python_login.sh hpc/prefetch_models.py
	@$(MAKE) --no-print-directory dataset
	@echo ""
	@echo "HPC setup complete. See hpc/README.md — adapt module loads / #SBATCH"
	@echo "values to your cluster before submitting hpc/grpo.slurm."

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

.PHONY: venv
venv:
	@if [ ! -d "$(VENV)" ]; then python3 -m venv "$(VENV)"; fi
	@source "$(VENV)/bin/activate" && pip install --upgrade pip && pip install torch --index-url $(TORCH_INDEX) && pip install -r requirements.txt
	@echo "Python environment ready: $(VENV)"
	@echo "Activate with: source $(VENV)/bin/activate"

## env-hpc does NOT use uv — Compute Canada / DRAC nodes don't reliably have it,
## and their wheelhouse (--no-index) packages need a plain venv against the
## loaded module's system site-packages. Delegates entirely to hpc/setup_cc.sh,
## which mirrors MixtureOfMathExperts/hpc/setup_cc.sh's phased approach.
.PHONY: env-hpc
env-hpc: _link-repos-hpc
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
	  uv pip install "lean-interact==0.11.5" huggingface_hub datasets pandas pyarrow zss
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
	git clone --branch $(VERL_TAG) --depth 1 $(VERL_REPO) repos/verl
	@$(MAKE) --no-print-directory _install-deps VENVDIR=$(VENV)

# ── Lean / Mathlib4 ───────────────────────────────────────────────────────────
# Pinned to 4.23.0 to match LoCoLib (proof-pair dataset). Own checkout, own build.

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

# LoCoLib proof-pair dataset is pre-prepared in data_locolib/ and included in the repo.
# The old Lean-Workbook dataset preparation (prepare_dataset.py) has been removed.
# If you need to work with a different dataset, see run.md or CLAUDE.md.

.PHONY: dataset
dataset:
	@echo "[dataset] ERROR: The Lean-Workbook dataset preparation has been removed."
	@echo "[dataset] We now use LoCoLib proof-pair data (pre-prepared in data_locolib/)."
	@echo "[dataset] See run.md for the current workflow."
	@exit 1

# ── Run ───────────────────────────────────────────────────────────────────────

.PHONY: smoke
smoke:
	@echo "[smoke] Smoke test: one GRPO step on data_locolib proof-pair data"
	@$(ACTIVATE) && \
	 TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=2 TOTAL_EPOCHS=1 \
	 PROJECT_NAME=beqplus_smoke EXPERIMENT_NAME=smoke \
	 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	 HF_HOME=$(MODELS_ROOT)/.hf_cache \
	 bash configs/run_grpo.sh \
	   data.train_files=$(PROJECT)/data_locolib/rl_proof.parquet \
	   data.val_files=$(PROJECT)/data_locolib/val_proof.parquet \
	   trainer.total_training_steps=1

# Two 30-step ablation arms, sequentially (single GPU: they cannot overlap).
.PHONY: train
train:
	@$(MAKE) --no-print-directory train-rl REWARD=composite      STEPS=30
	@$(MAKE) --no-print-directory train-rl REWARD=typecheck_only STEPS=30

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
	@$(ACTIVATE) && bash scripts/run_curriculum.sh

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

# SFT dataset is pre-prepared in data_locolib/ as sft_proof.parquet and val_sft_proof.parquet.

.PHONY: sft-dataset
sft-dataset:
	@echo "[sft-dataset] ERROR: The Lean-Workbook SFT dataset preparation has been removed."
	@echo "[sft-dataset] We now use LoCoLib proof-pair data (data_locolib/sft_proof.parquet)."
	@echo "[sft-dataset] See run.md for the current workflow."
	@exit 1

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
_INIT_MODEL = $(if $(INIT),$(INIT),$(MODEL_DIR))
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
train-sft: _check-init
	@MODEL_PATH=$(_INIT_MODEL) \
	 TRAIN_FILE=$${TRAIN_FILE:-$(PROJECT)/data_locolib/sft_proof.parquet} \
	 VAL_FILE=$${VAL_FILE:-$(PROJECT)/data_locolib/val_sft_proof.parquet} \
	 SAVE_PATH=$${SAVE_PATH:-$(SFT_CKPT)} \
	 bash scripts/train/run_sft.sh $(SFT_EXTRA)

# Defaults reflect the post-mortem of the 200-step guided run (BEq+ 38.8% ->
# 29.0%); configs/run_grpo.sh's header explains each one. The knobs that changed
# there are reward function, rollout_n, entropy, KL, and advantage
# normalisation; what changes HERE is only the step/validation cadence:
#   TEST_FREQ=10  -- validation now reports the real BEq+ rate as
#                    val-core/<data_source>/acc/mean@1 (reward functions emit
#                    `acc` = BEq+), so it is worth running often enough to
#                    actually select on. It costs a val pass over data/val.parquet.
#   SAVE_FREQ=20  -- checkpoint selection is only as good as its candidates, but
#                    each raw checkpoint is ~6.1GB (FSDP shards + optimizer), so
#                    granularity is bought with disk. 20 gives 5 candidates over
#                    a 100-step run for ~31GB. The disk has hit 100% here before.
#   MAX_CKPT_KEEP -- DERIVED from the two above, never set independently.
#                    verl's trainer.max_actor_ckpt_to_keep rmtree's the actor/
#                    subdir of older checkpoints, leaving a global_step_N/ shell
#                    holding only dataloader state. Its default of 2 against
#                    SAVE_FREQ=10 silently destroyed 8 of the 10 checkpoints of a
#                    100-step run -- the run completed, but there was almost
#                    nothing left to select over, and eval-sweep failed on the
#                    husks with a misleading "Repo id must be in the form ..."
#                    from transformers. Deriving it means the two knobs cannot
#                    drift apart again. Override only if you know the disk cost.
#
# PPO_MINI_BATCH_SIZE defaults to TRAIN_BATCH_SIZE, not a fixed number. verl's
# v1 trainer pads the batch up to lcm(dp_size, ppo_mini_batch_size * rollout.n)
# (trainer_base.py:_get_required_batch_multiple) -- if only TRAIN_BATCH_SIZE is
# lowered while PPO_MINI_BATCH_SIZE stays at its old value, the padder silently
# restores the ORIGINAL batch size with synthetic samples and the memory
# reduction you asked for never happens. Measured: TRAIN_BATCH_SIZE=4 alone
# with PPO_MINI_BATCH_SIZE left at 8 and ROLLOUT_N=8 logged "Upsampled batch
# from 32 to 64" and OOM'd on the exact same 64-sequence backward pass as
# before the "fix". Keeping them equal by default avoids this trap.
.PHONY: train-rl
train-rl: _check-init
	@$(ACTIVATE) && \
	 tbs=$${TRAIN_BATCH_SIZE:-8}; \
	 MODEL_PATH=$(_INIT_MODEL) \
	 REWARD_FN_NAME=$${REWARD_FN_NAME:-compute_score_$${REWARD:-gated}} \
	 EXPERIMENT_NAME=$${EXPERIMENT_NAME:-rl_$(_INIT_TAG)_$${REWARD_FN_NAME:-compute_score_$${REWARD:-gated}}} \
	 ROLLOUT_N=$${ROLLOUT_N:-8} AGENT_LOOP_WORKERS=$${AGENT_LOOP_WORKERS:-1} \
	 TRAIN_BATCH_SIZE=$$tbs PPO_MINI_BATCH_SIZE=$${PPO_MINI_BATCH_SIZE:-$$tbs} \
	 VALIDATION_DATA_DIR=$${VALIDATION_DATA_DIR:-$(PROJECT)/results/train/$${EXPERIMENT_NAME:-rl_$(_INIT_TAG)}/val_generations} \
	 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	 HF_HOME=$(MODELS_ROOT)/.hf_cache \
	 MAX_CKPT_KEEP=$${MAX_CKPT_KEEP:-$$(( $${TOTAL_STEPS:-$${STEPS:-100}} / $${SAVE_FREQ:-20} + 2 ))} \
	 bash configs/run_grpo.sh \
	   trainer.total_training_steps=$${TOTAL_STEPS:-$${STEPS:-100}} \
	   trainer.test_freq=$${TEST_FREQ:-10} trainer.save_freq=$${SAVE_FREQ:-20}

# Back-compat alias; `make train-sft` is the current entry point.
.PHONY: sft
sft: train-sft

# SFT checkpoints are FSDP-sharded like RL ones; merge before vLLM can load them.
.PHONY: merge-sft
merge-sft:
	@$(ACTIVATE) && \
	 step=$$(cat $(SFT_CKPT)/latest_checkpointed_iteration.txt 2>/dev/null); \
	 test -n "$$step" || { echo "No SFT checkpoint -- run 'make sft' first."; exit 1; }; \
	 python3 -m verl.model_merger merge --backend fsdp \
	   --local_dir $(SFT_CKPT)/global_step_$$step \
	   --target_dir $(PROJECT)/checkpoints/merged/sft-step$$step

# Thin wrappers over `train-rl` for the specific reward functions. Each accepts
# INIT= exactly like train-rl, e.g.
#   make train-guided INIT=checkpoints/merged/sft-step30
# The five former aliases (train-from-sft / train-guided / train-shaped /
# train-gated / train-gated-filtered) were each a one-line wrapper around
# train-rl with a different REWARD_FN_NAME. They are now flags on one target:
#
#   make train-rl REWARD=gated                 # semantic ladder (the default arm)
#   make train-rl REWARD=guided                # + similarity shaping
#   make train-rl REWARD=selfprove             # gold-free
#   make train-rl REWARD=typecheck_only        # exploitable proxy (probe only)
#   make train-rl REWARD=gated FILTER_GROUPS=1 # + DAPO group filtering
#   make train-rl INIT=$(SFT_MERGED)           # warm-start from the SFT policy
#
# REWARD is shorthand: REWARD=gated sets REWARD_FN_NAME=compute_score_gated.
# Pass REWARD_FN_NAME directly if you need a name that does not fit the pattern.
#
# NOTE these are LOCAL single-GPU runs, kept for smoke tests and 0.5B work. The
# 3B series runs on SLURM -- see hpc/grpo.slurm.

# ── Rejection-sampling (RFT) arm ──────────────────────────────────────────────
# The control that separates "BEq+ is a bad training signal" from "GRPO is the
# wrong way to consume it at this batch size".
#
# WHY THIS ARM EXISTS (measured, results/FINDINGS.md). At batch 4 x rollout_n 8
# only ~37% of rollout groups are informative, so each Adam update is driven by
# ~1.5 prompts. A placebo reward with NO semantic content but the same advantage
# geometry reproduces the gated arm's dead-step rate, advantage spread,
# grad_norm and KL drift -- i.e. the GRPO arms' training curves are consistent
# with pure noise. Rejection sampling has no advantage estimate at all, so it
# uses the SAME BEq+ verdicts with none of that variance.
#
# The three stages are split because their resource profiles differ by an order
# of magnitude: generation is GPU-bound and takes minutes, BEq+ scoring is
# CPU-bound and takes hours (and parallelises across processes offline, unlike
# during training where FSDP offload leaves no host RAM for extra Mathlibs).
ROLLOUT_DIR   := $(PROJECT)/data/rollouts
ROLLOUT_TAG   ?= sft390_k8
ROLLOUT_K     ?= 8
ROLLOUT_TEMP  ?= 1.15
SCORE_WORKERS ?= 4

.PHONY: rollouts
rollouts:
	@$(ACTIVATE) && \
	 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$(MODELS_ROOT)/.hf_cache \
	 python3 scripts/pool/generate_rollouts.py \
	   --checkpoint $${INIT:-$(SFT_MERGED)} --parquet $(PROJECT)/data/train.parquet \
	   --k $(ROLLOUT_K) --temperature $(ROLLOUT_TEMP) \
	   --out $(ROLLOUT_DIR)/$(ROLLOUT_TAG).jsonl

# Resumable and append-only: a pass over ~10k rollouts is hours long, so an
# interruption must not throw the completed work away.
.PHONY: score-rollouts
score-rollouts:
	@$(ACTIVATE) && python3 scripts/pool/score_rollouts.py \
	   --rollouts $(ROLLOUT_DIR)/$(ROLLOUT_TAG).jsonl \
	   --out $(ROLLOUT_DIR)/$(ROLLOUT_TAG).scored.jsonl \
	   --workers $(SCORE_WORKERS)

# Both arms come from the SAME scoring pass, so the only difference between them
# is the acceptance criterion -- which is exactly the paper's claim under test.
# The type-check filter accepts ~2x as many rollouts, so the second build is
# SIZE-MATCHED to the first: otherwise the comparison confounds reward quality
# with dataset size.
.PHONY: rft-data
rft-data:
	@$(ACTIVATE) && set -e; \
	 python3 scripts/misc/build_rft_dataset.py \
	   --scored $(ROLLOUT_DIR)/$(ROLLOUT_TAG).scored.jsonl \
	   --rollouts $(ROLLOUT_DIR)/$(ROLLOUT_TAG).jsonl \
	   --filter beq_plus --out-dir $(PROJECT)/data/rft_beq \
	   --stats-out $(PROJECT)/results/train/rollout_stats.json; \
	 n=$$(python3 -c "import pandas as pd,sys; print(len(pd.read_parquet('$(PROJECT)/data/rft_beq/train.parquet'))+len(pd.read_parquet('$(PROJECT)/data/rft_beq/val.parquet')))"); \
	 echo "[rft] size-matching type-check arm to $$n pairs"; \
	 python3 scripts/misc/build_rft_dataset.py \
	   --scored $(ROLLOUT_DIR)/$(ROLLOUT_TAG).scored.jsonl \
	   --rollouts $(ROLLOUT_DIR)/$(ROLLOUT_TAG).jsonl \
	   --filter typecheck --out-dir $(PROJECT)/data/rft_tc --match-size $$n

# Continue training from the SFT policy on its OWN verified generations.
# LR is an order of magnitude below the SFT stage's 1e-4: this is a refinement
# pass over a few thousand self-generated examples, and the starting policy is
# the best one we have -- the failure mode to avoid is exactly the drift the
# GRPO arms exhibit.
.PHONY: train-rft
train-rft:
	@test -n "$(RFT_DATA)" || { echo "Usage: make train-rft RFT_DATA=data/rft_beq TAG=beq"; exit 1; }
	@$(ACTIVATE) && \
	 MODEL_PATH=$${INIT:-$(SFT_MERGED)} \
	 SAVE_PATH=$(PROJECT)/checkpoints/rft/$${TAG:-beq} \
	 TRAIN_FILE=$(PROJECT)/$(RFT_DATA)/train.parquet \
	 VAL_FILE=$(PROJECT)/$(RFT_DATA)/val.parquet \
	 LR=$${LR:-1e-5} TOTAL_EPOCHS=$${TOTAL_EPOCHS:-2} SFT_SAVE_FREQ=$${SFT_SAVE_FREQ:-100} \
	 bash scripts/train/run_sft.sh

# ── Evaluation ────────────────────────────────────────────────────────────────
# The head-to-head comparison the PoC exists to produce. NOTE: verl's own
# `val-core/.../acc/mean@1` is just the mean of that run's reward function, so
# the two arms' validation numbers are on different scales and are NOT
# comparable to each other. `make evaluate` re-scores both final checkpoints
# with BOTH metrics (type-check rate and BEq+ rate) so they can be compared.

STEP        ?= 30
CKPT_ROOT   := $(PROJECT)/checkpoints/beqplus_rl_poc
CKPT_COMP   := $(CKPT_ROOT)/qwen25_coder_0_5b_compute_score_composite/global_step_$(STEP)
CKPT_TC     := $(CKPT_ROOT)/qwen25_coder_0_5b_compute_score_typecheck/global_step_$(STEP)
MERGED      := $(PROJECT)/checkpoints/merged

# verl keeps actor weights in FSDP-sharded .pt files; actor/huggingface/ holds
# only config+tokenizer. vLLM needs real HF weights, so merge them first.
.PHONY: merge-checkpoints
merge-checkpoints:
	@$(ACTIVATE) && \
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
# Regenerate every figure from the cached eval JSONs. Cheap and deterministic --
# no Lean, no GPU -- so it is safe to re-run after any eval lands.
# Regenerate the checkpoint table at the top of arms.md from cached results.
.PHONY: arms-table
arms-table:
	@$(ACTIVATE) && python3 scripts/figures/make_arms_table.py

# Roster, baseline and step grid are all discovered from results/eval/ at run
# time (see make_figures.py), so this tracks whatever series is on disk. Override
# with N=<prefix> (paired example count), STEPS=10,30,50,90 (pin the grid), or
# BASELINE_LABEL=/RUN_PREFIX= in the environment.
.PHONY: figures
figures:
	@$(ACTIVATE) && python3 scripts/figures/make_figures.py \
	  $${N:+--n $$N} $${STEPS:+--steps $$STEPS}

# Training-time curves (reward + loss per GRPO step) from verl's FileLogger
# output under results/train/train_metrics/. Runs are discovered off disk and
# share the eval figures' palette. ARMS=a,b limits it; MAX_STEP clips the x-axis.
.PHONY: train-figures
train-figures:
	@$(ACTIVATE) && python3 scripts/figures/fig_training.py \
	  $${ARMS:+--arms $$ARMS} $${MAX_STEP:+--max-step $$MAX_STEP}

# `plots` = every figure: the eval trajectories AND the training curves. The old
# scripts/plot_results.py parsed verl's stdout for per-step metrics, which only
# three arms ever wrote there; it is in git history with the reason.
.PHONY: plots
plots: figures train-figures

.PHONY: eval-ckpt
eval-ckpt:
	@test -n "$(CKPT)" || { echo "Usage: make eval-ckpt CKPT=<verl run dir | merged dir>"; exit 1; }
	@$(ACTIVATE) && set -e; \
	 src="$$(cd "$(CKPT)" && pwd)"; name=$$(basename "$$src"); \
	 if ls "$$src"/*.safetensors >/dev/null 2>&1; then \
	   merged="$$src"; \
	 else \
	   case "$$name" in \
	     global_step_*) \
	       step=$${name#global_step_}; \
	       name=$$(basename "$$(dirname "$$src")"); \
	       shard="$$src" ;; \
	     *) \
	       step=$$(cat "$$src/latest_checkpointed_iteration.txt" 2>/dev/null); \
	       test -n "$$step" || { echo "No latest_checkpointed_iteration.txt in $$src"; exit 1; }; \
	       shard="$$src/global_step_$$step" ;; \
	   esac; \
	   if [ -d "$$shard/actor" ]; then shard="$$shard/actor"; fi; \
	   if [ ! -f "$$shard/huggingface/config.json" ]; then \
	     echo "[eval-ckpt] SKIP $$name step $$step: no weights in $$shard."; \
	     echo "  verl's trainer.max_actor_ckpt_to_keep rmtree'd this checkpoint's actor/"; \
	     echo "  subdir, leaving only its dataloader state. Raise MAX_CKPT_KEEP (see"; \
	     echo "  train-rl) so every checkpoint you save also survives the run."; \
	     exit 3; \
	   fi; \
	   merged="$(PROJECT)/checkpoints/merged/$${name}-step$${step}"; \
	   if ! ls "$$merged"/*.safetensors >/dev/null 2>&1; then \
	     rm -rf "$$merged"; \
	     echo "[merge] $$shard -> $$merged"; \
	     python3 -m verl.model_merger merge --backend fsdp \
	       --local_dir "$$shard" --target_dir "$$merged"; \
	   fi; \
	 fi; \
	 bn=$$(basename $$merged); label=$${bn%-step*}; \
	 mkdir -p "$(PROJECT)/results/eval/$$label"; \
	 out="$(PROJECT)/results/eval/$$label/eval_$${bn}_n$${N_EVAL:-80}.json"; \
	 echo "[eval] $$merged -> $$out"; \
	 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$(MODELS_ROOT)/.hf_cache \
	 python3 scripts/eval/evaluate_checkpoints.py --checkpoint "$$merged" \
	   --n-eval $${N_EVAL:-80} --out "$$out"

# Evaluate EVERY saved step of one run dir, so checkpoint selection has
# candidates to choose between. Skips steps already scored (the result JSONs are
# keyed by checkpoint label), so re-running after more steps land is cheap.
#   make eval-sweep RUN=checkpoints/beqplus_rl_poc/rl_from_sft390_gated
.PHONY: eval-sweep
eval-sweep:
	@test -n "$(RUN)" || { echo "Usage: make eval-sweep RUN=<verl run dir> [N_EVAL=400]"; exit 1; }
	@set -e; skipped=0; \
	 for step in $$(ls -d $(RUN)/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n); do \
	   name=$$(basename $(RUN))-step$$step; label=$${name%-step*}; \
	   mkdir -p "$(PROJECT)/results/eval/$$label"; \
	   out="$(PROJECT)/results/eval/$$label/eval_$${name}_n$${N_EVAL:-400}.json"; \
	   if [ -f "$$out" ]; then echo "[eval-sweep] cached: $$name"; continue; fi; \
	   d=$(RUN)/global_step_$$step; \
	   if [ ! -f "$$d/actor/huggingface/config.json" ] && [ ! -f "$$d/huggingface/config.json" ]; then \
	     echo "[eval-sweep] SKIP $$name: weights pruned mid-run (max_actor_ckpt_to_keep)"; \
	     skipped=$$((skipped+1)); continue; \
	   fi; \
	   echo "[eval-sweep] ===== $$name ====="; \
	   $(MAKE) --no-print-directory eval-ckpt CKPT=$$d N_EVAL=$${N_EVAL:-400}; \
	 done; \
	 if [ $$skipped -gt 0 ]; then \
	   echo ""; \
	   echo "[eval-sweep] $$skipped checkpoint(s) had their weights pruned mid-run by"; \
	   echo "  verl's trainer.max_actor_ckpt_to_keep, so only their dataloader state"; \
	   echo "  remains. Selection below covers only what survived. Future runs derive"; \
	   echo "  MAX_CKPT_KEEP from SAVE_FREQ (see train-rl) so this cannot recur."; \
	 fi
	@$(MAKE) --no-print-directory select

# Aggregate every cached result JSON into one table (no re-scoring).
.PHONY: compare
compare:
	@$(ACTIVATE) && python3 scripts/misc/compare_results.py

# Pick the best checkpoint by BEq+ AND say whether the difference is real.
# `make compare` tabulates; this one applies the paired test, which is what
# stops a within-noise argmax from being reported as an improvement.
.PHONY: select
select:
	@$(ACTIVATE) && python3 scripts/eval/select_checkpoint.py $(SELECT_ARGS)

.PHONY: evaluate
evaluate: merge-checkpoints
	@$(ACTIVATE) && \
	 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	 HF_HOME=$(MODELS_ROOT)/.hf_cache \
	 python3 scripts/eval/evaluate_checkpoints.py \
	   --checkpoint base \
	   --checkpoint $(MERGED)/typecheck_only-step$(STEP) \
	   --checkpoint $(MERGED)/composite-step$(STEP) \
	   --n-eval $${N_EVAL:-80}

# ── HPC submission (SLURM) ────────────────────────────────────────────────────
# The train-* targets above are for local/interactive runs (they always activate
# $(VENV), not $(VENV_HPC)). On a cluster, submit hpc/grpo.slurm
# instead — these targets are just a one-line convenience wrapper for it.

# One submit target instead of one per arm.
#   make submit ARM=gated STEPS=90
.PHONY: submit
submit:
	sbatch --export=ALL,ARM=$${ARM:-gated},TOTAL_STEPS=$${STEPS:-90} hpc/grpo.slurm

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
	@echo "lean-gym-rl — a gym for RL on Lean 4 autoformalization"
	@echo ""
	@echo "SETUP"
	@echo "  make setup             Local: env + verl + Lean/Mathlib + model"
	@echo "  make venv              Create venv from requirements.txt (no uv required)"
	@echo "  make setup-hpc         Compute Canada / DRAC (read hpc/NARVAL_NOTES.md first)"
	@echo "  make model             Download the base model   (MODEL_ID=...)"
	@echo "  make check-toolchain   Verify the Lean/Mathlib pin"
	@echo ""
	@echo "TRAIN (local, single GPU — the 3B series runs on SLURM, see run.md)"
	@echo "  make smoke             ~1 step on LoCoLib proof-pair data; proves the loop runs"
	@echo "  make train-sft         SFT from the base model (on LoCoLib proof-pair data)"
	@echo "  make train-rl          GRPO. Flags:"
	@echo "                           REWARD=gated|typecheck_only|outcome"
	@echo "                           STEPS=100  INIT=<merged dir>"
	@echo "  make train-rft         Rejection-sampling arm (RFT_DATA=... TAG=...)"
	@echo ""
	@echo "DATA FOR RL"
	@echo "  make rollouts          Generate k rollouts from a checkpoint"
	@echo "  make score-rollouts    BEq+-score them (resumable)"
	@echo "  make rft-data          Build the size-matched RFT datasets"
	@echo ""
	@echo "EVALUATE"
	@echo "  make evaluate          Score checkpoints on BOTH metrics"
	@echo "  make eval-sweep RUN=…  Score every saved step of a run, then select"
	@echo "  make select            Best checkpoint by BEq+ + paired McNemar vs SFT"
	@echo "  make plots             Every figure: eval trajectories + training curves"
	@echo "  make figures           Eval figures only (results/figures/)"
	@echo "  make train-figures     Training reward + loss per GRPO step"
	@echo "  make arms-table        Refresh the checkpoint table in arms.md"
	@echo ""
	@echo "SLURM (the 3B series — these are where the real runs happen)"
	@echo "  See run.md for the current workflow (LoCoLib proof-pair, three arms)"
	@echo "  Shorthand: bash hpc/submit.sh grpo ARM=gated SERIES_TAG=locolib_proof_lr6"
	@echo ""
	@echo "HOUSEKEEPING"
	@echo "  make clean | clean-lean | clean-all | prune-checkpoints | kill-stale"


.DEFAULT_GOAL := help
