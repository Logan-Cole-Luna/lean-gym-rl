# Running this project on Narval (Compute Canada)

Eight environment issues had to be fixed before anything ran here. None were in
the research code; all are captured in `hpc/*.slurm` and in `Makefile`/`cc_env.sh`.

## Python environment (`$SCRATCH/ai4math_training_venv_hpc`)

The venv is built for **torch 2.11**. Keep this set mutually consistent:

    torch 2.11.0  torchvision 0.26.0  torchaudio 2.11.0
    vllm 0.20.2 (requires torch==2.11.0)   transformers 5.10.4
    flashinfer-python 0.6.8.post1 (vllm 0.20.2's exact pin)

* **Do not fix a mismatch by downgrading vllm.** vllm 0.19.0 pairs with torch
  2.10 but requires `transformers<5`, while verl 0.9.0 requires
  `transformers>=5.5.3,<5.11`. Irreconcilable.
* `tensordict` declares `torch~=2.10.0`, but its `_C.so` links neither libtorch
  nor c10, so that pin is advisory.
* **`import vllm` succeeds while vllm is broken** — `vllm/_C.abi3.so` loads
  lazily, so an ABI mismatch only surfaces when an LLM engine is constructed.
  SFT training never touches vllm and passes regardless. Test `import vllm._C`.
* Also installed by hand: `mpmath`, `scipy`, `pandas`. `pyarrow` is NOT missing —
  it comes from the `arrow/23.0.1` module `hpc/cc_env.sh` loads, so any check run
  without sourcing that script reports it absent incorrectly.

## SLURM

* Narval exports `ROCR_VISIBLE_DEVICES` alongside `CUDA_VISIBLE_DEVICES` on every
  GPU job; verl's `Worker._setup_env_cuda_visible_devices` hard-errors on that
  combination. Every job script must `unset ROCR_VISIBLE_DEVICES`.
* `--account=def-vganesh` (not `def-vganesh`).
* A100 40GB, so the 16GB-era constants in `configs/run_grpo.sh` are no longer
  binding. SFT at micro-batch 4 peaks at ~10.2GB allocated.

## Checkpoints

`/home` is 50GB and cannot hold them — a truncated optimizer write killed a run
at `unexpected pos ... vs ...`. `checkpoints/` is a symlink into `$SCRATCH`.
Save with `save_contents=[model,extra,hf_model]`: `hf_model` writes ready-to-load
HF weights under `<ckpt>/huggingface`, removing the `verl.model_merger` step, and
dropping `optimizer` saves ~4GB/checkpoint that RFT never resumes from.

## Lean / Mathlib — the big one

`import Mathlib` blows lean_interact's timeout on **compute nodes as well as
login nodes**. Mathlib is 36,700 files; read cold off Lustre, `cp -a` moved them
at ~18 files/s (1984s). It is not a broken build.

Stage the prebuilt archive to node-local NVMe instead — **41s vs 1984s**:

    tar -xf /scratch/logan03/mathlib4_v4.8.0-rc1.tar -C "$SLURM_TMPDIR"
    export MATHLIB_ROOT="$SLURM_TMPDIR/mathlib4"
    export LEAN_INTERACT_CACHE_DIR="$SLURM_TMPDIR/lean_interact_cache"

`BEQ_ENV_TIMEOUT` (added to `reward/beq_plus.py`) overrides the previously
hardcoded 600s as a safety net. Rebuild the archive with:

    cd $SCRATCH/ai4math_training_repos && tar -cf /scratch/logan03/mathlib4_v4.8.0-rc1.tar mathlib4

## Throughput

`evaluate_checkpoints.py --workers N` scores with N independent Lean REPLs
(~5GB each): **1.7s/example at 8 workers**, so a 400-example checkpoint takes
~20 min. For GRPO, `AGENT_LOOP_WORKERS` is the Lean-parallelism knob —
`reward.num_workers` does **not** control it.
