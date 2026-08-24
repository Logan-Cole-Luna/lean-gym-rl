# lean-gym-rl

Measuring the **training-time impact of BEq+** (deterministic, LLM-free
bidirectional Lean tactic equivalence, Poiroux et al. EMNLP'25) as an RL reward
for autoformalization. Qwen2.5-Coder-0.5B-Instruct, internlm/Lean-Workbook,
Lean 4.8.0-rc1 + Mathlib4, verl GRPO with colocated FSDP actor + vLLM rollout.

The paper never got BEq+ into the RL reward. That gap is this project.

## Read first

- `logook.md` — running log of results and TODOs. **Start here.**
- `results/FINDINGS.md` — the placebo-control writeup.
- `hpc/NARVAL_NOTES.md` — cluster gotchas. **Read before running anything on Narval.**

## Environment

Never run bare `python`. Always:

```bash
source hpc/cc_env.sh     # module stack + venv on $SCRATCH + HF/Lean cache paths
```

`pyarrow` comes from the `arrow/23.0.1` **module**, not pip — any check run
without sourcing `cc_env.sh` will wrongly report it missing. The venv lives at
`$SCRATCH/ai4math_training_venv_hpc` and must stay on the **torch 2.11** set
(see `hpc/NARVAL_NOTES.md`; do not "fix" a mismatch by downgrading vllm).

`import vllm` succeeds while vllm is broken — `_C.abi3.so` loads lazily. Test
`import vllm._C`. SFT training never touches vllm and passes regardless.

## Layout

```
reward/beq_plus.py       BEq+ behind a persistent Lean REPL (~4.3GB Mathlib resident)
reward/reward_fn.py      6 swappable rewards; compute_score = compute_score_gated
configs/run_grpo.sh      GRPO entrypoint. Read its header comments before tuning --
                         each constant records the failure that produced it.
scripts/                 rollout generation, offline scoring, RFT datasets, eval, stats
hpc/*.slurm              SLURM jobs, all Narval fixes baked in
```

`checkpoints/` and `repos/` are **symlinks into `$SCRATCH`**; model weights go
there too via the Makefile's `MODELS_ROOT` (no symlink). `/home` is 50GB and
cannot hold any of it — a truncated optimizer write killed a run at 46/50GB.

## Rewards

`compute_score_gated` is the default (semantic-signal ladder).
`compute_score_placebo` is the **control**: deterministic pseudo-random, keyed on
the hash of the rollout text, invokes no Lean. Its constants are tuned so its
*advantage geometry* matches the gated arm — it isolates how much of any effect
is the reward versus GRPO's update noise. Any claim that "RL helps/hurts" must be
made against the placebo, not against SFT.

## Conventions that matter

- **Eval naming**: `results/eval_<label>-step<N>_n<n>.json`, with a `per_example`
  array. `scripts/compare_arms.py` pairs on the filename; `select_checkpoint.py`
  merges records across files **by the label inside the JSON**, which
  `evaluate_checkpoints.py` sets from the *directory name*. Point it at a
  uniquely-named dir — aiming straight at `<ckpt>/huggingface` labels everything
  `"huggingface"` and silently collapses all records into one.
- **Always paired.** Same pinned 400-example validation slice, McNemar exact.
  Never difference two headline rates — a subset can be easier than the whole
  (SFT is 38.8% overall but 40.0% on the first 40).
- **n=400 detects ~5.6pp.** Most real effects here are 1–3pp and will land
  directionally right and statistically mute. Say so rather than over-reading.
- **Report retention vs gains**, not just net. Arms differ almost entirely in how
  much they *destroy*; the gain channel is flat at 4.9–7.3% across every signal.

## Running things

```bash
make rft-data                              # RFT datasets from a scoring pass
sbatch --export=ALL,ARM=beq  hpc/rft_eval.slurm
sbatch --export=ALL,ARM=gated hpc/grpo_bs16.slurm
sbatch --export=ALL,RUN=rl_gated_bs16 hpc/grpo_eval.slurm
python scripts/compare_arms.py rl_gated_clean rl_placebo_clean
python scripts/select_checkpoint.py --baseline sft-step390
```

Every SLURM job must `unset ROCR_VISIBLE_DEVICES` (Narval sets it alongside
`CUDA_VISIBLE_DEVICES`; verl's worker init hard-errors) and stage Mathlib to
`$SLURM_TMPDIR` from the prebuilt tar (41s vs 1984s; without it `import Mathlib`
exceeds the Lean timeout on compute nodes as well as login nodes).

Save checkpoints with `save_contents=[model,extra,hf_model]` — `hf_model` writes
ready-to-load HF weights, removing the `verl.model_merger` step.

## Throughput

- BEq+ eval: `--workers 8` → **1.7s/example**, ~20 min per 400-example checkpoint
- GRPO: `AGENT_LOOP_WORKERS` is the Lean-parallelism knob. `reward.num_workers`
  is **not** — tuning it has no effect on the number of Lean servers.
- Lean scoring dominates GRPO wall-clock: ~51s/step without it (placebo),
  ~10 min/step with it (gated), same batch.

## Baseline

`sft-step390` = **38.8% BEq+ / 76.2% type-check** at n=400. No RL checkpoint has
beaten it. Group composition on the disjoint RL pool: 47.0% starved / 22.1%
saturated / **30.9% informative**. pass@8 is 53.0%, so 14.1pp of headroom exists.
