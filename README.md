# lean-gym-rl

**A gym for reinforcement learning on Lean 4 autoformalization.**

Autoformalization — turning an informal mathematical statement into a formal
Lean 4 theorem — is unusually hard to reward. The statement is the model's
*output*, so "it compiles" is satisfied by `theorem x : 1 = 1`. Every reward that
is cheap is exploitable, and every reward that is sound is expensive. This repo
exists to make that trade-off **measurable**: swappable rewards, a real
Lean/Mathlib scoring backend, and an evaluation harness that insists on paired
tests against a *calibrated noise control*.

The flagship experiment is **BEq+** as a training-time RL reward. It is one arm
among several, and the harness is the point.

```
informal statement ──▶ policy ──▶ Lean statement ──▶ [ reward ] ──▶ GRPO
                                                        ▲
                          type-check · self-prove · BEq+ · similarity · placebo
```

---

## Why a gym, and not just a training script

Three things kept going wrong often enough to be designed against:

1. **A reward that looks like it works usually isn't working.** Type-check-only
   RL drives its own metric from 76.7% → 98.0% while semantic accuracy collapses
   41.2% → 8.0%. Any harness that reports only the reward would call that a
   success.
2. **GRPO with no signal at all is destructive**, so "RL made it worse" is not
   evidence about your reward. A run must be compared to a *fitted* placebo, not
   to the SFT baseline.
3. **Net accuracy hides the mechanism.** Arms differ mostly in how much they
   destroy. Splitting retention from gains is what separates "protects the
   policy" from "teaches it something".

So the harness ships a reward zoo, a calibrated control, and paired statistics as
first-class pieces.

---

## The reward zoo

`reward/reward_fn.py`. All share one signature and one diagnostic schema, so any
of them drops into `configs/run_grpo.sh` via `REWARD_FN_NAME`.

| reward | what it checks | Lean cost | exploitable? |
|---|---|---|---|
| `typecheck_only` | statement elaborates | 1 call | **yes, badly** |
| `selfprove` | elaborates + provable + not closed by `tauto` alone | ~4 calls | partially — drifts to true-but-unrelated |
| `gated` | BEq+ ladder: 0 / one direction / both directions | up to ~18 calls | no (gold-referenced) |
| `guided` | `gated` + type-check-gated structural similarity | same + CPU | similarity is capped below the semantic step |
| `shaped`, `composite` | weighted blends of the above | varies | — |
| `placebo` | deterministic pseudo-random on the rollout hash | **none** | n/a — it is the control |

**BEq+** (`reward/beq_plus.py`) is deterministic, LLM-free, bidirectional Lean
tactic equivalence (Poiroux et al., EMNLP'25), vendored from
`augustepoiroux/RLMEval` and wrapped in a persistent Mathlib-resident REPL
(~4.3GB per worker) so it can score rollouts online.

### The control is the load-bearing piece

`compute_score_placebo` invokes no Lean and carries zero information. Its
constants are **fitted to each policy** by `scripts/calibrate_placebo.py`, which
matches three measured statistics: the informative-group rate, the within-group
standard deviation, and the mean reward.

All three are necessary. Fitting only the first two leaves the problem
degenerate — within-group sd is symmetric under `p ↔ 1−p` — and the two
solutions have *mirrored advantage skew*, which changes how destructive the
control is. We shipped the wrong branch once and only caught it when a second
policy fitted to the opposite one.

---

## Experiments

A decisive per-arm breakdown — what each reward computes, where they overlap,
where they diverge — is in [`arms.md`](arms.md).

Each is a slurm arm plus an analysis. Results in
[`logook.md`](logook.md) and [`results/FINDINGS.md`](results/FINDINGS.md).

### 1. Does a gold-referenced semantic reward beat noise? *(flagship)*

`gated` vs the calibrated placebo, paired McNemar on a pinned 400-example slice.

| step | placebo | **gated** | diff | p |
|---|---|---|---|---|
| 30 | 22.2% | **40.5%** | **+18.2pp** | 7e-16 |
| 50 | 23.8% | 36.2% | +12.5pp | 6e-7 |
| 70 | 18.8% | 34.0% | +15.2pp | 2e-10 |

`gated-step30` is statistically **indistinguishable from the SFT baseline**
(40.5% vs 41.2%, p=0.78, retention 83.6%) while pure noise at the same step keeps
only 51.5% of the baseline's correct answers. BEq+ does not teach the policy much
— it very nearly fully protects it.

### 2. Is a type-check-only reward exploitable? *(reproduced, decisively)*

| step | 10 | 50 | 90 | 110 |
|---|---|---|---|---|
| BEq+ | 40.0% | 22.0% | 22.0% | **8.0%** |
| type-check | 88.0% | 93.5% | 97.2% | **98.0%** |

Monotone in both directions. By step 150 the training reward is exactly 1.000 and
`critic/advantages` has max = min = 0 — the objective is dead. Crucially this
ends up *below the placebo* on semantics while far above it on the proxy, which
is the difference between "an optimiser aimed at the wrong target" and "drift".

### 3. Can a gold-free reward substitute? *(open)*

`selfprove` is the strongest reward obtainable without ever consulting the gold.
If it keeps pace with `gated`, the case for gold-referenced semantics weakens a
lot. Two checkpoints so far: retention 82.4% / 78.2%, gain-rate 7.2% / 7.7%.
Running.

### 4. Does mid-training help? *(negative, and instructive)*

Following *On the Interplay of Pre-Training, Mid-Training, and RL*
(arXiv:2512.07783). Continued LM pre-training on Mathlib statements before SFT
moved nothing: 38.9% vs 39.4% BEq+, informative 42.9% vs 46.0%, pass@32 56.4% vs
55.2%.

The reason is worth knowing: **only ~2.3% of Mathlib declarations elaborate
standalone.** Mathlib is factored through `variable`/`section`/`namespace`
precisely so statements need not restate their context; autoformalization targets
are self-contained one-liners. The distributions barely intersect.

### 5. Curating RL data to the "edge of competence" *(running)*

GRPO's advantage is `A_i = r_i − mean(r_group)`, so a group scoring 0/8 and one
scoring 8/8 both contribute exactly zero. `gated_edge` restricts the pool to
prompts measured at 1–7/8, taking the informative fraction from 46% to ~100% by
construction.

### 6. Capability vs exploration *(settled)*

At k=32 on prompts the policy never solves at k=8, only **14.5%** are ever
recovered — and **84.3% type-check**. The model reliably writes well-formed Lean
that means the wrong thing. That is a capability limit, not an exploration one,
and it bounds what any reward can do.

---

## What the harness measures

Net accuracy is reported, but it is the least informative number.

**Retention vs gains.** Every arm is scored against the SFT baseline
example-by-example: how many correct answers it *kept*, and how many previously
wrong ones it *converted*.

**Gain-rate** (converted ÷ previously-wrong) separates the arms far better than
the headline does:

| reward class | gain-rate |
|---|---|
| gold-referenced semantics (`gated`) | **7.7–10.6%** |
| gold-free compiler signal (`selfprove`) | 7.2–7.7% |
| exploitable proxy (`typecheck`), decaying | 7.7% → 2.1% |
| pure noise (`placebo`) | 1.7–4.3% |

**pass@k.** `scripts/passk_report.py` uses the unbiased estimator (Chen et al.
2021). Differences visible at pass@1 but not at pass@32 are sharpening, not
capability.

**Always paired.** Same pinned slice, McNemar exact. Never difference two
headline rates — a subset can be easier than the whole.

---

## Figures

`make figures` regenerates everything in `results/figures/` from the cached eval
JSONs — no Lean, no GPU, deterministic.

Every step-indexed figure is drawn on the **reporting grid, `evalio.STEP_GRID` =
10/30/50/90**, and starts at step 0 on the SFT baseline all arms resume from.
Off-grid checkpoints are still trained and still evaluated — they are just not
what results are shown at. Override with `--steps`. `runtime.png` is the one
exception: per-step cost comes from mtime deltas between *consecutive*
checkpoints, so subsetting would degrade the estimate.

The **placebo is hidden by default** (`--hide rl3b_v2_placebo`). That is a
presentation choice, not a claim that `typecheck` replaces it: type-check is an
informative-but-exploitable reward, the placebo is calibrated zero-information
noise, and only the placebo can separate "RL helped" from "the policy drifted".
Its numbers stay in `arms.md` and `compare_arms.py`; `--hide ''` draws it again.

| figure | shows |
|---|---|
| `arm_trajectories_pass1.png` | **the reference figure.** pass@1 on BEq+ (left) and on compiling (right), same rollouts, same decoder — the only sound way to read the two verdicts against each other |
| `arm_trajectories.png` | BEq+ vs training step, every arm, against the SFT line. **Greedy decode (T=0)** — what the McNemar tests use, but not comparable point-for-point with any pass@k figure |
| `arm_trajectories_passk.png` | sampled at T=1.15: pass@1 (dotted) vs pass@32 (solid) on **BEq+**, same arms — sharpening vs capability |
| `arm_trajectories_passk_typecheck.png` | the same, on the **compiling** verdict. pass@32 is saturated (84.7–100%) so it cannot rank arms; pass@1 spans 40.8–94.6% and is where type-check RL's gain actually shows |
| `passk.png` | pass@k curves for the SFT baselines and every arm |
| `runtime.png` | seconds per GRPO step and cumulative GPU-hours, per reward, against the measured **Lean-free floor** (~32 s). Lean is 97% of a `gated` step and 1% of a `typecheck` step |
| `retention_gain.png` | what each arm kept vs converted, with the 4.9–7.3% ceiling band |
| `proxy_vs_semantic.png` | type-check on x, BEq+ on y: reward hacking as a trajectory |

Style follows Interplay-LM-Reasoning (arXiv:2512.07783), recovered by sampling
their published figure — they use **Okabe-Ito**. The series order in
`scripts/figstyle.py` is validated for colourblind separation and must not be
reshuffled; see that file's header.

---

## Layout

```
reward/
  beq_plus.py       BEq+ behind a persistent Lean REPL
  reward_fn.py      the reward zoo
  similarity.py     GTED-style structural + symbol similarity
  lean_tool.py      verl function_tool: gives the policy Lean's real diagnostics
configs/run_grpo.sh GRPO entrypoint; every constant's header records the failure
                    that produced it
scripts/
  prepare_dataset.py / prepare_sft_dataset.py / prepare_midtrain_dataset.py
  generate_rollouts.py / score_rollouts.py       rollout + offline BEq+ scoring
  build_rft_dataset.py / make_difficulty_subset.py / make_starved_subset.py
  calibrate_placebo.py                           fit the control to a policy
  evaluate_checkpoints.py / compare_arms.py / select_checkpoint.py
  passk_report.py / probe_gradient_signal.py
hpc/*.slurm         SLURM jobs; cluster fixes baked in (see hpc/NARVAL_NOTES.md)
```

---

## Quickstart

```bash
make setup                                   # env + verl + Mathlib4 + model + data
source hpc/cc_env.sh                         # never run bare `python`

# 1. splits
python scripts/prepare_sft_dataset.py --n-train 8000
python scripts/prepare_dataset.py --out-dir data_3b --n-val 1000 --n-train 4300

# 2. SFT baseline, then evaluate and pick the best epoch
sbatch hpc/sft_3b.slurm
sbatch hpc/eval_3b_sft.slurm

# 3. measure the group geometry and FIT THE CONTROL to this policy
sbatch hpc/rollouts_3b.slurm                 # also a go/no-go gate

# 4. arms
sbatch --export=ALL,ARM=gated   hpc/grpo_3b.slurm
sbatch --export=ALL,ARM=placebo hpc/grpo_3b.slurm

# 5. paired comparison
sbatch --export=ALL,RUN=rl3b_gated,N_EVAL=1000 hpc/grpo_eval.slurm
python scripts/compare_arms.py rl3b_gated rl3b_v2_placebo
```

Adding a reward is one function in `reward/reward_fn.py` returning
`{"score": float, ...diagnostics}` and one case in `hpc/grpo_3b.slurm`.

---

## Stack

| piece | choice | why |
|---|---|---|
| RL framework | [verl](https://github.com/volcengine/verl) | native GRPO, pluggable `custom_reward_function`, colocated FSDP + vLLM |
| algorithm | GRPO, full-parameter | no LoRA — see `configs/run_grpo.sh` |
| policy | `Qwen2.5-Coder-3B-Instruct` | code-pretrained; 0.5B series kept for cross-scale paired tests |
| data | [`internlm/Lean-Workbook`](https://huggingface.co/datasets/internlm/Lean-Workbook) | 13,297 unique NL↔Lean4 pairs after dedup |
| Lean | Mathlib4 @ `v4.8.0-rc1` | pinned to Lean-Workbook's target toolchain |

Compute notes, including why every job stages Mathlib to node-local NVMe, are in
[`hpc/NARVAL_NOTES.md`](hpc/NARVAL_NOTES.md).

---

## Status

`sft3b-step93` = **41.2% BEq+ / 79.0% type-check** on the pinned 400 slice. No RL
arm has *beaten* it; `gated` **ties** it while the calibrated control loses half
the baseline's correct answers. Running: `selfprove`, `guided`, `gated_edge`.

Full history, including the negative results and the bugs that produced them, is
in [`logook.md`](logook.md).
