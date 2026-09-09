# Reward-arm experiment: reward specification and measured effect

Scope: the three GRPO reward arms of `SERIES_TAG=locolib_proof_lr6` (sections 3 to
5) and the successor series `locolib_proof_c512` and `locolib_proof_c512cos`
(sections 6 to 7). Section 2 specifies every reward function in the repository,
including the derivation of each sub-reward. All results are on the pinned 760-row
LoCoLib proof-pair validation slice, greedy decode, paired per example.

Every figure was re-derived on 2026-09-08 under a single verified Lean toolchain
(`leanprover/lean4:v4.23.0`), and each `results/eval/**/eval_*.json` records the
toolchain that produced it. Pre-correction records are preserved alongside under
`_superseded_20260908/`, since several published claims were computed from them
and a correction is only checkable if both versions survive.

## 0. Correction: a Mathlib version mismatch between baseline and arms

The SFT baseline and the RL arms had been scored under different Mathlib versions.
`hpc/eval_sft.slurm` ran without the LoCoLib overrides, so `job_prelude.sh`'s
default `MATHLIB_TAR` (v4.8.0-rc1) applied, while `hpc/grpo_eval.slurm` ran with
them and used v4.23. LoCoLib golds import modules present only in 4.23, so the
baseline was scored against a Mathlib that could not elaborate its own reference
answers.

This was established directly rather than inferred. `scripts/eval/rescore_gen.py`
re-scores completions read back from a `gen_*.jsonl`, holding the text fixed by
construction, so the environment is the only free variable:

| | v4.8 elaborates | v4.8 BEq+ | v4.23 elaborates | v4.23 BEq+ |
|---|---|---|---|---|
| SFT-76 | 50.5% | 43.9% | 67.0% | 58.2% |
| gated-step10 | 50.8% | 44.5% | 67.8% | 59.2% |

Each published figure reproduces in its own cell. Of the 500 rows whose
completions are byte-identical between SFT-76 and gated-step10, 500/500 agree once
a single toolchain is used, including all 96 on which the two published
evaluations disagreed. There is no residual scorer noise. The toolchain moves the
same text by +14 to +17 pp; ten GRPO steps move it by +0.3 to +1.0 pp.

The correction sharpens the result rather than weakening it. The artifact added
approximately 14 pp to all three arms equally, inflating every delta and making
the exploitable proxy appear effective (+15.26 pp, p = 7.9e-21 as published).
Against a correctly scored baseline the proxy does not exceed SFT and the two
semantic rewards do, which is the hypothesis the ablation was constructed to test.

It also retires section 5 of the previous version of this document: the
"unexplained ~15 pp jump in the first 10 steps, identical across all three arms"
was this artifact, so the steps 1 to 9 investigation it recommended is
unnecessary.

## 1. Setup

| | |
|---|---|
| base model | Qwen2.5-Coder-3B-Instruct |
| task | proof-pair: emit a complete Lean 4 theorem and its own proof |
| corpus | LoCoLib, `data_locolib/rl_proof.parquet` (train), `val_proof.parquet` (760 rows, val) |
| resume point | `sft_3b_locolib_proof/global_step_76` (pinned in `data_locolib/best_sft_proof.txt`) |
| trainer | verl GRPO, colocated FSDP actor + vLLM rollout, 2x A100-40GB |
| `train_batch_size` / `rollout.n` / `ppo_epochs` | 16 / 8 / 1 |
| `max_response_length` | 128 tokens (a 16GB-card artifact; 512 repository-wide since, see section 5) |
| actor LR | 1e-6, `lr_warmup_ratio=0.05` |
| steps | 90, checkpointed and evaluated every 10 |
| Lean | 4.23.0 with Mathlib4 behind a persistent REPL, verified per record |

The three arms are identical except for `reward.custom_reward_function.name`,
verified per arm from the verl config dump in each job log (`logs/grpo_2013288/89`
typecheck, `2013291/92` gated, `2013294/95` outcome).

## 2. Reward specification

All seven arms score a rollout `x` with a scalar. They share one decision
procedure, `reward/reward.py:outcome_for`, which is not parameterised, and differ
only in the payout attached to its result.

### 2.1 Outcome assignment

`outcome_for(x)` evaluates four conditions in order and returns exactly one of six
outcomes. It reads booleans; `None` is never treated as success.

| row | outcome | condition |
|---|---|---|
| 1 | `no_answer` | no verdict, empty answer, or boilerplate shared across problems |
| 2 | `no_elaborate` | Lean rejected the submission |
| 3 | `incomplete` | elaborates, proof unfinished, statement unmatched |
| 4 | `compiles` | finished proof, statement unmatched, declares a non-trivial theorem |
| 5 | `incomplete_faithful` | statement matched under BEq+, proof unfinished |
| 6 | `solved` | statement matched and proof finished |

Two terms are distinguished throughout. *Type-correct* means Lean exited 0; since
`sorry` raises a warning rather than an error, a file whose only proof is `sorry`
is type-correct. *Proved* means type-correct, sorry-free, declaring a real
theorem, and free of smuggled axioms under `#print axioms`. Proved implies
type-correct.

Row 4 carries two admission tests, since it is the row most exposed to
exploitation: `declares_theorem`, and `statement_is_trivial is not True`. Failure
of either demotes the rollout to row 3.

### 2.2 The `outcome` family

Let `o = outcome_for(x)`. Rows 1 to 5 pay constants:

```
R(x) = 0.00 (row 1)   0.05 (row 2)   0.15 (row 3)   0.30 (row 4)   0.50 (row 5)
```

Row 6 is the only non-constant. With ceiling `V = 1.00`, faithfulness band
`beta_f = 0.15`, and brevity band `beta_b`:

```
R(x) = (V - beta_f) + beta_f * f - beta_b * (1 - b)
     = 0.85 + 0.15 * f - beta_b * (1 - b)
```

`f in [0,1]` is NL to FL faithfulness (2.4); `b in [0,1]` is FL to FL length
agreement (2.3). The conventions for unknown values are opposite by construction:
`f` is a payment, so an unmeasured `f` scores 0 and cannot raise the reward; `b`
is a deduction, so an unmeasured `b` is 1 and cannot lower it.

The four arms are one function body (`_outcome_core`) parameterised by two flags,
making `outcome` an exact control:

| arm | `beta_f` | measures `f` | `beta_b` | row 6 | range |
|---|---|---|---|---|---|
| `outcome` | 0.15 | no | 0.00 | 0.85 | {0.85} |
| `outcome_brevity` | 0.15 | no | 0.10 | 0.85 - 0.10(1-b) | [0.75, 0.85] |
| `outcome_faith` | 0.15 | yes | 0.00 | 0.85 + 0.15f | [0.85, 1.00] |
| `outcome_v2` | 0.15 | yes | 0.10 | 0.85 + 0.15f - 0.10(1-b) | [0.75, 1.00] |

The band is carved out whether or not an arm measures `f`: with `f` unmeasured,
row 6 scores 0.85, which is what all earlier arms recorded. Enabling faithfulness
does not lower the floor, it makes the upper 0.15 reachable. `outcome` and
`outcome_faith` therefore differ in exactly one respect.

Neither term reaches rows 1 to 5. A short non-proof lands on row 3 and cannot
access either.

### 2.3 Brevity, `b`

`reward/reward.py:brevity_for`. Let `L_p` and `L_g` be the candidate and gold
proof-body lengths in characters, obtained from the same splitter, and `s = 60`.

```
if L_g == 0 or L_p == 0:  b = None, treated as 1.0

d = ln( (L_p + s) / (L_g + s) )

    over = d,   free = ln(1.5) = 0.4055,  zero = ln(4)  = 1.3863    if d > 0
    over = -d,  free = ln(3)   = 1.0986,  zero = ln(10) = 2.3026    if d <= 0

b = 1                                        if over <= free
b = max(0, 1 - (over - free) / (zero - free))  otherwise
```

The charge against row 6 is `beta_b * (1 - b)`, bounded by `beta_b = 0.10`.

Three properties are deliberate:

- **Softening by `s = 60`.** A raw ratio `L_p/L_g` is degenerate for short golds:
  against a 5-character `rfl`, a reasonable `by simp [foo]` reads as a fourfold
  overrun. The additive constant makes the tolerance absolute for small golds and
  multiplicative for large ones. A 5-character gold is free to 38 characters; a
  1452-character gold is free from 444 to 2208.
- **Log space**, giving multiplicative tolerance across a corpus whose golds span
  a median of 67 characters to roughly 7300.
- **An asymmetric dead band.** Bloat is the target failure mode, charged from 1.5x.
  Proofs shorter than the gold are usually preferable, since the reference is one
  author's first draft, so the short side is free to gold/3 and charged only
  beyond it, as a guard against BEq+ proxy exploitation: an extremely short proof
  of a matched statement is the form a cascade false positive takes. A symmetric
  band was tested and rejected: against a 61-character gold, the proof `by omega`,
  strictly shorter and `proved` sorry-free and axiom-clean, scored `b = 0.88` and
  forfeited 0.012 of reward.

Representative values:

| `L_g` | `L_p` | ratio | `b` | charge at `beta_b` = 0.10 |
|---|---|---|---|---|
| 67 | 10 | 0.15x | 1.0000 | 0.0000 |
| 67 | 67 | 1.0x | 1.0000 | 0.0000 |
| 67 | 100 | 1.5x | 1.0000 | 0.0000 |
| 67 | 150 | 2.2x | 0.9006 | 0.0099 |
| 67 | 200 | 3.0x | 0.6829 | 0.0317 |
| 67 | 300 | 4.5x | 0.3511 | 0.0649 |
| 67 | 500 | 7.5x | 0.0000 | 0.1000 |
| 1452 | 400 | 0.28x | 0.9241 | 0.0076 |
| 1452 | 444 | 0.31x | 1.0000 | 0.0000 |
| 1452 | 2208 | 1.5x | 1.0000 | 0.0000 |
| 1452 | 100 | 0.07x | 0.0470 | 0.0953 |

`python -m reward.reward` prints the free window per gold percentile. `b` is
computed for every `outcome`-family arm even at `beta_b = 0`, since divergence of
`pred_proof_chars` from `gold_proof_chars` is the quantity the band exists to
detect and the control arm must report it.

### 2.4 Faithfulness, `f`

`f` is a judge verdict rather than a closed-form quantity, which is why it
requires a validity gate the other terms do not. `reward/faithfulness.py`
implements the AutoFaith pipeline:

```
informal statement + informal proof  --(judge)-->    NL blocks
candidate Lean proof                 --(Lean REPL)-> FL blocks   (all_tactics=True)
NL blocks + FL blocks                --(judge)-->    score in [0,1]
```

`f = clip(score, 0, 1)`, and `f = None` on every failure path.

An empty FL block list is treated as unknown, not zero. A term-mode proof
(`:= rfl`, `:= fun x => ...`, `:= Nat.add_comm a b`) is valid Lean emitting no
tactics, so it yields no FL blocks and AutoFaith's judge would return a hard 0.0.
48.7% of LoCoLib golds are term-mode (365/749), as are 43.8% of the policy's own
proofs, so admitting that path would score roughly half of all faithful proofs at
zero. The check precedes the judge call, leaving a structural unknown rate of
approximately 40 to 49% that `faith_known` reports alongside every faithfulness
figure.

`f` is evaluated on row 6 only. `reward_for_outcome` reads it nowhere else and
about 24% of rollouts reach that row, so the gate removes roughly 76% of judge
traffic without altering any score.

Two judges are used, one per side, and they must differ: scoring a policy with the
judge it trained against measures conformity to that judge. Training uses
`Qwen/Qwen2.5-7B-Instruct`, evaluation `microsoft/phi-4`. Both jobs assert the
served model on `/health`, so a stale server fails the job rather than collapsing
the separation silently.

### 2.5 `gated`

```
R(x) = 1.00   if o in {incomplete_faithful, solved}
     = 0.25   if o == incomplete and semantic_signal >= 1
     = 0.00   otherwise
```

No faithfulness or brevity term, and no payment for elaboration. The flat floor is
the design intent: under GRPO the advantage is reward minus group mean, so a group
carrying no semantic signal contributes no gradient, and any term varying within
such a group becomes the only climbable signal there.

The 0.25 step is applied as an override on row 3 rather than as a seventh outcome,
because `statement_matches_reference` is boolean and a one-directional BEq+ result
is not a match.

### 2.6 `typecheck`

```
R(x) = 1.00   if o in {incomplete, compiles, incomplete_faithful, solved}
     = 0.00   if o in {no_answer, no_elaborate}
```

Equivalently, 1.0 if and only if Lean accepted the submission. The arm is
exploitable by construction, since the statement is the model's own output and
`theorem x : 1 = 1 := rfl` satisfies it. Its BEq+ is expected to fall; the success
criterion is that the reward rises, separating "BEq+ does not teach" from "this
stack cannot optimise anything".

It uses `check_own_proof` rather than `typecheck_ex`, which would discard the
candidate's proof body and re-inject `sorry`, checking only the signature.

### 2.7 Summary

| arm | rows 1-5 | row 6 | `f` | `b` | Lean cost per step |
|---|---|---|---|---|---|
| `typecheck` | 0 / 0 / 1 / 1 / 1 | 1.00 | no | no | ~2 s |
| `gated` | 0 / 0 / 0* / 0 / 1 | 1.00 | no | no | ~935 s |
| `outcome` | 0.00 / 0.05 / 0.15 / 0.30 / 0.50 | 0.85 | no | reported | ~700 s |
| `outcome_brevity` | as above | 0.85 - 0.10(1-b) | no | charged | ~700 s |
| `outcome_faith` | as above | 0.85 + 0.15f | yes | reported | ~750 s plus judge |
| `outcome_v2` | as above | 0.85 + 0.15f - 0.10(1-b) | yes | charged | ~750 s plus judge |

\* `gated` pays 0.25 on row 3 when BEq+ proves one direction.

All constants are overridable (`BEQ_W_GATED_ONE_DIR`, `BEQ_BREVITY_BAND`,
`BEQ_FAITH_BAND`); no run reported here changed one. The arms are separate named
entry points rather than one function reading environment variables because verl
records `custom_reward_function.name` in its config dump, which is how a completed
run is later identified.

## 3. Results

Baseline: `sft3blocolib_proof-step76`, 58.16% BEq+ and 66.97% elaboration.

### Step 90 against that baseline (paired, McNemar exact)

| arm | BEq+ | elaborates | delta BEq+ | gained / lost | p |
|---|---|---|---|---|---|
| `gated` | 61.45% | 70.00% | +3.29 pp | 38 / 13 | 6.2e-04 |
| `outcome` | 60.53% | 69.47% | +2.37 pp | 40 / 22 | 0.030 |
| `typecheck` | 58.68% | 67.37% | +0.53 pp | 25 / 21 | 0.66 (n.s.) |

Only the two semantic rewards improve on the SFT baseline; the exploitable proxy
does not. For reference, the same table as previously published against the v4.8
baseline read gated +17.50, outcome +17.24, typecheck +15.26, all at p < 1e-20.

### Full trajectories (BEq+ %, n=760, all v4.23)

```
step        10     20     30     40     50     60     70     80     90
gated    59.21  59.08  59.08  58.95  59.34  59.87  58.42  60.66  61.45
outcome  58.55  59.08  59.21  59.34  59.47  59.74  60.26  61.18  60.53
typecheck 58.16 58.03  57.63  57.11  58.29  59.21  59.21  58.55  58.68
```

### Arm against arm at step 90 (paired)

| contrast | delta BEq+ | p |
|---|---|---|
| gated - typecheck | +2.76 pp | 0.0025 |
| outcome - typecheck | +1.84 pp | 0.070 |

The between-arm contrast was never affected by the toolchain bug: all three arms
were scored by the same job family under v4.23 throughout.

## 4. Principal finding

Every gain measured here is a reduction in the fraction of outputs that fail to
elaborate. BEq+ conditional on elaboration is flat at approximately 88% across
every checkpoint of every arm. GRPO is not improving mathematical content; it is
improving the rate at which the model emits Lean the elaborator accepts. The
reward that pays directly for elaboration is the least effective of the three at
producing it.

## 5. Caveats

- **SFT had not converged at its final checkpoint.** Re-scored under v4.23: step 38
  **50.92%**, step 76 **58.16%**, step 114 **60.92%**. `best_sft_proof.txt` pins 76.
  Measured against step 114 instead, GRPO's gain largely disappears (gated +0.53 pp,
  p=0.79). But step114 - step76 is itself only +2.76 pp at p=0.092, and picking the
  highest SFT checkpoint is selecting on the statistic. The defensible statement is
  narrow: the reward-attributable gain is of the same order as the variation
  between adjacent SFT checkpoints, and this setup cannot separate them because it has no
  selection split (`trainer.test_freq: -1`, no dev parquet). That needs a held-out
  split, not another re-score.
- **Neither metric compiles the model's proof at evaluation time.** `elaborates` means the
  generated *statement* elaborates; `BEq+` means that statement is bidirectionally
  equivalent to the gold *statement*. `proved` is 0.0% throughout.
- **`max_response_length=128` during training**, a 16GB-card artifact that outlived
  its hardware. `response_length/clip_ratio` ran 2.3 to 13.3% per step, and a
  truncated rollout scores 0 under every reward, so it was a standing gradient
  against longer answers. Now 512 repository-wide. Every figure in sections 3 to 5
  was produced at 128 and is not directly comparable to a 512 run; the
  `locolib_proof_c512` `outcome` arm bridges the two.
- **No checkpoint selection.** Every headline is step 90, fixed by the schedule.
- **Cost.** Lean scoring accounts for approximately 97% of a gated step and 1% of a
  typecheck step: ~962 s against ~26 s per step. The +2.76 pp advantage of the
  semantic reward over the proxy costs roughly 37x the wall-clock time per step.

## 6. Successor series: `c512` and `c512cos`

Two series, both at `max_response_length=512`, both resuming from
`sft_3b_locolib_proof/global_step_76`. Section 3's results were produced at 128
and are not directly comparable to either.

| series | LR schedule | arms | isolates |
|---|---|---|---|
| `locolib_proof_c512` | constant 1e-6 | `outcome`, `outcome_brevity`, `outcome_faith`, `outcome_v2` | the two new reward terms, against `outcome` |
| `locolib_proof_c512cos` | cosine 1e-6 to 1e-7 | `typecheck`, `gated`, `outcome` | the LR schedule, against `c512_outcome` |

### 6.1 Status

| arm | series | trained | eval |
|---|---|---|---|
| `outcome` | c512 | 90/90 | 9/9 |
| `outcome_brevity` | c512 | 90/90 | 9/9 |
| `outcome_faith` | c512 | 80/90 | queued |
| `outcome_v2` | c512 | 80/90 | queued |
| `typecheck` | c512cos | 90/90 | 8/9 |
| `gated` | c512cos | 40/90 | queued |
| `outcome` | c512cos | 40/90 | queued |

### 6.2 Effect of the response cap

BEq+ at step 90, n=760, v4.23: `lr6 outcome` (cap 128) 60.53%, `c512 outcome`
(cap 512) 59.87%. The endpoints are close, but the trajectories differ: at 128 the
arm climbs from 58.55 to 61.18, at 512 it is flat between 59.5 and 59.9 from step
10 onward. Raising the cap removed a standing gradient against long answers
(`response_length/clip_ratio` ran 2.3 to 13.3% per step at 128, and a truncated
rollout scores 0 under every reward). It did not by itself raise the endpoint.

### 6.3 Brevity has no measurable effect at 512

`outcome_brevity` against its own `outcome` control, paired on the same 760 rows,
McNemar exact:

| step | outcome | brevity | delta (pp) | p |
|---|---|---|---|---|
| 10 | 59.47 | 58.03 | -1.44 | 0.043 |
| 30 | 59.08 | 58.82 | -0.26 | 0.84 |
| 60 | 59.87 | 58.82 | -1.05 | 0.20 |
| 90 | 59.87 | 58.82 | -1.05 | 0.29 |

One of nine steps reaches p < 0.05, consistent with chance at alpha = 0.05 over
nine tests. The term confers no benefit and may impose a small cost.

The mechanism is measured. `val-aux/locolib/brevity/mean@1` at step 90 is 0.979 on
the brevity arm and 0.987 on the control, which computes `b` without spending it,
at 98.4% coverage. At `0.10 * (1 - b)` the mean charge is approximately 0.002,
applied to the 24% of rollouts reaching row 6. Response length is likewise
unchanged: both arms end at 60 to 64 tokens from an initial 67.

The earlier assessment that brevity requires a cap of at least 384 tokens was
necessary but not sufficient. Raising the cap made the band reachable without
making it binding, because `BREVITY_FREE_LONG = ln(1.5)` exempts everything up to
1.5x gold (2.3) and the policy's length distribution lies inside that dead band.
Testing brevity as a hypothesis requires narrowing the dead band to the observed
distribution, not increasing `beta_b`, which would steepen a slope no rollout
occupies.

### 6.4 The cosine schedule

`configs/run_grpo.sh` exposes `LR_SCHEDULER_TYPE`, `MIN_LR_RATIO` and
`LR_NUM_CYCLES`, all defaulting to verl's values so the schedule is opt-in. The
`c512cos` arms use cosine decay with `lr_warmup_ratio = 0.05` and
`min_lr_ratio = 0.1`: 1e-6 at step 5 decaying to 1e-7 at step 90, confirmed from
`actor/lr` in the typecheck arm.

`min_lr_ratio = 0.1` rather than 0.0, because arms are compared at matched steps
and a tail of zero-LR steps would leave the final comparison points uninformative.

The motivation is `train_batch_size = 16`, roughly 32x smaller than GRPO is
typically run at, so a late-run step at full LR is a full-size parameter update
driven by few prompts. The scheduler's position is stored in the checkpoint's
`extra` shard, which these arms save and load, so an 11h chunk boundary resumes
mid-curve rather than restarting the warmup.

## 7. Evaluation structure

Three measurements per model, on the pinned 760-row LoCoLib validation slice.

| metric | judge | question |
|---|---|---|
| BEq+ and elaboration rate | none | as in section 3 |
| NL to FL faithfulness | `microsoft/phi-4` | does the Lean proof carry out the informal argument? |
| FL to FL alignment | none | does it follow the reference proof's structure? |

The two faithfulness axes are distinct and neither substitutes for the other. NL
to FL is `reward/faithfulness.py`, the `f` of 2.4. FL to FL is
`scripts/eval/proof_alignment.py`, comparing intermediate goal states of two Lean
proofs with no judge and no natural language. It is scored as F1 over the two
trajectories rather than the source method's unnormalised total, because that
total is monotone in matched-checkpoint count and therefore pays for padding a
proof with intermediate `have` steps; dividing precision by the candidate's
checkpoint count makes padding cost what it buys.

miniF2F (out-of-domain) is deferred and opt-in via `WITH_MINIF2F=1 bash
hpc/submit_ood_matrix.sh`. Four single-checkpoint records computed before the
sweep was cancelled are retained under `results/eval/_partial_minif2f_20260909/`.
FL to FL cannot be run on miniF2F: all 244 golds terminate at `:= by` with an
empty body (0/244 carry a proof, against 760/760 on LoCoLib), leaving no reference
trajectory. `proof_alignment.py` drops statement-only golds with a count and exits
rather than returning 0.0 for every row, which would be indistinguishable from a
policy collapse. NL to FL is unaffected: 244/244 miniF2F prompts parse.

## 8. Reproduction

```bash
source hpc/cc_env.sh
python scripts/eval/compare_arms.py rl3b_locolib_proof_lr6_gated rl3b_locolib_proof_lr6_typecheck
python scripts/eval/select_checkpoint.py --baseline sft3blocolib_proof-step76

# section 6.3
python scripts/eval/compare_arms.py rl3b_locolib_proof_c512_outcome \
                                    rl3b_locolib_proof_c512_outcome_brevity

# section 2 invariants; stdlib only, no Lean and no cluster required
python -m reward.reward
```

Train and evaluate the new arms:

```bash
bash hpc/submit_c512cos.sh       # typecheck, gated, outcome at 512 with cosine LR
bash hpc/submit_ood_matrix.sh    # in-domain evaluation and both faithfulness axes
```

Re-derive any record from its cached generations under a named toolchain:

```bash
bash hpc/submit.sh hpc/rescore.slurm GEN=<gen_*.jsonl> TAG=v423 \
  MATHLIB_TAR=/scratch/logan03/mathlib4_v4.23_lake.tar MATHLIB_TAR_FLAT=1 \
  LEAN_INTERACT_CACHE_DIR=/scratch/logan03/ai4math_training_lean_interact_cache_v423
```

`scripts/eval/rescore_gen.py` refuses to run unless `MATHLIB_ROOT` pins the
requested toolchain, and records both the toolchain and the scoring mode
(shared-header or `--standalone`) in its output. Both were silent failure modes
before those guards existed.
