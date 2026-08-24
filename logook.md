# Done:

## 2026-08-24 — pass@1 as the reporting basis, and a fixed 10/30/50/90 grid

### Two verdicts, one measurement

Arms are now compared on **pass@1** — one T=1.15 sample per prompt over the 250
pinned prompts — reported as **BEq+ / compiles**. Both halves come off the *same*
rollouts, so the gap between them is a property of the policy.

The old pairing was `BEq+ greedy (T=0, n=400)` against `pass@32 (T=1.15, n=250)`.
That differs in decoder, prompt set AND k simultaneously, so the header had to
tell readers not to compare the two numbers in a cell — a table whose own
caption disowns its columns.

Baseline `sft3b-step93` at pass@1: **32.4% BEq+ / 67.3% compiles**
(greedy n=400: 41.2 / 79.0). The +8.8pp greedy-over-pass@1 gap is the decoder,
not training, and is stated in the table so it stops being rediscovered.

The compiling pass@k came free — the scored rollouts already carry `typecheck`
next to `beq_plus`, so `passk_report.py --metric typecheck` re-reads the same
files. Written as `results/passk_<label>_k32_tc.json`; `evalio.load_passk` now
filters on `metric`, because keying by label alone let whichever file globbed
last silently win.

**The compiling verdict discriminates only at k=1.**

| | pass@1 | pass@32 |
|---|---|---|
| compiles, SFT | 67.3% | 93.8% |
| compiles, `typecheck`-90 | 94.6% | 99.1% |
| compiles, `placebo`-30 | 40.8% | 84.7% |
| BEq+, SFT | 32.4% | 55.2% |

At k=32 every arm at every step lands in 84.7–100%: saturated, and mostly a
property of which prompts are malformed. At k=1 it is the widest-moving number
in the study, a 54-point spread. So `typecheck` RL moves single-sample
compilation **+27pp** and 32-sample compilation **+5pp** — concentration onto
outputs the policy could already produce, no new coverage, and BEq+ pass@32
falls 55.3 → 38.7 over the same 90 steps.

### The reporting grid: 10 / 30 / 50 / 90

`evalio.STEP_GRID`, applied by `make_figures.py` and `make_arms_table.py`.
Off-grid checkpoints are still trained and still evaluated (`typecheck` runs to
150) — they are just not what results are reported at.

It exists because the grid used to be whatever each arm happened to have.
`guided` was the only arm with a step-70 pass@k, which put a row in arms.md that
five of six arms could never fill; and the greedy trajectory (every 10) was
silently denser than the sampled one (every 20–40), so the two figures implied
different amounts of evidence.

Checked against the queue before adopting it: on this grid **nothing needed a
new eval**. Filling row 70 for gated/typecheck/placebo would have cost ~15
GPU-hours for a step between two we already have.

`typecheck`-50 landed (1587439, the re-run after the truncated
7,592-of-7,744-line scored file): **19.4% BEq+ / 92.6% compiles** at pass@1. It
completes the collapse trajectory — 31.1 → 24.9 → 19.4 → 20.7 BEq+ against
76.6 → 85.3 → 92.6 → 94.6 compiles across 10/30/50/90.

`gated_edge`-90 and `guided`-90 **do not exist yet**: those arms sit at steps 50
and 70, and the GRPO jobs that extend them to 90 are 1586330 (running) and
1586332 (queued). Their pass@k jobs are now queued behind them —
**1605485** (`afterany:1586330`, gated_edge-90) and **1605486**
(`afterany:1586332`, guided-90). `afterany` not `afterok`, matching the eval
chain: a GRPO chunk that hits its walltime exits non-zero, and `passk_3b`
validates the checkpoint dir itself and fails fast with a clear message.

1586331/1586333 remain the *greedy* evals for those arms; they were never pass@k
jobs, which is the gap 1605485/86 close.

`selfprove` still has no checkpoints (1586327 running). It will need four pass@k
jobs, one per grid step, once it trains.

### `selfprove` wedged a GPU for 20 minutes — and why the fix is two layers

Job 1586327 froze at `pending: 0, running: 1, finished: 5, failure: 10` for 20
consecutive one-minute samples with ~10h of walltime left and no checkpoints
written. The other two arms running alongside it showed `failure: 0`.

**The chain.** A Lean REPL process was gone; lean_interact surfaced that as
`AttributeError: 'NoneType' object has no attribute 'stdin'` (preceded by
`Broken pipe`) from inside `_execute_cmd_in_repl`. It reached
`beq_plus.typecheck_ex`, which caught only `(TimeoutError,
ConnectionAbortedError, JSONDecodeError)` — so the AttributeError escaped the
reward function, killed the verl agent-loop task holding that prompt, and the
GRPO step never closed. 17 tracebacks between 10:42 and 10:52, then silence: not
still failing, wedged.

**Why the REPL was gone is the un-obvious part.** `AutoLeanServer` is built with
`max_process_memory=0.8` precisely so it RECYCLES the REPL when it grows, and
BEq+/self-prove `exact?` searches trip that routinely once the policy
type-checks. **A dead REPL is a normal operating condition here, not a crash.**
The code was written as if it were exceptional.

**Layer 1 — `reward/beq_plus.py`, one choke point that recovers.** All Lean
traffic now goes through `_run()`, which catches anything and returns `None`;
callers translate that to an honest `infra` verdict. On a non-timeout failure it
calls `_restart()`.

The real bug was in what `_restart` has to throw away. `_env_cache` maps a header
string to an **integer env id living inside the REPL process**. Recycle the
process and every id is stale, but the dict survives — so a worker that merely
reconnected would keep submitting against environments that no longer exist and
return False for everything, silently poisoning the reward for the rest of the
run. A unit test with a fake always-dead server caught the first version of this
fix doing exactly that: it cleared the cache only on a *successful* rebuild, and
the likely case (restart also fails, because memory is still tight) left the
stale ids in place. `_restart` now invalidates first and unconditionally.

The vendored `check_theorem_equivalence` still catches only three types, so its
callers' outer `except Exception` now also calls `_restart()` when the error
classifies as `infra`.

**Layer 2 — `reward/reward_fn.py`, `@_never_raises` on all seven entry points.**
A reward function that raises does not lose a rollout, it loses the **job**. This
guarantees the step always closes whatever goes wrong later. Scoring 0 is not
neutral and is not pretended to be — a correct rollout lost to infrastructure
contributes a gradient against correctness, the same bias already documented for
Lean timeouts — so it emits `scorer_error=1` and the rate is visible in the run
metrics. Losing a rollout is recoverable; losing the job is not.

`stats["restarts"]` is now a first-class counter: if it climbs, REPLs are being
recycled often and each one costs a rollout.

**Also: fewer workers for this arm.** `selfprove` is the only arm that runs
`exact?` on *every* type-checking rollout, and 24 workers x the 8 GB
`BEQ_MEMORY_LIMIT_MB` cap is 192 GB against the job's `--mem=120G`. Even at
Mathlib's ~4.3 GB baseline, 24 workers is 103 GB with nothing left for vLLM's
host allocations. `AGENT_LOOP_WORKERS` defaults to **16** for `selfprove` only
(~69 GB typical). No reward-side guard can recover from a node OOM killing Ray
workers, so this is the part that had to change outside the reward code.

**Resubmitted:** 1607786 (chunk 1) -> 1607787 (chunk 2) -> 1607788 (eval at
n=1000). 1586327/28/29 cancelled.

### Where the Lean time actually goes

Prompted by a fair objection: how can `typecheck` cost the same per step as
`placebo`, when the placebo never opens Lean at all?

**Because verl logs no `timing_s/reward`.** The reward runs inside the agent
loop, so its cost lands in `timing_s/gen`. Over the 150 logged steps of each arm
(the only two that logged timings — the rest wrote to stderr in a form carrying
no metrics):

| arm | gen | old_log_prob | ref | update_actor | update_weights | **step** |
|---|---|---|---|---|---|---|
| placebo | 4.43s | 3.55s | 5.54s | 9.03s | 2.73s | **26.12s** |
| typecheck | 6.36s | 3.23s | 5.42s | 8.23s | 2.71s | **25.70s** |

Every non-gen term is identical, as it must be — same model, same batch, same
optimiser. **The entire type-check bill is the gen delta: ~1.9 s/step**, about
7% of the step, inside run-to-run variance. At `AGENT_LOOP_WORKERS=24` and 128
rollouts per step that is **~0.36 s of Lean per rollout** — one elaboration
against a header env already cached for that prompt.

So the two arms are not "the same speed by coincidence": ~26s is the **Lean-free
floor** of a GRPO step at this batch, and type-check sits on it.

**And yes, BEq+ really does cost that much more.** `gated` at ~962 s/step is
~935 s of Lean = **~175 Lean-seconds per rollout** at 24 workers. With
`BEQ_TIMEOUT_PER_PROOF=30` and up to 18 calls in the cascade, that is ~6
timed-out proof attempts per rollout — exactly the expected behaviour when most
rollouts are NOT equivalent to the gold and the cascade runs to exhaustion.

The magnitude is therefore not a constant factor on "a Lean call". It is **one
fast elaboration versus six half-minute searches**, and it is a property of the
*failure* path, which is where most of the mass sits.

`runtime.png` now draws the floor as a reference line and labels each arm with
its Lean share (gated 97%, guided 97%, edge pool 95%, typecheck 1%). The old
"31× between cheapest and dearest" callout was replaced: computing a Lean ratio
from the lollipops gives `(32 − 31)` in the denominator, i.e. mtime noise
amplified to four digits. The callout now quotes the logged `gen` numbers.

### Placebo hidden from the figures

`make figures --hide rl3b_v2_placebo` is now the default. **This is a
presentation change only and it is worth being clear about what it costs:**
`typecheck` does *not* substitute for the placebo. type-check is an informative
but exploitable reward — it answers "what happens if you optimise a cheap
proxy". The placebo carries calibrated *zero* information and answers "what
happens under pure update noise with matched advantage geometry". Only the
second can distinguish "RL helped" from "the policy drifted", which is the
distinction the whole control was built for.

Its numbers stay in `arms.md`, in `compare_arms.py`, and in FINDINGS. Restore it
in the figures with `--hide ''`.

### `passk_3b.slurm` now emits both verdicts

The job ran `passk_report.py` once, on `beq_plus` only, so every compiling curve
so far was generated by hand afterwards. Since pass@1 is reported as
**BEq+ / compiles**, the second call is not optional — and it is free, because
`score_rollouts.py` already writes `typecheck` next to `beq_plus` and the second
call only re-reads the file. Both records are written per job now,
`_k32.json` and `_k32_tc.json`.

### Figure fixes the grid exposed

- **`ax.legend()` replaces the axes legend.** `fig_arm_trajectories_passk` built
  the arm legend in `fs.finish()`, then called `ax.legend(handles=...)` for the
  k-encoding and `add_artist`-ed the *new* one. The arm legend had been silently
  deleted; the figure identified six arms by end label alone.
- **End labels only checked `y`.** `gated, edge pool` ending at step 50 printed
  through `guided` at step 70. `fs.finish` now boxes labels in display space.
- **The SFT reference line moved into `fs.finish(hline=...)`** so it can reserve
  its own label box — three arms finish within a point of 93.8% on the compiling
  figure and printed over it.
- **One-point arms no longer set the matched step.** `selfprove_t30` is a
  two-checkpoint probe with exactly one grid point, and it dragged
  `retention_gain` back to step 10 where nothing has diverged. Arms with a single
  grid point are excluded from the intersection; the panel now reports **step 50**
  (it was step 20 before the grid).

### New figures

- **`arm_trajectories_pass1.png`** — the reference figure. pass@1 on BEq+ (left)
  and on compiling (right), same rollouts, same decoder.
- **`arm_trajectories_passk_typecheck.png`** — pass@1 vs pass@32 on the compiling
  verdict, the counterpart to the BEq+ version.

All step-indexed figures now start at **step 0 on the SFT baseline** every arm
resumes from. `runtime.png` is the deliberate exception to the grid: per-step
cost is derived from mtime deltas between consecutive checkpoints, so subsetting
would average each value over 20–40 steps and integrate the cumulative curve from
four points.

### Edge pool: not the default yet

`gated_edge` beats `gated` at **5 of 5** matched steps (significant at 20,
p<1e-4, and 50, p=0.037) at **667 s/step vs 962** (−31%). Not adopted, because:

1. **The placebo stops being a control.** Its constants are fitted to the
   measured informative-group rate, which the edge pool sets to ~100% *by
   construction*. `BEQ_PLACEBO_GROUP_P=0.49` describes the full pool. Adopting
   edge as default requires a `placebo_edge` calibration first (~1 GPU-hour at
   31 s/step) or every gated-vs-placebo claim loses its control.
2. **1,123 prompts = 70 steps/epoch at batch 16.** Step 90 is the second pass;
   anything at 70–90 mixes pool decay with repetition.
3. **The arm was designed to decay** — a static filter on a moving target — and
   we have no data past step 50, which is where the prediction becomes testable.
   Job 1586330 supplies it.

## 2026-08-20 — Rejection sampling, the exploitability dose-response, and batch-16 GRPO

### Headline results

**1. Drift was the destroyer, not the reward — confirmed.** Rejection-sampling
fine-tuning (RFT) uses the identical BEq+ verdicts with *no advantage estimate at
all*. Every RFT arm holds SFT parity (±0.5pp) after 3 full epochs, where every
GRPO arm decayed (gated −4.5pp, placebo −14.3pp by step 50). Remove advantage
estimation and the decay vanishes entirely.

**2. But removing drift revealed no gain.** The sharp prediction — that BEq+
would turn net-positive once noise was gone — **failed**. Best arm is +0.5pp
(p=0.83). Most likely a self-distillation ceiling: the arm trains on the model's
own BEq+-certified output over the 631 prompts it already solves.

**3. The exploitability claim, quantified as a dose-response.** Three
size-matched arms (1,203 pairs each, one scoring pass, differing only in
acceptance criterion):

| training data BEq+-certified | mean Δ BEq+ vs SFT | mean Δ type-check |
|---|---|---|
| 100% (`rft_beq`)  | **+0.1pp** | +3.5pp |
| 48%  (`rft_tc`)   | **−1.0pp** | +5.2pp |
| 0%   (`rft_tcnb`) | **−2.1pp** | +5.3pp |

Monotonic in both columns. Linear interpolation at 48% predicts −1.04pp;
observed −1.0pp. Semantic accuracy is a near-linear function of how
BEq+-certified the training data was, in a setting with **zero advantage noise**.
Undiluted contrast (`rft_beq` vs `rft_tcnb`): −2.8 / −3.3 (p=0.047) / −0.5pp.

The arms are structurally indistinguishable (median length 104/109/114, vacuous
`∃a, a=…` at 0.2% in all three), so this is **not** surface-form degeneration —
it is well-formed Lean that means something else. Of the 3,590
type-checks-but-not-BEq+ rollouts: 86.9% semantically unrelated, 13.1% strictly
weaker, 0% stronger.

**4. Retention, not learning, is what BEq+ buys.** Decomposing every arm into
retention (of SFT's 155 correct) vs gains (of SFT's 245 failures):

| arm | kept | lost | gained | net |
|---|---|---|---|---|
| `rft_beq-17` | 145 (93.5%) | 10 | 12 (4.9%) | **+2** |
| `rft_beq-51` | 140 (90.3%) | 15 | 13 (5.3%) | −2 |
| GRPO gated bs4 @50 | 119 (76.8%) | 36 | 18 (7.3%) | −18 |
| GRPO placebo bs4 @50 | 90 (58.1%) | 65 | 8 (3.3%) | −57 |

Of the +9.8pp BEq+-vs-placebo gap at step 50, **29 of 39 examples (74%) are
retention**, only 10 are gains. Revises last session's framing: BEq+ mainly
*protects* capability under noisy updates; its teaching contribution is real
(7.3% vs 3.3% conversion) but small.

**5. The loop is starved, not broken.** Three independent proofs it works:
type-check-only drove type-check 76.2%→100% (saturation); every arm converts
real examples; the dose-response shows BEq+ is causally relevant. The arithmetic:

- group composition: **47.0% starved / 22.1% saturated / 30.9% informative**
- at batch 4: **1.24 informative groups per Adam update**
- gated @50: gains **+4.5pp**, losses **−9.0pp** → net −4.6pp

69% of every batch produces zero gradient. Fix the loss channel and BEq+ RL is
net-positive without one extra gain.

**6. Headroom is real.** SFT-step390 on the 1,191 RL prompts (temp 1.15):
pass@1 38.9% → pass@2 44.9% → pass@4 49.4% → **pass@8 53.0%**. 14.1pp of
prompts the policy *can* solve but doesn't reliably.

**7. Placebo geometry was off by 2×.** Measured against the gated arm's step-0
geometry on the exact RL pool (same policy, temp, k): the placebo at `p=0.30`
runs **+12.8% informative groups and +21.1% within-group σ** — not the ~11%
previously carried. `BEQ_PLACEBO_ROLLOUT_P=0.20` matches both within 1%.
NOT applied to the bs16 run, deliberately: keeping `p=0.30` preserves continuity
with `placebo_bs4` so the batch-size comparison uses an identical reward.

### Runs completed
- 3 RFT arms × 3 epochs = **9 checkpoints trained and evaluated** at n=400
- **`placebo_bs16`: all 80 steps**, 8 checkpoints (1:07:33; no Lean → ~51s/step)
- `gated_bs16`: running, ~10 min/step (Lean-dominated), projects to step 55–60

---

## 2026-08-20 (later) — batch-size confirmed, 3B migration, and two caught bugs

### Item 0 SETTLED: batch 16 halves the pure-noise damage

`placebo_bs4` vs `placebo_bs16` — **identical zero-information reward**, only the
batch differs, so any gap is batch size acting on update noise:

| step | bs4 kept | bs16 kept | bs4 lost | bs16 lost | bs4 gain | bs16 gain |
|---|---|---|---|---|---|---|
| 10 | 82.6% | 86.5% | 27 | 21 | 9 | 12 |
| 20 | 71.6% | **86.5%** | 44 | **21** | 12 | 8 |
| 30 | 71.0% | 77.4% | 45 | 35 | 11 | 12 |
| 40 | 65.2% | 72.3% | 54 | 43 | 11 | 9 |
| **50** | **58.1%** | **72.9%** | **65** | **42** | 8 | 14 |

At step 50: **−14.3pp → −7.0pp** (+7.3pp, p=0.0008). **Losses fall, gains stay
flat** (8–12 in both arms at every step) — a reward with zero information cannot
teach, so the entire improvement is noise-averaging. The gap widens with steps,
which is what accumulating-vs-cancelling noise looks like.

### 3B migration set up (Qwen2.5-Coder-3B-Instruct)

New `data_3b/` split, **verified all three sets disjoint** (val∩SFT, val∩RL,
SFT∩RL all zero):

| slice | size | note |
|---|---|---|
| val | 1,000 | MDE ~3.5pp vs ~5.6pp |
| SFT | 8,000 | |
| RL | **4,300** | **3.4× the old pool** — 268 steps/epoch at batch 16, so a 150-step run never repeats a prompt |

**The first 400 rows of `data_3b/val.parquet` are byte-identical to
`data/val.parquet`** (`VAL_OFFSET=400`, same seeded shuffle). So 3B evals at
n=400 pair directly with all 37 cached 0.5B evals under McNemar — the model-size
effect is measurable as a paired test, contradicting the earlier "never compare
a 3B number to 38.8%" caution.

Chain queued with SLURM dependencies: SFT → eval-and-pick-best → rollouts +
**go/no-go gate** + placebo fit → {gated, placebo, gated_filtered}. The gate
exits non-zero if informative <25% or saturated >45%, so `afterok` blocks all
three GRPO arms automatically if the scale-up lands past the informative peak.

`scripts/calibrate_placebo.py` fits the control's constants to the measured
policy geometry — they are policy-specific, and 0.5B's values would give a
control that is not a control. Validated by reproducing the 0.5B fit exactly
(0.37/0.20, within 1%).

### Lean cost (Q3)

Type-check short-circuit implemented and **verified output-identical offline**
against all 2,232 non-type-checking rollouts — the cascade never produced a
non-default result when type-check failed, so skipping it is exact, not an
approximation. Phase timers added. Memo-caching killed by data (0.7% repeats).
Direction short-circuit was already implemented upstream.

## 2026-08-20 (evening) — item 0 complete, 3B chain rebuilt, two more bugs

### Item 0 FINAL: batch 16 converts compounding decay into an equilibrium

All 8 `placebo_bs16` checkpoints evaluated. The level difference was expected;
the *shape* difference was not:

| step | bs4 BEq+ | bs16 BEq+ | bs4 lost | bs16 lost | bs16 gained |
|---|---|---|---|---|---|
| 10 | 34.2% | 36.5% | 27 | 21 | 12 |
| 20 | 30.8% | 35.5% | 44 | 21 | 8 |
| 30 | 30.2% | 33.0% | 45 | 35 | 12 |
| 40 | 28.0% | **30.2%** ← floor | 54 | 43 | 9 |
| 50 | **24.5%** | 31.8% | 65 | 42 | 14 |
| 60 | — | 33.5% | — | 32 | 11 |
| 70 | — | 33.5% | — | 32 | 11 |
| 80 | — | 32.0% | — | 38 | 11 |

**At batch 4 damage compounds without bound** (losses 27→65, still falling at
step 50 when the run ended). **At batch 16 it equilibrates**: losses peak at 43
then settle to 32–38, BEq+ bottoms at 30.2% and holds 32–33.5% for 40 more
steps. Gains are flat at 8–14 in both arms at every step — a zero-information
reward never teaches, so the entire difference is the loss channel.

Interpretation: with 4× more groups per update the random-walk step shrinks
enough that the KL anchor (`kl_loss_coef=0.01`) reaches equilibrium with the
noise instead of being overwhelmed. That is stronger than "batch 16 helps" — it
says a stable operating point exists at batch 16 that does not exist at batch 4.
Which sharpens what `gated_bs16` tests: BEq+ should sit clearly ABOVE the ~33%
noise equilibrium, not merely decay more slowly.

*(Steps 60 and 70 have identical aggregates. Checked rather than assumed: the
checkpoints differ, and their per-example vectors differ on 20 examples that
flipped both ways. Genuine coincidence.)*

### Item 2 in progress — and the binary framing was wrong

`starved_k32` at ~1,500/8,000 scorings, 47 complete prompts (all 0/8 at k=8):

- solved ≥1× at k=32: **17.0% ±10.7**
- type-check ≥1× at k=32: **85.1%**
- successes-per-prompt: `{0: 39, 1: 5, 3: 1, 4: 1, 10: 1}`

The plan said "near 0% ⇒ capability, clearly >0% ⇒ exploration". **That was too
binary.** Of the rescued prompts most succeed exactly 1/32 (p≈0.03 — such a
prompt shows 0/8 about 78% of the time, so it was rare rather than unreachable),
while 39 of 47 stay at zero across 32 samples *while type-checking 85% of the
time*. The model reliably writes well-formed Lean that means the wrong thing.

Provisional: 17% of the 47% starved pool ≈ 8% of all prompts recoverable by 4×
sampling → informative maybe 30.9% → 35–39%. Real but not transformative, and it
costs 4× the Lean scoring, which is already ~92% of GRPO wall-clock. Reading:
**exploration is a genuine but expensive minor lever; semantic capability is the
dominant limit.** Supports the 3B move, and suggests testing `rollout_n` there.

### Jobs restructured to 3h for scheduler priority

CC favours short jobs in backfill. Sizing against measured rates
(SFT 1.12 min/step, eval 1.7s/example, scoring 2.08s/example) put everything
at ≤3h — and exposed a job that could never have finished:

**`rollouts_3b` was impossible as written**: 4,300 prompts × k=8 = 34,400 BEq+
calls ≈ **19.9h against an 11h walltime**. It would have been killed mid-scoring
and `afterok` would have blocked all three GRPO arms. Fixed with a **600-prompt
geometry subsample** (2.8h) — that job measures two proportions, so a sample is
as good as a census (±3.7% on the informative fraction, far tighter than the
25%/45% gate needs). **GRPO still trains on all 4,300.**

GRPO now runs as **4 chained 3h chunks per arm** with `resume_mode=auto`, chained
`afterany` (a walltime kill is the expected chunk boundary, not a failure).
`test_freq` 25→50 halves in-run validation from 2.8h to 1.4h.

---

## 2026-08-22 — 3B arms land: the placebo gets *worse* at scale, and type-check dies of saturation

### 3B baseline (n=1000, `data_3b/val.parquet`)

| checkpoint | BEq+ | type-check | weaker-only |
|---|---|---|---|
| `sft3b-step31` (ep1) | 37.8% | 77.4% | 7.9% |
| `sft3b-step62` (ep2) | 37.9% | 75.9% | 5.6% |
| **`sft3b-step93` (ep3)** | **39.4%** | **76.7%** | 5.0% |

This is the number every 3B result is measured against. Note it is *only*
+0.6pp over the 0.5B `sft-step390` (38.8% at n=400) — 6x the parameters bought
almost nothing at SFT. The 3B gain showed up in **group geometry**, not in
pass@1: informative groups 30.9% -> 46.0%, via saturation collapsing
22.1% -> 8.7%.

### The placebo is MORE destructive at 3B, not less

`rl3b_placebo` (batch 16, recalibrated `GROUP_P=0.55 / ROLLOUT_P=0.20`),
evaluated at n=1000:

| step | BEq+ | type-check |
|---|---|---|
| — (SFT) | 39.4% | 76.7% |
| 10 | 33.9% | 74.3% |
| 20 | 29.8% | 67.6% |
| 30 | **21.2%** | 63.8% |
| 40 | 27.0% | 67.5% |
| 50 | 23.6% | 55.6% |
| 60 | 22.5% | **49.9%** |
| 70 | 26.4% | 60.4% |
| 80 | 27.1% | 61.3% |

**This contradicts the batch-16 equilibrium found at 0.5B.** There, pure noise
bottomed at 30.2% and held 32-33.5% for 40 steps against a 38.8% start (a ~5pp
crater). Here it craters ~12-18pp and never recovers, and it takes *type-check*
down with it — 76.7% -> 49.9%, a syntactic capability the 0.5B placebo never
touched. Same batch size, same `kl_loss_coef=0.01`, geometry-matched reward.

Working reading: batch 16 was tuned as a noise-vs-KL equilibrium for a 0.5B
policy. A 3B policy takes larger effective steps per update at the same LR
(`1e-6`), so the same anchor no longer holds it. **Do not carry the 0.5B batch
conclusion across scales.** The `bigbatch` arm (256) tests exactly this and is
now the load-bearing config question, not a side experiment.

Consequence for interpretation: the 3B placebo floor is *lower*, so a 3B gated
arm has a wider gap to open. Any BEq+ effect measured here will look larger than
the 0.5B +9.8pp for reasons that have nothing to do with BEq+. Report against
the 3B placebo only.

### Type-check-only RL dies of reward saturation — the mechanism, measured

`rl3b_typecheck` ran the full 150 steps (job 1516068). The reward curve:

| step | train `critic/score/mean` | val type-check | mean response length |
|---|---|---|---|
| 1 | 0.734 | — | 63.8 tok |
| 50 | 0.930 | 94.7% | 50.1 tok |
| 100 | 0.938 | 97.3% | 45.0 tok |
| 150 | **1.000** | **98.4%** | 45.8 tok |

At step 150 `critic/advantages` has **max = min = 0.0**. Every group is 8/8.
The reward is not merely weak, it is **identically zero gradient** — the arm
spent its last ~50 steps taking pure KL-regularised random walks with a dead
objective. Responses shrank 28% along the way.

First eval point, n=1000: `rl3b_typecheck-step10` = **87.6% type-check
(+10.9pp), 38.8% BEq+ (-0.6pp)**. The signature is exact: the arm buys a large
amount of the thing it is rewarded for and none of the thing we care about.
Later steps are evaluating (job 1516069).

**One negative result, stated so it is not overclaimed:** a syntactic check for
literal `hypothesis == goal` tautologies found 0.7% in the step-10 arm vs 0.9%
in SFT and 0.8% in the placebo — no difference. Whatever makes these statements
easy to elaborate at step 10, it is *not* the crude tautology hack. The
degeneracy visible so far is length collapse and reward saturation, not a
recognisable exploit template. Do not claim "the model learned to write
tautologies" without evidence we do not have.

### Multi-turn Lean feedback: ON by default from here

`hpc/grpo_3b.slurm` and `hpc/grpo_3b_bigbatch.slurm` now default
`MULTITURN=1` (set `MULTITURN=0` to disable), wiring
`reward/lean_tool.py::check_lean_statement` in as a verl `function_tool` so the
policy sees **Lean's actual diagnostics** rather than a category label.

**The current arm set is unaffected and stays internally comparable.** `sbatch`
snapshots the script at submit time, and every queued job (1515240/41,
1515248-50, 1516068, 1518169-71) was submitted pre-patch. Verified directly in
the config dump of 1516068: `multi_turn: {'enable': False}`. So
`gated / placebo / typecheck / selfprove / bigbatch` are all single-turn and
compare cleanly; multi-turn starts a **new** comparison set and its arms must
never be paired against these.

# TODO

## In progress

- **A. 3B single-turn arm set.** `gated` (1515240/41), `bigbatch` (1515248-50),
  `selfprove` (1518169-71) running; `typecheck` complete at 150 steps and
  evaluating (1516069); `placebo` complete and fully evaluated. All single-turn,
  all pairable at n=1000.
- **B. `guided` — the process-level reward arm** (1526596-98, eval 1526599).
  Submitted with `MULTITURN=0` so it pairs against the set above. Splits the
  32.3% "type-checks, earns nothing" dead band with gold-referenced similarity;
  `algorithm.norm_adv_by_std_in_grpo=False` is load-bearing, not a tuning choice.
- **C. Edge-of-competence pool** (score 1526600/01 -> build 1526607 ->
  `gated_edge` 1526608-10, eval 1526611). Two disjoint 900-prompt slices scored
  at k=8 in parallel, then the RL pool is restricted to prompts scoring 1-7/8.
- **D. pass@32 on the 3B SFT baseline** (1526602), on a pinned 250-prompt val
  subset. Closes the capability-vs-sharpening gap (see FINDINGS §7a).

## Next

0. **Mid-training stage** — designed, not yet built. See FINDINGS §7c and the
   design note below. Evaluable with the existing go/no-go gate (informative
   fraction + pass@32) *before* spending any RL job.
1. **Re-open batch size at 3B.** The 0.5B equilibrium did not transfer (see
   2026-08-22). `bigbatch` (256) is now the primary config test. If 256 also
   craters, the LR is the suspect, not the batch — try `1e-6 -> 3e-7` before
   spending another arm on batch size.
2. **First multi-turn arm** (`gated`, `MULTITURN=1`), evaluated against a
   **multi-turn placebo**, not against the single-turn set.
3. **`FILTER_GROUPS=1` (DAPO)** at 3B. Cancelled at 0.5B when refill measured
   ~90 min/step (8x, not the estimated 3x) because refill is one-prompt-at-a-
   time. Only worth retrying if the refill path can be batched.
4. **Process-level reward from the `semantic_signal` ladder** — see the
   Interplay-paper notes below. This is the highest-value untried idea.
5. **`rollout_n` at 3B.** Item 2 (below, settled) says k=32 recovers 14.5% of
   starved prompts at 4x Lean cost. Whether that pays depends on the 3B
   geometry, which is now known (46.0% informative) — probably not urgent.
6. **`results/` is still outside git.** `FINDINGS.md` untracked,
   `reward_impact.md` modified with the recovered rewrite, ~40 eval files
   untracked. A stray `git checkout` loses the rewrite silently.

## Done

0. ~~Batch-size effect on pure update noise.~~ **SETTLED at 0.5B**: bs4
   compounds without bound (losses 27->65, floor 24.5%), bs16 equilibrates
   (losses peak 43 then 32-38, BEq+ 32-33.5% through step 80). Gains flat 8-14
   in both — a zero-information reward never teaches. **Does not transfer to
   3B** (2026-08-22); superseded by Next-1.
2. ~~Capability vs exploration on the starved pool.~~ **SETTLED**: 242/250
   starved prompts at k=32 -> **14.5% recovered** (CI 10.0-18.9), 24/35 of the
   rescued succeed only <=2/32, and **84.3% type-check**. The model reliably
   writes well-formed Lean that means the wrong thing. **Capability limit, not
   exploration.**
3. ~~Eval power.~~ **DONE for 3B**: `data_3b/val.parquet` is n=1000 (MDE
   ~3.5pp) and its first 400 rows are byte-identical to `data/val.parquet`, so
   cross-scale paired tests still work. 0.5B left at n=400 deliberately.
4. ~~Model size.~~ **DONE**: full rebuild on Qwen2.5-Coder-3B-Instruct, fresh
   `data_3b/` split (8000 SFT / 4300 RL / 1000 val), SFT baseline 39.4%,
   go/no-go gate passed on geometry (informative 30.9% -> 46.0%).
5. ~~Tighten the placebo.~~ **DONE**: `scripts/calibrate_placebo.py`
   reproduces the 0.5B fit (0.37/0.20) and fitted 3B at
   `GROUP_P=0.55 / ROLLOUT_P=0.20`.
6. ~~Evaluate `gated_bs16` at 0.5B.~~ Superseded — the 3B series answers the
   same question with more power and a calibrated control.
9. ~~Read the Interplay paper.~~ **DONE** — notes and the four proposed
   alignments are in `results/FINDINGS.md`; actions carried into Next-1/2/4.

---

## 2026-08-22 (later) — queued: process reward, edge pool, pass@32

Three of the four Interplay-paper alignments are now jobs. New infrastructure:

| file | what it does |
|---|---|
| `hpc/grpo_3b.slurm` | new arms `guided` and `gated_edge`; `TRAIN_FILE` is now overridable; the `EXTRA` array was assembled but never passed to `run_grpo.sh` — fixed |
| `hpc/score_pool_3b.slurm` | scores one 900-prompt slice of the RL pool at k=8 (~4.2h); slices are disjoint by prompt text so they run in parallel |
| `hpc/build_edge_pool.slurm` | merges every scoring pass, writes `data_3b/train_edge.parquet`, and **fails** if the pool is under 700 prompts |
| `hpc/passk_3b.slurm` | pass@32 on a pinned 250-prompt val subset (`data_3b/passk_subset.parquet`, seed 7 — do not regenerate) |
| `scripts/make_difficulty_subset.py` | keeps prompts scoring `--min-wins..--max-wins` at k=8; joins on prompt TEXT with gold verification; reports steps/epoch and warns past 3 |
| `scripts/passk_report.py` | unbiased pass@k (Chen et al. 2021), not "did any of my n succeed" |

### Why `gated_edge` needed 1,800 more prompts scored first

Running the difficulty filter on the geometry pass alone kept **267 of 585**
prompts — 16.7 steps/epoch at batch 16, so 150 steps would be **9 epochs** over
267 prompts and memorisation would confound any gain. Two 900-prompt slices
bring the scored pool to ~2,385 and the edge pool to ~1,090, i.e. ~2.2 epochs.
The script's own guard printed this before anything was submitted; the guard is
now duplicated as a hard exit in `build_edge_pool.slurm`.

The dry run also caught **2 prompts whose gold disagreed** after the text join
and excluded them — the same class of defect that produced "starved" prompts
scoring 32/32 at 0.5B. Also worth knowing: the 4,300-row RL pool has only
**3,677 distinct prompt texts**.

### Similarity smoke test before launching `guided`

`zss` is installed (without it `structural_similarity` returns 0.0 for every
pair and silently disables half the signal — verified present). Against a gold
of `a < b -> a + 1 < b + 1`:

| variant | struct | symbol | total |
|---|---|---|---|
| renamed variables | 0.700 | 1.000 | 0.820 |
| **inequality INVERTED** | 0.950 | 1.000 | **0.970** |
| `theorem qux : 1 = 1` | 0.200 | 0.000 | 0.120 |

An inverted inequality scores *higher* than a correct renaming. This is exactly
why the similarity weight (0.10) sits strictly below the one-direction step
(0.15): similarity guides the search, only BEq+ certifies. It is also why
`norm_adv_by_std_in_grpo=False` is mandatory for this arm — a group that is
uniformly "type-checks only" has a reward spread of ~0.01, and verl's default
would renormalise that back to full scale, handing near-noise the same gradient
as proving equivalence.

### pass@k estimator self-test

`passk_report.py` on the existing k=8 geometry pass gives pass@1 35.6%,
pass@8 54.7% — matching the independently-measured 53.0%. Estimator sane.

---

# Design note: what mid-training would look like here

Not yet built. This is the plan, and the argument for it, so it can be costed.

## What the paper means by it — and what it is not

In arXiv:2512.07783 mid-training is **continued pre-training** (plain next-token
LM, no instruction format) on a controlled data mixture, inserted between
pre-training and RL. In their repo it is a LLaMA-Factory `pt` stage over
mixtures like `id2-10_0.2easy_0.3medium_0.5hard`. It is **not** task-specific
supervised fine-tuning — we already have that, and it is not the thing that
helped them. Their claim is that under *fixed compute*, moving budget from RL
into this stage wins.

## Why our own evidence points the same way, independently

Item 2 is the argument: on prompts the policy never solves, **84.3% still
type-check**. The model reliably writes well-formed Lean that means the wrong
thing. RL can only re-weight what the policy already samples, so a reward — any
reward, BEq+ included — cannot manufacture the missing knowledge. That is also
why the gain channel sat at 4.9-7.3% across every 0.5B arm regardless of signal,
and why 6x the parameters moved SFT BEq+ by only +0.6pp (38.8% -> 39.4%): the
bottleneck is not capacity, it is that neither model has seen enough Lean.

Corroborating: at 0.5B, **SFT scaling was the only intervention that ever raised
BEq+** (4k x 2ep -> 34.0%; 20k x 5ep -> 38.8%).

## Corpus

We have the material on disk already.

1. **Mathlib4 declarations — the primary corpus.** `repos/mathlib4/Mathlib` is
   4,187 `.lean` files carrying **107,993 top-level `theorem`/`lemma`
   declarations**. Statements-only (strip the proof after `:=`) is *exactly* the
   output distribution the task asks the model to produce: correct binder
   conventions, real lemma names, `Real.sqrt` not `sqrt`, the difference between
   `(h : 0 < x)` and `(hx : x > 0)`. Roughly 7M tokens at ~65 tokens each, out of 178MB of Mathlib source total.
2. **Mathlib4 full source** — statements with proofs and docstrings, for broader
   idiom. Larger, lower density of the thing we actually want.
3. **NOT more Lean-Workbook.** The 13,297 unique prompts are fully allocated
   (8,000 SFT / 4,300 RL / 1,000 val). Any Lean-Workbook text added here
   contaminates a split. External formal corpora (miniF2F, ProofNet, Compfiles)
   are the only safe additions, and they are small.

Contamination check to run before training, not after: Lean-Workbook golds are
named `lean_workbook_plus_NNNNN` and do not appear in Mathlib, but assert it
rather than assume it.

## The difficulty-mixture analogue

Their axis is operation count. Ours is **statement complexity** — binder count
is the cheapest proxy and is already parsed by `reward/similarity.py`'s binder
extraction. Bucket Mathlib declarations easy / medium / hard by binder count
(or token length), and mirror their weighting toward hard (`0.2/0.3/0.5`). This
is the part of their result least likely to transfer unchanged, so treat the
mixture as a knob to test, not a constant to copy.

## Where it goes in the pipeline — start from the BASE model

Current chain is `Qwen2.5-Coder-3B-Instruct -> our SFT -> GRPO`. Raw LM
continued pre-training on an already-instruction-tuned model degrades
instruction following, and we would then be repairing that damage in SFT.

Since **we do our own SFT anyway**, the clean chain is
`Qwen2.5-Coder-3B (base) -> mid-train on Mathlib -> our SFT -> GRPO`.
That removes the forgetting concern entirely and costs one extra model download.
It does mean the SFT baseline changes, so **every 3B number becomes a new
series** — the same discipline the 0.5B -> 3B move required.

## How to evaluate it WITHOUT spending an RL job — the important part

Mid-training is the most expensive item on the list, but it does not need RL to
be judged. The paper's own two preconditions are things we already measure:

| their claim | our existing measurement | job |
|---|---|---|
| pre-training must leave headroom | pass@32 on the pinned val subset | `hpc/passk_3b.slurm` (baseline running as 1526602) |
| RL data must hit the edge of competence | informative-group fraction at k=8 | `hpc/rollouts_3b.slurm` go/no-go gate |
| the stage itself helps | SFT BEq+ at n=1000 | `hpc/eval_3b_sft.slurm` |

So the decision procedure is: mid-train, SFT, then run those three. **If pass@32
and the informative fraction both rise, mid-training worked and the RL series is
worth re-running on top of it. If neither moves, stop — no reward can exploit
headroom that is not there.** That gate costs ~1 SFT run plus ~10h of scoring,
not a full arm set.

## Cost and risks, stated plainly

- **Compute is the real objection.** A 3B model over ~7M tokens of Mathlib
  statements is a few epochs of continued pre-training on 2xA100 — order one to
  two 11h chunks, which is cheap. Going to full Mathlib source is 10-50x that
  and is where the budget actually goes. **Start with statements-only**; it is
  the highest-density corpus and the cheapest test of the hypothesis.
- It **invalidates the current 3B baseline** and starts a new comparison series.
  Do not begin it until the arm set now running has been read out.
- The mixture weighting is the least transferable part of the paper's result.
- No verl involvement: this is a plain LM fine-tune, so it does not touch the
  fragile vllm/torch-2.11 path at all.

---

## 2026-08-22 (night) — type-check RL is now a demonstrated failure, and mid-training is built

### `rl3b_typecheck`: the exploitability claim, cleanly, at 3B

| step | BEq+ | type-check |
|---|---|---|
| SFT | 39.4% | 76.7% |
| 10 | 38.8% | 87.6% |
| 20 | 35.0% | 89.0% |
| 30 | 28.6% | 91.8% |
| 40 | 23.5% | 93.9% |
| 50 | **20.6%** | **94.7%** |

Monotone in both columns and in opposite directions: **+18.0pp type-check bought
with -18.8pp BEq+.** Later steps still evaluating.

The placebo is what makes this decisive rather than merely suggestive. At step 50
the placebo sits at 23.6% BEq+ / 55.6% type-check. The type-check arm is at
20.6% / 94.7% — **worse than pure noise on semantics while being 39pp better on
the proxy.** Undirected GRPO damage degrades both channels together; only an
optimiser pointed at the wrong objective degrades one while climbing the other.
So this is not "RL hurts", it is "this reward is actively adversarial to the
thing we measure".

Train-side confirmation from the same run: reward 0.73 -> 1.000, and by step 150
`critic/advantages` has max = min = 0.0 — every group 8/8, identically zero
gradient. Response length fell 63.8 -> 45.8 tokens.

This retires the "is there a sound non-BEq+ baseline?" question. There is not one
for autoformalization, for the structural reason already in
`configs/run_grpo.sh`: the statement is the model's OUTPUT, so a compiler-only
reward is satisfied by anything that elaborates. `selfprove` is the strongest
remaining gold-free contender and is still running.

### Mid-training implemented (built, not launched)

| file | what |
|---|---|
| `scripts/prepare_midtrain_dataset.py` | extracts Mathlib theorem statements (proofs stripped), dedups, difficulty-buckets, contamination-checks, writes parquet + a validation sample |
| `scripts/validate_midtrain_corpus.py` | type-checks a 500-statement sample and reports the standalone-valid rate and failure modes |
| `hpc/midtrain_3b.slurm` | the stage itself: 1 epoch, LR 2e-5, 2 GPUs, resumable; runs the corpus gate first |
| `hpc/sft_3b.slurm`, `eval_3b_sft.slurm`, `rollouts_3b.slurm`, `grpo_3b.slurm` | `TAG` / `BEST_SFT` / `SERIES_TAG` now scope checkpoints, eval labels, best-SFT pointers, geometry stats and placebo constants per series. **Defaults reproduce the existing paths byte-for-byte** — no running job changes behaviour. |

Corpus: **106,689 declarations parsed, 90,561 kept**, sampled to **60,374** at an
exact 0.2/0.3/0.5 easy/medium/hard mix. Contamination check clean against 4,639
split golds.

Three defects caught while building it, all by measuring rather than assuming:

1. **The difficulty mix was silently not being applied.** Binder counts are
   discrete and pile up, so tercile cuts gave 56,774 / 16,708 / 17,079 and a
   requested 0.2/0.3/0.5 degraded to 0.27/0.36/0.37 — the paper's one knob, not
   actually set. Now ranked on `(binders, length)` and split at rank thirds, so
   the buckets are equal by construction (30,187 each; easy 1-2 binders / median
   75 chars, hard 3-20 / median 150) and the total is capped at what the scarcest
   bucket supports so the mix is exact.
2. **40% of the gradient was going to Qwen boilerplate.** verl masks only
   `loss_mask[:len(generation_prompt)]`, i.e. `<|im_start|>system\n`. With a lone
   assistant message the chat template injects Qwen's DEFAULT system prompt and
   it lands *inside* the trained region — 48 of 51 tokens trained, 21 of them
   "You are Qwen, created by Alibaba Cloud...". Adding an explicit empty system
   message masks that block; trained text is now exactly the statement. Verified
   by decoding `input_ids[loss_mask.bool()]` for both variants.
3. **A quoted heredoc would have written a literal `${SUFFIX}`** into the geometry
   subset filename in `rollouts_3b.slurm`. `bash -n` passes on that. Fixed by
   passing the suffix as argv.

### Known limitation, stated before the numbers exist

Mathlib declarations lean on `variable` lines and open namespaces, so many are
not self-contained once lifted out of their file — and the task's outputs are.
`validate_midtrain_corpus.py` measures this on a 500-statement sample (+/-4.4%)
and runs as the first step of `midtrain_3b.slurm`. If the standalone-valid rate
is low the corpus still teaches vocabulary, but the claim about statement SHAPE
has to be dropped. Do not read the downstream numbers without that rate in hand.

### Launch recipe (deliberately not submitted)

```bash
sbatch hpc/midtrain_3b.slurm
L=$(ls -d checkpoints/midtrain_3b/global_step_* | sort -t_ -k3 -n | tail -1)/huggingface
sbatch --export=ALL,MODEL=$(readlink -f $L),TAG=mt hpc/sft_3b.slurm
sbatch --export=ALL,TAG=mt                        hpc/eval_3b_sft.slurm
sbatch --export=ALL,BEST_SFT=data_3b/best_sft_mt.txt,TAG=mt hpc/rollouts_3b.slurm
sbatch --export=ALL,LABEL=sft3bmt,CKPT_DIR=$(cat data_3b/best_sft_mt.txt) hpc/passk_3b.slurm
```
The last two ARE the gate: if the informative fraction and pass@32 both rise,
mid-training worked and the RL series is worth re-running on top
(`BEST_SFT=... SERIES_TAG=mt`). If neither moves, stop — no reward can exploit
headroom that is not there. That costs ~1 SFT run plus ~10h of scoring, not a
full arm set.

---

## 2026-08-23 — mid-training answered (negative), three infra defects, and a flaw in the control

### Mid-training: built, run, and it did not help

| | BEq+ (n=1000) | type-check | informative | pass@1 | pass@32 |
|---|---|---|---|---|---|
| `sft3b-step93` (baseline) | **39.4%** | 76.7% | **46.0%** | 32.4% | 55.2% |
| `sft3bmt-step93` (mid-trained) | 38.9% | 75.7% | 42.9% | 31.9% | 56.4% |

Epochs for the mid-trained SFT were 38.9 / 38.1 / 38.9 — flat, no trend.

Read honestly: **every headline number is within noise of the baseline and none
favours mid-training.** pass@32 is +1.2pp, informative is −3.1pp, BEq+ is −0.5pp
at n=1000 (MDE ~3.5pp). The paper's precondition — that mid-training raises
headroom — is not met here.

Both halves of the gate now exist, on the pinned 250-prompt subset, k=32:

```
sft3b-step93   pass@1 32.4  @2 39.6  @4 44.1  @8 47.9  @16 51.8  @32 55.2
sft3bmt        pass@1 31.9  @2 39.0  @4 44.0  @8 48.1  @16 52.1  @32 56.4
```
The curves are on top of each other. That is the cleanest statement of the
result: mid-training on Mathlib statements moved neither pass@1 nor the ceiling.

**Why this is a weak test, not a refutation.** The corpus gate reported
**13.0% standalone-valid** — 87% of extracted Mathlib statements do not
elaborate on their own. The failure modes say why, and it is NOT mainly what I
predicted: only 6.4% were `unknown identifier` (free `variable`s, the thing the
docstring warned about). **80.6% were "other"**, and reading them, they are
dominated by `failed to synthesize instance` — `Ring R`, `Semiring R`,
`NonAssocSemiring R`, `LE α`. Mathlib statements carry their typeclass context
in `variable` lines and section binders, so lifted out they are not merely
missing a name, they are missing their algebraic structure. (Correction to an earlier reading: I took
`IsCompl (A i) (` as evidence the 600-char cap was truncating statements
mid-expression. It was not -- that was my own `[:150]` display slice in the
inspection script. The cap DROPS long statements, it never cuts them, and an
explicit bracket-balance check added afterwards found 0 truncated declarations
in all 106,689.)

So the corpus taught the model to produce text that mostly does not compile.
That plausibly explains why type-check went DOWN (76.7 → 75.7). The idea is not
refuted; **this corpus** is. A corpus that would test it properly needs either
statements elaborated in their original context, or a filter that keeps only the
13% that stand alone (~11.7k of 90.5k, still a usable size).

### Three infrastructure defects, all costly

**1. `multiprocessing.Pool` hangs forever when a worker dies holding a task.**
Signature: `worker N ready (Mathlib loaded)` printed at the very tail, and the
output file exactly ONE rollout short. Cost: `score_pool` slice 1 and BOTH
`passk` jobs finished their real work and then sat hung until walltime —
**~9 GPU-hours burned** — and because they exited TIMEOUT rather than COMPLETED,
every `afterok` behind them was cancelled, killing the whole `gated_edge` chain.
Fixed in `scripts/score_rollouts.py`: the iterator is now driven by hand with
`it.next(timeout=--result-timeout)` (default 900s), turning a silent multi-hour
hang into a clean exit that keeps 99.99% of the data. Smoke-tested both paths.

All three stalled files were recoverable — the reports were regenerated offline
from the scored jsonl with no rerun.

**2. `passk_report.py` would have silently dropped the top of the curve.**
It truncated every group to the MINIMUM size; one ragged group from a killed run
would have made a 7,743/7,744-line file report pass@16 as the maximum. Now uses
the modal size and drops short groups (1 of 242 here).

**3. The placebo control was fitted on a degenerate objective.**
`calibrate_placebo.py` fitted only (informative rate, within-group sd). Within-
group sd is **symmetric under p <-> 1-p**, so the problem has two equally good
optima and the grid returned whichever it hit first. It did: the base 3B series
got `ROLLOUT_P=0.20` and the mid-trained series `ROLLOUT_P=0.79` from nearly
identical geometry.

This is not cosmetic. At k=8 a 2/8 group gives advantages `{+0.75 x2, -0.25 x6}`
and a 6/8 group gives `{+0.25 x6, -0.75 x2}`: same spread, **mirrored skew**. And
the shipped base fit has mean reward 0.55x0.20 = **0.11 against a measured
0.36** — a 3.2x miss on reward level.

Fixed by adding mean reward (the asymmetric statistic) to the objective. Both
series now fit uniquely and agree:

| series | old fit | new fit | informative | sd | mean reward |
|---|---|---|---|---|---|
| base 3B | 0.55 / 0.20 | **0.49 / 0.74** | -3.0% | +2.6% | +2.0% |
| mid-trained | 0.51 / 0.79 | **0.49 / 0.77** | +0.1% | +1.0% | +1.1% |

`calibrate_placebo.py` now REFUSES to overwrite an existing constants file
without `--force`, because rewriting them retroactively changes what a finished
control means. `data_3b/placebo_constants.sh` is therefore untouched.

**What this does and does not invalidate.** `rl3b_placebo` (150 steps, fully
evaluated) was trained on the wrong branch. It is still a zero-information
reward, so it remains a valid *noise* control, and the informative fraction and
within-group sd it was fitted to did match. What it is NOT is level-matched or
skew-matched to the real reward. Any claim that leans on the placebo's
*magnitude* rather than its direction needs the corrected arm. Re-running the
placebo at 0.49/0.74 is one 150-step job and should happen before the next
write-up.

### `rl3b_typecheck` complete: collapse to 6.0%

| step | 10 | 30 | 50 | 70 | 90 | 110 |
|---|---|---|---|---|---|---|
| BEq+ | 38.8 | 28.6 | 20.6 | 26.3 | 20.2 | **6.0** |
| type-check | 87.6 | 91.8 | 94.7 | 95.8 | 97.4 | **98.4** |

From 39.4% to **6.0%** BEq+ while type-check goes to 98.4%. The placebo over the
same span falls to 18.8% with type-check at 57.2%. So the type-check arm ends
**12.8pp below pure noise on semantics and 41pp above it on the proxy** —
an optimiser working hard in the wrong direction, not drift.

---

## 2026-08-23 (later) — Mathlib is not a viable mid-training corpus, and why

Three successive filters, each tested against Lean rather than assumed. The
standalone-valid rate on a 500-statement sample:

| corpus | filter | statements | standalone-valid |
|---|---|---|---|
| v1 | length + binders + dedup | 90,561 | **13.0%** |
| v2 | + reject instance binders / `Type*` / bare type-var binders | 43,905 | **12.0%** |
| v3 | + reject any unbound short/Greek identifier | **6,424** | **38.0%** |

**Two of my three hypotheses were wrong, and the failure messages said so.**
v1's failures were 80.6% "other", which I read as typeclass context missing --
hence v2. v2 changed nothing (13.0% -> 12.0%), because the real errors were
`function expected at f, term has type ?m.63`: an unsolved METAVARIABLE. Mathlib
declares `variable {α : Type*} (f : X → Y) (s : Set α)` at file or section
scope, so a lifted statement references `f, X, Y, s, ι, F, G` with nothing
binding them. Rejecting bare single-letter binder *types* caught only `(x : α)`
and left `Set α`, `F ≅ G`, `OrthonormalBasis ι ℝ F` untouched.

v3 tests the right thing — every short or Greek identifier must be bound by the
statement itself — and was validated in both directions before use: it keeps
**366/400 (91.5%) of our own validation golds** (the single rejection has a
genuinely free `c` and would not elaborate either) and rejects **18/20** of the
Mathlib statements Lean had already refused. It works: 13.0% -> 38.0%.

### The conclusion is about the corpus, not the filter

Of 106,689 Mathlib declarations, **6,424 pass the syntactic test and ~38% of
those elaborate standalone — about 2,440, or 2.3%.** At ~30 tokens each that is
roughly **75k tokens**. Continued pre-training on 75k tokens will do nothing.

This is structural, not a bug worth a fourth filter. Mathlib is written to be
maximally general and is factored through `variable` / `section` / `namespace`
precisely so statements need NOT restate their context. Our task's targets are
the opposite: fully self-contained one-liners. The two distributions barely
intersect.

**Stopping here.** The v1 corpus was already run end to end and produced
38.9% BEq+ / 42.9% informative / pass@32 56.4% against a 39.4% / 46.0% / 55.2%
baseline — no effect. A v3 run on 2,440 statements would be a weaker test of the
same idea, not a stronger one.

### If mid-training is revisited, the corpus has to come from elsewhere

1. **Emit statements with their file's `variable`/`open`/`namespace` preamble.**
   Makes them valid; changes the shape, since the model would learn to write
   preambles it must not produce at inference.
2. **A different corpus.** Lean-Workbook is fully allocated. miniF2F (488),
   ProofNet (371) and Compfiles are all far too small. The large
   autoformalization sets overlap competition problems and would risk val
   contamination — that would have to be checked, not assumed.
3. **Self-distillation.** Sample statements from the SFT policy, keep the ones
   that type-check, mid-train on those. Unlimited and cheap, but it teaches the
   model its own distribution, so it cannot add vocabulary it does not have —
   which is the whole thing we wanted mid-training for.

(3) is the honest read on why this line is hard: the missing ingredient is
Lean/Mathlib text shaped like our targets, and it does not appear to exist in
quantity.

### Corrected placebo launched

`rl3b_v2_placebo` (jobs 1558387/88, eval 1558389) at GROUP_P=0.49 /
ROLLOUT_P=0.74, confirmed in the job log. `rl3b_placebo` at 0.55/0.20 stays on
record unchanged. Decision rationale and the simulated advantage tables are in
`data_3b/placebo_constants_v2.sh`.

---

## 2026-08-23 (evening) — the headline result, and my correction rationale was backwards

### `gated` vs the corrected placebo: the project's central claim, measured

All rates on the SAME pinned first-400 validation slice. SFT baseline
`sft3b-step93` = 41.2% BEq+ / 79.0% type-check (165/400 correct).

| step | placebo OLD | placebo CORRECTED | **gated** |
|---|---|---|---|
| 10 | 35.0 | 38.2 | 38.2 |
| 20 | 31.2 | 27.8 | 35.8 |
| **30** | 20.8 | **22.2** | **40.5** |
| 40 | 27.0 | 21.8 | 39.5 |
| 50 | 22.5 | 23.8 | 36.2 |
| 60 | 21.5 | 24.8 | 35.0 |
| 70 | 28.0 | 18.8 | 34.0 |
| 150 | 20.2 | 13.0 | — |

Paired McNemar, gated against the corrected placebo at matched steps:

| step | 20 | 30 | 40 | 50 | 60 | 70 |
|---|---|---|---|---|---|---|
| diff | +8.0pp | **+18.2pp** | +17.8pp | +12.5pp | +10.2pp | +15.2pp |
| p | 7.3e-4 | **7.1e-16** | 5.4e-13 | 5.6e-7 | 2.0e-5 | 1.6e-10 |

Against the SFT baseline, `gated-step30` is **statistically indistinguishable**
(40.5% vs 41.2%, lost 27 / gained 24, retention 83.6%, **p=0.78**), while pure
noise at the same step falls to 22.2% (retention 51.5%, p=2.1e-19).

So the honest statement is: **BEq+ does not teach the policy much, but it very
nearly fully protects it from the damage GRPO does with no signal.** That has
always been the shape of the result; this is the first time it has been measured
against a properly-fitted control at n=400 with p < 1e-15.

### The gain channel finally moved

Converted, out of the 235 examples the baseline got wrong:

| arm | gained | rate |
|---|---|---|
| **gated-step30** | 24 | **10.2%** |
| gated-step70 | 21 | 8.9% |
| gated-step40 | 20 | 8.5% |
| selfprove-step20 | 18 | 7.7% |
| typecheck-step30 | 14 | 6.0% |
| placebo CORRECTED-step30 | 4 | 1.7% |

The gain channel was **flat at 4.9-7.3% across every 0.5B arm and every signal**,
including rejection sampling. `gated` at 3B reaches **10.2%**, six times the
control's 1.7%. That is the first time this number has cleared the ceiling this
project has been stuck behind, and it is the one measurement that distinguishes
"protects" from "teaches".

Caveat kept in front: `gated` still has not BEATEN the SFT baseline. Step 30 is a
tie, not a win. What changed is that it is now a tie instead of a loss, against a
control that loses half its correct answers.

### My reason for re-running the placebo was directionally wrong

I argued the old fit (0.55/0.20) was plausibly MORE destructive than the
corrected one because it hands a few random rollouts a large positive advantage,
and that this biased the comparison in `gated`'s favour. **The opposite is true.**
The corrected placebo is consistently more destructive from step 70 on and ends
at 13.0% against the old fit's 20.2%.

So the correction did not remove a bias favouring our own conclusion — it
removed one working AGAINST it, and `gated`'s margin is larger with the properly
fitted control than it was with the old one. The rerun was still right to do (an
unfitted control is not a control), but the stated rationale was wrong and the
simulation reasoning behind it should not be trusted as a predictor of training
dynamics. Advantage skew alone does not determine damage.

### Note on eval `n`

`hpc/grpo_eval.slurm` defaults to `N_EVAL=400`; the earlier 3B arms were run with
`N_EVAL=1000` explicitly and `rl3b_v2_placebo` was not. Harmless here because
everything above is computed on the first 400 rows of `data_3b/val.parquet`,
which are byte-identical across both files, but pass `N_EVAL=1000` for anything
that needs the extra power.

---

## 2026-08-23 (late) — the full arm table, and a config that has produced nothing

Everything on the pinned first-400 slice. Baseline `sft3b-step93` = 165/400 =
**41.2% BEq+ / 79.0% type-check**; gain-rate denominator is the 235 it got wrong.

### `gated` holds a gain channel above the ceiling at every step

| step | BEq+ | tc | retention | **gain-rate** |
|---|---|---|---|---|
| 10 | 38.2 | 78.8 | 78.8% | 9.8% |
| 20 | 35.8 | 80.0 | 73.3% | 9.4% |
| **30** | **40.5** | 81.8 | **83.6%** | **10.2%** |
| 40 | 39.5 | **83.0** | 83.6% | 8.5% |
| 50 | 36.2 | 79.8 | 72.7% | 10.6% |
| 60 | 35.0 | 77.0 | 69.7% | 10.6% |
| 70 | 34.0 | 76.8 | 69.7% | 8.9% |
| 80 | 35.0 | 79.8 | 72.1% | 8.9% |
| 90 | 33.8 | 79.2 | 70.9% | 7.7% |

The gain channel sits at **7.7-10.6% at every step from 10 to 90**. It was flat
at 4.9-7.3% across every 0.5B arm and every signal including rejection sampling,
so this is a sustained break from that ceiling rather than one lucky checkpoint.
Type-check also stays at or above the 79.0% baseline through step 90 — no other
arm preserves both channels.

### The four controls, at their most comparable

| arm | best BEq+ | BEq+ @150 | gain-rate range | tc @ end |
|---|---|---|---|---|
| **gated** | 40.5% (s30) | — (s90: 33.8%) | **7.7-10.6%** | 79.2% |
| selfprove | 38.2% (s10) | — (s20: 36.8%) | 7.2-7.7% | 89.8% |
| typecheck | 40.0% (s10) | 8.0% (s110) | 2.1-7.7% | **98.0%** |
| placebo OLD | 35.0% (s10) | 20.2% | 2.1-5.5% | 57.2% |
| placebo CORRECTED | 38.2% (s10) | **13.0%** | **1.7-4.3%** | 39.8% |

Read down the gain-rate column: it separates the arms more cleanly than BEq+
does. Gold-referenced semantics 7.7-10.6%, gold-free compiler signal 7.2-7.7%,
exploitable proxy decaying 7.7 -> 2.1%, pure noise 1.7-4.3%.

`selfprove` only has two checkpoints so far and is holding well (retention 82.4%
/ 78.2%, type-check 89.8%). It is the arm that matters most for the project's
thesis — if a gold-free compiler reward keeps pace with BEq+, the case for a
gold-referenced semantic reward weakens considerably. Two points is not enough
to say. Wait for the rest.

### `rl3b_bb_*` has burned ~28h and produced ZERO checkpoints

`hpc/grpo_3b_bigbatch.slurm` (batch 256) has run three 11h chunks
(1515248/49/50) and there is no `checkpoints/beqplus_rl_poc/rl3b_bb_*` directory
at all. Chunk 3 is 6.5h in and the log shows it still inside a single step
(`pending: 0, running: 26, finished: 230` of 256 prompts).

Cause: batch 256 x 8 rollouts = **2,048 Lean scorings per step**, 16x the batch-16
arm, and `SAVE_FREQ=10` was carried over unchanged. Ten steps at that size is
20-30h, so the first checkpoint has never been reached. **The bug is SAVE_FREQ,
not the batch size**: one step at batch 256 sees the same number of prompts as 16
steps at batch 16, so it must save every step. As configured this arm cannot
produce a comparable curve no matter how long it runs.

Not cancelled unilaterally — flagged for a decision. If it is restarted,
`SAVE_FREQ=1` and `TOTAL_STEPS` around 10 would make it comparable to batch-16
step 160 for roughly the same Lean spend.

### `compare_arms.py` could not compare the two arms that matter

Writing the docs surfaced it: the README's own example command failed.
`load_arm()` globbed `eval_<arm>-step*_n{n_eval}.json` with `--n-eval` defaulting
to 400, so `rl3b_gated` (written at `_n1000`) and `rl3b_v2_placebo` (at `_n400`)
reported **"No common steps"** despite being evaluated on the identical val
prefix. Every cross-arm comparison so far had to be recomputed by hand.

Fixed: `--n-eval 0` (now the default) matches any `n`, records are truncated to
the shortest, and rates are **recomputed over that prefix** instead of read from
`*_rate`, which describes whatever `n` the file happened to be written at. Valid
because `evaluate_checkpoints.py` always scores the first `n` rows of the same
parquet in order. It reproduces the hand-computed table exactly and extends it:

```
metric=beq_plus  A=rl3b_v2_placebo  B=rl3b_gated  (paired on the first n=400)
 step     A%     B%  Δ(B-A) pp   A>B   B>A        p
   10   38.2   38.2       +0.0    25    25   1.0000
   20   27.8   35.8       +8.0    27    59   0.0007*
   30   22.2   40.5      +18.2     9    82   0.0000*
   40   21.8   39.5      +17.8    16    87   0.0000*
   50   23.8   36.2      +12.5    25    75   0.0000*
   60   24.8   35.0      +10.2    25    66   0.0000*
   70   18.8   34.0      +15.2    17    78   0.0000*
   80   18.8   35.0      +16.2    16    81   0.0000*
   90   20.8   33.8      +13.0    18    70   0.0000*
```

Significant at every step from 20 to 90. The discordant counts are the real
story: at step 30, 82 examples that noise destroyed BEq+ kept, against 9 the
other way.

### Docs updated

`README.md` reframed from "a BEq+ proof-of-concept" to **a gym for RL on Lean 4
autoformalization** — reward zoo, calibrated control, paired-evaluation harness —
with BEq+ as experiment 1 of 6. `CLAUDE.md` rewritten around the reward zoo, the
series-scoping conventions (`TAG` / `BEST_SFT` / `SERIES_TAG`), and a traps
section listing the failures that have actually cost time. Every number quoted in
the README was recomputed from `results/` before publishing; all 7 checks matched.

---

## 2026-08-23 (audit) — six bugs found, all fixed; the headline result survives

A deliberate bug hunt over everything written in the last few days.

### The check that mattered most, first

`hpc/grpo_eval.slurm` defaults `VAL_PARQUET=data/val.parquet` — the **0.5B**
400-row set — and I submitted `rl3b_guided`, `rl3b_gated_edge` and
`rl3b_v2_placebo` without overriding it. So the corrected placebo, which the
+18.2pp headline is measured against, was evaluated on a different file from
`gated`.

Verified directly rather than trusting the note in CLAUDE.md: the first 400
prompts **and** golds of `data/val.parquet` and `data_3b/val.parquet` are
identical and in the same order. **The headline result stands.** But it was safe
by luck, so the default is now series-aware (`rl3b*` → `data_3b/val.parquet`).

Also verified while there: every arm trained on the file it was supposed to
(`gated_edge` on `train_edge.parquet`, the rest on `data_3b/train.parquet`) — the
`set -x` trace shows both the run_grpo.sh default and the later override, and
`"$@"` comes last so the override wins.

### The findings

| # | bug | severity |
|---|---|---|
| 1 | `--result-timeout 0` documented as "disables", but `it.next(timeout=0)` **aborts on the first result** | high — my own fix from yesterday, inverted |
| 2 | `evaluate_checkpoints.py` does `.head(n_eval)`, so a 400-row parquet with `--n-eval 1000` writes an `_n1000` file holding 400 records | high — silent mislabelling |
| 3 | `grpo_eval.slurm` defaulted to the 0.5B val set for 3B runs | medium — harmless here, verified |
| 4 | `passk_3b.slurm` defaulted `LABEL=$(basename CKPT_DIR)` = **"huggingface"** for every model | high — latent |
| 5 | `make_difficulty_subset.py` accepted ragged groups, so a **7/7 saturated** prompt passes a `1..7` test | low |
| 6 | my `guided` comment implied `norm_adv_by_std_in_grpo=False` was arm-specific | doc error |

**(4) is the nastiest.** Every checkpoint path ends in `/huggingface`, so the
default label was that string for every model. Two runs would collide on
`data_3b/rollouts/passk_huggingface_k32.jsonl`, and since generation is skipped
when that file exists, **the second model would be scored on the first model's
rollouts and reported as its own pass@k** — no error, wrong number. Same class as
the eval-label collapse already documented in CLAUDE.md. It never fired because I
passed `LABEL` explicitly both times. Now derives `<run>-step<N>` from the path
and hard-refuses `huggingface`, empty, or a bare `global_step_*`.

**(6) is worth stating plainly**: `configs/run_grpo.sh:189` already defaults
`norm_adv_by_std=${NORM_ADV_BY_STD:-False}`, so **every arm has always run with
it off**. My `EXTRA` addition for `guided` is redundant — a guard against that
default changing, not something that distinguishes the arm. No gated-vs-guided
difference may be attributed to it.

**(5) touched the running arm.** `gated_edge` is training on the 1,123-prompt
pool built before the fix; the corrected build gives 1,122. One saturated prompt
in 1,123 is 0.09% of the pool and cannot move the result — **not restarting it**.

### Checks that came back clean

- Placebo constants reach the Ray workers: v2's observed mean training reward is
  **0.379** against the predicted 0.49x0.74 = 0.363. (This mattered — verl passes
  an explicit `runtime_env.env_vars` list that does not include `BEQ_PLACEBO_*`.)
- `_placebo_hash` is md5, not Python's per-process-randomised `hash()`.
  Byte-identical across three fresh interpreters.
- All rewards return the same diagnostic schema, so verl's group filtering and
  metric aggregation see a consistent shape across arms.
- No eval JSON is labelled `huggingface`.
- Every `.py` parses; every `.slurm` passes `bash -n`; no other quoted heredoc
  references a shell variable.
- `guided` is confirmed running `compute_score_guided`.

### Not fixed, deliberately

`is_concrete()` rejects any statement containing the substring `Type`, which also
drops legitimate names like `Subtype.foo`. It is over-broad but only ever removes
data from an already-abandoned corpus, so it is not worth the churn.

---

## 2026-08-23 (consolidation) — one checkout, one prelude, 34% less SLURM

A simplification pass. Nothing running was disturbed (`sbatch` snapshots at
submit time, so queued jobs keep their own copies), and every runtime dependency
was checked present afterwards.

### The finding that justified the pass

`hpc/grpo_eval.slurm` set `PROJECT_ROOT=/scratch/logan03/lean-gym-rl-rl_reward`
and put it on `PYTHONPATH`. **Every eval job has been importing code from a
second tree last touched 2026-08-20**, while training ran from `/home`. That is
why yesterday's `n_eval` guard in `evaluate_checkpoints.py` would never have
protected an eval job.

Audited before changing anything:

- `reward/beq_plus.py`: **85 substantive lines only in home, 0 only in scratch.**
  `score`, `typecheck_ex` and `check_theorem_equivalence` identical. The extras
  are `self_prove`, `_prove_with`, `typecheck_message` — all added later and all
  unused by eval.
- Home is a strict file superset. Of 14 files with a newer mtime in scratch,
  **all 14 are byte-identical** — the mtime is just the copy.

**So no cached eval number is affected.** Consolidated to one tree regardless.

### What changed

| | before | after |
|---|---|---|
| checkouts | 2 | **1** |
| slurm jobs | 17 | 12 (+5 archived) |
| slurm LOC | 1,439 | **952** + a 48-line shared prelude |
| python scripts | 18 | 17 (+1 archived) |
| midtrain corpora | 3 | 2 |

- **`hpc/job_prelude.sh`** replaces a ~20-line preamble that had been copy-pasted
  into 12 files. It also *improves* on the original: the old inline staging block
  silently did nothing when `SLURM_TMPDIR` or the tar was missing, which is the
  setup where Lean then blows `BEQ_ENV_TIMEOUT`. `stage_mathlib()` warns loudly.
- **Archived** `grpo_bs16`, `rft_arms`, `rft_eval`, `rft_tcnb`, `starved_k32` to
  `hpc/archive/` with a table naming each one's replacement. All were
  unreferenced by the Makefile and docs.
- **`make_starved_subset.py` is exactly `make_difficulty_subset.py --min-wins 0
  --max-wins 0`** — verified identical (both 559 prompts, symmetric difference 0)
  before archiving. That exposed a message bug in the generalisation: it printed
  "informative fraction ~100% by construction" for a selection that produces
  *zero* advantage. Now conditional.
- **`data_3b/README.md`** documents every file in the split, which two are pinned
  (`passk_subset.parquet`, `placebo_constants.sh`) and why. `midtrain_v2` was
  deleted — regenerable in ~5 min at a fixed seed, and its only result is
  preserved in `results/midtrain_v2_validation.json`.

### Left alone deliberately

The Makefile's 46 targets all resolve (checked: the four "missing" hits are
trailing periods in prose, not real references). Its `train-*` targets are 0.5B
single-GPU workflows superseded by `hpc/grpo_3b.slurm`, but they work and cost
nothing to keep, and deleting targets someone may be using is a worse trade than
the clutter.

---

## 2026-08-23 (figures + Makefile) — plots in the Interplay style, and one honesty correction

### Where the style came from

The Interplay repo does **not** ship its plotting source — `scripts/composition/`
contains only `__pycache__/*.pyc` (`plot_composition_one_figure`,
`plot_graph_pass_bar_line`) and a data dir. So the style was recovered by
sampling `assets/findings.png` directly. The dominant saturated colours came back
as `#009E73`, `#0072B2`, `#E69F00` — **Okabe-Ito**, the standard colourblind-safe
qualitative palette. The rest (bold axis labels, dotted grid, markers on every
point, in-axes legend, value labels over bars, red dashed callouts, pastel panel
tints) is read off that figure.

### The palette order is validated, not chosen by eye

Assigned in fixed order and checked: **blue → green → orange → pink → sky**
passes lightness band, chroma floor, CVD separation and normal-vision floor. Two
orderings that look equally sensible put `#CC79A7` adjacent to `#009E73`, which
collapses to **deutan ΔE 7.6 — below the 8.0 floor**. Three hues sit under 3:1
contrast against white, which obliges "relief", so every series is
**direct-labelled at its line end** rather than identified by colour alone.

The control is deliberately *not* categorical: grey, dashed, recessive. It fails
the chroma floor on purpose, and the dash pattern is its secondary encoding.

### Four figures, `make figures`

`scripts/figstyle.py` + `scripts/make_figures.py`, driven only by cached
`results/eval_*.json` and `results/passk_*.json` — no Lean, no GPU, deterministic,
safe to re-run after any eval lands.

| figure | mirrors | shows |
|---|---|---|
| `arm_trajectories` | their panel 1 | BEq+ vs step, all arms, against the SFT line |
| `passk` | their panel 2 | pass@k baseline vs mid-trained — curves on top of each other |
| `retention_gain` | their panel 4 | kept vs converted, with the 4.9–7.3% ceiling band |
| `proxy_vs_semantic` | — | type-check on x, BEq+ on y: the exploit as a trajectory |

**Every rate is recomputed on the common prefix of `per_example`**, never read
from `*_rate`, because arms were evaluated at different `--n-eval`.

### The matched-step fix changed the story, and the earlier version flattered us

The first `retention_gain` used each arm's *best available* step, which put
`gated` at step 30 against the placebo at step 150 — one arm near its peak
against the control at its worst. Fixed to the latest step **every** arm shares.

That step is **20**, capped by `selfprove` having only two checkpoints, and at
step 20 the picture is much less flattering: retention `gated` **73.3%** vs
`selfprove` **78.2%** — selfprove *retains more* — and gain 9.4% vs 7.7%. The
arms do not separate until later; `arm_trajectories` is where the separation is
visible. The figure now says so in its subtitle. The step-30 numbers quoted
earlier are real, but they are not a matched comparison against the control and
should not be presented as one.

### Makefile: 46 targets → 38, and a real interface

`train-from-sft`, `train-guided`, `train-shaped`, `train-gated`,
`train-gated-filtered` were each a **one-line alias** for `train-rl` with a
different `REWARD_FN_NAME`; `train-composite`/`train-typecheck` were near-identical
five-line blocks; `submit-composite`/`submit-typecheck` differed by one word. All
collapsed into flags:

```
make train-rl REWARD=gated|guided|selfprove|typecheck_only|composite \
              STEPS=100 INIT=<merged dir> FILTER_GROUPS=1
make submit   REWARD=gated STEPS=200
```

`REWARD=x` expands to `compute_score_x`; an explicit `REWARD_FN_NAME` still wins
(verified). `help` is regrouped into SETUP / TRAIN / DATA / EVALUATE / SLURM /
HOUSEKEEPING and now points at the SLURM jobs, which is where the real runs are.

---

## 2026-08-23 (late) — selfprove had silently stalled; pass@k trajectories queued

### `selfprove` was not running, and could never have finished

Asked whether we still use it, and the honest answer was **no, by accident**: it
sat at step 20 with nothing queued. Its two 11h chunks (1518169/70) both hit
walltime — **22h for 20 steps**, against `gated`'s 90 steps in the same budget.

Cause found in `_prove_with`: the ladder short-circuits on success, so the full
cost falls on statements nothing proves — which is most of them. `tauto`,
`simp_all_arith!`, `noncomm_ring`, then **`exact?`, a search over all of Mathlib
that reliably burns its whole budget**, all at the BEq+ `timeout_per_proof` of
30s. That is ~120s per unprovable rollout, i.e. ~4.5x gated per step, i.e. 150
steps would take ~165h.

Fixed with a **separate `BEQ_PROBE_TIMEOUT` (default 10s)** for the probe ladder
only; BEq+ equivalence keeps its 30s. Worst case falls ~120s -> ~40s. This does
change the metric — a proof needing >10s of `exact?` no longer counts — but the
metric was always "provable within a budget", and an unfinishable arm measures
nothing. Runs with different values are not comparable, so:

- the old run is preserved as **`rl3b_selfprove_t30`** (checkpoints, eval and gen
  files renamed, and the label *inside* each JSON rewritten too — `select_checkpoint`
  merges on that, not on the filename);
- **`rl3b_selfprove` restarted from scratch** (1585990/91, eval 1585992).

This arm matters more than any other: if a gold-free compiler reward keeps pace
with BEq+, the case for a gold-referenced semantic reward largely collapses. At
step 20 it retained MORE than gated (78.2% vs 73.3%). Two points settle nothing.

### pass@k per checkpoint did not exist — now queued

Per-checkpoint evals are **k=1**, so the requested pass@k-vs-step figure could not
be built from cache. Queued `hpc/passk_3b.slurm` on seven checkpoints
(1585737–43): `gated` {10,30,50,90}, `v2_placebo` {30,90}, `typecheck` {110}.
~3.5h each, 1 GPU, run in parallel.

That required a fix to the LABEL derivation: **RL checkpoints live at
`global_step_N/actor/huggingface`**, not `global_step_N/huggingface`, so the
derivation returned `global_step_N` and hit the guard added in the audit. It now
strips a trailing `actor` and handles both shapes. (The guard did its job —
loud failure rather than a silent collision.)

### New figure: `passk_trajectories`

Three panels — pass@1, pass@8, pass@32 — against GRPO step, same arms and colours
as `arm_trajectories`, so BEq+ and pass@k read side by side. **CORRECTION (2026-08-23, later): the claim below that
`arm_trajectories` "IS pass@1" is WRONG.** `evaluate_checkpoints.py` decodes
GREEDILY at temperature 0.0; the pass@k jobs sample at 1.15. On sft3b-step93 that
is 41.2% vs 32.4%, 8.8pp apart. The sharpening-vs-capability comparison therefore
has to be made INSIDE the pass@k figure, where pass@1 and pass@32 come from the
same rollouts, and a step-for-step difference BETWEEN the two figures means
nothing. Original text:
The pairing is the
point: BEq+ in `arm_trajectories` is pass@1, which sharpening alone can move;
pass@32 moves only if the policy reaches answers it previously could not at any
temperature. The Interplay paper treats only the second as a capability gain, and
our arms differ almost entirely in retention — the signature of sharpening.

The plotting code was verified against synthesised inputs and the synthetic files
deleted immediately, rather than left to fail three hours from now when the real
jobs land. Its title is deliberately neutral: an earlier draft asserted
"pass@1 moves while pass@32 does not", which is the expected result, not a
measured one, and the figure exists to test it.

---

## 2026-08-23 (figures II) — runtime, pass@k trajectories, and a 90-step cap

### Runtime: `runtime.png`

Per-step wall-clock for every arm, from **checkpoint mtimes** rather than the
logs. verl writes `timing_s/step` to stdout, but only three arms have it — the
rest logged to stderr in a form carrying no metrics, and those are precisely the
expensive arms worth timing. Consecutive checkpoint mtimes / SAVE_FREQ covers all
of them.

**Cross-validated before use**: for `rl3b_typecheck` the mtime method gives 31s
against a logged median of 26s, and the 5s gap is the ~50s checkpoint save
amortised over 10 steps. So it measures step time *plus* save overhead —
consistent, and the right number for "what does a step cost". Intervals spanning
a chunk boundary carry SLURM queue time (one `gated` interval is 5,223s against a
962s median) and are dropped at 3x the median.

| arm | s/step | |
|---|---|---|
| gated (BEq+) | 972 | **16.2 min** |
| guided | 960 | 16.0 min |
| gated, edge pool | 700 | 11.7 min |
| type-check only | 32 | |
| placebo | 31 | |

**31x between the cheapest and dearest reward, same batch and model — Lean
scoring is the entire difference.** Reaching step 90 costs ~25.6 GPU-hours on
`gated` against ~0.8 on the placebo.

Worth noting: **`gated_edge` is 28% cheaper per step than `gated`** on the same
reward and batch. The edge pool is all-informative by construction, so it holds
fewer of the pathological rollouts that drive BEq+ into its 30s timeouts.

The left panel is a **lollipop, not bars**: the range is ~30x, so linear bars make
the cheap arms invisible and log bars are worse — bar *length* stops encoding
ratio once the baseline is not zero. A dot encodes by position, which is
legitimate on a log axis.

### `arm_trajectories_passk.png`

Replaces the three-panel `passk_trajectories` with a single panel laid out
**identically to `arm_trajectories`** — same palette, markers, x-axis, baseline
line, end labels — with pass@32 on y instead of pass@1. (Superseded: see the correction above --
`arm_trajectories` is GREEDY, not pass@1, so divergence between the two figures
is not interpretable. pass@1 is now drawn inside the pass@k figure itself.) Renders as soon as
jobs 1585737-43 land.

### Steps capped at 90, in figures AND experiments

`scripts/make_figures.py --max-step 90` (default) and
`hpc/grpo_3b.slurm TOTAL_STEPS` 150 -> 90.

Justified by the data already in hand: `gated` sits at 34-40% from step 30 to 90,
the placebo is on its floor by 30, and typecheck's collapse is unambiguous by 50.
The extra 60 steps cost ~16 GPU-hours per BEq+ arm and changed no conclusion.
Arms are compared at matched steps anyway, so the usable range is set by the
shortest arm regardless.

Applied to the queue while everything was still PENDING, so no compute was
wasted: `selfprove` (1586327/28, eval 1586329), `gated_edge` (1586330/31) and
`guided` (1586332/33) resubmitted at 90; their 150-step tails cancelled.

### Placebo presentation

`rl3b_placebo` (the 0.55/0.20 degenerate fit) is **dropped from every figure** and
`rl3b_v2_placebo` is now labelled simply **"placebo"**. The old arm stays on disk
for provenance. "placebo" in any figure means the corrected 0.49/0.74 fit.

---

## 2026-08-23 (arms) — bigbatch killed, eval n standardised, and a gap named

### `bigbatch` killed

Three chunks, **30h 6m of compute, zero checkpoints.** Batch 256 with
`SAVE_FREQ=10` carried over from batch 16 puts the first save 20-30h away.
Cancelled 1515250. If ever retried: `SAVE_FREQ=1`, `TOTAL_STEPS` ~10.

### Eval n was inconsistent, and it hit the control

| arm | n used |
|---|---|
| `rl3b_gated`, `rl3b_typecheck`, `rl3b_placebo` | 1000 |
| **`rl3b_v2_placebo`** | **400** |

`hpc/grpo_eval.slurm` defaulted to 400 and I did not override it for the
corrected placebo — **the control the +18.2pp headline is measured against.**
Harmless so far because every cross-arm number is computed on the shared 400-row
prefix, but the control deserves the same power as the arms.

Default is now **n=1000** (MDE ~3.5pp vs ~5.6pp; most real effects here are
1-3pp). `rl3b_v2_placebo` re-running at 1000 (1586742). n=400 stays available for
cross-scale paired tests, since the first 400 rows are byte-identical to
`data/val.parquet`.

### Missing evals queued

`guided` (7 checkpoints to step 70) and `gated_edge` (5 to step 50) both had
**zero evals**. Queued at n=1000: 1586740, 1586741.

### `arms.md`

Per-arm breakdown organised around the observation that makes the arms
comparable: they all share the same `elaborates` gate and differ almost entirely
in **what they pay for a well-formed statement that is not the gold** — the
32.3% dead band.

| arm | reward on that band |
|---|---|
| `typecheck` | 1.0 (the exploit) |
| `selfprove` | 0.2, or 1.0 if true and non-trivial |
| `guided` | 0.10-0.20, graded by resemblance |
| `gated`, `gated_edge` | 0.0 — same as garbage |

### The gap: BEq+ **and** self-prove has never been run

Checked rather than recalled: no reward combines them. `gated` uses BEq+ and never
asks whether a non-equivalent statement is *true*; `selfprove` asks exactly that
and never consults the gold. They are orthogonal and would compose on the same
32.3% band:

```
both directions        1.0    (BEq+)
one direction          0.25   (BEq+)
true and non-trivial   0.15   (self-prove -- currently 0.0)
elaborates only        0.0
```

Same band `guided` attacks, but with a **verified** signal rather than a
*resemblance* one — similarity cannot see an inverted inequality, self-prove can,
because a false statement is not provable. More principled, and the most
expensive arm in the set (~20+ min/step: BEq+ cascade plus the prove ladder).

**Decision deferred to `guided`'s eval** (1586740): if grading that band helps at
all, this is the better way to grade it; if it does not, neither is worth the
compute.

---

## 2026-08-23 (contamination + cache) — val split verified clean, placebo matched

### Contamination check on the val split — CLEAN

Asked whether the extra rows the n=1000 eval uses were ever trained on. Checked
rather than assumed, both halves separately, against every training set:

| training set | vs `val[0:400]` | vs `val[400:1000]` |
|---|---|---|
| SFT train (8,000) | 0 prompt / 0 gold | 0 prompt / 0 gold |
| SFT val (200) | 0 / 0 | 0 / 0 |
| RL pool (4,300) | 0 / 0 | 0 / 0 |
| RL edge pool (1,122) | 0 / 0 | 0 / 0 |

The SFT parquets carry a `messages` schema, not `prompt`/`reward_model`, so a
naive comparison silently SKIPs them — the first pass did exactly that and had to
be redone against the real schema. Golds were normalised (`:= by sorry` stripped,
whitespace collapsed) so a formatting-only difference could not hide a real
overlap. **n=1000 is uncontaminated, including the 600 rows beyond the shared
prefix.**

### Evals now reuse a cached result at any n

`hpc/grpo_eval.slurm` skipped only on an exact `_n<N>` filename match, so raising
`N_EVAL` re-scored an arm from scratch. `REUSE_CACHED=1` (default) now keeps an
existing eval at any n; `REUSE_CACHED=0` forces the requested n.

Applied immediately: the `rl3b_v2_placebo` re-run at n=1000 was **cancelled** —
all 15 steps are already cached at n=400, that is ~6h of Lean for power only used
on rows 400-1000, and every cross-arm number is computed on the shared 400-row
prefix regardless.

### Placebo pass@k coverage matched to gated

The control had {30, 90} against gated's {10, 30, 50, 90}. Added steps 10 and 50
(1586946, 1586947) so `arm_trajectories_passk` compares equal step coverage
rather than interpolating the control across a wider gap. Nine pass@k jobs queued.

---

## 2026-08-23 — arms.md checkpoint table, and the SFT n=1000 question

### The SFT baseline was already evaluated at n=1000

Asked whether to run it; it was done. All three epochs carry 1,000 `per_example`
records:

| checkpoint | BEq+ | type-check |
|---|---|---|
| `sft3b-step31` | 37.8% | 77.4% |
| `sft3b-step62` | 37.9% | 75.9% |
| **`sft3b-step93`** | **39.4%** | **76.7%** |

Nothing to run. Worth keeping straight: **39.4% is the n=1000 number and 41.2% is
the same checkpoint on the shared 400-row prefix.** The prefix is a slightly
easier slice, which is exactly why headline rates are never differenced across
different n — and why the table below fixes n rather than using each arm's own.

### `make arms-table`

`scripts/make_arms_table.py` regenerates the table between markers at the top of
`arms.md`, so it refreshes as evals land instead of going stale. Cells are
**BEq+ / pass@32**, blank where nothing has run.

The header states plainly that the two halves are **different measurements** —
BEq+ is greedy (T=0) on 400 shared rows, pass@32 is sampled (T=1.15) on the 250
pinned prompts — so the gap between them within a cell is not a sharpening
measurement. That is the same conflation corrected earlier today; putting the two
numbers side by side in one cell makes it easy to make again, so the caveat sits
directly above the table.

BEq+ is computed on the shared prefix rather than read from `*_rate`, because
arms were evaluated at n=400 and n=1000 and `*_rate` describes whatever n its own
file used.

Current state: `gated`, `typecheck` and `placebo` populated across steps 10-90;
`guided`, `gated_edge` and `selfprove` blank pending 1586740/41 and the restarted
run; every pass@32 cell blank pending the nine queued jobs.

---

## 2026-08-23 — `passk.png` now carries the arms, and a colour-identity bug

### `passk.png` rewritten

Was: pass@k vs k for the two SFT baselines only, with colours assigned by
`enumerate(files)`.

**That indexing was a real bug, not just a limitation.** Colour was following
file order, so adding any arm would have repainted every existing series — and
the two baselines were already holding `CATEGORICAL[0]` and `[1]`, i.e. gated's
blue and selfprove's green. The same model would have been a different colour in
`passk.png` than in `arm_trajectories.png`.

Now: arms use their `fs.ARM_STYLE` colour — the one they carry in every other
figure — and the baselines are neutral greys with star markers, which also keeps
them visually subordinate to the thing under test. One checkpoint per arm (the
latest at or below the 90 cap), because a line per checkpoint is ~30 lines and
adjacent steps overlap.

Verified against synthesised inputs in both states (baselines only, and with four
arms) before the real jobs land; synthetic files deleted immediately.

### Two things that fell out of the render

- **`typecheck`'s pass@k was queued at step 110**, above the 90-step cap, so the
  figure filtered it straight back out. Cancelled 1585743, requeued at step 90
  (1587253).
- **The placebo's `x` marker was invisible.** `x` has no face, so the
  `markeredgecolor="white"` every series carries erased it. Changed to `s`.

### pass@k coverage now queued (11 jobs)

| arm | steps |
|---|---|
| `gated` | 10, 30, 50, 90 |
| `placebo` | 10, 30, 50, 90 |
| `typecheck` | 90 |
| `guided` | 70 |
| `gated_edge` | 50 |
| `selfprove` | — (restarted run has no checkpoints yet) |

`gated` and the control have matched step coverage so the trajectory comparison
does not interpolate one across a wider gap than the other.

---

## 2026-08-23 (audit II) — a misleading noise figure, a classification bug, and a shared reader

### The finding that mattered: `compare_results.py` was telling us to ignore real effects

It printed

```
re-measurement spread (same checkpoint, different eval runs):
  _meta: 38.9% (rollout_stats.json), 35.6% (rollout_stats_3b.json),
         37.3% (rollout_stats_3b_mt.json)   -> spread 3.3 pp
  Treat differences smaller than this spread as noise.
```

**Those are three different policies** — 0.5B, 3B, and mid-trained 3B — not one
checkpoint re-scored. `_meta` is a metadata block that every `rollout_stats*.json`
carries, so all three collided under one label. It was also being rendered as a
row in the main checkpoint table.

**3.3pp is the size of the real effects in this project.** Anyone following that
instruction would have dismissed the mid-training result, the gain-channel break,
and most of the arm differences as noise. Labels starting with `_` are now
filtered at load, and the caveat above the spread block says what invalidates it.

### `evalio.py` — one reader instead of four

`make_figures`, `make_arms_table`, `compare_arms` and `select_checkpoint` had each
grown a copy of "glob the eval JSONs, pull `per_example`, compute a rate", and
they had already drifted on two axes: whether a step evaluated at several n keeps
the largest or the first found, and whether the rate is recomputed over a fixed
prefix or read from `*_rate`. The second is the dangerous one — `*_rate` describes
whatever n its own file used, and arms here were evaluated at both 400 and 1000.

`scripts/evalio.py` is now the single implementation; it never reads `*_rate`.
Verified behaviour-preserving: `compare_arms` output, the `arms.md` table, and all
five figure checksums are **byte-identical** before and after.

Writing it immediately exposed a bug the old copies had:
`sft3b-step93` matches `"<name>-step<N>"`, so pass@k results for the BASELINE were
being filed as an arm called `sft3b`. `make_figures` happened to be safe (it
guarded on `in fs.ARM_STYLE`) but silently **dropped `sft3bmt` entirely**, since
that label matched neither branch. Baselines are now keyed by label so several
coexist.

### Removed / archived

- **`fig_passk_trajectories`** — dead since `fig_arm_trajectories_passk` replaced
  it; nothing referenced it.
- **`scripts/plot_results.py` -> `scripts/archive/`.** It parsed per-step metrics
  out of verl's stdout, which only three arms ever wrote there; on the 3B series
  it exits with "No parseable step metrics in those logs". `runtime.png` gets the
  same information from checkpoint mtimes, which works for every arm.
  `make plots` is now an alias for `make figures`.
- Unused `import sys` in `compare_arms.py` (added by me two passes ago).
- A comment in `figstyle.py` that two successive string replaces had run together
  into one unreadable line.

### Checked, nothing wrong

- `check_lean_statement` in `reward/lean_tool.py` reads as dead to a call-graph
  scan but is invoked by verl through the `function_tool` registry. Kept.
- Every other "unused import" hit was `from __future__ import annotations`.
- `probe_gradient_signal.py`, `run_curriculum.sh`, `test_lean_interact.py` and
  `compare_results.py` are all still referenced and still work.

---

## 2026-08-24 — pass@32 lands: no arm raises the ceiling

14 of 19 pass@k jobs done, plus the first-ever evals of `guided` and `gated_edge`.

### The capability question, answered — and the answer is negative

pass@32 on the 250-prompt pinned subset. **SFT baseline = 55.2%.**

| arm | s10 | s30 | s50 | s70 | s90 |
|---|---|---|---|---|---|
| `gated` | 51.1 | 51.7 | 46.2 | — | **48.1** |
| `gated_edge` | 52.5 | 48.3 | 46.6 | — | — |
| `guided` | — | — | — | 44.3 | — |
| `typecheck` | 55.3 | — | — | — | 38.7 |
| `placebo` | 52.5 | 35.1 | 38.4 | — | 36.4 |

**Not one arm exceeds the baseline at any step.** Every arm's ceiling falls; they
differ only in how fast. Under the Interplay paper's criterion — a capability gain
is a pass@128 (here pass@32) improvement — **BEq+ RL produces no capability gain.
It slows the loss.** `gated` ends 7.1pp below baseline where noise ends 18.8pp
below, so the +18.2pp headline against the placebo is real and remains real, but
it is a statement about **damage avoided, not ability added.**

### The sharpening signature is now directly visible

Same figure, pass@1 dotted against pass@32 solid. For `gated`, pass@1 runs
33.7 -> 30.4 -> 34.8 across steps 10-90 — roughly flat, ending slightly up — while
pass@32 falls 51.1 -> 48.1. **The distribution is concentrating without
extending**, which is exactly what sharpening looks like and exactly what the
paper says does not count. This is the cleanest evidence yet for the reading that
has been implicit since the 0.5B series: BEq+ protects and sharpens; it does not
teach.

### `gated_edge` is the strongest arm on every retention measure

Paired against the SFT baseline, greedy, n=400 prefix:

| arm-step | BEq+ | lost | gain | retention | gain-rate | p |
|---|---|---|---|---|---|---|
| **`gated_edge`-20** | **43.2%** | 19 | 27 | **88.5%** | **11.5%** | 0.30 |
| `gated_edge`-30 | 41.8% | 24 | 26 | 85.5% | 11.1% | 0.89 |
| `gated_edge`-50 | 39.0% | 37 | 28 | 77.6% | **11.9%** | 0.32 |
| `gated`-30 | 40.5% | 27 | 24 | 83.6% | 10.2% | 0.78 |
| `guided`-70 | 36.2% | 36 | 16 | 78.2% | 6.8% | **0.008** |

`gated_edge`-20 is **the first arm to score above the SFT baseline** (43.2% vs
41.2%). Stated honestly: **+2.0pp at p=0.30 is not significant** — the MDE at
n=400 is ~5.6pp. What is notable is that it holds the best retention (88.5%) and
the best gain-rate (11.5-11.9%) of any arm, at **28% lower cost per step**. Data
curation is doing more per GPU-hour than any reward change tried so far.

### `guided` does not work

Never exceeds the baseline at any step, and the degradation is significant by
step 70 (p=0.008). Grading the 32.3% dead band by *resemblance* to the gold did
not help — which raises the value of the untried BEq+ **and** self-prove ladder,
since that grades the same band by *verified truth* instead.

### Note on this session's `/simplify`

All four cleanup agents terminated immediately on an account spend limit, so the
simplification pass did **not** run. Nothing was reviewed and nothing was changed
by it. It needs re-running once the limit resets.
