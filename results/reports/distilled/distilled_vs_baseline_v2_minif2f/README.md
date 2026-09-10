# Every trained SFT variant on miniF2F (out-of-domain)

All 7 trained models plus the untrained base, on the same pinned
244-row `data_minif2f/val_proof.parquet` slice -- competition problems the
models never saw. Distilled/CoT models are scored with `score_standalone`
(they emit self-contained snippets), gold-target models with the plain path;
all use a 1536-token budget where a `<think>` block is involved. The untrained
model is the dash-dot floor. Note 32 of the 244 miniF2F golds do not elaborate under Mathlib v4.23, so BEq+ is effectively capped near 87%.

## Headline

Each row is that run's **final** checkpoint, not a checkpoint selected on this
table's own BEq+ column. Both metrics score the generated *statement* only --
see Caveats.

| series | step | elaborates | BEq+ | Δ BEq+ vs untrained | McNemar p |
|---|---|---|---|---|---|
| untrained base | 0 | 45.5% | 5.3% | — | — |
| Baseline (capped, gold) | 114 | 46.7% | 10.7% | +5.3 pp | 0.015 |
| Baseline (full data, gold) | 225 | 52.0% | 13.9% | +8.6 pp | 0.00019 |
| Matched-gold | 96 | 52.5% | 14.8% | +9.4 pp | 5.6e-06 |
| CoT + gold target | 192 | 46.3% | 10.7% | +5.3 pp | 0.0044 |
| Distilled (cosine LR) | 160 | 67.2% | 15.2% | +9.8 pp | 1.9e-05 |
| Distilled (constant LR) | 175 | 69.7% | 16.4% | +11.1 pp | 1.1e-07 |
| Distilled, BEq+-clean | 175 | 58.2% | 18.0% | +12.7 pp | 7.8e-07 |

## Trajectories (BEq+ %)

- **Baseline (capped, gold)**  38:10.7  76:12.7  114:10.7
- **Baseline (full data, gold)**  40:12.3  80:14.3  120:12.7  160:11.5  200:9.4  225:13.9
- **Matched-gold**  33:8.2  66:10.7  96:14.8
- **CoT + gold target**  32:4.5  64:5.7  96:9.0  128:9.4  160:10.2  192:10.7
- **Distilled (cosine LR)**  32:11.9  64:14.3  96:16.0  128:14.8  160:15.2
- **Distilled (constant LR)**  33:12.3  66:13.5  96:16.8  100:16.4  125:15.2  150:13.1  175:16.4
- **Distilled, BEq+-clean**  25:11.5  50:15.2  75:18.9  100:13.9  125:16.0  150:15.2  175:18.0

## What the target choice buys

Replacing the teacher's *statement* with the gold statement, everything
else held fixed:

| training target | BEq+ | elaborates |
|---|---|---|
| teacher statement + CoT | 15.2% | 67.2% |
| teacher statement, BEq+-filtered + CoT | 18.0% | 58.2% |
| gold statement + CoT | 10.7% | 46.3% |

Out of domain the ladder is not monotone: swapping in the gold
statement (`cotgold`) *hurts* here on both axes, and BEq+-filtering
the teacher targets (`beqok`) is what helps. The in-domain finding
that the gold statement is the better target does not transfer -- see
the Read section.

## Read

The out-of-domain ranking inverts the in-domain one. On LoCoLib the
gold-target models lead; on miniF2F the distilled/CoT models do:

- distilled family final-checkpoint BEq+ 15.2-18.0%, elaboration 58.2-69.7%
- gold-target family 10.7-14.8%, elaboration 46.3-52.5%

The likely cause is prompt-format transfer: miniF2F is self-contained
competition statements with a trivial preamble, which is the shape the
distilled models learned to emit (their teacher wrote standalone Lean).
The gold-target models learned to emit a bare theorem that slots into
LoCoLib's rich namespace context, and that habit does not carry over.
`CoT + gold target` (10.7%) sits at the bottom of the range with the
capped baseline -- the `<think>` prefix plus a gold-context bare theorem
transfers poorly on both axes.

`BEq+-clean` (distilled targets filtered to statements BEq+ says match
the gold) tops the distilled band at 18.0%, though the band is a
few points wide and the runs cross over between checkpoints -- read
`table_eval_by_step.tex`, not the single number.

Every trained model still beats the untrained base (5.3% BEq+ / 45.5% elaboration) with paired significance -- see `table_results.tex`.


## Caveats

- **Proof validity is not measured, by either metric.** Scoring goes through
  `BEqPlusScorer.typecheck_ex`, which calls
  `clean_last_theorem_string(..., add_sorry=True)`: the model's proof body is
  stripped and replaced with `sorry` before Lean is ever called. So
  *elaborates* means the generated **statement** elaborates, and *BEq+* means
  that statement is bidirectionally equivalent to the gold **statement**.
  Neither says whether the model's proof closes the goal.
  `BEqPlusScorer.check_own_proof` would check exactly that and is never called
  by `scripts/eval/evaluate_checkpoints.py`. Elaboration also hard-gates BEq+
  (`BEQ_SKIP_CASCADE_ON_TYPECHECK_FAIL=1`), so the two are nested, not
  independent.
- **No checkpoint is selected on the reported metric.** Every table and figure
  annotation uses each run's *final* checkpoint, fixed by the training
  schedule. There is no separate selection split here (`trainer.test_freq: -1`,
  no dev parquet), so an argmax over the reported BEq+ column would be
  selecting on the statistic it then reports. Read
  `table_eval_by_step.tex` -- the full trajectory -- as the result.
- The untrained number is an upper bound: standalone (union-context) scoring,
  a 1536-token generation budget, and zero-shot with no few-shot exemplars.
- Adjacent checkpoints still disagree on a sizeable minority of the 244 examples
  where the aggregate rate barely moves, so read the curve, not any single
  checkpoint. The LR is annealed over the run, which keeps the trajectory
  monotone; an earlier constant-LR run of the same corpus swung several
  points between neighbouring checkpoints.

## Contents

```
figures/   fig_beq_plus_rate, fig_elaborate_rate, fig_training_loss,
           fig_outcome_breakdown, fig_output_length, fig_target_length,
           fig_domain_distribution            (.png + .pdf)
tables/    table_results, table_eval_by_step, table_outcome_breakdown,
           table_output_lengths, table_config, table_target_lengths,
           table_training_distribution        (.tex, booktabs)
eval/      the per-checkpoint eval JSON (with per_example) and the raw
           generations, copied verbatim from results/eval/<label>/
training/  the SFT job logs these curves were parsed from, plus
           train_loss.csv (series, step, loss)
```

Regenerate: `source hpc/cc_env.sh && python scripts/misc/compare_distilled_baseline.py [--full]`
