# CoT-distilled SFT vs the untrained base model

Same pinned 760-row `data_locolib/val_proof.parquet` (in-domain),
same scoring path
(`score_standalone`) and the same 1536-token generation budget for both, so
nothing here is confounded by eval settings. The untrained model is a level,
not a trajectory, so it is drawn as a dash-dot reference line.

## Headline

Each row is that run's **final** checkpoint, not a checkpoint selected on this
table's own BEq+ column. Both metrics score the generated *statement* only --
see Caveats.

| series | step | elaborates | BEq+ | Δ BEq+ vs untrained | McNemar p |
|---|---|---|---|---|---|
| untrained base | 0 | 20.9% | 11.8% | — | — |
| Distilled | 160 | 63.6% | 49.7% | +37.9 pp | 4.2e-68 |

## Trajectories (BEq+ %)

- **Distilled**  32:41.7  64:45.8  96:48.9  128:50.7  160:49.7

## Read

SFT does two separable jobs, and `table_outcome_breakdown.tex` splits them:

1. **Make a statement that elaborates at all.** The untrained model elaborates 159/760 (20.9%) of the slice; after distilled SFT that is 483/760 (63.6%). This is the bulk of the gain.
2. **Make the *intended* theorem.** Conditional on elaborating, the untrained model is right 56.6% of the time and the distilled model 78.3%. The `weaker_only` bucket (51 vs 9) is statement drift: a theorem implied by the gold rather than equivalent to it, traced to the ~24% of the training targets that BEq+ says are not the gold theorem.

Distilled data trains the model: it moves BEq+ from 11.8% to 49.7% and the statement-elaboration rate from 20.9% to 63.6%, both with overwhelming paired significance.

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
- Adjacent checkpoints still disagree on a sizeable minority of the 760 examples
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
