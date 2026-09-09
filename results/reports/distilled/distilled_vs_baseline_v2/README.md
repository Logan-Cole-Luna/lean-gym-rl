# CoT-distilled SFT vs the untrained base model

Same pinned 760-row `data_locolib/val_proof.parquet` (in-domain),
same scoring path
(`score_standalone`) and the same 1536-token generation budget for both, so
nothing here is confounded by eval settings. The untrained model is a level,
not a trajectory, so it is drawn as a dash-dot reference line.

## Headline

| series | step | type-check | BEq+ | Δ BEq+ vs untrained | McNemar p |
|---|---|---|---|---|---|
| untrained base | 0 | 20.9% | 11.8% | — | — |
| Distilled | 128 | 66.4% | 50.7% | +38.8 pp | 5e-70 |

## Trajectories (BEq+ %)

- **Distilled**  32:41.7  64:45.8  96:48.9  128:50.7  160:49.7

## Read

SFT does two separable jobs, and `table_outcome_breakdown.tex` splits them:

1. **Make Lean that elaborates at all.** The untrained model compiles 159/760 (20.9%) of the slice; after distilled SFT that is 505/760 (66.4%). This is the bulk of the gain.
2. **Make the *intended* theorem.** Conditional on compiling, the untrained model is right 56.6% of the time and the distilled model 76.2%. The residual `weaker_only` count (54 vs 9) is statement drift: the model states a theorem that is implied by the gold rather than equivalent to it, traced to the ~24% of its training targets that BEq+ says are not the gold theorem.

Distilled data trains the model: it moves BEq+ from 11.8% to 50.7% and the compile rate from 20.9% to 66.4%, both with overwhelming paired significance.

## Caveats

- The untrained number is an upper bound: standalone (union-context) scoring,
  a 1536-token generation budget, and zero-shot with no few-shot exemplars.
- Adjacent checkpoints still disagree on a sizeable minority of the 760 examples
  where the aggregate rate barely moves, so read the curve, not any single
  checkpoint. The LR is annealed over the run, which keeps the trajectory
  monotone; an earlier constant-LR run of the same corpus swung several
  points between neighbouring checkpoints.

## Contents

```
figures/   fig_beq_plus_rate, fig_compile_rate, fig_training_loss,
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
