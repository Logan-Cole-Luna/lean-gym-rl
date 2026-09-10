# results/

Two kinds of directory. **Job-written** directories are the output of a SLURM
job and are named by the code that writes them; do not rename or nest them.
**Report** directories are assembled from those by a script and are free to move.

```
eval/                     job-written. One directory per checkpoint SERIES.
    <label>/                eval_<label>-step<N>_n<n>.json, matching gen_*.jsonl,
                            passk_*.json. Discovery over this level returns the
                            series still under study.
    _completed/             finished studies, held out of discovery
        sft_corpus_ablation/  SFT recipe comparison (base, matchedgold, beqok,
                              cotgold, distilled, distilledcos, uncapped)
        minif2f_sweep/        the miniF2F transfer sweep for that study
        _partial_minif2f_20260909/
    archive/                Lean-Workbook era, scored under v4.8.0-rc1
train/                    job-written. train_metrics/ (verl FileLogger),
                          val_generations_*/, _probes/
figures/                  make_figures.py. c512/ holds the 512-cap series;
                          the top level holds the 128-cap lr6 series
rescore/                  rescore_array.slurm, one JSON per re-scored gen file
faithfulness/             faith_eval.slurm, one directory per gen file scored.
                          gate_<judge>/ holds the judge-selection runs
alignment/                alignment.slurm, one directory per gen file scored
manifests/                inputs to the batch jobs
    rescore_manifest*.txt   which gen files a rescore array covers
    genlists/               file lists passed as GENS_FILE
reports/                  assembled studies, each self-contained
    reward_arms_lr6/        README + eval/ + figures/ + tables/ + training/
    minif2f_corpus_ablation/
    distilled/
tables/                   shared LaTeX fragments
_archive/                 superseded material, kept for provenance
REWARD_ARMS.md            the reward specification and the arm results
RUNS.md                   the run index
```

## Conventions

A directory whose name is `archive` or begins with `_` is held out of every
discovery path (`evalio.discover_arms`, `discover_baseline_label`,
`select_checkpoint.load_records`). Nesting a series there removes it from the
live comparison without deleting it.

Records superseded by a re-derivation move to `_superseded_<stamp>/` inside their
own series directory rather than being overwritten, so a change in a published
number stays checkable against the record it replaced.

The baseline every arm is compared against is resolved from the pinned pointer
file (`data_locolib/best_sft_proof.txt`), not from directory write times.
`BASELINE_LABEL` overrides it.

Every eval record written from 2026-09-07 onward carries `toolchain` and
`max_new_tokens`. A record without them was produced at 128 tokens under an
unrecorded Lean version and is not comparable to one that has them.
