# Archived scripts

| archived | replacement |
|---|---|
| `make_starved_subset.py` | `make_difficulty_subset.py --min-wins 0 --max-wins 0` |

Verified equivalent before archiving: both produce the same 559 prompts from
`data/rollouts/sft390_k8.scored.jsonl`, symmetric difference 0.
| `plot_results.py` | `scripts/make_figures.py` (`make figures`) |

`plot_results.py` parsed per-step metrics out of verl's stdout. Only three arms
ever logged them there — the rest wrote to stderr in a form carrying no metrics —
so on the 3B series it exits with "No parseable step metrics in those logs".
`runtime.png` gets the same wall-clock information from checkpoint mtimes, which
works for every arm.
