# Original Baseline Eval Jobs

This directory contains evaluation jobs that run the original LeWM baseline
(`third_party/lewm/eval.py`) across multiple environments.

## Scripts

- `pusht_eval.sh`: baseline PushT eval.
- `pusht_eval_withmetrics.sh`: baseline eval + per-eval pass/fail manifest.
- `pusht_eval_withmetrics_budget.sh`: metric eval variant with larger eval budget.
- `pusht_eval_withmetrics_horizon.sh`: metric eval variant with longer planning horizon.
- `pusht/matrix/`: CPU array sweep for the original flat PushT baseline using
  `roadmap/baseline_matrix_sweep.csv`.
- `cube/matrix/`: CPU array sweep for the original flat OGBench Cube baseline,
  pinned to the single-cube config/dataset.

## Runtime outputs

Slurm outputs are written under `out/` by these scripts.
Those `*.out`/`*.err` files are intentionally ignored by Git.
