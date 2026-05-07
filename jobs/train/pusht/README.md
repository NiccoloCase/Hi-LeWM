# PushT Train Jobs

This directory holds PushT training and benchmark jobs.

## Main scripts

- `train.sh`: base hierarchical PushT P2 training run.
- `train_hope1.sh`: scratch-node training variant using local TMPDIR reads. Supports `MAX_EPOCHS` and `LATENT_ACTION_DIM` environment overrides.
- `train_hope2.sh`: dedicated scratch-node P2 run with defaults `MAX_EPOCHS=15` and `LATENT_ACTION_DIM=32`.
- `train_hope3.sh`: dedicated scratch-node P2 run with frozen lower level and defaults `MAX_EPOCHS=15` and `LATENT_ACTION_DIM=8`.
- `train_latent_action_dim_8_stride_5.sh`: dedicated scratch-node P2 run with frozen lower level, `LATENT_ACTION_DIM=8`, and fixed waypoint sampling stride `5` (`max_span=20`).
- `train_latent_action_dim_8_stride_5_n4.sh`: dedicated scratch-node P2 run with frozen lower level, `LATENT_ACTION_DIM=8`, `waypoints.num=4`, and fixed waypoint sampling stride `5` without changing the default `max_span=15`.
- `train_joint_levels.sh`: scratch-node joint P1+P2 training run with defaults `MAX_EPOCHS=50` and `LATENT_ACTION_DIM=32`, plus per-epoch object and training-state checkpoints.
- `train_joint_levels_resume.sh`: resume an existing joint P1+P2 run in place from its saved `..._weights.ckpt`, keeping the same run directory and W&B id.
- `train_hope1_smoke.sh`: fast smoke version of `train_hope1.sh`.
- `benchmark.sh`: short P2 throughput benchmark.
- `benchmark_ab_io.sh`: A/B benchmark (shared scratch vs node-local TMPDIR I/O).
- `benchmark_cpu_optimization.sh`: single node-local benchmark path.

## Local artifacts

Generated runtime artifacts (Slurm outputs and benchmark logs) are intentionally untracked:

- `*.out`, `*.err`, `*.log`
- `out/` runtime logs created by benchmark scripts
- `old/` historical local log snapshots
