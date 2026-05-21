# hope2 Experiment 2: Teacher-Forced vs Open-Loop High-Level Error

Source: [hi_diagnostics_report_2026-05-09.md](/gpfs/home2/scur0200/main/roadmap/analysis/hi_diagnostics_report_2026-05-09.md:346)

This note keeps only the `hope2` rows from the `Actual results` table.

## What is happening

For `hh1`, the high-level model is stable once the macro-action sequence is fixed. At both `d25` and `d50`, `open_loop_true_mse == teacher_mse` (`0.1177` and `0.5378`), so there is no extra error from rolling the predictor forward for a single high-level step. In this experiment, the CEM-selected macro is actually easier for the model than the true dataset macro, with `open_loop_cem_mse` dropping to `0.0108` at `d25` and `0.0598` at `d50`. That does not mean `hh1` is globally good; it only means the high-level latent predictor evaluates a one-step plan consistently.

For `hh2`, compounding rollout error appears immediately. At `d25`, `teacher_mse` rises from `0.0281` to `open_loop_true_mse = 0.0466` (`1.6569x`), and the CEM-selected sequence makes it worse again at `0.0837` (`1.7970x` over open-loop true). Both `predictor_flag` and `planner_flag` are `True`, so this is the clearest `hope2` case where both model rollout and planned macro selection are failing.

At `d50 hh2`, the pattern is weaker but similar. `teacher_mse` increases from `0.0924` to `0.1486` in open loop (`1.6082x`), so the predictor is still unstable over two stages. But the CEM plan only increases error to `0.1764` (`1.1873x` over open-loop true), and `planner_flag` stays `False`. That means the dominant issue here is predictor instability, not a severe planner-prior mismatch.

## Takeaway

For `hope2`, this experiment isolates a high-level modeling problem that shows up when the horizon is split into two stages. `hh1` looks stable in this diagnostic because it only asks the model to evaluate one macro transition. `hh2` exposes the real weakness: once the model must roll forward across multiple macro steps, error compounds, and at `d25` the planner also selects sequences that the model handles badly.
