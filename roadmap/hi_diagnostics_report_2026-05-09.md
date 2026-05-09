# Hierarchical Diagnostics Report

Date: 2026-05-09

## Scope

This report summarizes the four offline diagnostics experiments run from the matrix launcher under `jobs/eval/hi/diagnostics/`.

Completed job arrays:

- `22605021` for `hi_lewm_p2_train_hope2_22253175`
- `22605041` for `hi_lewm_p2_train_latent_action_dim_8_stride_5_n4_22518175`
- `22605043` for `hi_lewm_p2_train_latent_action_dim_32_stride_5_n4_22569364`

Completion status:

- all `60/60` diagnostic tasks finished
- `squeue -j 22605021,22605041,22605043` is empty
- no `Traceback`, `Exception`, or `ERROR` markers were found in the final logs

Primary summary files:

- [summary_macro_manifold.tsv](/gpfs/home2/scur0200/main/jobs/eval/hi/diagnostics/logs/summary_macro_manifold.tsv)
- [summary_teacher_vs_open_loop.tsv](/gpfs/home2/scur0200/main/jobs/eval/hi/diagnostics/logs/summary_teacher_vs_open_loop.tsv)
- [summary_dataset_subgoal_reachability.tsv](/gpfs/home2/scur0200/main/jobs/eval/hi/diagnostics/logs/summary_dataset_subgoal_reachability.tsv)
- [summary_generated_subgoal_reachability.tsv](/gpfs/home2/scur0200/main/jobs/eval/hi/diagnostics/logs/summary_generated_subgoal_reachability.tsv)

Model shorthand used below:

- `hope2` = `hi_lewm_p2_train_hope2_22253175`
- `latent8` = `hi_lewm_p2_train_latent_action_dim_8_stride_5_n4_22518175`
- `latent32` = `hi_lewm_p2_train_latent_action_dim_32_stride_5_n4_22569364`

## Shared Diagnostic Settings

These settings were shared across the diagnostics unless noted otherwise.

### Data and checkpoint loading

- dataset name: `pusht_expert_train`
- dataset backend: `stable_worldmodel.data.HDF5Dataset`
- cached dataset key: `action`
- policy loader: `stable_worldmodel.policy.AutoCostModel`
- image size for latent encoding: `224`
- diagnostics cache root: `${STABLEWM_HOME}` via the launcher unless overridden

### Sampling protocol

- `num_eval_samples = 256`
- `num_empirical_chunks = 4096`
- `reference_latent_pool_size = 4096`
- `seed = 42`
- starts were sampled only from contiguous within-episode windows
- contiguity was checked with `episode_idx` or `ep_idx`, and with `step_idx` when present
- if the valid pool was smaller than the request, sampling switched to replacement

### Tokenization and horizon conversion

- `frame_skip = 5`
- planner token count was computed as `ceil(goal_offset_steps / frame_skip)`
- this gives:
  - `d25 -> 5` high-level tokens
  - `d50 -> 10` high-level tokens
- token partitions used by `high_horizon` were:
  - `d25, hh1 -> [5]`
  - `d25, hh2 -> [2, 3]`
  - `d50, hh1 -> [10]`
  - `d50, hh2 -> [5, 5]`
- each token span was converted to raw action rows using the model-specific macro grouping:
  - `raw_span = span_tokens * group`
  - `group = macro_input_dim / raw_action_dim`

### Action preprocessing

- raw actions were standardized with a `StandardScaler` fit on all non-NaN dataset actions
- macro-action chunks were grouped into model input tokens before encoding
- low-level action tokens used the same normalized action representation

### CEM settings

- `cem_bound_mode = none`
- no hard quantile clipping was applied to CEM candidates in this run
- CLI default `cem_elite_frac = 0.1` existed, but in practice the matrix jobs set explicit `topk`, so `topk` controlled elite selection

### Device and execution

- diagnostics were launched on CPU
- Slurm resources per task:
  - `cpus-per-task = 8`
  - `gpus = 0`
  - `time limit = 06:00:00`

### Budget presets by distance

For all `d25` rows:

- high-level CEM: `900` samples, `20` iterations, `topk = 10`
- low-level CEM: `300` samples, `30` iterations, `topk = 150`

For all `d50` rows:

- high-level CEM: `1500` samples, `40` iterations, `topk = 10`
- low-level CEM: `900` samples, `20` iterations, `topk = 150`

## Experiment 1: Macro-Action Manifold

### What this experiment was about

This experiment compares real dataset-encoded macro-action latents against the macro-action latents selected by the high-level CEM planner. The question is whether the planner is choosing latents that move off the empirical training manifold.

### Hypothesis

The original hypothesis was:

- selected CEM macro-actions would have larger norm and Mahalanobis distance than dataset macro-actions
- `hh2` would drift more than `hh1`
- `d50` would drift more than `d25`

### Expectation

If the high-level planner is going off-manifold, then:

- selected macro-action norm should be noticeably larger than dataset macro norm
- elite macro statistics should also shift away from the dataset distribution
- `hh2`, especially at long horizon, should look worst

### Detailed settings

Configuration sweep per model:

- `d25, hh1, lh2`
- `d25, hh2, lh2`
- `d50, hh1, lh2`
- `d50, hh2, lh2`

Important implementation details:

- this experiment only uses the high-level macro-action space
- `low_horizon=2` is carried by the matrix row label but is not used by the metric computation
- for each span length, the code sampled `4096` empirical macro-action chunks from the dataset to build the reference distribution
- reference statistics were built separately for each macro span length:
  - `2`, `3`, `5`, and `10` tokens depending on row
- dataset chunks were encoded with `model.encode_macro_actions`
- planned chunks came from final-iteration high-level CEM candidates, elites, selected mean, and selected best sequence
- Mahalanobis distance was computed against the empirical macro-action reference distribution for that exact span length
- covariance was regularized with `1e-4 * I` before inversion

Cost and planning settings:

- the planner optimized macro-action sequences against the final goal latent
- high-level cost was final latent MSE between predicted final high-level latent and encoded goal latent
- `d25` used `900 x 20` high-level CEM
- `d50` used `1500 x 40` high-level CEM

### Actual results

`hh1` rows:

| Model | d | Span Tokens | Dataset Mean Norm | Selected Mean Norm | Selected Best Mean Norm | Elite Mean Norm | Selected MD2 P50 | Elite MD2 P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hope2 | 25 | 5 | 13.7297 | 15.8607 | 15.8726 | 15.8650 | 50718.6484 | 50846.2656 |
| latent32 | 25 | 5 | 14.9719 | 17.6152 | 17.6301 | 17.6199 | 73387.2344 | 73560.4688 |
| latent8 | 25 | 5 | 10.2311 | 10.7368 | 10.7369 | 10.7368 | 40.9262 | 40.9208 |
| hope2 | 50 | 10 | 14.6444 | 18.8051 | 18.8057 | 18.8051 | 107211.3125 | 107211.5469 |
| latent8 | 50 | 10 | 9.6376 | 15.2226 | 15.2226 | 15.2226 | 117.9486 | 117.9427 |
| latent32 | 50 | 10 | 14.5113 | 24.2393 | 24.2398 | 24.2393 | 156483.7188 | 156526.5000 |

`hh2` rows:

| Model | d | Step 1 Span | Step 1 Dataset Norm | Step 1 Selected Norm | Step 1 Selected Best | Step 1 Elite Norm | Step 1 MD2 P50 | Step 1 Elite MD2 P50 | Step 2 Span | Step 2 Dataset Norm | Step 2 Selected Norm | Step 2 Selected Best | Step 2 Elite Norm | Step 2 MD2 P50 | Step 2 Elite MD2 P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| latent8 | 25 | 2 | 10.2506 | 10.6481 | 10.6486 | 10.6483 | 104.0253 | 103.9607 | 3 | 10.2075 | 9.8180 | 9.8184 | 9.8181 | 52.0488 | 51.9142 |
| hope2 | 25 | 2 | 14.1380 | 14.9204 | 14.9825 | 14.9579 | 64205.2500 | 63831.6758 | 3 | 13.9237 | 15.4221 | 15.4572 | 15.4440 | 58536.4219 | 58775.2891 |
| latent32 | 25 | 2 | 14.5073 | 16.3541 | 16.4059 | 16.3925 | 118896.5781 | 119781.4844 | 3 | 14.6700 | 16.4603 | 16.5149 | 16.4928 | 91617.4922 | 92521.6719 |
| latent8 | 50 | 5 | 10.2311 | 11.1737 | 11.1737 | 11.1737 | 39.8901 | 39.8929 | 5 | 10.2311 | 10.1118 | 10.1119 | 10.1118 | 33.9362 | 33.9380 |
| latent32 | 50 | 5 | 14.9719 | 17.5647 | 17.5660 | 17.5649 | 68651.0781 | 68650.8047 | 5 | 14.9719 | 16.9901 | 16.9898 | 16.9902 | 60719.8125 | 60733.6602 |
| hope2 | 50 | 5 | 13.7297 | 15.6738 | 15.6754 | 15.6739 | 45158.4844 | 45202.8516 | 5 | 13.7297 | 15.4800 | 15.4798 | 15.4801 | 45407.3398 | 45391.9414 |

Useful norm ratios, `selected_mean_norm / dataset_mean_norm`:

| Model | d25 hh1 | d25 hh2 step1 | d25 hh2 step2 | d50 hh1 | d50 hh2 step1 | d50 hh2 step2 |
|---|---:|---:|---:|---:|---:|---:|
| hope2 | 1.1552 | 1.0553 | 1.1076 | 1.2841 | 1.1416 | 1.1275 |
| latent8 | 1.0494 | 1.0388 | 0.9618 | 1.5793 | 1.0921 | 0.9883 |
| latent32 | 1.1766 | 1.1273 | 1.1227 | 1.6704 | 1.1732 | 1.1341 |

### Interpretation

- The expected drift exists, but the strongest drift is not `hh2`.
- The worst case is `d50 hh1`, especially for `latent8` and `latent32`.
- `d50 hh1` is substantially farther from the dataset manifold than `d50 hh2`.
- This means one long macro-action is a worse abstraction than two shorter macro-actions.
- The original hypothesis that `hh2` would be the most off-manifold is not supported.

## Experiment 2: Teacher-Forced vs Open-Loop High-Level Error

### What this experiment was about

This experiment compares high-level rollout quality in three modes:

- one-step teacher-forced with true macro-actions
- multi-step open-loop with true macro-actions
- multi-step open-loop with CEM-selected macro-actions

The purpose is to separate:

- predictor instability
- planner/prior mismatch

### Hypothesis

The original hypothesis was:

- if open-loop with true macros fails, the high-level predictor is unstable
- if open-loop with true macros is fine but CEM macros fail, the planner/prior is the issue

### Expectation

- `hh1` should look stable
- `hh2` should expose whether the main problem is the predictor or the planned macro-action sequence

### Detailed settings

Configuration sweep per model:

- `d25, hh1, lh2`
- `d25, hh2, lh2`
- `d50, hh1, lh2`
- `d50, hh2, lh2`

Important implementation details:

- this experiment uses the high-level latent transition model only
- `low_horizon=2` is carried by the matrix row label but is not used by the metric computation
- `256` valid contiguous start points were sampled per row
- for each row:
  - `z_init` was encoded from the start image
  - true macro-action sequences were built from real dataset action chunks
  - target future latents were encoded from the future image at each macro boundary
- teacher-forced rollout:
  - the model predicted one step at a time
  - after each prediction, the next step started from the true target latent
- open-loop true rollout:
  - the model rolled out through the whole macro sequence without resetting to true targets
- open-loop CEM rollout:
  - the high-level CEM first optimized macro-action latents toward the final goal latent
  - then the model rolled out open-loop under that selected macro sequence
- flags were computed from the summary ratios:
  - predictor instability from `open_loop_true / teacher_forced`
  - planner/prior issue from `open_loop_cem / open_loop_true`

Cost and planning settings:

- high-level CEM optimized final latent MSE to the goal latent
- `d25` used `900 x 20` high-level CEM
- `d50` used `1500 x 40` high-level CEM

### How the procedure works

For each row, the experiment first sampled `256` valid contiguous starts from the dataset. For each sampled start it built:

- `z_init`: the encoded current latent at the start image
- `macro_seq`: the sequence of true macro-action latents built from real dataset action chunks
- `target_seq`: the sequence of true future latents at each macro boundary
- `z_goal`: the final latent target, equal to the last element of `target_seq`

It then compared three rollout modes.

1. Teacher-forced rollout

- the high-level model predicts one step at a time
- after each prediction, the next step starts from the true latent target instead of the predicted latent
- this isolates one-step transition quality
- it removes compounding rollout drift

2. Open-loop rollout with true macro-actions

- the high-level model is rolled out across the full true macro-action sequence
- after step 1, step 2 starts from the model's own predicted latent rather than the true latent
- this measures how much error compounds when the model feeds on its own predictions

3. Open-loop rollout with CEM-selected macro-actions

- first, high-level CEM optimizes a macro-action latent sequence toward the final goal latent
- then the high-level model rolls out open-loop under that selected macro-action sequence
- this tests whether the planned macro sequence is worse than the true macro sequence under the model

### Why `hh1` is mainly a sanity check

When `high_horizon = 1`, there is only one high-level step. That means:

- teacher-forced rollout and open-loop true rollout are effectively the same procedure
- there is no second step where prediction drift can accumulate
- therefore `open_loop_true_mse_mean` should match `teacher_forced_mse_mean`

That is exactly what happened for every `hh1` row, which is a useful sanity check that the diagnostic is behaving as intended.

### How to read the key metrics

The experiment reports two especially important ratios.

1. `open_loop_true_over_teacher`

- this compares multi-step open-loop error under true macro-actions against teacher-forced error
- if it is close to `1.0`, the predictor is stable across the rollout
- if it is much larger than `1.0`, the model accumulates error when rolled out on its own predictions
- in this implementation, `> 1.5` triggers `high_predictor_instability_flag = True`

2. `open_loop_cem_over_open_true`

- this compares open-loop error under CEM-selected macro-actions against open-loop error under true macro-actions
- if it is close to `1.0`, planned macros are about as model-compatible as true macros
- if it is much larger than `1.0`, planned macros are worse than true macros
- if it is smaller than `1.0`, the planned macros are actually easier for the model to roll out than the true macros
- in this implementation, `> 1.5` triggers `planner_prior_issue_flag = True`

### How to read the per-step metrics

For `hh2`, the report stores separate step-1 and step-2 errors for all three rollout modes.

This is important because:

- step 1 teacher-forced and step 1 open-loop true usually match, since both begin from the true `z_init`
- the real divergence typically appears at step 2
- if step 2 open-loop true is much worse than step 2 teacher-forced, that is direct evidence of compounding model error
- if step 2 open-loop CEM is much worse than step 2 open-loop true, that is evidence that the planned macro-action sequence is problematic under rollout

### What this experiment can and cannot conclude

This experiment is fully offline and latent-space only.

It can tell us:

- whether one-step high-level prediction is accurate
- whether multi-step high-level prediction is stable
- whether CEM-selected macro-actions are worse than true macro-actions under the model

It cannot tell us:

- whether the low-level planner can physically realize those macro plans in the environment
- whether the generated subgoals are semantically useful for control
- whether online replanning causes harmful subgoal churn during real execution

So when `open_loop_cem` looks good, that only means the planned macro sequence is compatible with the high-level latent model. It does not yet mean the full hierarchical controller will succeed online.

### Actual results

| Model | d | hh | Teacher MSE | Open-Loop True MSE | Open-Loop CEM MSE | Open True / Teacher | Open CEM / Open True | Planner Flag | Predictor Flag | Step 1 Teacher | Step 1 Open True | Step 1 Open CEM | Step 2 Teacher | Step 2 Open True | Step 2 Open CEM |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| hope2 | 25 | 1 | 0.1177 | 0.1177 | 0.0108 | 1.0000 | 0.0914 | False | False | 0.1177 | 0.1177 | 0.0108 |  |  |  |
| latent8 | 25 | 1 | 0.1713 | 0.1713 | 0.0325 | 1.0000 | 0.1899 | False | False | 0.1713 | 0.1713 | 0.0325 |  |  |  |
| latent32 | 25 | 1 | 0.1746 | 0.1746 | 0.0194 | 1.0000 | 0.1114 | False | False | 0.1746 | 0.1746 | 0.0194 |  |  |  |
| latent8 | 25 | 2 | 0.3499 | 0.5722 | 0.1850 | 1.6353 | 0.3234 | False | True | 0.4084 | 0.4084 | 0.3589 | 0.2914 | 0.7359 | 0.0112 |
| latent32 | 25 | 2 | 0.3752 | 0.5961 | 0.1689 | 1.5885 | 0.2833 | False | True | 0.4383 | 0.4383 | 0.3296 | 0.3122 | 0.7538 | 0.0081 |
| hope2 | 25 | 2 | 0.0281 | 0.0466 | 0.0837 | 1.6569 | 1.7970 | True | True | 0.0195 | 0.0195 | 0.1614 | 0.0367 | 0.0737 | 0.0060 |
| latent32 | 50 | 1 | 1.2929 | 1.2929 | 0.3251 | 1.0000 | 0.2515 | False | False | 1.2929 | 1.2929 | 0.3251 |  |  |  |
| latent8 | 50 | 1 | 1.2727 | 1.2727 | 0.4030 | 1.0000 | 0.3167 | False | False | 1.2727 | 1.2727 | 0.4030 |  |  |  |
| hope2 | 50 | 1 | 0.5378 | 0.5378 | 0.0598 | 1.0000 | 0.1112 | False | False | 0.5378 | 0.5378 | 0.0598 |  |  |  |
| hope2 | 50 | 2 | 0.0924 | 0.1486 | 0.1764 | 1.6082 | 1.1873 | False | True | 0.0812 | 0.0812 | 0.3398 | 0.1036 | 0.2160 | 0.0131 |
| latent32 | 50 | 2 | 0.1376 | 0.1843 | 0.1341 | 1.3396 | 0.7272 | False | False | 0.1023 | 0.1023 | 0.2474 | 0.1730 | 0.2664 | 0.0207 |
| latent8 | 50 | 2 | 0.1313 | 0.1830 | 0.1301 | 1.3936 | 0.7110 | False | False | 0.1003 | 0.1003 | 0.2441 | 0.1623 | 0.2657 | 0.0162 |

### Interpretation

- `hh1` is stable in every case: `open_loop_true_mse_mean == teacher_forced_mse_mean`.
- `d25 hh2` shows clear predictor instability for all three models.
- `hope2 d25 hh2` is the only setting with an explicit planner/prior failure flag.
- For `latent8` and `latent32`, `d50 hh2` does not look like a planner failure in latent-MSE terms.
- This means the final long-horizon online failure is not explained by this high-level latent prediction diagnostic alone.

## Experiment 3: Dataset-Subgoal Reachability

### What this experiment was about

This experiment asks the low-level planner to reach real future dataset latents from the same trajectory, using offsets:

- `+2`
- `+3`
- `+5`

The purpose is to test whether the low-level planner can reach sensible subgoals before blaming generated high-level subgoals.

### Hypothesis

The original hypothesis was:

- if the low-level cannot reach real dataset subgoals, generated subgoals have no chance

There was also an informal expectation that:

- `lh2` might look better than `lh5`

### Expectation

- `lh1` should be too short
- `lh2` should be usable
- this experiment should tell us whether low-level capability is fundamentally broken

### Detailed settings

Configuration sweep per model:

- `d25, lh1`
- `d25, lh2`
- `d25, lh3`
- `d25, lh5`
- `d50, lh1`
- `d50, lh2`
- `d50, lh3`
- `d50, lh5`

Important implementation details:

- `high_horizon=1` is carried by the matrix row label but is not used by this experiment
- low-level targets were real future dataset latents from the same trajectory
- offsets were fixed to:
  - `+2`
  - `+3`
  - `+5`
- for each row:
  - `256` valid contiguous starts were sampled
  - future target latents were encoded directly from the dataset images at those offsets
  - the low-level planner started from the encoded current latent and optimized action-token sequences toward the true future latent
- the maximum raw contiguous span used for start sampling was `max(subgoal_offsets) * group`
- the low-level CEM operated in action-token space using normalized action tokens
- terminal error was computed as latent MSE between the best low-level rollout endpoint and the target future latent

Cost and planning settings:

- low-level cost was final latent MSE to the true future subgoal latent
- `d25` used `300 x 30` low-level CEM
- `d50` used `900 x 20` low-level CEM

### Actual results

`d25` results:

| Model | Low Horizon | Offset +2 Error | Offset +3 Error | Offset +5 Error | Overall Error |
|---|---:|---:|---:|---:|---:|
| hope2 | 1 | 0.0170 | 0.0749 | 0.4330 | 0.1750 |
| latent8 | 1 | 0.0170 | 0.0738 | 0.4329 | 0.1746 |
| latent32 | 1 | 0.0171 | 0.0741 | 0.4363 | 0.1758 |
| latent32 | 2 | 0.0075 | 0.0113 | 0.1556 | 0.0582 |
| latent8 | 2 | 0.0076 | 0.0113 | 0.1608 | 0.0599 |
| hope2 | 2 | 0.0076 | 0.0112 | 0.1578 | 0.0589 |
| hope2 | 3 | 0.0092 | 0.0101 | 0.0851 | 0.0348 |
| latent8 | 3 | 0.0093 | 0.0101 | 0.0890 | 0.0361 |
| latent32 | 3 | 0.0093 | 0.0101 | 0.0901 | 0.0365 |
| hope2 | 5 | 0.0150 | 0.0154 | 0.0721 | 0.0341 |
| latent32 | 5 | 0.0149 | 0.0154 | 0.0698 | 0.0334 |
| latent8 | 5 | 0.0150 | 0.0154 | 0.0693 | 0.0332 |

`d50` results:

| Model | Low Horizon | Offset +2 Error | Offset +3 Error | Offset +5 Error | Overall Error |
|---|---:|---:|---:|---:|---:|
| hope2 | 1 | 0.0152 | 0.0653 | 0.3860 | 0.1555 |
| latent32 | 1 | 0.0153 | 0.0653 | 0.3861 | 0.1556 |
| latent8 | 1 | 0.0152 | 0.0652 | 0.3818 | 0.1541 |
| hope2 | 2 | 0.0062 | 0.0091 | 0.1073 | 0.0409 |
| latent8 | 2 | 0.0062 | 0.0091 | 0.1038 | 0.0397 |
| latent32 | 2 | 0.0063 | 0.0091 | 0.1032 | 0.0395 |
| hope2 | 3 | 0.0075 | 0.0077 | 0.0420 | 0.0191 |
| latent32 | 3 | 0.0074 | 0.0077 | 0.0434 | 0.0195 |
| latent8 | 3 | 0.0074 | 0.0078 | 0.0417 | 0.0190 |
| hope2 | 5 | 0.0113 | 0.0111 | 0.0365 | 0.0196 |
| latent8 | 5 | 0.0116 | 0.0112 | 0.0375 | 0.0201 |
| latent32 | 5 | 0.0115 | 0.0112 | 0.0381 | 0.0202 |

### Interpretation

- The low-level planner can reliably reach real dataset subgoals.
- `lh1` is clearly too short.
- `lh2` is much better than `lh1`, but it is not the best horizon.
- `lh3` and `lh5` are the best settings overall.
- The expectation that `lh2` might beat `lh5` is not supported.
- The hierarchy is therefore not failing because the low-level planner is incapable of reaching sensible latent targets.

## Experiment 4: Generated-Subgoal Reachability

### What this experiment was about

This experiment feeds the low-level planner high-level generated subgoals rather than real dataset future latents. It measures:

- low-level terminal cost
- low-level terminal latent error
- distance to nearest dataset latent
- distance to nearest same-trajectory future latent

The purpose is to test whether generated subgoals are both:

- reachable
- close to realistic futures

### Hypothesis

The original hypothesis was:

- high-level generated subgoals would be off-manifold and hard to reach

### Expectation

- bad generated subgoals should have higher terminal error
- bad generated subgoals should be farther from same-trajectory future latents
- the worst settings should correlate with poor online success

### Detailed settings

Configuration sweep per model:

- `d25, hh1, lh2`
- `d25, hh2, lh2`
- `d50, hh1, lh2`
- `d50, hh2, lh2`

Important implementation details:

- this experiment combines both high-level and low-level planning
- `low_horizon` was fixed to `2` for the whole sweep
- `256` valid contiguous starts were sampled per row
- the high-level planner first optimized a macro-action latent sequence toward the final goal latent
- the model then rolled out the selected high-level macro sequence to produce generated subgoals at each stage boundary
- for each generated subgoal:
  - the low-level planner optimized a 2-step action-token sequence toward that generated latent
  - terminal cost and terminal latent error were recorded
  - distance to the nearest dataset latent was measured using a `4096`-latent reference pool sampled from the whole dataset
  - distance to the nearest same-trajectory future latent was measured against all future latents from that start within the relevant future raw span
  - Mahalanobis distance was computed in state-latent space against the sampled dataset latent pool reference distribution
- for `hh2`, step 1 and step 2 were analyzed separately

Cost and planning settings:

- high-level cost was final latent MSE to the encoded goal latent
- low-level cost was final latent MSE to the generated subgoal latent
- `d25` used:
  - high-level `900 x 20`
  - low-level `300 x 30`
- `d50` used:
  - high-level `1500 x 40`
  - low-level `900 x 20`

### Actual results

`d25` results:

| Model | hh | Low Horizon | Step 1 Terminal Cost | Step 1 Terminal Error | Step 1 Dataset Distance | Step 1 Same-Traj Distance | Step 2 Terminal Cost | Step 2 Terminal Error | Step 2 Dataset Distance | Step 2 Same-Traj Distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hope2 | 1 | 2 | 0.1639 | 0.1639 | 5.0071 | 1.2698 |  |  |  |  |
| latent8 | 1 | 2 | 0.1479 | 0.1479 | 5.1541 | 1.9395 |  |  |  |  |
| latent32 | 1 | 2 | 0.1645 | 0.1645 | 5.0641 | 1.4939 |  |  |  |  |
| latent8 | 2 | 2 | 0.0593 | 0.0593 | 5.7274 | 6.6730 | 0.0212 | 0.0212 | 4.9763 | 1.3398 |
| hope2 | 2 | 2 | 0.0128 | 0.0128 | 4.9118 | 2.7779 | 0.0161 | 0.0161 | 4.8708 | 0.9386 |
| latent32 | 2 | 2 | 0.0551 | 0.0551 | 5.7084 | 6.1163 | 0.0204 | 0.0204 | 4.9522 | 1.1825 |

`d50` results:

| Model | hh | Low Horizon | Step 1 Terminal Cost | Step 1 Terminal Error | Step 1 Dataset Distance | Step 1 Same-Traj Distance | Step 2 Terminal Cost | Step 2 Terminal Error | Step 2 Dataset Distance | Step 2 Same-Traj Distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| latent8 | 1 | 2 | 0.2522 | 0.2522 | 7.7049 | 6.8489 |  |  |  |  |
| hope2 | 1 | 2 | 0.5289 | 0.5289 | 5.6839 | 2.8625 |  |  |  |  |
| latent32 | 1 | 2 | 0.2834 | 0.2834 | 7.3606 | 5.9942 |  |  |  |  |
| latent8 | 2 | 2 | 0.0980 | 0.0980 | 5.6353 | 4.5769 | 0.1160 | 0.1160 | 5.3475 | 1.3176 |
| hope2 | 2 | 2 | 0.1086 | 0.1086 | 5.5547 | 5.2670 | 0.1173 | 0.1173 | 5.3101 | 1.1151 |
| latent32 | 2 | 2 | 0.0910 | 0.0910 | 5.4976 | 4.6356 | 0.1132 | 0.1132 | 5.4258 | 1.2781 |

### Interpretation

- `d50 hh1` is the clearly worst regime.
- The generated subgoals from one long macro-action are harder to reach and farther from realistic future latents.
- `hh2` generated subgoals are much more reachable than `hh1` generated subgoals.
- In `hh2`, step 2 is usually much closer to a realistic future than step 1.
- This result agrees with the macro-manifold diagnostic: splitting the horizon into shorter macro-actions improves subgoal quality.

## Cross-Experiment Summary

The combined story from the diagnostics is:

- the low-level planner is not fundamentally broken
- real future latent subgoals are reachable
- one long macro-action is the worst abstraction and goes off-manifold
- two shorter macro-actions produce better generated subgoals than one long macro
- `hh2` still shows predictor instability, especially in `d25`
- for `latent8` and `latent32`, the final long-horizon online failure is not explained by low-level inability or by a simple planner-prior mismatch alone

The most likely failure point is therefore the online hierarchical control loop:

- high-level rollout and replanning dynamics
- how generated subgoals are updated and handed to the low level
- how multi-stage subgoals are used in closed loop

## Bottom-Line Conclusions

1. The diagnostics do not support the theory that the low-level planner is the primary problem.
2. The diagnostics do support the theory that long single macro-actions are structurally poor and go off-manifold.
3. The diagnostics support using split high-level horizons over one long macro, at least in terms of subgoal plausibility and reachability.
4. The remaining gap between better offline subgoal quality and weaker final online performance points to instability in online hierarchical execution rather than a failure of the low-level planner in isolation.
