# PushT Planning Config Matrix

This file summarizes:

1. The current planning configs used by `jobs/eval/hi/matrix/eval_fixed_stride_matrix.sh`
2. Recommended configs to test for Hope2 on PushT at `d=25`, `d=50`, and `d=75`

Common assumptions across all hierarchical Hope2 configs here:

- Env is always `PushT`
- `planning.mode=hierarchical`
- `EVAL_DEVICE=cpu`
- `HIGH_RECEDING_HORIZON=1`
- `LOW_ACTION_BLOCK=5`
- `HIGH_ACTION_BLOCK=1`

Active checkpoint in the current matrix:

- `hi_lewm_cube_train_hope1_22606708`, epoch `10` from [checkpoints_fixed_stride.txt](/gpfs/home2/scur0200/main/jobs/eval/hi/matrix/checkpoints_fixed_stride.txt:3)

Historical reference success rates in the current-matrix table come from the older Hope2 run recorded in [fixed_stride_matrix_params.csv](/gpfs/home2/scur0200/main/jobs/eval/hi/matrix/fixed_stride_matrix_params.csv:1).

## Current Matrix

| idx | label | d | budget | HH | LH | H-RH | L-RH | H-replan | H samples/iters/topk | L samples/iters/topk | Hist. success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `d25_hh1_lh2_lrh1` | 25 | 50 | 1 | 2 | 1 | 1 | 5 | `900 / 20 / 10` | `300 / 30 / 150` | `84.0` |
| 2 | `d25_hh1_lh5_lrh1` | 25 | 50 | 1 | 5 | 1 | 1 | 5 | `900 / 20 / 10` | `300 / 30 / 150` | `44.0` |
| 3 | `d25_hh2_lh2_lrh1` | 25 | 50 | 2 | 2 | 1 | 1 | 5 | `900 / 20 / 10` | `300 / 30 / 150` | `20.0` |
| 4 | `d50_hh1_lh2_lrh1` | 50 | 50 | 1 | 2 | 1 | 1 | 5 | `1500 / 40 / 10` | `900 / 20 / 150` | `44.0` |
| 5 | `d50_hh2_lh2_lrh1` | 50 | 50 | 2 | 2 | 1 | 1 | 5 | `1500 / 40 / 10` | `900 / 20 / 150` | `32.0` |
| 6 | `d50_hh2_lh5_lrh1` | 50 | 50 | 2 | 5 | 1 | 1 | 5 | `1500 / 40 / 10` | `900 / 20 / 150` | `14.0` |
| 7 | `d50_hh2_lh5_lrh5` | 50 | 50 | 2 | 5 | 1 | 5 | 5 | `1500 / 40 / 10` | `900 / 20 / 150` | `12.0` |

## Recommended Tests

### d=25

| prio | label | budget | HH | LH | H-replan | H samples/iters/topk | L samples/iters/topk | Why this makes sense for Hope2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `d25_hh1_lh2_bestprev` | 50 | 1 | 2 | 5 | `900 / 20 / 10` | `300 / 30 / 150` | Strong short-horizon baseline: Hope2 keeps macro planning shallow while low-level execution stays on short latent segments. |
| 2 | `d25_hh1_lh2_replan3` | 50 | 1 | 2 | 3 | `900 / 20 / 10` | `300 / 30 / 150` | Tests whether Hope2 subgoals are useful but need faster closed-loop refresh. |
| 3 | `d25_hh1_lh2_searchboost` | 50 | 1 | 2 | 5 | `900 / 20 / 10` | `600 / 40 / 200` | Checks whether good Hope2 subgoals are currently being lost at the low-level execution stage due to weak CEM search. |
| 4 | `d25_hh2_lh2_baseline` | 50 | 2 | 2 | 5 | `900 / 20 / 10` | `300 / 30 / 150` | Tests whether even short PushT goals benefit from explicit two-stage macro decomposition. |
| 5 | `d25_hh2_lh2_searchboost` | 50 | 2 | 2 | 5 | `1200 / 30 / 20` | `600 / 40 / 200` | Separates “`HH=2` is bad” from “`HH=2` needs stronger search” in the Hope2 hierarchy. |
| 6 | `d25_hh1_lh1_searchboost` | 50 | 1 | 1 | 5 | `1200 / 30 / 20` | `600 / 40 / 200` | Tests whether Hope2 low-level rollout error compounds even over 2 grouped steps and prefers tighter replanning. |
| 7 | `d25_hh1_lh3_bridge` | 50 | 1 | 3 | 5 | `900 / 20 / 10` | `300 / 30 / 150` | Clean bridge between `LH=2` and `LH=5` to locate where latent rollout drift begins. |
| 8 | `d25_hh1_lh2_highsearch_only` | 50 | 1 | 2 | 5 | `1200 / 30 / 20` | `300 / 30 / 150` | Isolates whether macro subgoal selection is the limiting factor rather than low-level execution. |
| 9 | `d25_hh1_lh2_hightopk30` | 50 | 1 | 2 | 5 | `900 / 20 / 30` | `300 / 30 / 150` | Useful if Hope2’s macro latent manifold is multi-modal and greedy top-k pruning drops good subgoals too early. |
| 10 | `d25_hh1_lh2_budget75` | 75 | 1 | 2 | 5 | `900 / 20 / 10` | `300 / 30 / 150` | Checks whether planning is already adequate at `d=25` and failures are mostly timeout/execution-budget related. |

### d=50

| prio | label | budget | HH | LH | H-replan | H samples/iters/topk | L samples/iters/topk | Why this makes sense for Hope2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `d50_hh1_lh2_pscaled` | 50 | 1 | 2 | 5 | `1500 / 40 / 10` | `900 / 20 / 150` | Best-supported medium-horizon baseline: keep macro structure simple and spend extra search budget. |
| 2 | `d50_hh2_lh2_pscaled` | 50 | 2 | 2 | 5 | `1500 / 40 / 10` | `900 / 20 / 150` | Natural test of whether Hope2 needs two macro stages for medium-range PushT goals. |
| 3 | `d50_hh2_lh2_searchboost` | 50 | 2 | 2 | 5 | `1500 / 40 / 20` | `900 / 40 / 200` | Strong test of a two-stage Hope2 decomposition under heavier search. |
| 4 | `d50_hh2_lh2_replan3_searchboost` | 50 | 2 | 2 | 3 | `1500 / 40 / 20` | `900 / 40 / 200` | Adds faster macro correction for the case where Hope2’s staged plans drift during longer execution. |
| 5 | `d50_hh3_lh2_midcompute` | 50 | 3 | 2 | 5 | `1200 / 30 / 10` | `300 / 30 / 150` | Tests whether Hope2 benefits from a deeper macro hierarchy at `d=50` without exploding compute. |
| 6 | `d50_hh1_lh2_searchboost` | 50 | 1 | 2 | 5 | `1500 / 40 / 20` | `900 / 40 / 200` | Keeps the simpler `HH=1` structure and only strengthens search, which is useful if execution is the real bottleneck. |
| 7 | `d50_hh1_lh2_replan3_searchboost` | 50 | 1 | 2 | 3 | `1500 / 40 / 20` | `900 / 40 / 200` | Combines simple macro intent with faster subgoal updates for a more reactive Hope2 controller. |
| 8 | `d50_hh1_lh1_searchboost` | 50 | 1 | 1 | 5 | `1500 / 40 / 20` | `900 / 40 / 200` | Worth testing if Hope2’s low-level execution gets unreliable over 2 grouped actions at medium horizon. |
| 9 | `d50_hh2_lh3_pscaled` | 50 | 2 | 3 | 5 | `1500 / 40 / 10` | `900 / 20 / 150` | Middle point between `LH=2` and `LH=5`, useful for measuring when longer low-level commitment starts hurting. |
| 10 | `d50_hh2_lh2_searchboost_budget75` | 75 | 2 | 2 | 5 | `1500 / 40 / 20` | `900 / 40 / 200` | Checks whether the Hope2 two-stage plan is reasonable but the standard budget is too tight to complete it. |

### d=75

| prio | label | budget | HH | LH | H-replan | H samples/iters/topk | L samples/iters/topk | Why this makes sense for Hope2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `d75_hh1_lh2_pscaled` | 150 | 1 | 2 | 5 | `1500 / 40 / 10` | `900 / 20 / 150` | Simplest long-horizon baseline: shallow macro planning with stronger search. |
| 2 | `d75_hh1_lh2_searchboost` | 150 | 1 | 2 | 5 | `1500 / 40 / 20` | `900 / 40 / 200` | Tests whether long-horizon Hope2 is mainly search-limited rather than structurally wrong. |
| 3 | `d75_hh1_lh2_replan3` | 150 | 1 | 2 | 3 | `1500 / 40 / 20` | `900 / 40 / 200` | More frequent macro replanning is especially sensible when stale long-horizon Hope2 intents may drift. |
| 4 | `d75_hh2_lh2_pscaled` | 150 | 2 | 2 | 5 | `1500 / 40 / 10` | `900 / 20 / 150` | First serious test of whether Hope2 needs two macro stages for the longest PushT horizon. |
| 5 | `d75_hh2_lh2_searchboost` | 150 | 2 | 2 | 5 | `1500 / 40 / 20` | `900 / 40 / 200` | Strongest existing direct test of a two-stage Hope2 decomposition at `d=75`. |
| 6 | `d75_hh1_lh1_searchboost` | 150 | 1 | 1 | 5 | `1200 / 30 / 20` | `600 / 40 / 200` | Useful if Hope2 low-level rollouts break down fastest on long tasks and need very short execution chunks. |
| 7 | `d75_hh3_lh2_midcompute` | 150 | 3 | 2 | 5 | `1200 / 30 / 10` | `300 / 30 / 150` | Directly probes whether Hope2 needs a deeper latent macro hierarchy for very long goals. |
| 8 | `d75_hh2_lh2_replan3_searchboost` | 150 | 2 | 2 | 3 | `1500 / 40 / 20` | `900 / 40 / 200` | High-probability “strong sensible” candidate: two macro stages, strong search, faster replanning. |
| 9 | `d75_hh3_lh2_pscaled` | 150 | 3 | 2 | 5 | `1500 / 40 / 10` | `900 / 20 / 150` | Tests the same `HH=3` structural idea as above, but with full paper-scaled search rather than mid-compute. |
| 10 | `d75_hh2_lh3_searchboost` | 150 | 2 | 3 | 5 | `1500 / 40 / 20` | `900 / 40 / 200` | Intermediate low horizon checks whether long-horizon Hope2 wants slightly longer low-level commitment than `LH=2`. |

## Notes

- The `d=25` recommendations are anchored by the short-horizon pattern in [PLANNING_HPARAM_RESULTS.md](/gpfs/home2/scur0200/main/jobs/eval/hi/PLANNING_HPARAM_RESULTS.md:33), where lower low-level horizon helped substantially.
- The `d=50` recommendations are anchored by the existing Hope2 wrappers in [hope2/README.md](/gpfs/home2/scur0200/main/jobs/eval/hi/hope2/README.md:8) and the current matrix outcomes in [fixed_stride_matrix_params.csv](/gpfs/home2/scur0200/main/jobs/eval/hi/matrix/fixed_stride_matrix_params.csv:1).
- The `d=75` recommendations are anchored by the existing `d75` matrix in [eval_hope2_d75_matrix.sh](/gpfs/home2/scur0200/main/jobs/eval/hi/d75/eval_hope2_d75_matrix.sh:1).
- `LOW_ACTION_BLOCK=5` is kept fixed because your eval guide treats that as the safe setting for this checkpoint family in [EVAL_CONFIG_GUIDE.md](/gpfs/home2/scur0200/main/jobs/eval/hi/EVAL_CONFIG_GUIDE.md:64).
