# Hierarchical PushT Executive Report

Date: 2026-05-09

## Mission

The project goal is to turn the frozen LeWorldModel baseline into a hierarchical controller for PushT:

- keep the upstream low-level visual world model and short-horizon action model
- add a trainable high-level model that predicts future waypoint latents
- use hierarchical planning at inference time so the high level proposes subgoals or macro-actions and the low level realizes them
- outperform or at least match flat planning on short-horizon PushT, then extend that to harder settings such as `d=50`

Operationally, the current question is not whether the representation can work at all. It can. The current question is whether the hierarchical system is robust enough to work outside a narrow “good” configuration.

## Architecture

The repo is split intentionally:

- frozen upstream baseline in `third_party/lewm`
- local hierarchical code in root-level `hi_*` files and `config/*/hi_*.yaml`

The core hierarchical model is `HiJEPA`:

- image encoder from the baseline
- baseline low-level predictor and low-level action encoder
- a new latent action encoder that converts an action chunk into a macro-action latent
- a new high-level autoregressive predictor that maps waypoint context latents plus macro-action latents to the next waypoint latent

Training works like this:

1. sample waypoint indices from a configured waypoint policy
2. encode the waypoint frames into latents
3. extract the action chunk between consecutive waypoints
4. encode each action chunk into a macro-action latent
5. predict the next waypoint latent from the current waypoint latent plus the macro-action latent
6. optimize high-level latent prediction loss

In the P2-only frozen runs, this is almost entirely a high-level training problem:

- low-level modules are loaded from a pretrained LeWM checkpoint
- encoder, low predictor, low-level action encoder, projector, and low-level prediction head are frozen
- only the high-level path is effectively trained
- loss is usually `alpha=0.0, beta=1.0`, so the main objective is the high-level next-waypoint latent loss

At eval time, `hi_eval.py` builds either:

- flat planning, or
- hierarchical planning with a high-level CEM solver over latent macro-actions and a low-level CEM solver over grouped actions

Important architectural fact:

- training macro-actions are dataset-derived chunks between sampled waypoints
- inference macro-actions are search-generated CEM candidates

That gap matters.

## Base Training Configuration

The default training config in `config/train/hi_lewm.yaml` is:

- `history_size=3`
- `embed_dim=192`
- default `latent_action_dim=192`
- waypoint training policy: `num=5`, `strategy=random_sorted`, `min_stride=1`, `max_span=15`
- `trainer.max_epochs=100` in config, but all concrete cluster jobs override this
- optimizer `AdamW`, lr `5e-5`, batch size `128`
- by default `train_low_level=False`

The PushT training dataset config is:

- dataset `pusht_expert_train`
- `frameskip=5`
- `num_steps = history_size + max_span = 18`

That means the standard random-sorted training recipe sees variable waypoint gaps inside a total future span of at most 15 steps.

## What We Trained

### 1. Historical hope1 family

The historical eval scripts and planning notes revolve around the earlier checkpoint family:

- `hi_lewm_p2_train_hope1_21983875`

Those logs show the first clear finding:

- flat planning was competitive
- default hierarchical planning was poor
- hierarchical planning improved a lot once the low-level horizon was reduced

There is also a later local W&B hope1 run:

- `hi_lewm_p2_train_hope1_22246065`

Its saved config shows:

- P2-only frozen training
- `latent_action_dim=192`
- `alpha=0.0`, `beta=1.0`

Its W&B summary ended around epoch 10 with:

- validation loss `0.0194`
- validation macro-action norm `29.0`
- mean waypoint gap `2.38`

I do not see matching local eval logs for this later `22246065` run, so it should not be conflated with the older `21983875` hope1 eval family.

### 2. hope2 family

Run:

- `hi_lewm_p2_train_hope2_22253175`

Training setup:

- P2-only frozen training
- `latent_action_dim=32`
- still variable-gap waypoint training (`random_sorted`, `num=5`, `max_span=15`)
- `max_epochs=15`
- `alpha=0.0`, `beta=1.0`

W&B summary at epoch 15:

- validation loss `0.0163`
- validation macro-action norm `14.1`
- mean waypoint gap `2.37`

This became the main post-hope1 evaluated family.

### 3. hope3 family

Run:

- `hi_lewm_p2_train_hope3_22515001`

Training setup:

- P2-only frozen training
- `latent_action_dim=8`
- same non-fixed waypoint recipe as hope2

The dedicated eval scripts describe this run as timed out, with the last available checkpoint at epoch 13.

### 4. Fixed-stride latent8 ablation

Run:

- `hi_lewm_p2_train_latent_action_dim_8_stride_5_n4_22518175`

Training setup:

- P2-only frozen training
- `latent_action_dim=8`
- explicit fixed-stride waypoint training
- `waypoints.num=4`
- `waypoints.strategy=fixed_stride`
- `waypoints.stride=5`
- total waypoint span kept at 15

This is the first training recipe that makes train-time waypoint gaps match the common eval-time grouped-action scale more directly.

W&B summary at epoch 15:

- validation loss `0.0330`
- validation macro-action norm `10.2`
- waypoint gap mean `5.0`

### 5. Fixed-stride latent32 ablation

Run:

- `hi_lewm_p2_train_latent_action_dim_32_stride_5_n4_22569364`

Training setup:

- P2-only frozen training
- `latent_action_dim=32`
- `waypoints.num=4`
- `waypoints.strategy=fixed_stride`
- `waypoints.stride=5`

This is the clean stride-5 ablation on top of the stronger dim-32 family.

W&B summary at epoch 15:

- validation loss `0.0327`
- validation macro-action norm `14.9`
- waypoint gap mean `5.0`

### 6. Joint end-to-end training

Runs:

- `hi_lewm_joint_22361354`
- `hi_lewm_joint_22368719`

Joint training changes the regime substantially:

- `training.train_low_level=True`
- nothing is frozen in the low-level path
- `alpha=1.0`, `beta=1.0`
- `loss.sigreg.weight=0.2`
- micro-batch `16` with gradient accumulation `8`
- target effective batch size `128`

What happened:

- `22361354` failed immediately with CUDA OOM
- `22368719` resumed successfully and reached epoch 25

W&B summary for the successful resumed run:

- validation loss `0.2201`
- validation `l1_pred_loss=0.00095`
- validation `l2_pred_loss=0.00547`
- validation `sigreg_loss=1.068`

Important status note:

- I do not find local eval output logs for the joint run yet
- eval scripts exist for `d=25` and `d=50`, but I cannot confirm local results

## Evaluation Configurations We Tested

### Task difficulty

- `d=25`: short horizon, goal 25 steps ahead
- `d=50`: longer horizon, goal 50 steps ahead
- budget usually `50`, with some older ablations at `30` and `75`

### Planning modes

- flat planning
- hierarchical planning
- staged hierarchical diagnostic mode was prepared, but I do not see local output logs

### High-level planner ablations

- horizon `1`, `2`, `3`, and historical `4`
- high replan interval `5`, plus an older `3`
- samples/iters/topk ranging from:
  - `900/20/10`
  - `1200/30/10`
  - `1500/40/10`
  - searchboost variants with larger sample counts and `topk`

### Low-level planner ablations

- low horizon `1`, `2`, `3`, `5`
- low receding horizon `1` and `5`
- low samples/iters/topk from `300/30/30` up to `900/40/200`

### Safety and compatibility finding

- `LOW_ACTION_BLOCK=5` is effectively fixed for the current checkpoint families
- changing it to `3` caused a channel mismatch crash because the low-level action encoder expects grouped actions with size `10 = 5 * 2`

## Ablations We Completed

### Training ablations

- P2-only frozen vs joint end-to-end
- `latent_action_dim`: `192`, `32`, `8`
- variable waypoint gaps (`random_sorted`) vs fixed-stride training (`stride=5`, `N=4`)

### Evaluation ablations

- flat vs hierarchical
- short horizon `d=25` vs longer horizon `d=50`
- high-level horizon sweeps
- low-level horizon sweeps
- CEM search budget sweeps
- budget sweeps (`30`, `50`, `75`)
- high replan interval sweep (`5` vs `3`)

## Results So Far

### A. Historical hope1 result: the first important lesson

From the archived planning summary:

- default hierarchical was bad: `14%` at `d=25`, `10%` at `d=50`
- flat planning at `d=25` reached `62%`
- the best historical hierarchical result reached `84%` at `d=25`
- the main improvement came from reducing low-level horizon from `5` to `2`

That is the first strong sign that the hierarchy is not uniformly broken; it is highly sensitive to planner shape.

### B. hope2 dedicated evals

For `hi_lewm_p2_train_hope2_22253175`:

- `d=25, hh1, lh2, lrh1` -> `84%`
- `d=25, hh2, lh2, lrh1` -> `20%`
- `d=25, hh2` with searchboost -> `34%`
- `d=25, hh1, lh2` with replan interval `3` -> `84%`
- `d=50, hh1, lh2, lrh1` -> `44%`
- `d=50, hh2, lh2, lrh1` -> roughly `30-34%` depending exact search budget
- `d=50, hh3, lh2` -> `24%`
- `d=50` searchboost on `hh2` did not help materially -> `24%`

Interpretation:

- `hh1/lh2` is strong
- pushing the high-level horizon beyond `1` hurts fast
- adding more search does not fix the failure mode

### C. Dedicated hope3 and stride5n4 evals

- `hope3` epoch 13: `40%` at `d=25`, `30%` at `d=50`
- `latent8_stride5n4` dedicated `d=50` run: `28%`

So the first latent-dim-8 attempt underperformed the stronger dim-32 family, and fixed-stride training did not immediately solve `d=50`.

### D. Current matrix results

The current matrix logs in `jobs/eval/hi/matrix/logs` give:

#### `d = 25`

| Config | hope2 | latent8 stride5 n4 | latent32 stride5 n4 |
| --- | ---: | ---: | ---: |
| `hh1_lh2_lrh1` | 84 | 84 | 90 |
| `hh1_lh5_lrh1` | 44 | 44 | 42 |
| `hh2_lh2_lrh1` | 20 | not found | not found |

#### `d = 50`

| Config | hope2 | latent8 stride5 n4 | latent32 stride5 n4 |
| --- | ---: | ---: | ---: |
| `hh1_lh2_lrh1` | 44 | 38 | 40 |
| `hh2_lh2_lrh1` | 32 | 20 | 26 |
| `hh2_lh5_lrh1` | 14 | 14 | 20 |
| `hh2_lh5_lrh5` | 12 | 8 | 6 |

Important matrix caveats:

- the current `checkpoints_fixed_stride.txt` only lists `hope2`, so the latent8/latent32 logs are historical matrix outputs rather than part of the current checkpoint file
- I do not find the `d25 hh2_lh2_lrh1` matrix cell for the two fixed-stride families in local logs

## What These Results Mean

### The good news

- the hierarchy can work very well on short-horizon PushT
- the best observed short-horizon numbers are strong: `84-90%`
- fixed-stride training with `latent_action_dim=32` looks like the strongest current short-horizon family

### The bad news

The system is still brittle and incomplete.

1. Performance collapses when the hierarchy is asked to plan farther ahead.

- `d=50` is consistently much weaker than `d=25`
- even the best `d=50` results are only around `40-44%`

2. Performance collapses when the high-level horizon is increased.

- on `hope2`, going from `hh1` to `hh2` at `d=25` drops `84 -> 20`
- on `d=50`, higher high-level horizons also hurt

That strongly suggests compounding error in high-level rollout or poor subgoal quality, not just insufficient search compute.

3. Longer low-level open-loop rollouts are fragile.

- `lh2` is much better than `lh5`
- `lh5` pushes the controller into error accumulation before it replans

4. More search budget is not the main answer.

- searchboost helped a little in some places, but not enough
- at `d=50`, bigger search often failed to recover the performance gap

That means the bottleneck is likely representational or train/eval mismatch, not only solver compute.

## Why The Results Are Bad

This is my best grounded diagnosis.

### 1. Train/eval mismatch on macro-actions

The early strong families (`hope1`, `hope2`, `hope3`) train with:

- variable waypoint gaps
- variable-length action chunks between waypoints

But eval typically assumes:

- fixed low-level action block `5`
- latent prior calibration with fixed `chunk_len=5`
- a CEM search over macro-actions that were never directly sampled from the train distribution

This mismatch is explicitly acknowledged in the repo notes. Training teaches the high-level model to predict latents generated from dataset chunks. Eval asks it to optimize over search-generated chunks. That is a distribution shift.

### 2. P2-only frozen training optimizes latent prediction, not closed-loop control

For the main evaluated families:

- low-level is frozen
- objective is almost entirely `beta * l2_pred_loss`
- there is no direct control objective

So good validation latent loss can coexist with weak closed-loop planning.

### 3. High-level horizon > 1 is beyond the stable operating region

The pattern is too consistent to ignore:

- `hh1` often works
- `hh2+` often degrades badly

That means the high-level model is not yet reliable enough for multi-step latent MPC over longer horizons.

### 4. `d=50` is still outside the hierarchy’s comfort zone

At `d=50`, the planner needs:

- better subgoal quality
- better long-horizon consistency
- or a better coupling between high-level and low-level dynamics

Right now it has neither enough robustness nor enough representational alignment.

## Tests We Completed Versus Tests That Are Only Prepared

### Completed and evidenced by local logs

- historical hope1 planning sweeps
- flat vs hierarchical comparisons
- low-level horizon sweeps
- budget sweeps
- searchboost sweeps
- dedicated hope2 evals
- dedicated hope3 evals
- dedicated latent8 stride-5 evals
- matrix sweep logs for hope2, latent8 stride-5, and latent32 stride-5
- multiple frozen-P2 training runs
- one successful resumed joint training run

### Prepared in the repo, but I do not find local output logs

- joint `d=25` eval for `hi_lewm_joint_22368719`
- joint `d=50` eval for `hi_lewm_joint_22368719`
- staged high-plan diagnostic
- macro-action manifold diagnostic

These are important because they are exactly the tests needed to validate the current diagnosis.

## Executive Assessment

The project has reached a meaningful intermediate milestone, but not the mission finish line.

- The hierarchy is real: it can hit `84-90%` on short-horizon PushT.
- The current best family appears to be `latent_action_dim=32` with fixed-stride training.
- The hierarchy is not robust: it breaks under longer horizons, longer low-level plans, and multi-step high-level planning.
- The dominant failure mode looks structural, not just “needs more samples.”

In plain terms:

- short-horizon hierarchical control works
- long-horizon hierarchical control still does not
- the main unresolved issue is train/eval mismatch in the macro-action abstraction, plus compounding error when the high-level planner is asked to do more than one reliable step

## Recommended Next Moves

1. Evaluate `hi_lewm_joint_22368719` immediately at `d=25` and `d=50`.
   This is the most important missing result in the local record.

2. Treat `latent32_stride5_n4` with `d25, hh1, lh2` as the current short-horizon best model.

3. Do not spend more time on `hh2+` sweeps until the diagnostic tests explain why high-level multi-step rollout is collapsing.

4. Run the prepared diagnostics:
   - `macro_action_manifold_cpu.sh`
   - `staged_high_plan_cpu.sh`

5. Align eval-time latent prior calibration with the train-time chunk distribution.
   The repo already notes that fixed `chunk_len=5` calibration is only an approximation for variable-gap training.

6. If long-horizon performance remains poor, bias future training toward eval-consistent waypoint structure.
   The fixed-stride `N=4, stride=5` family is the right direction, especially with `latent_action_dim=32`.
