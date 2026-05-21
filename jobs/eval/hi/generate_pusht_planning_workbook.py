#!/usr/bin/env python3
"""Generate an Excel workbook summarizing current and proposed PushT planning configs."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape


HERE = Path(__file__).resolve().parent
MATRIX_DIR = HERE / "matrix"
OUTPUT_PATH = HERE / "pusht_planning_config_matrix.xlsx"


def parse_active_checkpoints(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        rows.append(
            {
                "run_name": parts[0],
                "checkpoint_epoch": parts[1],
            }
        )
    return rows


def read_historical_success(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return {
            row["label"]: row.get("success_rate", "")
            for row in reader
            if row.get("label")
        }


CURRENT_MATRIX_CONFIGS = [
    dict(
        config_index=1,
        label="d25_hh1_lh2_lrh1",
        env_name="PushT",
        goal_offset_steps_d=25,
        eval_budget=50,
        high_horizon=1,
        low_horizon=2,
        high_receding_horizon=1,
        low_receding_horizon=1,
        high_replan_interval=5,
        high_action_block=1,
        low_action_block=5,
        high_num_samples=900,
        high_n_steps=20,
        high_topk=10,
        low_num_samples=300,
        low_n_steps=30,
        low_topk=150,
    ),
    dict(
        config_index=2,
        label="d25_hh1_lh5_lrh1",
        env_name="PushT",
        goal_offset_steps_d=25,
        eval_budget=50,
        high_horizon=1,
        low_horizon=5,
        high_receding_horizon=1,
        low_receding_horizon=1,
        high_replan_interval=5,
        high_action_block=1,
        low_action_block=5,
        high_num_samples=900,
        high_n_steps=20,
        high_topk=10,
        low_num_samples=300,
        low_n_steps=30,
        low_topk=150,
    ),
    dict(
        config_index=3,
        label="d25_hh2_lh2_lrh1",
        env_name="PushT",
        goal_offset_steps_d=25,
        eval_budget=50,
        high_horizon=2,
        low_horizon=2,
        high_receding_horizon=1,
        low_receding_horizon=1,
        high_replan_interval=5,
        high_action_block=1,
        low_action_block=5,
        high_num_samples=900,
        high_n_steps=20,
        high_topk=10,
        low_num_samples=300,
        low_n_steps=30,
        low_topk=150,
    ),
    dict(
        config_index=4,
        label="d50_hh1_lh2_lrh1",
        env_name="PushT",
        goal_offset_steps_d=50,
        eval_budget=50,
        high_horizon=1,
        low_horizon=2,
        high_receding_horizon=1,
        low_receding_horizon=1,
        high_replan_interval=5,
        high_action_block=1,
        low_action_block=5,
        high_num_samples=1500,
        high_n_steps=40,
        high_topk=10,
        low_num_samples=900,
        low_n_steps=20,
        low_topk=150,
    ),
    dict(
        config_index=5,
        label="d50_hh2_lh2_lrh1",
        env_name="PushT",
        goal_offset_steps_d=50,
        eval_budget=50,
        high_horizon=2,
        low_horizon=2,
        high_receding_horizon=1,
        low_receding_horizon=1,
        high_replan_interval=5,
        high_action_block=1,
        low_action_block=5,
        high_num_samples=1500,
        high_n_steps=40,
        high_topk=10,
        low_num_samples=900,
        low_n_steps=20,
        low_topk=150,
    ),
    dict(
        config_index=6,
        label="d50_hh2_lh5_lrh1",
        env_name="PushT",
        goal_offset_steps_d=50,
        eval_budget=50,
        high_horizon=2,
        low_horizon=5,
        high_receding_horizon=1,
        low_receding_horizon=1,
        high_replan_interval=5,
        high_action_block=1,
        low_action_block=5,
        high_num_samples=1500,
        high_n_steps=40,
        high_topk=10,
        low_num_samples=900,
        low_n_steps=20,
        low_topk=150,
    ),
    dict(
        config_index=7,
        label="d50_hh2_lh5_lrh5",
        env_name="PushT",
        goal_offset_steps_d=50,
        eval_budget=50,
        high_horizon=2,
        low_horizon=5,
        high_receding_horizon=1,
        low_receding_horizon=5,
        high_replan_interval=5,
        high_action_block=1,
        low_action_block=5,
        high_num_samples=1500,
        high_n_steps=40,
        high_topk=10,
        low_num_samples=900,
        low_n_steps=20,
        low_topk=150,
    ),
]


def proposed_configs() -> list[dict[str, object]]:
    common = {
        "env_name": "PushT",
        "high_receding_horizon": 1,
        "low_receding_horizon": 1,
        "high_action_block": 1,
        "low_action_block": 5,
    }
    return [
        {
            **common,
            "priority": 1,
            "d": 25,
            "label": "d25_hh1_lh2_bestprev",
            "eval_budget": 50,
            "high_horizon": 1,
            "low_horizon": 2,
            "high_replan_interval": 5,
            "high_num_samples": 900,
            "high_n_steps": 20,
            "high_topk": 10,
            "low_num_samples": 300,
            "low_n_steps": 30,
            "low_topk": 150,
            "source_family": "existing Hope2 wrapper",
            "why_this_makes_sense_for_hope2": "Direct carryover of the strongest short-horizon hierarchical pattern: Hope2 keeps macro planning shallow while the low-level planner only has to realize short 2-token latent segments.",
        },
        {
            **common,
            "priority": 2,
            "d": 25,
            "label": "d25_hh1_lh2_replan3",
            "eval_budget": 50,
            "high_horizon": 1,
            "low_horizon": 2,
            "high_replan_interval": 3,
            "high_num_samples": 900,
            "high_n_steps": 20,
            "high_topk": 10,
            "low_num_samples": 300,
            "low_n_steps": 30,
            "low_topk": 150,
            "source_family": "existing Hope2 wrapper",
            "why_this_makes_sense_for_hope2": "More frequent macro replanning is sensible when Hope2 subgoals are useful but can drift after a few environment steps; this isolates reactivity without changing the low planner.",
        },
        {
            **common,
            "priority": 3,
            "d": 25,
            "label": "d25_hh1_lh2_searchboost",
            "eval_budget": 50,
            "high_horizon": 1,
            "low_horizon": 2,
            "high_replan_interval": 5,
            "high_num_samples": 900,
            "high_n_steps": 20,
            "high_topk": 10,
            "low_num_samples": 600,
            "low_n_steps": 40,
            "low_topk": 200,
            "source_family": "existing Hope2 wrapper",
            "why_this_makes_sense_for_hope2": "If Hope2 already proposes reachable short subgoals, extra low-level CEM depth should help execute them more reliably without increasing macro complexity.",
        },
        {
            **common,
            "priority": 4,
            "d": 25,
            "label": "d25_hh2_lh2_baseline",
            "eval_budget": 50,
            "high_horizon": 2,
            "low_horizon": 2,
            "high_replan_interval": 5,
            "high_num_samples": 900,
            "high_n_steps": 20,
            "high_topk": 10,
            "low_num_samples": 300,
            "low_n_steps": 30,
            "low_topk": 150,
            "source_family": "existing Hope2 wrapper",
            "why_this_makes_sense_for_hope2": "Two macro stages test whether Hope2 benefits from explicitly partitioning even a short PushT goal into consecutive latent subgoals rather than committing to a single macro target.",
        },
        {
            **common,
            "priority": 5,
            "d": 25,
            "label": "d25_hh2_lh2_searchboost",
            "eval_budget": 50,
            "high_horizon": 2,
            "low_horizon": 2,
            "high_replan_interval": 5,
            "high_num_samples": 1200,
            "high_n_steps": 30,
            "high_topk": 20,
            "low_num_samples": 600,
            "low_n_steps": 40,
            "low_topk": 200,
            "source_family": "existing Hope2 wrapper",
            "why_this_makes_sense_for_hope2": "This separates 'H=2 is bad' from 'H=2 needs more search': if Hope2's second macro stage is useful but harder to optimize, the stronger CEM should expose that.",
        },
        {
            **common,
            "priority": 6,
            "d": 25,
            "label": "d25_hh1_lh1_searchboost",
            "eval_budget": 50,
            "high_horizon": 1,
            "low_horizon": 1,
            "high_replan_interval": 5,
            "high_num_samples": 1200,
            "high_n_steps": 30,
            "high_topk": 20,
            "low_num_samples": 600,
            "low_n_steps": 40,
            "low_topk": 200,
            "source_family": "existing Hope2 wrapper",
            "why_this_makes_sense_for_hope2": "A 1-token low horizon is a direct stress test of the Hope2 low-level decoder: if rollout error compounds quickly, shorter execution chunks may outperform longer open-loop segments.",
        },
        {
            **common,
            "priority": 7,
            "d": 25,
            "label": "d25_hh1_lh3_bridge",
            "eval_budget": 50,
            "high_horizon": 1,
            "low_horizon": 3,
            "high_replan_interval": 5,
            "high_num_samples": 900,
            "high_n_steps": 20,
            "high_topk": 10,
            "low_num_samples": 300,
            "low_n_steps": 30,
            "low_topk": 150,
            "source_family": "new targeted ablation",
            "why_this_makes_sense_for_hope2": "This is the clean midpoint between the successful `lh=2` and weak `lh=5` settings, useful for locating where Hope2's low-level latent rollout starts to drift.",
        },
        {
            **common,
            "priority": 8,
            "d": 25,
            "label": "d25_hh1_lh2_highsearch_only",
            "eval_budget": 50,
            "high_horizon": 1,
            "low_horizon": 2,
            "high_replan_interval": 5,
            "high_num_samples": 1200,
            "high_n_steps": 30,
            "high_topk": 20,
            "low_num_samples": 300,
            "low_n_steps": 30,
            "low_topk": 150,
            "source_family": "new targeted ablation",
            "why_this_makes_sense_for_hope2": "This isolates whether the limiting factor is macro subgoal selection rather than low-level execution, which matters for a hierarchical latent model like Hope2.",
        },
        {
            **common,
            "priority": 9,
            "d": 25,
            "label": "d25_hh1_lh2_hightopk30",
            "eval_budget": 50,
            "high_horizon": 1,
            "low_horizon": 2,
            "high_replan_interval": 5,
            "high_num_samples": 900,
            "high_n_steps": 20,
            "high_topk": 30,
            "low_num_samples": 300,
            "low_n_steps": 30,
            "low_topk": 150,
            "source_family": "new targeted ablation",
            "why_this_makes_sense_for_hope2": "Keeping more macro candidates can help if Hope2's latent macro manifold is multi-modal and the best subgoal is being pruned too early by a very greedy high-level CEM.",
        },
        {
            **common,
            "priority": 10,
            "d": 25,
            "label": "d25_hh1_lh2_budget75",
            "eval_budget": 75,
            "high_horizon": 1,
            "low_horizon": 2,
            "high_replan_interval": 5,
            "high_num_samples": 900,
            "high_n_steps": 20,
            "high_topk": 10,
            "low_num_samples": 300,
            "low_n_steps": 30,
            "low_topk": 150,
            "source_family": "new evaluation ablation",
            "why_this_makes_sense_for_hope2": "This checks whether Hope2 is already planning well at `d=25` but occasionally times out during execution, which is a different failure mode from choosing bad subgoals.",
        },
        {
            **common,
            "priority": 1,
            "d": 50,
            "label": "d50_hh1_lh2_pscaled",
            "eval_budget": 50,
            "high_horizon": 1,
            "low_horizon": 2,
            "high_replan_interval": 5,
            "high_num_samples": 1500,
            "high_n_steps": 40,
            "high_topk": 10,
            "low_num_samples": 900,
            "low_n_steps": 20,
            "low_topk": 150,
            "source_family": "existing Hope2 wrapper",
            "why_this_makes_sense_for_hope2": "This is the best-supported medium-horizon baseline: keep macro planning simple, but allocate more CEM compute because longer-goal Hope2 search is harder than `d=25`.",
        },
        {
            **common,
            "priority": 2,
            "d": 50,
            "label": "d50_hh2_lh2_pscaled",
            "eval_budget": 50,
            "high_horizon": 2,
            "low_horizon": 2,
            "high_replan_interval": 5,
            "high_num_samples": 1500,
            "high_n_steps": 40,
            "high_topk": 10,
            "low_num_samples": 900,
            "low_n_steps": 20,
            "low_topk": 150,
            "source_family": "existing Hope2 wrapper",
            "why_this_makes_sense_for_hope2": "At `d=50`, two macro stages are structurally plausible for Hope2 because the model may need one subgoal to line up the object and a second to finish control near the goal.",
        },
        {
            **common,
            "priority": 3,
            "d": 50,
            "label": "d50_hh2_lh2_searchboost",
            "eval_budget": 50,
            "high_horizon": 2,
            "low_horizon": 2,
            "high_replan_interval": 5,
            "high_num_samples": 1500,
            "high_n_steps": 40,
            "high_topk": 20,
            "low_num_samples": 900,
            "low_n_steps": 40,
            "low_topk": 200,
            "source_family": "existing Hope2 wrapper",
            "why_this_makes_sense_for_hope2": "This is the right follow-up when `hh=2` has the right decomposition bias but seems search-limited; both macro and micro CEMs get more room to refine latent plans.",
        },
        {
            **common,
            "priority": 4,
            "d": 50,
            "label": "d50_hh2_lh2_replan3_searchboost",
            "eval_budget": 50,
            "high_horizon": 2,
            "low_horizon": 2,
            "high_replan_interval": 3,
            "high_num_samples": 1500,
            "high_n_steps": 40,
            "high_topk": 20,
            "low_num_samples": 900,
            "low_n_steps": 40,
            "low_topk": 200,
            "source_family": "existing Hope2 wrapper",
            "why_this_makes_sense_for_hope2": "For longer PushT problems, faster macro refresh can correct accumulated subgoal error before the low-level controller commits too far to a stale Hope2 plan.",
        },
        {
            **common,
            "priority": 5,
            "d": 50,
            "label": "d50_hh3_lh2_midcompute",
            "eval_budget": 50,
            "high_horizon": 3,
            "low_horizon": 2,
            "high_replan_interval": 5,
            "high_num_samples": 1200,
            "high_n_steps": 30,
            "high_topk": 10,
            "low_num_samples": 300,
            "low_n_steps": 30,
            "low_topk": 150,
            "source_family": "existing Hope2 wrapper",
            "why_this_makes_sense_for_hope2": "Three macro stages explicitly test whether Hope2 benefits from a more granular latent decomposition at `d=50`, while using moderate compute to keep the search problem tractable.",
        },
        {
            **common,
            "priority": 6,
            "d": 50,
            "label": "d50_hh1_lh2_searchboost",
            "eval_budget": 50,
            "high_horizon": 1,
            "low_horizon": 2,
            "high_replan_interval": 5,
            "high_num_samples": 1500,
            "high_n_steps": 40,
            "high_topk": 20,
            "low_num_samples": 900,
            "low_n_steps": 40,
            "low_topk": 200,
            "source_family": "new targeted ablation",
            "why_this_makes_sense_for_hope2": "If the current medium-horizon weakness is mainly low-level execution quality, this keeps the empirically stronger `hh=1` structure and only makes the search more robust.",
        },
        {
            **common,
            "priority": 7,
            "d": 50,
            "label": "d50_hh1_lh2_replan3_searchboost",
            "eval_budget": 50,
            "high_horizon": 1,
            "low_horizon": 2,
            "high_replan_interval": 3,
            "high_num_samples": 1500,
            "high_n_steps": 40,
            "high_topk": 20,
            "low_num_samples": 900,
            "low_n_steps": 40,
            "low_topk": 200,
            "source_family": "new targeted ablation",
            "why_this_makes_sense_for_hope2": "This combines the simpler one-stage macro plan with faster subgoal updates, which is reasonable if Hope2 can choose good coarse intents but needs frequent closed-loop correction.",
        },
        {
            **common,
            "priority": 8,
            "d": 50,
            "label": "d50_hh1_lh1_searchboost",
            "eval_budget": 50,
            "high_horizon": 1,
            "low_horizon": 1,
            "high_replan_interval": 5,
            "high_num_samples": 1500,
            "high_n_steps": 40,
            "high_topk": 20,
            "low_num_samples": 900,
            "low_n_steps": 40,
            "low_topk": 200,
            "source_family": "new targeted ablation",
            "why_this_makes_sense_for_hope2": "A shorter low horizon is worth testing if the Hope2 low-level rollout becomes unreliable over 2 grouped actions once the task horizon grows to `d=50`.",
        },
        {
            **common,
            "priority": 9,
            "d": 50,
            "label": "d50_hh2_lh3_pscaled",
            "eval_budget": 50,
            "high_horizon": 2,
            "low_horizon": 3,
            "high_replan_interval": 5,
            "high_num_samples": 1500,
            "high_n_steps": 40,
            "high_topk": 10,
            "low_num_samples": 900,
            "low_n_steps": 20,
            "low_topk": 150,
            "source_family": "new targeted ablation",
            "why_this_makes_sense_for_hope2": "This is the natural bridge between the promising `lh=2` family and the weak `lh=5` family, and it tells you whether medium-horizon Hope2 wants slightly longer low-level commitment but not full `lh=5`.",
        },
        {
            **common,
            "priority": 10,
            "d": 50,
            "label": "d50_hh2_lh2_searchboost_budget75",
            "eval_budget": 75,
            "high_horizon": 2,
            "low_horizon": 2,
            "high_replan_interval": 5,
            "high_num_samples": 1500,
            "high_n_steps": 40,
            "high_topk": 20,
            "low_num_samples": 900,
            "low_n_steps": 40,
            "low_topk": 200,
            "source_family": "new evaluation ablation",
            "why_this_makes_sense_for_hope2": "This checks whether Hope2 is producing acceptable two-stage plans at `d=50` but failing because the standard budget is too tight to finish execution cleanly.",
        },
        {
            **common,
            "priority": 1,
            "d": 75,
            "label": "d75_hh1_lh2_pscaled",
            "eval_budget": 150,
            "high_horizon": 1,
            "low_horizon": 2,
            "high_replan_interval": 5,
            "high_num_samples": 1500,
            "high_n_steps": 40,
            "high_topk": 10,
            "low_num_samples": 900,
            "low_n_steps": 20,
            "low_topk": 150,
            "source_family": "existing d75 matrix",
            "why_this_makes_sense_for_hope2": "This is the simplest long-horizon baseline: keep macro structure shallow and rely on heavier search, which is appropriate when you do not yet know whether Hope2 needs more macro stages at `d=75`.",
        },
        {
            **common,
            "priority": 2,
            "d": 75,
            "label": "d75_hh1_lh2_searchboost",
            "eval_budget": 150,
            "high_horizon": 1,
            "low_horizon": 2,
            "high_replan_interval": 5,
            "high_num_samples": 1500,
            "high_n_steps": 40,
            "high_topk": 20,
            "low_num_samples": 900,
            "low_n_steps": 40,
            "low_topk": 200,
            "source_family": "existing d75 matrix",
            "why_this_makes_sense_for_hope2": "If long-horizon Hope2 is primarily search-limited rather than structurally wrong, this should dominate the plain paper-scaled baseline without changing the macro design.",
        },
        {
            **common,
            "priority": 3,
            "d": 75,
            "label": "d75_hh1_lh2_replan3",
            "eval_budget": 150,
            "high_horizon": 1,
            "low_horizon": 2,
            "high_replan_interval": 3,
            "high_num_samples": 1500,
            "high_n_steps": 40,
            "high_topk": 20,
            "low_num_samples": 900,
            "low_n_steps": 40,
            "low_topk": 200,
            "source_family": "existing d75 matrix",
            "why_this_makes_sense_for_hope2": "At `d=75`, replanning more often is a natural Hope2 test because stale macro intents are more likely to hurt over a long execution window.",
        },
        {
            **common,
            "priority": 4,
            "d": 75,
            "label": "d75_hh2_lh2_pscaled",
            "eval_budget": 150,
            "high_horizon": 2,
            "low_horizon": 2,
            "high_replan_interval": 5,
            "high_num_samples": 1500,
            "high_n_steps": 40,
            "high_topk": 10,
            "low_num_samples": 900,
            "low_n_steps": 20,
            "low_topk": 150,
            "source_family": "existing d75 matrix",
            "why_this_makes_sense_for_hope2": "Two macro stages are the first serious long-range decomposition test for Hope2; this tells you whether a single macro subgoal is too coarse at `d=75`.",
        },
        {
            **common,
            "priority": 5,
            "d": 75,
            "label": "d75_hh2_lh2_searchboost",
            "eval_budget": 150,
            "high_horizon": 2,
            "low_horizon": 2,
            "high_replan_interval": 5,
            "high_num_samples": 1500,
            "high_n_steps": 40,
            "high_topk": 20,
            "low_num_samples": 900,
            "low_n_steps": 40,
            "low_topk": 200,
            "source_family": "existing d75 matrix",
            "why_this_makes_sense_for_hope2": "This is the strongest direct test of a two-stage Hope2 decomposition before moving to even longer macro horizons.",
        },
        {
            **common,
            "priority": 6,
            "d": 75,
            "label": "d75_hh1_lh1_searchboost",
            "eval_budget": 150,
            "high_horizon": 1,
            "low_horizon": 1,
            "high_replan_interval": 5,
            "high_num_samples": 1200,
            "high_n_steps": 30,
            "high_topk": 20,
            "low_num_samples": 600,
            "low_n_steps": 40,
            "low_topk": 200,
            "source_family": "existing d75 matrix",
            "why_this_makes_sense_for_hope2": "Very short low-level plans are worth testing at `d=75` if Hope2's low-level rollout quality deteriorates fastest on long tasks and needs aggressive closed-loop correction.",
        },
        {
            **common,
            "priority": 7,
            "d": 75,
            "label": "d75_hh3_lh2_midcompute",
            "eval_budget": 150,
            "high_horizon": 3,
            "low_horizon": 2,
            "high_replan_interval": 5,
            "high_num_samples": 1200,
            "high_n_steps": 30,
            "high_topk": 10,
            "low_num_samples": 300,
            "low_n_steps": 30,
            "low_topk": 150,
            "source_family": "existing d75 matrix",
            "why_this_makes_sense_for_hope2": "Three macro stages explicitly probe whether Hope2 needs a deeper latent hierarchy to solve the longest PushT setting, while still using manageable solver compute.",
        },
        {
            **common,
            "priority": 8,
            "d": 75,
            "label": "d75_hh2_lh2_replan3_searchboost",
            "eval_budget": 150,
            "high_horizon": 2,
            "low_horizon": 2,
            "high_replan_interval": 3,
            "high_num_samples": 1500,
            "high_n_steps": 40,
            "high_topk": 20,
            "low_num_samples": 900,
            "low_n_steps": 40,
            "low_topk": 200,
            "source_family": "new targeted ablation",
            "why_this_makes_sense_for_hope2": "This is the high-probability 'strongest sensible' long-range candidate: two macro stages, strong search, and frequent replanning together target the main hierarchical failure modes.",
        },
        {
            **common,
            "priority": 9,
            "d": 75,
            "label": "d75_hh3_lh2_pscaled",
            "eval_budget": 150,
            "high_horizon": 3,
            "low_horizon": 2,
            "high_replan_interval": 5,
            "high_num_samples": 1500,
            "high_n_steps": 40,
            "high_topk": 10,
            "low_num_samples": 900,
            "low_n_steps": 20,
            "low_topk": 150,
            "source_family": "new targeted ablation",
            "why_this_makes_sense_for_hope2": "If `hh=3` is the right macro granularity, the existing mid-compute version may under-search; this variant tests the same structural idea with full paper-scaled compute.",
        },
        {
            **common,
            "priority": 10,
            "d": 75,
            "label": "d75_hh2_lh3_searchboost",
            "eval_budget": 150,
            "high_horizon": 2,
            "low_horizon": 3,
            "high_replan_interval": 5,
            "high_num_samples": 1500,
            "high_n_steps": 40,
            "high_topk": 20,
            "low_num_samples": 900,
            "low_n_steps": 40,
            "low_topk": 200,
            "source_family": "new targeted ablation",
            "why_this_makes_sense_for_hope2": "This intermediate low horizon checks whether long-horizon Hope2 needs slightly longer low-level commitment than `lh=2`, but not the much weaker `lh=5` style rollout length.",
        },
    ]


def col_letter(index: int) -> str:
    result = []
    n = index
    while n > 0:
        n, rem = divmod(n - 1, 26)
        result.append(chr(65 + rem))
    return "".join(reversed(result))


def xml_cell(row_idx: int, col_idx: int, value: object, style_id: int = 0) -> str:
    ref = f"{col_letter(col_idx)}{row_idx}"
    style_attr = f' s="{style_id}"' if style_id else ""
    if value is None:
        return f'<c r="{ref}"{style_attr} t="inlineStr"><is><t></t></is></c>'
    if isinstance(value, bool):
        return f'<c r="{ref}"{style_attr} t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"{style_attr}><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}"{style_attr} t="inlineStr"><is><t>{text}</t></is></c>'


def build_sheet_xml(
    columns: list[str],
    rows: Iterable[dict[str, object]],
    widths: dict[str, int] | None = None,
    wrap_columns: set[str] | None = None,
) -> str:
    widths = widths or {}
    wrap_columns = wrap_columns or set()
    col_xml = []
    for idx, col in enumerate(columns, start=1):
        width = widths.get(col, max(12, min(60, len(col) + 2)))
        col_xml.append(
            f'<col min="{idx}" max="{idx}" width="{width}" customWidth="1"/>'
        )
    row_xml = []
    header_cells = [xml_cell(1, idx, col, style_id=1) for idx, col in enumerate(columns, start=1)]
    row_xml.append(f'<row r="1">{"".join(header_cells)}</row>')
    for row_idx, row in enumerate(rows, start=2):
        cells = []
        for col_idx, col in enumerate(columns, start=1):
            style_id = 2 if col in wrap_columns else 0
            cells.append(xml_cell(row_idx, col_idx, row.get(col, ""), style_id=style_id))
        row_xml.append(f'<row r="{row_idx}">{"".join(cells)}</row>')
    last_col = col_letter(len(columns))
    last_row = len(row_xml)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews>'
        '<sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '</sheetView>'
        '</sheetViews>'
        f'<dimension ref="A1:{last_col}{last_row}"/>'
        f'<cols>{"".join(col_xml)}</cols>'
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        f'<autoFilter ref="A1:{last_col}{last_row}"/>'
        '</worksheet>'
    )


def write_workbook(output_path: Path, sheets: list[tuple[str, str]]) -> None:
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for idx in range(1, len(sheets) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    workbook_xml = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        "<sheets>",
    ]
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
    ]
    for idx, (name, _) in enumerate(sheets, start=1):
        escaped_name = escape(name)
        workbook_xml.append(
            f'<sheet name="{escaped_name}" sheetId="{idx}" r:id="rId{idx + 1}"/>'
        )
        workbook_rels.append(
            f'<Relationship Id="rId{idx + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        )
    workbook_xml.extend(["</sheets>", "</workbook>"])
    workbook_rels.append("</Relationships>")

    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )

    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2">'
        '<font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/></font>'
        '</fonts>'
        '<fills count="2">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFD9EAF7"/><bgColor indexed="64"/></patternFill></fill>'
        '</fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="3">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="1" borderId="0" xfId="0" applyFont="1" applyFill="1"/>'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf>'
        '</cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "".join(content_types))
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", "".join(workbook_xml))
        zf.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        zf.writestr("xl/styles.xml", styles_xml)
        for idx, (_, sheet_xml) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", sheet_xml)


def main() -> None:
    checkpoints = parse_active_checkpoints(MATRIX_DIR / "checkpoints_fixed_stride.txt")
    historical_success = read_historical_success(MATRIX_DIR / "fixed_stride_matrix_params.csv")

    current_rows: list[dict[str, object]] = []
    for checkpoint_index, checkpoint in enumerate(checkpoints, start=1):
        for cfg in CURRENT_MATRIX_CONFIGS:
            row = dict(cfg)
            row["active_checkpoint_index"] = checkpoint_index
            row["run_name"] = checkpoint["run_name"]
            row["checkpoint_epoch"] = checkpoint["checkpoint_epoch"]
            row["eval_device"] = "cpu"
            row["historical_success_rate_hope2_ep15"] = historical_success.get(cfg["label"], "")
            row["historical_note"] = (
                "Reference only: pulled from fixed_stride_matrix_params.csv for the older Hope2 ep15 run."
                if cfg["label"] in historical_success
                else ""
            )
            current_rows.append(row)

    recommended_rows = proposed_configs()

    current_columns = [
        "active_checkpoint_index",
        "run_name",
        "checkpoint_epoch",
        "config_index",
        "label",
        "env_name",
        "eval_device",
        "goal_offset_steps_d",
        "eval_budget",
        "high_horizon",
        "low_horizon",
        "high_receding_horizon",
        "low_receding_horizon",
        "high_replan_interval",
        "high_action_block",
        "low_action_block",
        "high_num_samples",
        "high_n_steps",
        "high_topk",
        "low_num_samples",
        "low_n_steps",
        "low_topk",
        "historical_success_rate_hope2_ep15",
        "historical_note",
    ]
    current_widths = {
        "run_name": 34,
        "label": 22,
        "historical_note": 68,
    }

    recommended_columns = [
        "priority",
        "d",
        "label",
        "env_name",
        "eval_budget",
        "high_horizon",
        "low_horizon",
        "high_receding_horizon",
        "low_receding_horizon",
        "high_replan_interval",
        "high_action_block",
        "low_action_block",
        "high_num_samples",
        "high_n_steps",
        "high_topk",
        "low_num_samples",
        "low_n_steps",
        "low_topk",
        "source_family",
        "why_this_makes_sense_for_hope2",
    ]
    recommended_widths = {
        "label": 28,
        "source_family": 22,
        "why_this_makes_sense_for_hope2": 88,
    }

    current_sheet = build_sheet_xml(
        current_columns,
        current_rows,
        widths=current_widths,
        wrap_columns={"historical_note"},
    )
    recommended_sheet = build_sheet_xml(
        recommended_columns,
        recommended_rows,
        widths=recommended_widths,
        wrap_columns={"why_this_makes_sense_for_hope2"},
    )

    write_workbook(
        OUTPUT_PATH,
        [
            ("Current Matrix", current_sheet),
            ("Recommended Tests", recommended_sheet),
        ],
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
