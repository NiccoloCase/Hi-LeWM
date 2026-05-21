#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OFFLINE_CONFIGS = {
    1: dict(experiment_kind="macro_manifold", goal_offset_steps=25, high_horizon=1, low_horizon=2),
    2: dict(experiment_kind="macro_manifold", goal_offset_steps=25, high_horizon=2, low_horizon=2),
    3: dict(experiment_kind="macro_manifold", goal_offset_steps=50, high_horizon=1, low_horizon=2),
    4: dict(experiment_kind="macro_manifold", goal_offset_steps=50, high_horizon=2, low_horizon=2),
    5: dict(experiment_kind="teacher_vs_open_loop", goal_offset_steps=25, high_horizon=1, low_horizon=2),
    6: dict(experiment_kind="teacher_vs_open_loop", goal_offset_steps=25, high_horizon=2, low_horizon=2),
    7: dict(experiment_kind="teacher_vs_open_loop", goal_offset_steps=50, high_horizon=1, low_horizon=2),
    8: dict(experiment_kind="teacher_vs_open_loop", goal_offset_steps=50, high_horizon=2, low_horizon=2),
    9: dict(experiment_kind="dataset_subgoal_reachability", goal_offset_steps=25, high_horizon=1, low_horizon=1),
    10: dict(experiment_kind="dataset_subgoal_reachability", goal_offset_steps=25, high_horizon=1, low_horizon=2),
    11: dict(experiment_kind="dataset_subgoal_reachability", goal_offset_steps=25, high_horizon=1, low_horizon=3),
    12: dict(experiment_kind="dataset_subgoal_reachability", goal_offset_steps=25, high_horizon=1, low_horizon=5),
    13: dict(experiment_kind="dataset_subgoal_reachability", goal_offset_steps=50, high_horizon=1, low_horizon=1),
    14: dict(experiment_kind="dataset_subgoal_reachability", goal_offset_steps=50, high_horizon=1, low_horizon=2),
    15: dict(experiment_kind="dataset_subgoal_reachability", goal_offset_steps=50, high_horizon=1, low_horizon=3),
    16: dict(experiment_kind="dataset_subgoal_reachability", goal_offset_steps=50, high_horizon=1, low_horizon=5),
    17: dict(experiment_kind="generated_subgoal_reachability", goal_offset_steps=25, high_horizon=1, low_horizon=2),
    18: dict(experiment_kind="generated_subgoal_reachability", goal_offset_steps=25, high_horizon=2, low_horizon=2),
    19: dict(experiment_kind="generated_subgoal_reachability", goal_offset_steps=50, high_horizon=1, low_horizon=2),
    20: dict(experiment_kind="generated_subgoal_reachability", goal_offset_steps=50, high_horizon=2, low_horizon=2),
}

ACTING_CONFIGS = {
    1: dict(experiment_kind="oracle_subgoal_acting", goal_offset_steps=50, high_horizon=2, low_horizon=2, low_receding_horizon=1),
    2: dict(experiment_kind="oracle_subgoal_acting", goal_offset_steps=50, high_horizon=2, low_horizon=3, low_receding_horizon=1),
    3: dict(experiment_kind="oracle_subgoal_acting", goal_offset_steps=50, high_horizon=2, low_horizon=5, low_receding_horizon=1),
    4: dict(experiment_kind="low_level_reality_gap", goal_offset_steps=50, high_horizon=1, low_horizon=2, low_receding_horizon=1),
    5: dict(experiment_kind="low_level_reality_gap", goal_offset_steps=50, high_horizon=1, low_horizon=3, low_receding_horizon=1),
    6: dict(experiment_kind="low_level_reality_gap", goal_offset_steps=50, high_horizon=1, low_horizon=5, low_receding_horizon=1),
    7: dict(experiment_kind="generated_subgoal_acting", goal_offset_steps=50, high_horizon=1, low_horizon=2, low_receding_horizon=1),
    8: dict(experiment_kind="generated_subgoal_acting", goal_offset_steps=50, high_horizon=2, low_horizon=2, low_receding_horizon=1),
    9: dict(experiment_kind="online_hierarchical_logging", goal_offset_steps=50, high_horizon=1, low_horizon=2, low_receding_horizon=1),
    10: dict(experiment_kind="online_hierarchical_logging", goal_offset_steps=50, high_horizon=2, low_horizon=2, low_receding_horizon=1),
    11: dict(experiment_kind="online_hierarchical_logging", goal_offset_steps=50, high_horizon=2, low_horizon=5, low_receding_horizon=1),
    12: dict(experiment_kind="online_hierarchical_logging", goal_offset_steps=50, high_horizon=2, low_horizon=5, low_receding_horizon=5),
}

OFFLINE_EXPLANATIONS = {
    "macro_manifold": "Checks whether the high-level CEM planner selects macro latents that stay close to the dataset latent manifold. Lower Mahalanobis and norm drift are better.",
    "teacher_vs_open_loop": "Separates high-level model quality from planner quality. Teacher-forced error probes one-step prediction; open-loop error shows rollout drift; high ratios and boolean flags suggest instability or planner prior mismatch.",
    "dataset_subgoal_reachability": "Uses true future latents from the dataset as subgoals and tests whether the low-level optimizer can reach them. Lower terminal latent error means the low level can reliably hit realistic subgoals.",
    "generated_subgoal_reachability": "Uses generated high-level subgoals, then asks whether they are reachable and look like real future states. Lower terminal error and smaller dataset/same-trajectory distances are better.",
}

ACTING_EXPLANATIONS = {
    "oracle_subgoal_acting": "Runs online control with oracle future subgoals taken from the real trajectory. This isolates low-level execution and stage handoff quality from high-level subgoal generation quality.",
    "low_level_reality_gap": "Executes short oracle subgoals online and compares predicted latent progress with actual latent progress. Larger positive reality gap means the model is more optimistic than reality.",
    "generated_subgoal_acting": "Runs online control with planner-generated stage targets. This mixes high-level target quality with low-level executability; large negative offset error means generated subgoals correspond to nearer-than-intended future states.",
    "online_hierarchical_logging": "Runs the full closed-loop hierarchical policy with replanning and detailed logging. This is the closest diagnostic to the deployed acting setup.",
}

HEADER_PATTERNS = {
    "config_row": re.compile(r"^Config row:\s+(\d+)\s*/\s*(\d+)\s*$", re.M),
    "run_name": re.compile(r"^Run name:\s+(.+?)\s*$", re.M),
    "checkpoint_epoch": re.compile(r"^Checkpoint epoch:\s+(.+?)\s*$", re.M),
    "policy": re.compile(r"^Policy:\s+(.+?)\s*$", re.M),
    "experiment_kind": re.compile(r"^Experiment kind:\s+(.+?)\s*$", re.M),
    "goal_offset_steps": re.compile(r"^Goal offset steps:\s+(.+?)\s*$", re.M),
    "eval_budget": re.compile(r"^Eval budget:\s+(.+?)\s*$", re.M),
    "high_horizon": re.compile(r"^High horizon:\s+(.+?)\s*$", re.M),
    "low_horizon": re.compile(r"^Low horizon:\s+(.+?)\s*$", re.M),
    "low_receding_horizon": re.compile(r"^Low receding horizon:\s+(.+?)\s*$", re.M),
    "artifact_dir": re.compile(r"^Artifact dir:\s+(.+?)\s*$", re.M),
    "json_path": re.compile(r"^JSON path:\s+(.+?)\s*$", re.M),
    "npz_path": re.compile(r"^NPZ path:\s+(.+?)\s*$", re.M),
}


@dataclass
class FolderSpec:
    mode: str
    summary_marker: str
    finished_marker: str
    configs: dict[int, dict[str, Any]]
    explanations: dict[str, str]
    csv_name: str
    md_name: str


SPECS = {
    "offline": FolderSpec(
        mode="offline",
        summary_marker="=== Diagnostic Summary ===",
        finished_marker="Diagnostic finished.",
        configs=OFFLINE_CONFIGS,
        explanations=OFFLINE_EXPLANATIONS,
        csv_name="results.csv",
        md_name="README.md",
    ),
    "acting": FolderSpec(
        mode="acting",
        summary_marker="=== Acting Diagnostic Summary ===",
        finished_marker="Acting diagnostic finished.",
        configs=ACTING_CONFIGS,
        explanations=ACTING_EXPLANATIONS,
        csv_name="results.csv",
        md_name="README.md",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize diagnostic log folders into CSV + Markdown.")
    parser.add_argument("mode", choices=sorted(SPECS))
    parser.add_argument("folder", type=Path)
    return parser.parse_args()


def parse_scalar(text: str | None) -> Any:
    if text is None:
        return None
    text = text.strip()
    if text.isdigit():
        return int(text)
    try:
        return float(text)
    except ValueError:
        return text


def extract_header(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, pattern in HEADER_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        out[key] = parse_scalar(match.group(1))
        if key == "config_row":
            out["config_row"] = int(match.group(1))
            out["config_row_total"] = int(match.group(2))
    return out


def extract_summary(text: str, marker: str) -> dict[str, Any]:
    marker_idx = text.find(marker)
    if marker_idx < 0:
        raise ValueError(f"Missing summary marker: {marker}")
    brace_idx = text.find("{", marker_idx)
    if brace_idx < 0:
        raise ValueError("Missing JSON object after summary marker.")
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(text[brace_idx:])
    if not isinstance(payload, dict):
        raise ValueError("Summary JSON root is not an object.")
    return payload


def get_step(step_items: list[dict[str, Any]], step: int) -> dict[str, Any]:
    for item in step_items:
        if int(item.get("step", -1)) == step:
            return item
    return {}


def get_offset(offset_items: list[dict[str, Any]], offset: int) -> dict[str, Any]:
    for item in offset_items:
        if int(item.get("offset_tokens", -1)) == offset:
            return item
    return {}


def flatten_common(meta: dict[str, Any], summary: dict[str, Any], out_file: Path, spec: FolderSpec) -> dict[str, Any]:
    row = {
        "log_file": out_file.name,
        "err_file": out_file.with_suffix(".err").name,
        "completed": spec.finished_marker in out_file.read_text(),
        "config_row": meta.get("config_row"),
        "config_row_total": meta.get("config_row_total"),
        "run_name": summary.get("run_name", meta.get("run_name")),
        "checkpoint_epoch": summary.get("checkpoint_epoch", meta.get("checkpoint_epoch")),
        "policy": summary.get("policy", meta.get("policy")),
        "experiment_kind": summary.get("experiment_kind", meta.get("experiment_kind")),
        "goal_offset_steps": summary.get("goal_offset_steps", meta.get("goal_offset_steps")),
        "goal_tokens": summary.get("goal_tokens"),
        "high_horizon": summary.get("high_horizon", meta.get("high_horizon")),
        "low_horizon": summary.get("low_horizon", meta.get("low_horizon")),
        "low_receding_horizon": summary.get("low_receding_horizon", meta.get("low_receding_horizon")),
        "eval_budget": summary.get("eval_budget", meta.get("eval_budget")),
        "num_eval": summary.get("num_eval"),
        "num_eval_samples": summary.get("num_eval_samples"),
        "device": summary.get("device"),
        "seed": summary.get("seed"),
        "result_status": summary.get("result_status"),
        "artifact_dir": meta.get("artifact_dir"),
        "json_path": meta.get("json_path"),
        "npz_path": meta.get("npz_path"),
    }
    if row["config_row"] in spec.configs:
        for key, value in spec.configs[int(row["config_row"])].items():
            row[f"expected_{key}"] = value
    return row


def flatten_offline(row: dict[str, Any], summary: dict[str, Any]) -> None:
    kind = summary.get("experiment_kind")
    if kind == "macro_manifold":
        steps = summary.get("step_metrics", [])
        for step_num in (1, 2):
            step = get_step(steps, step_num)
            row[f"step{step_num}_span_tokens"] = step.get("span_tokens")
            row[f"step{step_num}_dataset_mean_norm"] = (((step.get("dataset") or {}).get("mean_norm")) if step else None)
            row[f"step{step_num}_selected_mean_norm"] = (((step.get("selected_mean") or {}).get("mean_norm")) if step else None)
            row[f"step{step_num}_selected_best_mean_norm"] = (((step.get("selected_best") or {}).get("mean_norm")) if step else None)
            row[f"step{step_num}_elite_mean_norm"] = (((step.get("elite_cloud") or {}).get("mean_norm")) if step else None)
            row[f"step{step_num}_selected_mean_md2_p50"] = ((((step.get("selected_mean") or {}).get("mahalanobis") or {}).get("p50")) if step else None)
            row[f"step{step_num}_elite_md2_p50"] = ((((step.get("elite_cloud") or {}).get("mahalanobis") or {}).get("p50")) if step else None)
    elif kind == "teacher_vs_open_loop":
        for key in (
            "teacher_forced_mse_mean",
            "open_loop_true_mse_mean",
            "open_loop_cem_mse_mean",
            "open_loop_true_over_teacher",
            "open_loop_cem_over_open_true",
            "planner_prior_issue_flag",
            "high_predictor_instability_flag",
        ):
            row[key] = summary.get(key)
        for step_num, key_base in ((1, 0), (2, 1)):
            teacher = summary.get("teacher_forced_mse_per_step", [])
            open_true = summary.get("open_loop_true_mse_per_step", [])
            open_cem = summary.get("open_loop_cem_mse_per_step", [])
            row[f"step{step_num}_teacher_mse"] = teacher[key_base] if len(teacher) > key_base else None
            row[f"step{step_num}_open_true_mse"] = open_true[key_base] if len(open_true) > key_base else None
            row[f"step{step_num}_open_cem_mse"] = open_cem[key_base] if len(open_cem) > key_base else None
    elif kind == "dataset_subgoal_reachability":
        offsets = summary.get("offset_results", [])
        row["overall_terminal_error_mean"] = summary.get("terminal_latent_error_overall_mean")
        for offset in (2, 3, 5):
            item = get_offset(offsets, offset)
            row[f"offset{offset}_terminal_error_mean"] = item.get("terminal_latent_error_mean")
    elif kind == "generated_subgoal_reachability":
        steps = summary.get("step_results", [])
        for step_num in (1, 2):
            step = get_step(steps, step_num)
            row[f"step{step_num}_terminal_cost_mean"] = step.get("low_level_best_cem_terminal_cost_mean")
            row[f"step{step_num}_terminal_error_mean"] = step.get("achieved_terminal_latent_error_mean")
            row[f"step{step_num}_dataset_distance_mean"] = step.get("nearest_dataset_latent_distance_mean")
            row[f"step{step_num}_same_traj_distance_mean"] = step.get("nearest_same_trajectory_future_distance_mean")
            row[f"step{step_num}_offset_error_token_mean"] = step.get("offset_error_token_mean")


def flatten_acting(row: dict[str, Any], summary: dict[str, Any]) -> None:
    kind = summary.get("experiment_kind")
    if kind == "oracle_subgoal_acting":
        for key in (
            "stage1_terminal_latent_error_mean",
            "final_terminal_latent_error_mean",
            "goal_progress_mean",
            "reality_gap_mean",
            "success_rate",
        ):
            row[key] = summary.get(key)
    elif kind == "low_level_reality_gap":
        offsets = summary.get("offset_results", [])
        row["overall_actual_error_mean"] = summary.get("overall_actual_error_mean")
        row["overall_reality_gap_mean"] = summary.get("overall_reality_gap_mean")
        for offset in (2, 3, 5):
            item = get_offset(offsets, offset)
            row[f"offset{offset}_actual_error_mean"] = item.get("actual_error_mean")
    elif kind == "generated_subgoal_acting":
        steps = summary.get("step_results", [])
        for step_num in (1, 2):
            step = get_step(steps, step_num)
            row[f"step{step_num}_stage_end_actual_error_mean"] = step.get("stage_end_actual_error_mean")
            row[f"step{step_num}_offset_error_token_mean"] = step.get("offset_error_token_mean")
        row["final_terminal_latent_error_mean"] = summary.get("final_terminal_latent_error_mean")
        row["goal_progress_mean"] = summary.get("goal_progress_mean")
        row["success_rate"] = summary.get("success_rate")
    elif kind == "online_hierarchical_logging":
        for key in (
            "success_rate",
            "final_terminal_latent_error_mean",
            "mean_distance_current_to_subgoal",
            "mean_distance_current_to_final_goal",
            "mean_subgoal_churn_mse",
            "mean_reality_gap",
        ):
            row[key] = summary.get(key)


def summarize_folder(spec: FolderSpec, folder: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    out_files = sorted(folder.glob("*.out"))
    if not out_files:
        raise FileNotFoundError(f"No .out files found in {folder}")
    for out_file in out_files:
        text = out_file.read_text()
        meta = extract_header(text)
        summary = extract_summary(text, spec.summary_marker)
        row = flatten_common(meta, summary, out_file, spec)
        if spec.mode == "offline":
            flatten_offline(row, summary)
        else:
            flatten_acting(row, summary)
        rows.append(row)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    return rows, fieldnames


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def config_table_md(spec: FolderSpec) -> str:
    lines = [
        "| config_row | experiment_kind | key params | what it means |",
        "| --- | --- | --- | --- |",
    ]
    for config_row in sorted(spec.configs):
        cfg = spec.configs[config_row]
        params = [f"d{cfg['goal_offset_steps']}", f"hh{cfg['high_horizon']}", f"lh{cfg['low_horizon']}"]
        if "low_receding_horizon" in cfg:
            params.append(f"lrh{cfg['low_receding_horizon']}")
        lines.append(
            f"| {config_row} | `{cfg['experiment_kind']}` | `{', '.join(params)}` | {spec.explanations[cfg['experiment_kind']]} |"
        )
    return "\n".join(lines)


def metric_notes_md(spec: FolderSpec) -> str:
    if spec.mode == "offline":
        notes = [
            "- `d25` / `d50`: final goal is 25 or 50 environment steps into the future.",
            "- `hh*`: number of high-level macro stages used to cover the goal horizon.",
            "- `lh*`: low-level optimization horizon in latent-action tokens.",
            "- `step*_selected_mean_md2_p50` and `step*_elite_md2_p50`: median Mahalanobis distance to the dataset reference distribution; lower means more dataset-like.",
            "- `open_loop_true_over_teacher > 1` means multi-step rollout drift is worse than teacher-forced prediction; `> 1.5` raises `high_predictor_instability_flag`.",
            "- `open_loop_cem_over_open_true > 1` means planner-selected macro actions are harder for the model to roll out than true dataset macro actions; `> 1.5` raises `planner_prior_issue_flag`.",
            "- `offset*_terminal_error_mean`: low-level terminal latent error when targeting a real future latent `offset` tokens ahead; lower is better.",
            "- `step*_dataset_distance_mean` and `step*_same_traj_distance_mean`: how far a generated subgoal is from any dataset future state and from futures on the same trajectory; lower is better.",
        ]
    else:
        notes = [
            "- `d50`: acting diagnostics use a 50-step future goal window.",
            "- `hh*`: number of high-level stages or stage partitions across the goal window.",
            "- `lh*`: low-level planning horizon in latent-action tokens.",
            "- `lrh*`: low-level receding horizon; `lrh5` means actions are committed in larger blocks before replanning.",
            "- `success_rate`: fraction of evaluation episodes that solved PushT under this diagnostic.",
            "- `reality_gap_mean` / `overall_reality_gap_mean`: actual latent error minus model-predicted latent error. Positive means the world is harder than the model predicts.",
            "- `goal_progress_mean`: reduction in latent distance to the final goal over the rollout; higher is better.",
            "- `step*_offset_error_token_mean`: for generated subgoals, signed error between intended future offset and nearest matched future latent. Negative means the planner chose an effectively nearer subgoal than intended.",
            "- `mean_subgoal_churn_mse`: how much the high-level planner changes subgoals when replanning; higher means more instability.",
        ]
    return "\n".join(notes)


def write_markdown(path: Path, spec: FolderSpec, rows: list[dict[str, Any]]) -> None:
    run_names = sorted({str(row.get("run_name")) for row in rows if row.get("run_name")})
    status_counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("result_status"))
        status_counts[key] = status_counts.get(key, 0) + 1
    status_summary = ", ".join(f"`{key}`: {value}" for key, value in sorted(status_counts.items()))
    content = "\n".join(
        [
            f"# {'Acting ' if spec.mode == 'acting' else ''}Diagnostics Summary",
            "",
            f"- Folder: `{path.parent}`",
            f"- Runs covered: {', '.join(f'`{name}`' for name in run_names)}",
            f"- Log count: {len(rows)}",
            f"- Result statuses: {status_summary}",
            f"- CSV: `{spec.csv_name}`",
            "",
            "## Naming",
            "",
            "- `d*` is the goal offset in environment steps.",
            "- `hh*` is the high-level horizon, i.e. how many macro stages the goal window is partitioned into.",
            "- `lh*` is the low-level horizon in latent-action tokens.",
            "- `lrh*` appears only in acting diagnostics and is the low-level receding horizon.",
            "",
            "## Config Rows",
            "",
            config_table_md(spec),
            "",
            "## Metric Notes",
            "",
            metric_notes_md(spec),
        ]
    )
    path.write_text(content + "\n")


def main() -> int:
    args = parse_args()
    spec = SPECS[args.mode]
    folder = args.folder.resolve()
    rows, fieldnames = summarize_folder(spec, folder)
    rows.sort(key=lambda row: (int(row.get("config_row") or 0), str(row.get("log_file"))))
    write_csv(folder / spec.csv_name, rows, fieldnames)
    write_markdown(folder / spec.md_name, spec, rows)
    print(f"Wrote {folder / spec.csv_name}")
    print(f"Wrote {folder / spec.md_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
