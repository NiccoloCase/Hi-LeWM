#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Iterable

SUCCESS_RE = re.compile(r"success_rate['\"]?\s*[:=]\s*([0-9.]+)")
JOB_FILE_RE = re.compile(r"_(\d+)_(\d+)\.out$")

RAW_FIELDNAMES = [
    "task_id",
    "job_id",
    "config_row",
    "total_configs",
    "seed",
    "determinism",
    "sweep_name",
    "task_label",
    "display_label",
    "policy",
    "config_name",
    "eval_device",
    "num_eval",
    "goal_offset_steps",
    "eval_budget",
    "plan_horizon",
    "plan_receding_horizon",
    "plan_action_block",
    "solver_num_samples",
    "solver_n_steps",
    "success_rate",
    "status",
    "result_file",
    "episode_manifest",
    "out_file",
    "err_file",
]

SUMMARY_GROUP_FIELDS = [
    "sweep_name",
    "task_label",
    "policy",
    "config_name",
    "eval_device",
    "num_eval",
    "goal_offset_steps",
    "eval_budget",
    "plan_horizon",
    "plan_receding_horizon",
    "plan_action_block",
    "solver_num_samples",
    "solver_n_steps",
]


def extract_prefixed_value(text: str, prefix: str) -> str:
    pattern = re.compile(rf"^{re.escape(prefix)}\s*(.*)$", re.MULTILINE)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def parse_success_rate(text: str) -> str:
    matches = SUCCESS_RE.findall(text)
    return matches[-1] if matches else ""


def detect_status(success_rate: str, out_text: str, err_text: str) -> str:
    if success_rate:
        return "completed"
    upper_err = err_text.upper()
    if "TIME LIMIT" in upper_err:
        return "time_limit"
    if "CANCELLED" in upper_err:
        return "cancelled"
    if "TRACEBACK" in out_text or "TRACEBACK" in err_text or "ERROR:" in out_text or "ERROR:" in err_text:
        return "error"
    return "missing_result"


def parse_job_ids(path: Path) -> tuple[str, str]:
    match = JOB_FILE_RE.search(path.name)
    return (match.group(1), match.group(2)) if match else ("", "")


def parse_counts(text: str, prefix: str) -> tuple[str, str]:
    value = extract_prefixed_value(text, prefix)
    match = re.match(r"(\d+)\s*/\s*(\d+)", value)
    return (match.group(1), match.group(2)) if match else ("", "")


def read_text_if_exists(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def build_result_paths(out_text: str) -> tuple[str, str]:
    checkpoint_path = extract_prefixed_value(out_text, "Expected checkpoint:")
    subdir = extract_prefixed_value(out_text, "Artifacts subdir:")
    filename = extract_prefixed_value(out_text, "Result filename:")
    if not (checkpoint_path and subdir and filename):
        return ("", "")
    base_dir = Path(checkpoint_path).parent
    result_path = base_dir / subdir / filename
    manifest_path = result_path.with_name(f"{result_path.stem}_episodes.tsv")
    return (str(result_path), str(manifest_path))


def parse_original_out_file(out_path: Path) -> dict[str, str]:
    err_path = out_path.with_suffix(".err")
    out_text = read_text_if_exists(out_path)
    err_text = read_text_if_exists(err_path)
    job_id, task_id = parse_job_ids(out_path)
    config_row, total_configs = parse_counts(out_text, "Config row:")
    if not config_row:
        config_row = task_id
    result_file, episode_manifest = build_result_paths(out_text)

    row = {
        "task_id": task_id,
        "job_id": job_id,
        "config_row": config_row,
        "total_configs": total_configs,
        "seed": extract_prefixed_value(out_text, "Seed:"),
        "determinism": extract_prefixed_value(out_text, "Determinism:"),
        "sweep_name": extract_prefixed_value(out_text, "Sweep name:"),
        "task_label": extract_prefixed_value(out_text, "Task label:"),
        "display_label": extract_prefixed_value(out_text, "Display label:"),
        "policy": extract_prefixed_value(out_text, "Policy:"),
        "config_name": extract_prefixed_value(out_text, "Config name:"),
        "eval_device": extract_prefixed_value(out_text, "Eval device:"),
        "num_eval": extract_prefixed_value(out_text, "Num eval:"),
        "goal_offset_steps": extract_prefixed_value(out_text, "Goal offset steps:"),
        "eval_budget": extract_prefixed_value(out_text, "Eval budget:"),
        "plan_horizon": extract_prefixed_value(out_text, "Plan horizon:"),
        "plan_receding_horizon": extract_prefixed_value(out_text, "Plan receding horizon:"),
        "plan_action_block": extract_prefixed_value(out_text, "Plan action block:"),
        "solver_num_samples": extract_prefixed_value(out_text, "Solver num samples:"),
        "solver_n_steps": extract_prefixed_value(out_text, "Solver n steps:"),
        "success_rate": parse_success_rate(out_text),
        "status": detect_status(parse_success_rate(out_text), out_text, err_text),
        "result_file": result_file,
        "episode_manifest": episode_manifest,
        "out_file": str(out_path),
        "err_file": str(err_path),
    }
    return row


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in SUMMARY_GROUP_FIELDS)
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, str]] = []
    for key, group_rows in grouped.items():
        successes = [float(row["success_rate"]) for row in group_rows if row.get("success_rate")]
        completed = [row for row in group_rows if row.get("status") == "completed"]
        summary = {field: value for field, value in zip(SUMMARY_GROUP_FIELDS, key)}
        summary["num_rows"] = str(len(group_rows))
        summary["num_seeds_completed"] = str(len(completed))
        summary["success_rate_mean"] = f"{statistics.mean(successes):.6g}" if successes else ""
        summary["success_rate_std"] = f"{statistics.stdev(successes):.6g}" if len(successes) > 1 else ("0" if successes else "")
        summary["success_rate_min"] = f"{min(successes):.6g}" if successes else ""
        summary["success_rate_max"] = f"{max(successes):.6g}" if successes else ""
        statuses = sorted({row.get("status", "") for row in group_rows if row.get("status")})
        summary["statuses"] = ",".join(statuses)
        summary_rows.append(summary)
    return sorted(summary_rows, key=lambda row: (row.get("sweep_name", ""), row.get("task_label", "")))


def collect_rows(log_root: Path) -> list[dict[str, str]]:
    rows = [parse_original_out_file(path) for path in sorted(log_root.rglob("*.out"))]
    return sorted(rows, key=lambda row: (row.get("job_id", ""), int(row.get("task_id") or 0)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract original matrix eval results into CSV.")
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()

    rows = collect_rows(args.log_root)
    write_csv(args.output, RAW_FIELDNAMES, rows)

    if args.summary_output:
        summary_rows = build_summary_rows(rows)
        summary_fields = SUMMARY_GROUP_FIELDS + [
            "num_rows",
            "num_seeds_completed",
            "success_rate_mean",
            "success_rate_std",
            "success_rate_min",
            "success_rate_max",
            "statuses",
        ]
        write_csv(args.summary_output, summary_fields, summary_rows)


if __name__ == "__main__":
    main()
