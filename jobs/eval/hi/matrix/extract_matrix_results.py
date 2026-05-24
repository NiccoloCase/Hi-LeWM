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
    "checkpoint_row",
    "total_checkpoints",
    "config_row",
    "total_configs",
    "seed",
    "determinism",
    "run_name",
    "checkpoint_epoch",
    "label",
    "display_label",
    "config_note",
    "goal_offset_tag",
    "goal_offset_steps",
    "eval_budget",
    "num_eval",
    "planning_mode",
    "eval_device",
    "high_horizon",
    "high_receding_horizon",
    "high_action_block",
    "high_num_samples",
    "high_n_steps",
    "high_topk",
    "high_replan_interval",
    "low_horizon",
    "low_receding_horizon",
    "low_action_block",
    "low_num_samples",
    "low_n_steps",
    "low_topk",
    "empirical_macro_enabled",
    "empirical_macro_num_sequences",
    "empirical_macro_chunk_len",
    "empirical_macro_residual_scale",
    "empirical_macro_min_residual_std",
    "empirical_macro_return_top_candidates",
    "empirical_macro_encode_batch_size",
    "empirical_macro_stage_sampling",
    "success_rate",
    "status",
    "result_file",
    "episode_manifest",
    "out_file",
    "err_file",
]

SUMMARY_GROUP_FIELDS = [
    "run_name",
    "checkpoint_epoch",
    "label",
    "config_note",
    "goal_offset_tag",
    "goal_offset_steps",
    "eval_budget",
    "num_eval",
    "planning_mode",
    "eval_device",
    "high_horizon",
    "high_receding_horizon",
    "high_action_block",
    "high_num_samples",
    "high_n_steps",
    "high_topk",
    "high_replan_interval",
    "low_horizon",
    "low_receding_horizon",
    "low_action_block",
    "low_num_samples",
    "low_n_steps",
    "low_topk",
    "empirical_macro_enabled",
    "empirical_macro_num_sequences",
    "empirical_macro_chunk_len",
    "empirical_macro_residual_scale",
    "empirical_macro_min_residual_std",
    "empirical_macro_return_top_candidates",
    "empirical_macro_encode_batch_size",
    "empirical_macro_stage_sampling",
]


def extract_prefixed_value(text: str, prefix: str) -> str:
    pattern = re.compile(rf"^{re.escape(prefix)}\s*(.*)$", re.MULTILINE)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def extract_split_counts(value: str) -> tuple[str, str]:
    match = re.match(r"(\d+)\s*/\s*(\d+)", value)
    return (match.group(1), match.group(2)) if match else ("", "")


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


def read_text_if_exists(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def extract_config_block(result_text: str) -> str:
    match = re.search(r"==== CONFIG ====\n(.*?)\n==== ", result_text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"==== CONFIG ====\n(.*)", result_text, re.DOTALL)
    return match.group(1) if match else ""


def extract_named_section(block: str, section_name: str) -> str:
    lines = block.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped != f"{section_name}:":
            continue
        section_lines: list[str] = []
        for follow in lines[idx + 1 :]:
            if not follow.strip():
                section_lines.append("")
                continue
            follow_indent = len(follow) - len(follow.lstrip())
            if follow_indent <= indent:
                break
            section_lines.append(follow[indent + 2 :])
        return "\n".join(section_lines)
    return ""


def extract_section_scalar(section_text: str, key: str) -> str:
    match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*(.+?)\s*$", section_text)
    return match.group(1).strip() if match else ""


def build_result_paths(row: dict[str, str]) -> tuple[str, str]:
    stable_home = extract_prefixed_value(row["_out_text"], "STABLEWM_HOME:")
    output_subdir = extract_prefixed_value(row["_out_text"], "Output subdir:")
    result_filename = extract_prefixed_value(row["_out_text"], "Result filename:")
    run_name = row.get("run_name", "")
    if not (stable_home and output_subdir and result_filename and run_name):
        return ("", "")
    result_path = Path(stable_home) / "runs" / run_name / output_subdir / result_filename
    manifest_path = result_path.with_name(f"{result_path.stem}_episodes.tsv")
    return (str(result_path), str(manifest_path))


def parse_hi_out_file(out_path: Path) -> dict[str, str]:
    err_path = out_path.with_suffix(".err")
    out_text = read_text_if_exists(out_path)
    err_text = read_text_if_exists(err_path)
    job_id, task_id = parse_job_ids(out_path)
    checkpoint_row, total_checkpoints = extract_split_counts(extract_prefixed_value(out_text, "Checkpoint row:"))
    config_row, total_configs = extract_split_counts(extract_prefixed_value(out_text, "Config row:"))
    result_file, episode_manifest = build_result_paths({"_out_text": out_text, "run_name": extract_prefixed_value(out_text, "Run name:")})

    row = {
        "task_id": task_id,
        "job_id": job_id,
        "checkpoint_row": checkpoint_row,
        "total_checkpoints": total_checkpoints,
        "config_row": config_row,
        "total_configs": total_configs,
        "seed": extract_prefixed_value(out_text, "Seed:"),
        "determinism": extract_prefixed_value(out_text, "Determinism:"),
        "run_name": extract_prefixed_value(out_text, "Run name:"),
        "checkpoint_epoch": extract_prefixed_value(out_text, "Checkpoint epoch:"),
        "label": extract_prefixed_value(out_text, "Label:"),
        "display_label": extract_prefixed_value(out_text, "Display label:"),
        "config_note": extract_prefixed_value(out_text, "Config note:"),
        "goal_offset_tag": extract_prefixed_value(out_text, "Goal offset tag:"),
        "goal_offset_steps": extract_prefixed_value(out_text, "Goal offset steps:"),
        "eval_budget": extract_prefixed_value(out_text, "Eval budget:"),
        "num_eval": extract_prefixed_value(out_text, "Num eval:"),
        "planning_mode": extract_prefixed_value(out_text, "Planning mode:"),
        "eval_device": extract_prefixed_value(out_text, "Eval device:"),
        "high_horizon": "",
        "high_receding_horizon": "",
        "high_action_block": "",
        "high_num_samples": "",
        "high_n_steps": "",
        "high_topk": "",
        "high_replan_interval": "",
        "low_horizon": "",
        "low_receding_horizon": "",
        "low_action_block": "",
        "low_num_samples": "",
        "low_n_steps": "",
        "low_topk": "",
        "empirical_macro_enabled": "",
        "empirical_macro_num_sequences": "",
        "empirical_macro_chunk_len": "",
        "empirical_macro_residual_scale": "",
        "empirical_macro_min_residual_std": "",
        "empirical_macro_return_top_candidates": "",
        "empirical_macro_encode_batch_size": "",
        "empirical_macro_stage_sampling": "",
        "success_rate": parse_success_rate(out_text),
        "status": "",
        "result_file": result_file,
        "episode_manifest": episode_manifest,
        "out_file": str(out_path),
        "err_file": str(err_path),
        "_out_text": out_text,
        "_err_text": err_text,
    }

    high_match = re.search(
        r"High planner: horizon=(\S+), receding=(\S+), block=(\S+), samples=(\S+), iters=(\S+), topk=(\S+), replan=(\S+)",
        out_text,
    )
    if high_match:
        (
            row["high_horizon"],
            row["high_receding_horizon"],
            row["high_action_block"],
            row["high_num_samples"],
            row["high_n_steps"],
            row["high_topk"],
            row["high_replan_interval"],
        ) = high_match.groups()

    low_match = re.search(
        r"Low planner: horizon=(\S+), receding=(\S+), block=(\S+), samples=(\S+), iters=(\S+), topk=(\S+)",
        out_text,
    )
    if low_match:
        (
            row["low_horizon"],
            row["low_receding_horizon"],
            row["low_action_block"],
            row["low_num_samples"],
            row["low_n_steps"],
            row["low_topk"],
        ) = low_match.groups()

    result_text = read_text_if_exists(Path(result_file)) if result_file else ""
    config_block = extract_config_block(result_text)
    empirical_section = extract_named_section(config_block, "empirical_macro")
    if empirical_section:
        row["empirical_macro_enabled"] = extract_section_scalar(empirical_section, "enabled")
        row["empirical_macro_num_sequences"] = extract_section_scalar(empirical_section, "num_sequences")
        row["empirical_macro_chunk_len"] = extract_section_scalar(empirical_section, "chunk_len")
        row["empirical_macro_residual_scale"] = extract_section_scalar(empirical_section, "residual_scale")
        row["empirical_macro_min_residual_std"] = extract_section_scalar(empirical_section, "min_residual_std")
        row["empirical_macro_return_top_candidates"] = extract_section_scalar(empirical_section, "return_top_candidates")
        row["empirical_macro_encode_batch_size"] = extract_section_scalar(empirical_section, "encode_batch_size")
        row["empirical_macro_stage_sampling"] = extract_section_scalar(empirical_section, "stage_sampling")

    row["status"] = detect_status(row["success_rate"], out_text, err_text)
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
    return sorted(summary_rows, key=lambda row: (row.get("run_name", ""), row.get("checkpoint_epoch", ""), row.get("label", "")))


def collect_rows(log_root: Path) -> list[dict[str, str]]:
    rows = [parse_hi_out_file(path) for path in sorted(log_root.rglob("*.out"))]
    return sorted(rows, key=lambda row: (row.get("job_id", ""), int(row.get("task_id") or 0)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract hi matrix eval results into CSV.")
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()

    rows = collect_rows(args.log_root)
    for row in rows:
        row.pop("_out_text", None)
        row.pop("_err_text", None)
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
