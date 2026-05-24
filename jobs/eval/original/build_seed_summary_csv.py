#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


COMPACT_FIELDS = [
    "goal_offset_steps",
    "eval_budget",
    "plan_horizon",
    "plan_receding_horizon",
    "plan_action_block",
    "solver_num_samples",
    "solver_n_steps",
    "task_label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact config-level summary CSV from one raw baseline "
            "extract CSV per seed."
        )
    )
    parser.add_argument(
        "--seed-csv",
        action="append",
        required=True,
        metavar="SEED=PATH",
        help="Raw extractor CSV for one seed, for example 42=/tmp/pusht42_raw.csv",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV path.",
    )
    return parser.parse_args()


def parse_seed_csv_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected SEED=PATH, got: {value}")
    seed, path_str = value.split("=", 1)
    seed = seed.strip()
    if not seed or not seed.isdigit():
        raise ValueError(f"Invalid seed in --seed-csv: {value}")
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Seed CSV not found: {path}")
    return seed, path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def row_rank(row: dict[str, str]) -> tuple[int, int, int]:
    status_rank = 1 if row.get("status") == "completed" else 0
    job_id = int(row.get("job_id") or 0)
    task_id = int(row.get("task_id") or 0)
    return status_rank, job_id, task_id


def select_latest_completed_per_config(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        label = row.get("task_label", "").strip()
        if not label:
            continue
        current = selected.get(label)
        if current is None or row_rank(row) > row_rank(current):
            selected[label] = row
    return selected


def format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def build_summary_rows(
    per_seed_rows: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, str]]:
    all_labels = set()
    for rows in per_seed_rows.values():
        all_labels.update(rows.keys())

    sorted_seeds = sorted(per_seed_rows.keys(), key=int)
    output_rows: list[dict[str, str]] = []
    for label in sorted(all_labels):
        row = {"label": label}
        seed_values: list[float] = []

        template_row = None
        for seed in sorted_seeds:
            seed_row = per_seed_rows[seed].get(label)
            if seed_row and template_row is None:
                template_row = seed_row
            value = ""
            if seed_row and seed_row.get("status") == "completed" and seed_row.get("success_rate"):
                numeric_value = float(seed_row["success_rate"])
                value = format_float(numeric_value)
                seed_values.append(numeric_value)
            row[f"seed_{seed}"] = value

        if template_row is not None:
            row["d"] = template_row.get("goal_offset_steps", "")
            row["b"] = template_row.get("eval_budget", "")
            row["h"] = template_row.get("plan_horizon", "")
            row["rh"] = template_row.get("plan_receding_horizon", "")
            row["blk"] = template_row.get("plan_action_block", "")
            row["ns"] = template_row.get("solver_num_samples", "")
            row["it"] = template_row.get("solver_n_steps", "")
        else:
            row["d"] = ""
            row["b"] = ""
            row["h"] = ""
            row["rh"] = ""
            row["blk"] = ""
            row["ns"] = ""
            row["it"] = ""

        if seed_values:
            row["mean"] = format_float(statistics.mean(seed_values))
            row["std"] = (
                format_float(statistics.stdev(seed_values))
                if len(seed_values) > 1
                else "0"
            )
        else:
            row["mean"] = ""
            row["std"] = ""

        output_rows.append(row)

    return sorted(
        output_rows,
        key=lambda item: (
            int(item.get("d") or 0),
            int(item.get("b") or 0),
            int(item.get("h") or 0),
            int(item.get("rh") or 0),
            int(item.get("blk") or 0),
            int(item.get("ns") or 0),
            int(item.get("it") or 0),
            item.get("label", ""),
        ),
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    args = parse_args()
    seed_paths = dict(parse_seed_csv_arg(value) for value in args.seed_csv)
    sorted_seeds = sorted(seed_paths.keys(), key=int)

    per_seed_rows: dict[str, dict[str, dict[str, str]]] = {}
    for seed, path in seed_paths.items():
        per_seed_rows[seed] = select_latest_completed_per_config(load_rows(path))

    rows = build_summary_rows(per_seed_rows)
    fieldnames = (
        ["mean", "std"]
        + [f"seed_{seed}" for seed in sorted_seeds]
        + ["d", "b", "h", "rh", "blk", "ns", "it", "label"]
    )
    write_csv(args.output, fieldnames, rows)


if __name__ == "__main__":
    main()
