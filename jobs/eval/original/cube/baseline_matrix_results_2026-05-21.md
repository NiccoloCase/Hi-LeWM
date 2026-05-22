# Cube Baseline Matrix Results

Final status: `14/14 COMPLETED` for Slurm array `23022524` (`orig_cube_matrix`).

Detailed machine-readable source:
- [orig_cube_matrix_status_2026-05-21.csv](/home/scur0200/main/roadmap/results/orig_cube_matrix_status_2026-05-21.csv:1)

Raw logs:
- [orig_cube_single_cpu_matrix_20260521_172100](/home/scur0200/main/jobs/eval/original/cube/matrix/logs/orig_cube_single_cpu_matrix_20260521_172100:1)

## Summary

| Goal Offset | Rows | Avg Success Rate | Best | Worst |
| --- | ---: | ---: | ---: | ---: |
| `D25` | 1 | `72.0` | `72.0` | `72.0` |
| `D50` | 6 | `45.33` | `56.0` | `36.0` |
| `D75` | 7 | `48.86` | `56.0` | `46.0` |

## Per-Task Results

| Task | Goal Offset | Eval Budget | Low Horizon | Low Receding Horizon | Low Action Block | Low Num Samples | Low N Steps | Success Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `D50` | `50` | `5` | `5` | `5` | `300` | `30` | `56.0` |
| `2` | `D50` | `100` | `5` | `5` | `5` | `300` | `30` | `56.0` |
| `3` | `D50` | `50` | `10` | `5` | `5` | `300` | `30` | `36.0` |
| `4` | `D50` | `100` | `10` | `5` | `5` | `300` | `30` | `36.0` |
| `5` | `D75` | `75` | `5` | `5` | `5` | `300` | `30` | `56.0` |
| `6` | `D75` | `150` | `5` | `5` | `5` | `300` | `30` | `56.0` |
| `7` | `D75` | `75` | `10` | `5` | `5` | `300` | `30` | `46.0` |
| `8` | `D75` | `100` | `10` | `5` | `5` | `300` | `30` | `46.0` |
| `9` | `D75` | `150` | `10` | `5` | `5` | `300` | `30` | `46.0` |
| `10` | `D75` | `75` | `15` | `5` | `5` | `300` | `30` | `46.0` |
| `11` | `D75` | `150` | `15` | `5` | `5` | `300` | `30` | `46.0` |
| `12` | `D25` | `50` | `5` | `5` | `5` | `300` | `30` | `72.0` |
| `13` | `D50` | `50` | `5` | `1` | `5` | `300` | `30` | `44.0` |
| `14` | `D50` | `100` | `5` | `1` | `5` | `300` | `30` | `44.0` |

## Notes

- This report reflects the successful rerun array `23022524`.
- Earlier cube baseline arrays failed during setup, but those failures are not part of the final result set above.
- For LLM or scripting use, prefer the linked CSV because it also includes `job_id`, `state`, `elapsed`, `start`, `end`, `exit_code`, `valid_starting_points`, `last_cem_solve_time_s`, and exact log paths.
