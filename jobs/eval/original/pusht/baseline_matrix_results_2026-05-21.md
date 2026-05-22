# PushT Baseline Matrix Results

Final status: `14/14 COMPLETED` for Slurm array `23022525` (`orig_pusht_matrix`).

Detailed machine-readable source:
- [orig_pusht_matrix_status_2026-05-21.csv](/home/scur0200/main/roadmap/results/orig_pusht_matrix_status_2026-05-21.csv:1)

Raw logs:
- [orig_pusht_cpu_matrix_20260521_172102](/home/scur0200/main/jobs/eval/original/pusht/matrix/logs/orig_pusht_cpu_matrix_20260521_172102:1)

## Summary

| Goal Offset | Rows | Avg Success Rate | Best | Worst |
| --- | ---: | ---: | ---: | ---: |
| `D25` | 1 | `98.0` | `98.0` | `98.0` |
| `D50` | 6 | `39.33` | `54.0` | `24.0` |
| `D75` | 7 | `9.43` | `12.0` | `4.0` |

## Per-Task Results

| Task | Goal Offset | Eval Budget | Low Horizon | Low Receding Horizon | Low Action Block | Low Num Samples | Low N Steps | Success Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `D50` | `50` | `5` | `5` | `5` | `300` | `30` | `52.0` |
| `2` | `D50` | `100` | `5` | `5` | `5` | `300` | `30` | `54.0` |
| `3` | `D50` | `50` | `10` | `5` | `5` | `300` | `30` | `24.0` |
| `4` | `D50` | `100` | `10` | `5` | `5` | `300` | `30` | `34.0` |
| `5` | `D75` | `75` | `5` | `5` | `5` | `300` | `30` | `10.0` |
| `6` | `D75` | `150` | `5` | `5` | `5` | `300` | `30` | `10.0` |
| `7` | `D75` | `75` | `10` | `5` | `5` | `300` | `30` | `8.0` |
| `8` | `D75` | `100` | `10` | `5` | `5` | `300` | `30` | `12.0` |
| `9` | `D75` | `150` | `10` | `5` | `5` | `300` | `30` | `12.0` |
| `10` | `D75` | `75` | `15` | `5` | `5` | `300` | `30` | `4.0` |
| `11` | `D75` | `150` | `15` | `5` | `5` | `300` | `30` | `10.0` |
| `12` | `D25` | `50` | `5` | `5` | `5` | `300` | `30` | `98.0` |
| `13` | `D50` | `50` | `5` | `1` | `5` | `300` | `30` | `30.0` |
| `14` | `D50` | `100` | `5` | `1` | `5` | `300` | `30` | `42.0` |

## Notes

- This report reflects the successful rerun array `23022525`.
- Earlier PushT baseline arrays failed during setup, but those failures are not part of the final result set above.
- For LLM or scripting use, prefer the linked CSV because it also includes `job_id`, `state`, `elapsed`, `start`, `end`, `exit_code`, `valid_starting_points`, `last_cem_solve_time_s`, and exact log paths.
