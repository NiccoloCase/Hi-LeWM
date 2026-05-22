#!/bin/bash

# Submit the full hope hierarchical matrix sweep.
#
# Run it with:
#   ./submit_hope_hierarchical_matrix.sh
#
# Override SWEEP_FILE to run a smaller ablation sweep instead.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-${SCRIPT_DIR}/checkpoints_hope_hierarchical.txt}"
SWEEP_FILE="${SWEEP_FILE:-${SCRIPT_DIR}/full_hierarchical_matrix_sweep.csv}"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/eval_hope_hierarchical_matrix.sh}"

if [[ ! -f "${CHECKPOINT_FILE}" ]]; then
  echo "ERROR: checkpoint list not found: ${CHECKPOINT_FILE}" >&2
  exit 2
fi

if [[ ! -f "${SWEEP_FILE}" ]]; then
  echo "ERROR: sweep file not found: ${SWEEP_FILE}" >&2
  exit 3
fi

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "ERROR: job script not found: ${JOB_SCRIPT}" >&2
  exit 4
fi

NUM_CHECKPOINTS="$(grep -Evc '^[[:space:]]*($|#)' "${CHECKPOINT_FILE}")"
NUM_CONFIGS="$(awk '
  BEGIN { count = 0 }
  NR == 1 { next }
  /^[[:space:]]*#/ { next }
  /^[[:space:]]*$/ { next }
  { count++ }
  END { print count }
' "${SWEEP_FILE}")"

if ! [[ "${NUM_CHECKPOINTS}" =~ ^[0-9]+$ ]] || (( NUM_CHECKPOINTS <= 0 )); then
  echo "ERROR: checkpoint list is empty: ${CHECKPOINT_FILE}" >&2
  exit 5
fi

if ! [[ "${NUM_CONFIGS}" =~ ^[0-9]+$ ]] || (( NUM_CONFIGS <= 0 )); then
  echo "ERROR: sweep is empty: ${SWEEP_FILE}" >&2
  exit 6
fi

mapfile -t CHECKPOINT_ROWS < <(grep -Ev '^[[:space:]]*($|#)' "${CHECKPOINT_FILE}")
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs_hope_hierarchical_matrix}"
mkdir -p "${LOG_ROOT}"

echo "Checkpoint file: ${CHECKPOINT_FILE}"
echo "Sweep file: ${SWEEP_FILE}"
echo "Job script: ${JOB_SCRIPT}"
echo "Checkpoints: ${NUM_CHECKPOINTS}"
echo "Configs per checkpoint: ${NUM_CONFIGS}"
echo "Log root: ${LOG_ROOT}"

for i in "${!CHECKPOINT_ROWS[@]}"; do
  read -r RUN_NAME CHECKPOINT_EPOCH <<< "${CHECKPOINT_ROWS[i]}"
  LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
  mkdir -p "${LOG_DIR}"

  echo
  echo "Submitting checkpoint $((i + 1))/${NUM_CHECKPOINTS}: ${RUN_NAME} (epoch ${CHECKPOINT_EPOCH})"
  echo "Log dir: ${LOG_DIR}"

  sbatch \
    --array="1-${NUM_CONFIGS}" \
    --output="${LOG_DIR}/eval_hope_hierarchical_matrix_%A_%a.out" \
    --error="${LOG_DIR}/eval_hope_hierarchical_matrix_%A_%a.err" \
    --export="ALL,CHECKPOINT_ROW_INDEX=$((i + 1))" \
    "${JOB_SCRIPT}"
done
