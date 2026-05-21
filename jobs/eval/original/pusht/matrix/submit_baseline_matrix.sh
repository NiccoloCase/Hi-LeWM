#!/bin/bash

# Submit the original PushT baseline CPU matrix sweep as one Slurm array.
#
# Usage:
#   cd /gpfs/home2/scur0200/main/jobs/eval/original/pusht/matrix
#   ./submit_baseline_matrix.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PARAMS_FILE="${PARAMS_FILE:-${SCRIPT_DIR}/baseline_matrix_sweep.csv}"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/eval_baseline_matrix.sh}"
SWEEP_NAME="${SWEEP_NAME:-orig_pusht_cpu_matrix_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT_BASE="${LOG_ROOT_BASE:-${SCRIPT_DIR}/logs}"
LOG_ROOT="${LOG_ROOT:-${LOG_ROOT_BASE}/${SWEEP_NAME}}"

if [[ ! -f "${PARAMS_FILE}" ]]; then
  echo "ERROR: parameter CSV not found: ${PARAMS_FILE}" >&2
  exit 2
fi

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "ERROR: job script not found: ${JOB_SCRIPT}" >&2
  exit 3
fi

NUM_CONFIGS="$(
  awk 'NR > 1 && $0 !~ /^[[:space:]]*($|#)/ { count++ } END { print count + 0 }' "${PARAMS_FILE}"
)"
if ! [[ "${NUM_CONFIGS}" =~ ^[0-9]+$ ]] || (( NUM_CONFIGS <= 0 )); then
  echo "ERROR: parameter CSV is empty: ${PARAMS_FILE}" >&2
  exit 4
fi

mkdir -p "${LOG_ROOT}"

echo "Parameter CSV: ${PARAMS_FILE}"
echo "Job script: ${JOB_SCRIPT}"
echo "Sweep name: ${SWEEP_NAME}"
echo "Configs: ${NUM_CONFIGS}"
echo "Log root: ${LOG_ROOT}"
echo "Array: 1-${NUM_CONFIGS}"

sbatch \
  --array="1-${NUM_CONFIGS}" \
  --output="${LOG_ROOT}/baseline_matrix_%A_%a.out" \
  --error="${LOG_ROOT}/baseline_matrix_%A_%a.err" \
  --export="ALL,SWEEP_NAME=${SWEEP_NAME}" \
  "${JOB_SCRIPT}"
