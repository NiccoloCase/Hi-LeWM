#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/eval_hope2_d75_matrix.sh}"
RUN_NAME="${RUN_NAME:-hi_lewm_p2_train_hope2_22253175}"
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-15}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs}"
LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
NUM_CONFIGS=7

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "ERROR: job script not found: ${JOB_SCRIPT}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"

echo "Job script: ${JOB_SCRIPT}"
echo "Run name: ${RUN_NAME}"
echo "Checkpoint epoch: ${CHECKPOINT_EPOCH}"
echo "Configs: ${NUM_CONFIGS}"
echo "Log dir: ${LOG_DIR}"

sbatch \
  --array="1-${NUM_CONFIGS}" \
  --output="${LOG_DIR}/eval_hope2_d75_matrix_%A_%a.out" \
  --error="${LOG_DIR}/eval_hope2_d75_matrix_%A_%a.err" \
  --export="ALL,RUN_NAME=${RUN_NAME},CHECKPOINT_EPOCH=${CHECKPOINT_EPOCH},GOAL_OFFSET_STEPS=75,EVAL_BUDGET=150,EVAL_DEVICE=cpu" \
  "${JOB_SCRIPT}"
