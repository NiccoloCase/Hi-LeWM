#!/bin/bash

# Array-driven d=75 eval sweep for the hope2 PushT hierarchical checkpoint.
#
# Current hardcoded configs:
#   1. d75_hh1_lh2_pscaled
#   2. d75_hh1_lh2_searchboost
#   3. d75_hh1_lh2_replan3
#   4. d75_hh2_lh2_pscaled
#   5. d75_hh2_lh2_searchboost
#   6. d75_hh1_lh1_searchboost
#   7. d75_hh3_lh2_midcompute
#
# Submit with:
#   cd /gpfs/home2/scur0200/main/jobs/eval/hi/d75
#   ./submit_hope2_d75_matrix.sh

#SBATCH --partition=rome
#SBATCH --job-name=hi_eval_hope2_d75
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --chdir=/gpfs/home2/scur0200/main/jobs/eval/hi/d75
#SBATCH --output=eval_hope2_d75_matrix_%A_%a.out
#SBATCH --error=eval_hope2_d75_matrix_%A_%a.err

set -euo pipefail

resolve_d75_dir() {
  local c p
  for c in \
    "${D75_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}" \
    "${PWD:-}" \
    "${PROJECT_ROOT:-}/jobs/eval/hi/d75" \
    "/gpfs/home2/${USER}/main/jobs/eval/hi/d75"; do
    [[ -z "${c}" ]] && continue
    if p="$(cd "${c}" >/dev/null 2>&1 && pwd)"; then
      if [[ -f "${p}/eval_hope2_d75_matrix.sh" ]]; then
        echo "${p}"
        return 0
      fi
    fi
  done
  return 1
}

if ! D75_DIR_RESOLVED="$(resolve_d75_dir)"; then
  echo "ERROR: could not resolve d75 directory." >&2
  exit 2
fi

PROJECT_ROOT_DEFAULT="$(cd "${D75_DIR_RESOLVED}/../../../.." >/dev/null 2>&1 && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"

BASE_SCRIPT="${BASE_SCRIPT:-${D75_DIR_RESOLVED}/../hope2/hope2_pusht_eval_base.sh}"
if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "ERROR: base eval script not found: ${BASE_SCRIPT}" >&2
  exit 3
fi

NUM_CONFIGS=7
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-}}"
if [[ -z "${TASK_ID}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set." >&2
  echo "Submit with ./submit_hope2_d75_matrix.sh or pass TASK_ID manually." >&2
  exit 4
fi

if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]] || (( TASK_ID < 1 || TASK_ID > NUM_CONFIGS )); then
  echo "ERROR: task id ${TASK_ID} is out of range 1-${NUM_CONFIGS}" >&2
  exit 5
fi

export RUN_NAME="${RUN_NAME:-hi_lewm_p2_train_hope2_22253175}"
export CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-15}"
RUN_SHORT="${RUN_NAME#hi_lewm_p2_train_}"
export MODEL_LABEL="${MODEL_LABEL:-${RUN_SHORT}_ep${CHECKPOINT_EPOCH}}"

export CONFIG_NAME="${CONFIG_NAME:-hi_pusht}"
export EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
export GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-75}"
export EVAL_BUDGET="${EVAL_BUDGET:-150}"

export HIGH_RECEDING_HORIZON=1
export HIGH_ACTION_BLOCK=1
export LOW_RECEDING_HORIZON=1
export LOW_ACTION_BLOCK=5

case "${TASK_ID}" in
  1)
    LABEL="d75_hh1_lh2_pscaled"
    export HIGH_HORIZON=1
    export HIGH_REPLAN_INTERVAL=5
    export HIGH_NUM_SAMPLES=1500
    export HIGH_N_STEPS=40
    export HIGH_TOPK=10
    export LOW_HORIZON=2
    export LOW_NUM_SAMPLES=900
    export LOW_N_STEPS=20
    export LOW_TOPK=150
    ;;
  2)
    LABEL="d75_hh1_lh2_searchboost"
    export HIGH_HORIZON=1
    export HIGH_REPLAN_INTERVAL=5
    export HIGH_NUM_SAMPLES=1500
    export HIGH_N_STEPS=40
    export HIGH_TOPK=20
    export LOW_HORIZON=2
    export LOW_NUM_SAMPLES=900
    export LOW_N_STEPS=40
    export LOW_TOPK=200
    ;;
  3)
    LABEL="d75_hh1_lh2_replan3"
    export HIGH_HORIZON=1
    export HIGH_REPLAN_INTERVAL=3
    export HIGH_NUM_SAMPLES=1500
    export HIGH_N_STEPS=40
    export HIGH_TOPK=20
    export LOW_HORIZON=2
    export LOW_NUM_SAMPLES=900
    export LOW_N_STEPS=40
    export LOW_TOPK=200
    ;;
  4)
    LABEL="d75_hh2_lh2_pscaled"
    export HIGH_HORIZON=2
    export HIGH_REPLAN_INTERVAL=5
    export HIGH_NUM_SAMPLES=1500
    export HIGH_N_STEPS=40
    export HIGH_TOPK=10
    export LOW_HORIZON=2
    export LOW_NUM_SAMPLES=900
    export LOW_N_STEPS=20
    export LOW_TOPK=150
    ;;
  5)
    LABEL="d75_hh2_lh2_searchboost"
    export HIGH_HORIZON=2
    export HIGH_REPLAN_INTERVAL=5
    export HIGH_NUM_SAMPLES=1500
    export HIGH_N_STEPS=40
    export HIGH_TOPK=20
    export LOW_HORIZON=2
    export LOW_NUM_SAMPLES=900
    export LOW_N_STEPS=40
    export LOW_TOPK=200
    ;;
  6)
    LABEL="d75_hh1_lh1_searchboost"
    export HIGH_HORIZON=1
    export HIGH_REPLAN_INTERVAL=5
    export HIGH_NUM_SAMPLES=1200
    export HIGH_N_STEPS=30
    export HIGH_TOPK=20
    export LOW_HORIZON=1
    export LOW_NUM_SAMPLES=600
    export LOW_N_STEPS=40
    export LOW_TOPK=200
    ;;
  7)
    LABEL="d75_hh3_lh2_midcompute"
    export HIGH_HORIZON=3
    export HIGH_REPLAN_INTERVAL=5
    export HIGH_NUM_SAMPLES=1200
    export HIGH_N_STEPS=30
    export HIGH_TOPK=10
    export LOW_HORIZON=2
    export LOW_NUM_SAMPLES=300
    export LOW_N_STEPS=30
    export LOW_TOPK=150
    ;;
  *)
    echo "ERROR: unsupported task id ${TASK_ID}" >&2
    exit 6
    ;;
esac

export EVAL_SUBDIR="${EVAL_SUBDIR:-eval_d75_${LABEL}_job_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID:-${TASK_ID}}}"
export RESULT_FILENAME="${RESULT_FILENAME:-${MODEL_LABEL}_${LABEL}_results.txt}"

echo "Task: ${TASK_ID} / ${NUM_CONFIGS}"
echo "Run name: ${RUN_NAME}"
echo "Checkpoint epoch: ${CHECKPOINT_EPOCH}"
echo "Label: ${LABEL}"
echo "Goal offset steps: ${GOAL_OFFSET_STEPS}"
echo "Eval budget: ${EVAL_BUDGET}"
echo "High planner: horizon=${HIGH_HORIZON}, replan=${HIGH_REPLAN_INTERVAL}, samples=${HIGH_NUM_SAMPLES}, iters=${HIGH_N_STEPS}, topk=${HIGH_TOPK}"
echo "Low planner: horizon=${LOW_HORIZON}, samples=${LOW_NUM_SAMPLES}, iters=${LOW_N_STEPS}, topk=${LOW_TOPK}"

exec bash "${BASE_SCRIPT}"
