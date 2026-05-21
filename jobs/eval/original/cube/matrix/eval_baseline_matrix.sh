#!/bin/bash

# Array-driven baseline LeWM sweep for OGBench Cube.
#
# This launcher intentionally stays on the original baseline path:
# - Runs `original_eval_with_manifest.py`
# - Uses `third_party/lewm/config/eval/cube.yaml`
# - Uses the original flat LeWM checkpoint `cube/lewm`
# - Does NOT call `hi_eval.py` or any hierarchical planner path
#
# CSV mapping:
# - `goal_offset` -> `eval.goal_offset_steps` after stripping the `D` prefix
# - `eval_budget` -> `eval.eval_budget`
# - `low_horizon` -> `plan_config.horizon`
# - `low_receding_horizon` -> `plan_config.receding_horizon`
# - `low_action_block` -> `plan_config.action_block`
# - `low_num_samples` -> `solver.num_samples`
# - `low_n_steps` -> `solver.n_steps`
# - `solver.topk` stays at the baseline config default unless overridden manually
#
# Cube-specific guardrails:
# - Uses the single-cube baseline config `third_party/lewm/config/eval/cube.yaml`
# - Explicitly forces `eval.dataset_name=ogbench/cube_single_expert`
# - Checks both known single-cube cache layouts under `${STABLEWM_HOME}` and
#   uses the one that actually exists
#
# CPU notes:
# - The baseline submodule config defaults the solver device to CUDA.
# - This job forces both the eval wrapper device and solver device to CPU.
# - `original_eval_with_manifest.py` resolves that device and moves the baseline
#   model there, so this array can run on `rome` without a GPU.
#
# Output layout:
# - Logs: `jobs/eval/original/cube/matrix/logs/<sweep-name>/`
# - Eval artifacts: `${STABLEWM_HOME}/cube/<sweep-name>/<task-label>/`
# - `sweep-name` defaults to a timestamped name, so rerunning the submitter
#   creates a new non-overwriting folder by default.
#
# Submit with:
#   cd /gpfs/home2/scur0200/main/jobs/eval/original/cube/matrix
#   ./submit_baseline_matrix.sh

#SBATCH --partition=rome
#SBATCH --job-name=orig_cube_matrix
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --chdir=/gpfs/home2/scur0200/main/jobs/eval/original/cube/matrix
#SBATCH --output=baseline_matrix_%A_%a.out
#SBATCH --error=baseline_matrix_%A_%a.err

set -euo pipefail

resolve_matrix_dir() {
  local candidate resolved
  for candidate in \
    "${MATRIX_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}" \
    "${PWD:-}" \
    "${PROJECT_ROOT:-}/jobs/eval/original/cube/matrix" \
    "/gpfs/home2/${USER}/main/jobs/eval/original/cube/matrix"; do
    [[ -z "${candidate}" ]] && continue
    if resolved="$(cd "${candidate}" >/dev/null 2>&1 && pwd)"; then
      if [[ -f "${resolved}/baseline_matrix_sweep.csv" ]]; then
        echo "${resolved}"
        return 0
      fi
    fi
  done
  return 1
}

if ! MATRIX_DIR_RESOLVED="$(resolve_matrix_dir)"; then
  echo "ERROR: could not resolve jobs/eval/original/cube/matrix." >&2
  exit 2
fi

PROJECT_ROOT_DEFAULT="$(cd "${MATRIX_DIR_RESOLVED}/../../../.." >/dev/null 2>&1 && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"

PARAMS_FILE="${PARAMS_FILE:-${MATRIX_DIR_RESOLVED}/baseline_matrix_sweep.csv}"
POLICY="${POLICY:-cube/lewm}"
CONFIG_NAME="${CONFIG_NAME:-cube}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
NUM_EVAL="${NUM_EVAL:-50}"
SWEEP_NAME="${SWEEP_NAME:-orig_cube_single_cpu_matrix_$(date +%Y%m%d_%H%M%S)}"
HF_URL="${HF_URL:-https://huggingface.co/quentinll/lewm-cube/tree/main}"
export STABLEWM_HOME="${STABLEWM_HOME:-/scratch-shared/${USER}/stablewm_data}"

resolve_dataset_path() {
  local candidate
  for candidate in \
    "${DATASET_PATH:-}" \
    "${STABLEWM_HOME}/cube_single_expert.h5" \
    "${STABLEWM_HOME}/ogbench/cube_single_expert.h5"; do
    [[ -z "${candidate}" ]] && continue
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  echo "${STABLEWM_HOME}/cube_single_expert.h5"
  return 0
}

DATASET_PATH="$(resolve_dataset_path)"

if [[ ! -f "${PARAMS_FILE}" ]]; then
  echo "ERROR: parameter CSV not found: ${PARAMS_FILE}" >&2
  exit 3
fi

mapfile -t PARAM_ROWS < <(
  awk 'NR > 1 && $0 !~ /^[[:space:]]*($|#)/ { print }' "${PARAMS_FILE}"
)
NUM_CONFIGS="${#PARAM_ROWS[@]}"

if (( NUM_CONFIGS == 0 )); then
  echo "ERROR: no parameter rows found in ${PARAMS_FILE}" >&2
  exit 4
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-}}"
if [[ -z "${TASK_ID}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set." >&2
  echo "Submit with ./submit_baseline_matrix.sh or pass TASK_ID manually." >&2
  exit 5
fi

if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]] || (( TASK_ID < 1 || TASK_ID > NUM_CONFIGS )); then
  echo "ERROR: task id ${TASK_ID} is out of range 1-${NUM_CONFIGS}" >&2
  exit 6
fi

IFS=';' read -r GOAL_OFFSET_LABEL EVAL_BUDGET PLAN_HORIZON PLAN_RECEDING_HORIZON PLAN_ACTION_BLOCK SOLVER_NUM_SAMPLES SOLVER_N_STEPS <<< "${PARAM_ROWS[TASK_ID - 1]}"

if [[ ! "${GOAL_OFFSET_LABEL}" =~ ^[Dd]([0-9]+)$ ]]; then
  echo "ERROR: invalid goal_offset value '${GOAL_OFFSET_LABEL}' in ${PARAMS_FILE}" >&2
  exit 7
fi
GOAL_OFFSET_STEPS="${BASH_REMATCH[1]}"

for value_name in \
  EVAL_BUDGET \
  PLAN_HORIZON \
  PLAN_RECEDING_HORIZON \
  PLAN_ACTION_BLOCK \
  SOLVER_NUM_SAMPLES \
  SOLVER_N_STEPS \
  NUM_EVAL; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${value_name} must be an integer, got '${value}'." >&2
    exit 8
  fi
done

if [[ "${EVAL_DEVICE}" != "cpu" ]]; then
  echo "ERROR: this baseline matrix is intended for CPU runs; got EVAL_DEVICE=${EVAL_DEVICE}" >&2
  exit 9
fi

TASK_LABEL="d${GOAL_OFFSET_STEPS}_b${EVAL_BUDGET}_h${PLAN_HORIZON}_rh${PLAN_RECEDING_HORIZON}_blk${PLAN_ACTION_BLOCK}_ns${SOLVER_NUM_SAMPLES}_it${SOLVER_N_STEPS}"
JOB_TOKEN="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID:-${TASK_ID}}"
EVAL_SUBDIR="${SWEEP_NAME}/${TASK_LABEL}_job_${JOB_TOKEN}"
RESULT_FILENAME="ogb_cube_results_${TASK_LABEL}.txt"

mkdir -p "${STABLEWM_HOME}"

REPO_ROOT="${PROJECT_ROOT}"

module purge
module load 2025
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
if conda env list | grep -E '(^|[[:space:]])lewm-gpu([[:space:]]|$)' >/dev/null 2>&1; then
  conda activate lewm-gpu
elif conda env list | grep -E '(^|[[:space:]])lewm([[:space:]]|$)' >/dev/null 2>&1; then
  conda activate lewm
else
  echo "ERROR: Could not find conda environment 'lewm-gpu' or 'lewm'" >&2
  echo "Run jobs/setup/setup_env.sh first, or create the environment from environment-gpu.yml" >&2
  exit 10
fi

cd "${REPO_ROOT}"

if [[ ! -f "original_eval_with_manifest.py" ]]; then
  echo "ERROR: original_eval_with_manifest.py not found at ${REPO_ROOT}" >&2
  exit 11
fi

if [[ ! -f "third_party/lewm/eval.py" ]]; then
  echo "ERROR: third_party/lewm/eval.py not found at ${REPO_ROOT}" >&2
  exit 12
fi

CKPT_OBJECT_PATH="${STABLEWM_HOME}/${POLICY}_object.ckpt"

echo "Matrix dir: ${MATRIX_DIR_RESOLVED}"
echo "Project root: ${REPO_ROOT}"
echo "Parameter CSV: ${PARAMS_FILE}"
echo "Task: ${TASK_ID}/${NUM_CONFIGS}"
echo "Sweep name: ${SWEEP_NAME}"
echo "Task label: ${TASK_LABEL}"
echo "Policy: ${POLICY}"
echo "Config name: ${CONFIG_NAME}"
echo "Eval device: ${EVAL_DEVICE}"
echo "Num eval: ${NUM_EVAL}"
echo "Eval dataset: ogbench/cube_single_expert"
echo "Goal offset steps: ${GOAL_OFFSET_STEPS}"
echo "Eval budget: ${EVAL_BUDGET}"
echo "Plan horizon: ${PLAN_HORIZON}"
echo "Plan receding horizon: ${PLAN_RECEDING_HORIZON}"
echo "Plan action block: ${PLAN_ACTION_BLOCK}"
echo "Solver num samples: ${SOLVER_NUM_SAMPLES}"
echo "Solver n steps: ${SOLVER_N_STEPS}"
echo "Artifacts subdir: ${EVAL_SUBDIR}"
echo "Result filename: ${RESULT_FILENAME}"
echo "Expected checkpoint: ${CKPT_OBJECT_PATH}"
echo "Expected dataset: ${DATASET_PATH}"
echo "Baseline entrypoint: third_party/lewm/eval.py via original_eval_with_manifest.py"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "ERROR: missing dataset ${DATASET_PATH}" >&2
  echo "Run setup first, for example:" >&2
  echo "  sbatch --export=ALL,STABLEWM_HOME=${STABLEWM_HOME} jobs/setup/download_cube.sh" >&2
  exit 13
fi

if [[ ! -f "${CKPT_OBJECT_PATH}" ]]; then
  echo "Checkpoint object not found. Converting from Hugging Face..."
  python scripts/convert_hf_weights_to_object_ckpt.py \
    --hf-url "${HF_URL}" \
    --run-name "${POLICY}"
fi

CMD=(
  python original_eval_with_manifest.py
  --config-name="${CONFIG_NAME}"
  "policy=${POLICY}"
  "eval.dataset_name=ogbench/cube_single_expert"
  "eval.num_eval=${NUM_EVAL}"
  "eval.goal_offset_steps=${GOAL_OFFSET_STEPS}"
  "eval.eval_budget=${EVAL_BUDGET}"
  "+eval.device=${EVAL_DEVICE}"
  "plan_config.horizon=${PLAN_HORIZON}"
  "plan_config.receding_horizon=${PLAN_RECEDING_HORIZON}"
  "plan_config.action_block=${PLAN_ACTION_BLOCK}"
  "solver.num_samples=${SOLVER_NUM_SAMPLES}"
  "solver.n_steps=${SOLVER_N_STEPS}"
  "solver.device=${EVAL_DEVICE}"
  "output.filename=${RESULT_FILENAME}"
  "+output.subdir=${EVAL_SUBDIR}"
)

echo "Command:"
printf '  %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Baseline matrix eval finished."
echo "Artifacts should be under: ${STABLEWM_HOME}/$(dirname "${POLICY}")/${EVAL_SUBDIR}"
echo "Result file should be under: ${STABLEWM_HOME}/$(dirname "${POLICY}")/${EVAL_SUBDIR}/${RESULT_FILENAME}"
echo "Episode manifest should be under: ${STABLEWM_HOME}/$(dirname "${POLICY}")/${EVAL_SUBDIR}/${RESULT_FILENAME%.*}_episodes.tsv"
