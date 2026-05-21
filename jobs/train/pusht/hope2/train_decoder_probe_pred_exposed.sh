#!/bin/bash

# Convenience wrapper for Phase B of the frozen HOPE2 decoder probe.
# Initializes from a Phase A decoder checkpoint and exposes the decoder to
# predicted HOPE2 waypoint latents without updating HOPE2 itself.
#
# Usage:
#   cd jobs/train/pusht
#   INIT_DECODER_CKPT=/scratch-shared/$USER/stablewm_data/runs/hi_decoder_probe_true_x/hi_decoder_probe_true_x_probe.pt \
#     sbatch hope2/train_decoder_probe_pred_exposed.sh
#
# Optional overrides:
#   MAX_EPOCHS=15 sbatch hope2/train_decoder_probe_pred_exposed.sh
#   TRAIN_RUN_NAME=hi_decoder_probe_pred_custom sbatch hope2/train_decoder_probe_pred_exposed.sh

#SBATCH --partition=gpu_a100
#SBATCH --constraint=scratch-node
#SBATCH --gpus=1
#SBATCH --job-name=hi_decoder_probe_pred
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=10:00:00
#SBATCH --output=train_decoder_probe_pred_%j.out
#SBATCH --error=train_decoder_probe_pred_%j.err

set -euo pipefail

if [[ -z "${INIT_DECODER_CKPT:-}" ]]; then
  echo "ERROR: INIT_DECODER_CKPT must point to a Phase A decoder probe checkpoint." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=pred_exposed
export MODE

if [[ -z "${TRAIN_RUN_NAME:-}" ]]; then
  export TRAIN_RUN_NAME="hi_decoder_probe_pred_exposed_${SLURM_JOB_ID:-manual}"
fi

exec "${SCRIPT_DIR}/train_decoder_probe.sh"
