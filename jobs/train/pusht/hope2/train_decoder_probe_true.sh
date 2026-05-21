#!/bin/bash

# Convenience wrapper for Phase A of the frozen HOPE2 decoder probe.
# Trains the decoder only on true waypoint latents.
#
# Usage:
#   cd jobs/train/pusht
#   sbatch hope2/train_decoder_probe_true.sh
#   MAX_EPOCHS=10 sbatch hope2/train_decoder_probe_true.sh
#   TRAIN_RUN_NAME=hi_decoder_probe_true_custom sbatch hope2/train_decoder_probe_true.sh

#SBATCH --partition=gpu_a100
#SBATCH --constraint=scratch-node
#SBATCH --gpus=1
#SBATCH --job-name=hi_decoder_probe_true
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=10:00:00
#SBATCH --output=train_decoder_probe_true_%j.out
#SBATCH --error=train_decoder_probe_true_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=true_only
export MODE

if [[ -z "${TRAIN_RUN_NAME:-}" ]]; then
  export TRAIN_RUN_NAME="hi_decoder_probe_true_${SLURM_JOB_ID:-manual}"
fi

exec "${SCRIPT_DIR}/train_decoder_probe.sh"
