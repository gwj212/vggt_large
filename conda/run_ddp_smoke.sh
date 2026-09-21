#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-vggt-reconstruction}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

GPUS="${GPUS:-0,1,2}"
NPROC="${NPROC:-$(awk -F, '{print NF}' <<<"${GPUS}")}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/smoke_conda_ddp}"

CUDA_VISIBLE_DEVICES="${GPUS}" \
TRAIN_DIR="${REPO_ROOT}/ddp_shape_smoke_data" \
LOW_VRAM_MODE=1 \
FREEZE_DPT_FEATURE_HEAD=1 \
SKIP_TEST_PRECOMPUTE=1 \
TRAIN_VIEWS_PER_STEP=1 \
MAX_IMG_SIDE=56 \
HARD_STOP_STEP="$((NPROC * 2))" \
NUM_EPOCHS=1 \
LOG_INTERVAL=1 \
EVAL_INTERVAL=999 \
SAVE_INTERVAL=999 \
LOG_ALL_RANK_SHAPES=1 \
OUTPUT_DIR="${OUTPUT_DIR}" \
torchrun --standalone --nproc_per_node="${NPROC}" "${REPO_ROOT}/train_nogt.py"
