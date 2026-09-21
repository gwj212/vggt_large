#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-$(awk -F, '{print NF}' <<<"${GPUS}")}"

: "${TRAIN_DIR:?set TRAIN_DIR to the training scene root}"
: "${TEST_DIR:?set TEST_DIR to the test scene root}"
: "${VGGT_MODEL:?set VGGT_MODEL to the local VGGT-1B directory}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a writable output directory}"

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${GPUS}" \
torchrun --standalone --nproc_per_node="${NPROC}" "${REPO_ROOT}/train_nogt.py"
