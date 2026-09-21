#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${MASTER_ADDR:?set MASTER_ADDR to the routable address of node 0}"
: "${MASTER_PORT:=29500}"
: "${NNODES:?set NNODES to the physical node count}"
: "${NODE_RANK:?set NODE_RANK to this node's zero-based rank}"
: "${GPUS_PER_NODE:?set GPUS_PER_NODE to the process/GPU count per node}"
: "${TRAIN_DIR:?set TRAIN_DIR to the shared training scene root}"
: "${TEST_DIR:?set TEST_DIR to the shared test scene root}"
: "${VGGT_MODEL:?set VGGT_MODEL to the shared VGGT-1B directory}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a shared writable output directory}"

cd "${REPO_ROOT}"
torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/train_nogt.py"
