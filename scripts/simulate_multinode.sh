#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-vggt-reconstruction}"
GPUS_NODE0="${GPUS_NODE0:-0,1}"
GPUS_NODE1="${GPUS_NODE1:-2,3}"
NPROC_NODE0="$(awk -F, '{print NF}' <<<"${GPUS_NODE0}")"
NPROC_NODE1="$(awk -F, '{print NF}' <<<"${GPUS_NODE1}")"

if [[ "${NPROC_NODE0}" -ne "${NPROC_NODE1}" ]]; then
  echo "Both simulated nodes must expose the same GPU count." >&2
  exit 2
fi

NPROC_PER_NODE="${NPROC_NODE0}"
WORLD_SIZE="$((NPROC_PER_NODE * 2))"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29672}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/smoke_2node_sim}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

COMMON_ENV=(
  "TRAIN_DIR=${REPO_ROOT}/ddp_shape_smoke_data"
  "LOW_VRAM_MODE=1"
  "FREEZE_DPT_FEATURE_HEAD=1"
  "SKIP_TEST_PRECOMPUTE=1"
  "TRAIN_VIEWS_PER_STEP=1"
  "MAX_IMG_SIDE=56"
  "HARD_STOP_STEP=$((WORLD_SIZE * 2))"
  "NUM_EPOCHS=1"
  "LOG_INTERVAL=1"
  "EVAL_INTERVAL=999"
  "SAVE_INTERVAL=999"
  "LOG_ALL_RANK_SHAPES=1"
  "OUTPUT_DIR=${OUTPUT_DIR}"
)

cleanup() {
  [[ -n "${PID0:-}" ]] && kill "${PID0}" 2>/dev/null || true
  [[ -n "${PID1:-}" ]] && kill "${PID1}" 2>/dev/null || true
}
trap cleanup INT TERM

env CUDA_VISIBLE_DEVICES="${GPUS_NODE0}" "${COMMON_ENV[@]}" \
  torchrun --nnodes=2 --nproc_per_node="${NPROC_PER_NODE}" --node_rank=0 \
  --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/train_nogt.py" >"${OUTPUT_DIR}/node0_torchrun.log" 2>&1 &
PID0=$!

env CUDA_VISIBLE_DEVICES="${GPUS_NODE1}" "${COMMON_ENV[@]}" \
  torchrun --nnodes=2 --nproc_per_node="${NPROC_PER_NODE}" --node_rank=1 \
  --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/train_nogt.py" >"${OUTPUT_DIR}/node1_torchrun.log" 2>&1 &
PID1=$!

set +e
wait "${PID0}"
RC0=$?
wait "${PID1}"
RC1=$?
set -e

printf 'node0_exit=%s node1_exit=%s output=%s\n' "${RC0}" "${RC1}" "${OUTPUT_DIR}"
if [[ "${RC0}" -ne 0 || "${RC1}" -ne 0 ]]; then
  exit 1
fi
