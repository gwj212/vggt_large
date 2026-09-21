# VGGT Gaussian Reconstruction for Distributed Training

This repository contains the modified VGGT feed-forward Gaussian reconstruction
model and its PyTorch DistributedDataParallel (DDP) training workflow. The
training objective does not use GT Gaussians.

The verified baseline uses Python 3.10, PyTorch 2.3.1 + CUDA 12.1, and NVIDIA
V100 GPUs. It has passed both a single-host four-GPU test and a simulated
two-node, two-GPU-per-node test with mixed image sizes.

## Features

- Feed-forward Gaussian reconstruction built on VGGT.
- No GT Gaussian generation or supervision.
- DDP-only training with one process per GPU.
- Deterministic scene sharding and equal step counts on all ranks.
- Different image sizes may be used by different ranks in the same step.
- Synchronized invalid-step handling to prevent collective deadlocks.
- Rank-0 checkpointing and globally unique per-rank log files.
- Portable single-host, multi-node, and multi-node simulation launch scripts.
- Mirror-friendly Conda and pip configuration for restricted networks.

## Repository Layout

```text
.
|-- train_nogt.py                 # DDP training entry point
|-- vggt/train_nogt.py            # Canonical training implementation
|-- train_nogt_ddp.py             # Saved byte-identical DDP copy
|-- vggt/                          # Modified reconstruction model
|-- scripts/run_single_node.sh    # Single-host launcher
|-- scripts/run_multinode.sh      # Physical multi-node launcher
|-- scripts/simulate_multinode.sh # Two-agent simulation on one host
|-- conda/run_ddp_smoke.sh        # Small mixed-shape DDP smoke test
|-- ddp_nccl_smoke.py             # Basic NCCL diagnostic
|-- ddp_shape_smoke_data/         # Independent five-scene test fixture
|-- conda/environment.yml         # Python 3.10 Conda environment
`-- requirements.txt              # Verified reconstruction dependencies
```

The original VGGT README is preserved at `docs/README_VGGT_UPSTREAM.md`.

## Requirements

- Linux x86_64
- Python 3.10
- NVIDIA driver compatible with CUDA 12.1
- One or more CUDA GPUs
- NCCL connectivity between all participating processes
- Shared code, model, dataset, and output paths for physical multi-node jobs

The validated `gsplat` version is 1.5.3. Pip installs it from the configured
mirror. Build it on the target cluster when a compatible binary is unavailable.

## Installation

### 1. Clone

Direct GitHub connection:

```bash
git clone https://github.com/gwj212/vggt_large.git
cd vggt_large
```

Verified mirror fallback:

```bash
git clone https://ghproxy.net/https://github.com/gwj212/vggt_large.git
cd vggt_large
```

### 2. Install Miniconda

```bash
wget -O /tmp/miniconda.sh \
  https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash
```

### 3. Configure mirrors and create the environment

```bash
cp conda/condarc "$HOME/.condarc"
mkdir -p "$HOME/.config/pip"
cp conda/pip.conf "$HOME/.config/pip/pip.conf"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda env create -f conda/environment.yml
conda activate vggt-reconstruction
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip check
```

### 4. Prepare VGGT weights

Store the VGGT-1B weights on local or shared storage. Weights and checkpoints
are intentionally excluded from Git.

```text
/shared/models/VGGT-1B/
|-- config.json
`-- model.safetensors
```

Set `VGGT_MODEL` to the actual directory used by the cluster.

## Dataset

The training and test roots contain one directory per scene. Each scene must
contain at least two readable images.

```text
my_train_20v/
|-- scene_0001/
|   |-- 00.jpg
|   `-- 01.jpg
`-- scene_0002/
    |-- 00.png
    `-- 01.png
```

The source server uses:

```text
/data2/vggt/my_train_20v
/data2/vggt/my_test_multiview
```

Training reads images on demand and does not modify either dataset.

## Validation Before Training

### Basic NCCL test

```bash
conda activate vggt-reconstruction
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 ddp_nccl_smoke.py
```

### Single-host mixed-shape smoke test

```bash
GPUS=0,1,2,3 \
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh" \
CONDA_ENV=vggt-reconstruction \
bash conda/run_ddp_smoke.sh
```

Expected completion markers:

```text
valid=4/4
DDP rank update counts agree: 2
DDP optimizer updates: 2
```

### Simulated two-node test

This starts two independent `torchrun` agents on one physical host. GPUs 0 and
1 represent node 0; GPUs 2 and 3 represent node 1.

```bash
GPUS_NODE0=0,1 \
GPUS_NODE1=2,3 \
CONDA_ENV=vggt-reconstruction \
bash scripts/simulate_multinode.sh
```

The simulation validates rendezvous, global and local ranks, scene sharding,
NCCL synchronization, forward, render loss, backward, and optimizer updates. It
does not validate the physical inter-node network.

## Training

`train_nogt.py` is DDP-only. Do not run it with plain `python`.

### Single host

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vggt-reconstruction
cd /shared/code/vggt_large

GPUS=0,1,2,3 \
TRAIN_DIR=/shared/data/my_train_20v \
TEST_DIR=/shared/data/my_test_multiview \
VGGT_MODEL=/shared/models/VGGT-1B \
OUTPUT_DIR=/shared/outputs/vggt_run01 \
bash scripts/run_single_node.sh
```

Equivalent direct command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TRAIN_DIR=/shared/data/my_train_20v \
TEST_DIR=/shared/data/my_test_multiview \
VGGT_MODEL=/shared/models/VGGT-1B \
OUTPUT_DIR=/shared/outputs/vggt_run01 \
torchrun --standalone --nproc_per_node=4 train_nogt.py
```

### Physical multi-node

Run the command once on every node. `NNODES`, `MASTER_ADDR`, and `MASTER_PORT`
must match; `NODE_RANK` must be unique and zero-based.

Node 0:

```bash
MASTER_ADDR=10.0.0.10 \
MASTER_PORT=29500 \
NNODES=2 \
NODE_RANK=0 \
GPUS_PER_NODE=4 \
TRAIN_DIR=/shared/data/my_train_20v \
TEST_DIR=/shared/data/my_test_multiview \
VGGT_MODEL=/shared/models/VGGT-1B \
OUTPUT_DIR=/shared/outputs/vggt_run01 \
bash scripts/run_multinode.sh
```

Node 1 runs the same command with `NODE_RANK=1`. `MASTER_ADDR` must be the
routable address of node 0, not `127.0.0.1`.

### Slurm example

Start one task per node and let each task launch the local GPU workers:

```bash
#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vggt-reconstruction
cd /shared/code/vggt_large

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=29500
export NNODES="$SLURM_NNODES"
export GPUS_PER_NODE=4
export TRAIN_DIR=/shared/data/my_train_20v
export TEST_DIR=/shared/data/my_test_multiview
export VGGT_MODEL=/shared/models/VGGT-1B
export OUTPUT_DIR=/shared/outputs/vggt_run01

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -c '
  export NODE_RANK="$SLURM_NODEID"
  bash scripts/run_multinode.sh
'
```

Adapt resource directives to the cluster scheduler and GPU policy.

## Training Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `TRAIN_DIR` | `/data2/vggt/my_train_20v` | Training scene root |
| `TEST_DIR` | `/data2/vggt/my_test_multiview` | Test scene root |
| `VGGT_MODEL` | `/root/models/VGGT-1B` | Local VGGT weight directory |
| `OUTPUT_DIR` | `/root/vggt/output/nogt_geoinit_pure_ddp` | Logs and checkpoints |
| `TRAIN_VIEWS_PER_STEP` | `5` | Random views per scene and rank |
| `MAX_IMG_SIDE` | `0` | Long-side limit; `0` disables resizing |
| `NUM_EPOCHS` | `10` | Epoch limit |
| `LOW_VRAM_MODE` | `0` | FP16 frozen backbone and CPU camera mode |
| `FREEZE_DPT_FEATURE_HEAD` | `0` | Freeze DPT feature head when set to `1` |
| `SKIP_TEST_PRECOMPUTE` | `0` | Skip test cache/evaluation when set to `1` |
| `RESUME` | empty | Checkpoint path or `auto` |

The effective global batch grows with world size while the learning rate stays
fixed. Retune the learning rate when scaling to substantially more GPUs.

## Low-Memory Test Configuration

Use these only for interface and DDP validation:

```bash
LOW_VRAM_MODE=1
FREEZE_DPT_FEATURE_HEAD=1
TRAIN_VIEWS_PER_STEP=1
MAX_IMG_SIDE=56
SKIP_TEST_PRECOMPUTE=1
```

Before a production job, run 50-100 steps at the intended image resolution and
view count. Monitor the smallest GPU and verify checkpoint save/resume.

## Multi-Node Troubleshooting

### Rendezvous timeout

Verify node 0 is reachable from every node:

```bash
nc -vz "$MASTER_ADDR" "$MASTER_PORT"
```

Open the selected TCP port or choose a permitted port. Do not use localhost for
physical multi-node jobs.

### NCCL selects the wrong interface

Select the routable network interface explicitly:

```bash
export NCCL_SOCKET_IFNAME=eth0
```

For InfiniBand clusters, use the interface recommended by the administrator. If
IB is unavailable or misconfigured, test TCP fallback:

```bash
export NCCL_IB_DISABLE=1
```

### Need detailed NCCL diagnostics

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,COLL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

### A rank exits or hangs

- Confirm every node uses the same code commit and Python environment.
- Confirm every node can read the same dataset and model paths.
- Confirm `NNODES`, `GPUS_PER_NODE`, and rank assignments are correct.
- Inspect `nogt.log` and every `nogt_rankN.log` in `OUTPUT_DIR`.
- Use the included smoke dataset before testing production data.

The training loop pads scene indices so all ranks execute the same number of
collectives. If any rank rejects a step, all ranks skip that optimizer update.

### Out of memory

Reduce settings in this order:

1. Set `MAX_IMG_SIDE` to `448`, `392`, or lower.
2. Reduce `TRAIN_VIEWS_PER_STEP`.
3. Enable `LOW_VRAM_MODE=1`.
4. Freeze the DPT head with `FREEZE_DPT_FEATURE_HEAD=1`.

Use the smallest-memory GPU as the configuration limit.

## Verified Results

| Test | Topology | Result |
| --- | --- | --- |
| Mixed-size DDP smoke | 1 host x 4 V100 | Passed, 2 synchronized updates |
| Multi-node simulation | 2 agents x 2 V100 | Passed, global ranks 0-3 |
| Different H/W by rank | Same optimizer step | Passed without deadlock |
| Low-memory peak | Smoke configuration | About 2.2 GB per rank |

Physical Ethernet or InfiniBand behavior must still be validated on the target
cluster before a long production run.

## Attribution

This repository builds on VGGT. Keep `LICENSE.txt` and the upstream attribution
files when redistributing the code.
