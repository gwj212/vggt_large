# Conda Environment

The complete installation and execution guide is in the repository root
`README.md`.

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

Validation commands:

```bash
GPUS=0,1,2,3 bash conda/run_ddp_smoke.sh
GPUS_NODE0=0,1 GPUS_NODE1=2,3 bash scripts/simulate_multinode.sh
```

The validated `gsplat` version is 1.5.3. Build it on the target cluster when a
compatible binary is unavailable from the configured mirror.
