# VGGT Reconstruction Environment

Use the complete installation procedure in [`README.md`](README.md#installation).

The validated training environment is:

- Python 3.10
- PyTorch 2.3.1 + CUDA 12.1
- torchvision 0.18.1 + CUDA 12.1
- gsplat 1.5.3 built for PyTorch 2.3 and CUDA 12.1

Create it with:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda env create -f conda/environment.yml
conda activate vggt-reconstruction
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip check
```

Mirror settings are provided in `conda/condarc` and `conda/pip.conf`.
