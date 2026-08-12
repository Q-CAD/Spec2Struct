#!/bin/bash
# Build the `dosmatgen-cu128` conda environment: Python 3.12 + PyTorch 2.7.0
# built against CUDA 12.8, for Blackwell-class GPUs (sm_100) on a SLURM cluster.
# The CUDA 12.8 build also runs on older Ada/Ampere cards, so one environment
# covers both.
#
# The PyTorch Geometric compiled extensions (pyg_lib, torch_scatter, ...) are
# the part that genuinely needs the matching CUDA build; they are installed
# from the torch-2.7.0+cu128 wheel index below.
#
# Written for UF HiPerGator. Adapt the conda bootstrap and any `module load`
# lines to your system; everything after that is site-independent.
#
# Usage:  bash build_cu128_env.sh   (~15 min)
set -e

# --- site-specific: point this at your conda installation -------------------
# e.g. module load conda   /   source ~/miniconda3/etc/profile.d/conda.sh
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "==== [1] create env ===="
conda create -n dosmatgen-cu128 python=3.12 -y
conda activate dosmatgen-cu128
python --version

echo "==== [2] torch 2.7.0 + cu128 ===="
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

echo "==== [3] torch_geometric 2.7.0 ===="
pip install torch_geometric==2.7.0

echo "==== [4] PyG compiled extensions (torch-2.7.0+cu128 wheels) ===="
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

echo "==== [5] lightning + deps ===="
pip install pytorch-lightning ase pymatgen omegaconf einops wandb p-tqdm scipy

echo "==== [6] eval deps ===="
pip install matminer smact pyxtal

echo "==== [7] sanity: versions ===="
python - <<'PY'
import torch, numpy
print("torch", torch.__version__)
print("numpy", numpy.__version__)
import pytorch_lightning as pl
print("pytorch_lightning", pl.__version__)
import torch_geometric as pyg
print("torch_geometric", pyg.__version__)
import torch_scatter, torch_sparse, torch_cluster, torch_spline_conv
print("pyg ext OK")
print("arch_list", torch.cuda.get_arch_list())   # must include sm_100 for Blackwell
PY

echo "==== DONE ===="
echo "Next: conda activate dosmatgen-cu128 && python gpu_smoke_b200.py"
