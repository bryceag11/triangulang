# Installation Guide

## Requirements

- Linux (tested on Ubuntu 22.04)
- NVIDIA GPU with CUDA 12.6+
- 80GB+ GPU memory recommended (A100) for training
- Conda or Miniconda

## Automated Installation

```bash
bash setup/install.sh
```

This creates a `triangulang` conda environment. Pass a custom name: `bash setup/install.sh myenv`.

## Step-by-Step Manual Installation

### 1. Create Conda Environment

```bash
conda create -n triangulang python=3.12 -y
conda activate triangulang
```

### 2. Install PyTorch with CUDA

```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu126
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### 3. Initialize Git Submodules

```bash
git submodule init
git submodule update --recursive
```

This pulls SAM3, Depth Anything V3, and optional benchmarking/pointmap submodules (LERF, LangSplat-V2, MapAnything, Pi3) under `third_party/`. The `third_party/` submodules are only needed for reproducing baselines or using world-frame pointmap alternatives to DA3.

### 4. Install SAM3

```bash
cd sam3
pip install -e .
cd ..
```

SAM3 model weights require HuggingFace authentication:

1. Go to [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) and request access
2. Run `hf auth login` and enter your token

### 5. Install Depth Anything V3

```bash
cd depth_anything_v3
pip install -e .
cd ..
```

### 6. Install Requirements

```bash
pip install -r requirements.txt
```

### 7. Set PYTHONPATH

Add to your `.bashrc` or run before each session:

```bash
export PYTHONPATH=$PWD:$PWD/sam3:$PWD/depth_anything_v3/src
```

### 8. Verify Installation

```bash
python -c "
from triangulang.training.train import TrianguLangModel
from triangulang.evaluation.benchmark import load_model
from triangulang.models.gasa import GASAEncoder
print('All imports OK')
"
```

## Optional Dependencies

### PyTorch3D (for ScanNet++ mesh rasterization)

```bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install --no-build-isolation -e .
cd ..
```

Requires CUDA toolkit matching your PyTorch CUDA version.

### pydensecrf (for CRF post-processing)

```bash
conda install -y gxx_linux-64
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

## Data Setup

See the main [README](../README.md#data) for dataset download instructions.

Pre-cached depth maps and rasterized masks are available at:
[huggingface.co/datasets/bag100/triangulang-scannetpp-cache](https://huggingface.co/datasets/bag100/triangulang-scannetpp-cache)

## Troubleshooting

**`ModuleNotFoundError: No module named 'triangulang'`**
Make sure PYTHONPATH includes the repo root: `export PYTHONPATH=$PWD:$PWD/sam3:$PWD/depth_anything_v3/src`

**`CUDA out of memory`**
Reduce `--batch-size` to 1 or reduce `--views` to 4.

**`xformers` build errors**
Install the prebuilt wheel: `pip install xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu126`

**SAM3 weight download fails**
Ensure you've been granted access on HuggingFace and logged in with `hf auth login`.
