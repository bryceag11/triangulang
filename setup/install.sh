#!/bin/bash
# TrianguLang Installation Script
# Creates conda environment and installs all dependencies

set -e

ENV_NAME="${1:-triangulang}"

echo "Setting up TrianguLang environment: ${ENV_NAME}"

# Create conda environment
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment '${ENV_NAME}'..."
    conda create -n "${ENV_NAME}" python=3.12 -y
fi

echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# PyTorch with CUDA 12.6
echo "Installing PyTorch 2.7.0 with CUDA 12.6..."
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# xformers (required by DA3)
echo "Installing xformers..."
pip install xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu126

# Initialize submodules
echo "Initializing git submodules..."
git submodule init
git submodule update --recursive

# Install SAM3
echo "Installing SAM3..."
cd sam3
pip install -e .
cd ..

# Install Depth Anything V3
echo "Installing Depth Anything V3..."
cd depth_anything_v3
pip install -e .
cd ..

# Install remaining requirements
echo "Installing additional requirements..."
pip install -r requirements.txt

# Optional: PyTorch3D (needed for ScanNet++ mesh rasterization)
echo ""
echo "Optional: Install PyTorch3D for mesh rasterization:"
echo "  git clone https://github.com/facebookresearch/pytorch3d.git"
echo "  cd pytorch3d && pip install --no-build-isolation -e . && cd .."

# Optional: pydensecrf
echo ""
echo "Optional: Install pydensecrf for CRF post-processing:"
echo "  conda install -y gxx_linux-64"
echo "  pip install git+https://github.com/lucasb-eyer/pydensecrf.git"

echo ""
echo "Setup complete! Activate the environment with:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "IMPORTANT: SAM3 requires HuggingFace authentication for model weights."
echo "  1. Request access at: https://huggingface.co/facebook/sam3"
echo "  2. Run: hf auth login"
echo ""
echo "Set PYTHONPATH before running:"
echo "  export PYTHONPATH=\$PWD:\$PWD/sam3:\$PWD/depth_anything_v3/src"
echo ""
echo "Test the setup:"
echo "  python -c \"from triangulang.models.triangulang_model import TrianguLangModel; print('OK')\""
