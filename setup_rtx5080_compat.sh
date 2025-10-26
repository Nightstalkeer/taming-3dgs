#!/bin/bash

# Setup script for Taming 3DGS on RTX 5080 with CUDA compatibility mode
# This script uses CUDA forward compatibility to run on RTX 5080

set -e  # Exit on error

echo "=========================================="
echo "Taming 3DGS Setup for RTX 5080"
echo "Using CUDA Forward Compatibility Mode"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found!"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda found: $(which conda)"
echo ""

# Check if environment already exists
if conda env list | grep -q "^3dgs "; then
    echo "WARNING: Environment '3dgs' already exists!"
    read -p "Do you want to remove it and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n 3dgs -y
    else
        echo "Exiting. Please remove the environment manually or use a different name."
        exit 1
    fi
fi

# Use the original environment file with PyTorch 2.1
cat > environment_3dgs_compat.yml << 'EOF'
name: 3dgs
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - pytorch::pytorch=2.1.0
  - pytorch::torchvision=0.16.0
  - pytorch::torchaudio=2.1.0
  - pytorch::pytorch-cuda=12.1
  - numpy<2.0
  - plyfile
  - tqdm
  - websockets
  - pip:
    - lpips
EOF

echo "Step 1: Creating conda environment..."
echo "This may take 5-10 minutes..."
conda env create -f environment_3dgs_compat.yml

echo ""
echo "Step 2: Setting up CUDA compatibility for RTX 5080..."
echo ""

# Activate environment
eval "$(conda shell.bash hook)"
conda activate 3dgs

# Set environment variables for CUDA compatibility
# RTX 5080 is sm_120, but we'll tell PyTorch to use sm_90 (Ada/Hopper) which is compatible
ENV_DIR="$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$ENV_DIR"

cat > "$ENV_DIR/cuda_compat.sh" << 'ENVEOF'
#!/bin/bash
# CUDA compatibility settings for RTX 5080 (Blackwell)
# Use sm_90 (Hopper/Ada) as fallback since PyTorch 2.1 doesn't support sm_120

export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Force PyTorch to treat RTX 5080 as compatible with sm_90
export CUDA_FORCE_PTX_JIT=1
ENVEOF

chmod +x "$ENV_DIR/cuda_compat.sh"
source "$ENV_DIR/cuda_compat.sh"

echo "✓ CUDA compatibility environment variables set"
echo ""

echo "Step 3: Installing custom CUDA extensions..."
echo "Note: This will compile CUDA code targeting sm_90, which is compatible with RTX 5080"
echo ""

# Install diff-gaussian-rasterization with sm_90 target
echo "  - Installing diff-gaussian-rasterization..."
TORCH_CUDA_ARCH_LIST="9.0" pip install submodules/diff-gaussian-rasterization/

# Install simple-knn
echo "  - Installing simple-knn..."
TORCH_CUDA_ARCH_LIST="9.0" pip install submodules/simple-knn/

# Install fused-ssim
echo "  - Installing fused-ssim..."
TORCH_CUDA_ARCH_LIST="9.0" pip install submodules/fused-ssim/

echo ""
echo "Step 4: Verifying installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'Compute capability: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}')
print('')
print('Testing CUDA extensions...')
try:
    import diff_gaussian_rasterization
    print('✓ diff_gaussian_rasterization loaded successfully')
except Exception as e:
    print(f'✗ diff_gaussian_rasterization failed: {e}')
try:
    import simple_knn
    print('✓ simple_knn loaded successfully')
except Exception as e:
    print(f'✗ simple_knn failed: {e}')
try:
    import fused_ssim
    print('✓ fused_ssim loaded successfully')
except Exception as e:
    print(f'✗ fused_ssim failed: {e}')
"

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: CUDA Compatibility Mode Enabled"
echo "  - Extensions compiled for sm_90 (compatible with RTX 5080)"
echo "  - CUDA_FORCE_PTX_JIT=1 enables runtime JIT compilation"
echo "  - First run may be slower due to JIT compilation"
echo ""
echo "To use the environment:"
echo "  conda activate 3dgs"
echo ""
echo "To run training:"
echo "  ./train_rtx5080.sh"
echo ""
echo "To test a single scene:"
echo "  python train.py -s data/bonsai -i images_2 -m ./test_output --budget 2 --mode multiplier --quiet --eval"
echo ""
