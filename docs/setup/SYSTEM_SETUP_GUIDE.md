# Complete System Setup Guide for 3D Gaussian Splatting Training

**Based on:** RTX 5080 (Blackwell sm_120) successful training setup
**Last Updated:** October 25, 2025
**Target Use Case:** 3D Gaussian Splatting (taming-3dgs) on modern NVIDIA GPUs

---

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [System Prerequisites](#system-prerequisites)
3. [NVIDIA Driver and CUDA Installation](#nvidia-driver-and-cuda-installation)
4. [Conda Environment Setup](#conda-environment-setup)
5. [PyTorch Installation](#pytorch-installation)
6. [Python Dependencies](#python-dependencies)
7. [CUDA Submodule Compilation](#cuda-submodule-compilation)
8. [Environment Variables](#environment-variables)
9. [Verification](#verification)
10. [Troubleshooting](#troubleshooting)

---

## Hardware Requirements

### Minimum Requirements

- **GPU:** NVIDIA GPU with 8GB+ VRAM, Compute Capability ≥ 7.5
  - Turing (RTX 20xx): sm_75
  - Ampere (RTX 30xx): sm_86
  - Ada Lovelace (RTX 40xx): sm_89
  - Blackwell (RTX 50xx): sm_120
- **RAM:** 16GB system memory (32GB recommended for large scenes)
- **Storage:** 50GB free disk space (100GB+ recommended for datasets)
- **CPU:** Multi-core processor (4+ cores recommended)

### Our Tested Configuration

- **GPU:** NVIDIA GeForce RTX 5080 (16GB VRAM, sm_120)
- **RAM:** 64GB DDR5
- **Storage:** 706GB SSD with 480GB available
- **CPU:** Modern multi-core x86_64 processor
- **OS:** Manjaro Linux (kernel 6.16.8)

**Environment Size:** ~13GB for complete conda environment

---

## System Prerequisites

### Operating System

**Recommended:**
- Ubuntu 20.04/22.04 LTS
- Manjaro Linux (tested)
- Arch Linux
- Other modern Linux distributions

**Requirements:**
- Linux kernel 5.x or higher
- X11 or Wayland (for GUI applications)
- systemd (for service management)

### System Packages

Install essential build tools:

```bash
# Manjaro/Arch
sudo pacman -S base-devel gcc cmake git ninja

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential cmake git ninja-build

# Common packages for all distributions
sudo <package-manager> install \
    wget curl \
    libgl1-mesa-glx \
    libglib2.0-0
```

### Compiler Versions

**GCC/G++:**
- Minimum: GCC 9.x
- Recommended: GCC 11.x - 15.x
- Tested: GCC 15.2.1

**Verification:**
```bash
gcc --version
g++ --version
```

---

## NVIDIA Driver and CUDA Installation

### NVIDIA Driver

**Minimum Driver Version:** 525+ (for CUDA 12.x support)
**Recommended:** Latest stable driver for your GPU
**Tested:** Driver 580.82.09

#### Installation Methods

**Method 1: Distribution Package Manager (Recommended)**

```bash
# Manjaro/Arch
sudo pacman -S nvidia nvidia-utils

# Ubuntu
sudo apt-get install nvidia-driver-550
```

**Method 2: NVIDIA Official Installer**

Download from: https://www.nvidia.com/Download/index.aspx

```bash
# Disable nouveau driver first
sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
sudo update-initramfs -u
sudo reboot

# Install driver
sudo bash NVIDIA-Linux-x86_64-*.run
```

**Verification:**
```bash
nvidia-smi
# Should show driver version and CUDA version
```

### CUDA Toolkit

**Required:** CUDA 12.1 or higher
**Recommended:** CUDA 12.8 or 12.9
**Tested:** CUDA 12.9

#### Installation

**Download from:** https://developer.nvidia.com/cuda-downloads

```bash
# Example for Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_550.54.14_linux.run
sudo sh cuda_12.9.0_550.54.14_linux.run
```

**Installation Options:**
- ☑ CUDA Toolkit 12.9
- ☐ Driver (skip if already installed)
- ☑ CUDA Samples
- ☑ CUDA Documentation

**Add to PATH** (add to `~/.bashrc` or `~/.zshrc`):
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Verification:**
```bash
nvcc --version
# Should show: Cuda compilation tools, release 12.9, V12.9.86
```

---

## Conda Environment Setup

### Install Miniconda/Anaconda

**If not already installed:**

```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install
bash Miniconda3-latest-Linux-x86_64.sh

# Follow prompts, accept license, choose installation path
# Default: ~/miniconda3

# Initialize conda
~/miniconda3/bin/conda init bash  # or zsh
source ~/.bashrc  # or ~/.zshrc
```

### Create Dedicated Environment

```bash
# Create Python 3.10 environment
conda create -n rtx5080_3dgs python=3.10 -y

# Activate environment
conda activate rtx5080_3dgs

# Verify Python version
python --version
# Should show: Python 3.10.x
```

**Why Python 3.10?**
- Maximum compatibility with PyTorch 2.9.0
- Stable ABI for CUDA extensions
- Well-tested with scientific Python packages

---

## PyTorch Installation

### Critical: GPU Architecture Compatibility

**Understanding Compute Capabilities:**

| GPU Series | Architecture | Compute Capability | PyTorch 2.9.0 Support |
|-----------|--------------|-------------------|----------------------|
| RTX 20xx | Turing | sm_75 (7.5) | ✓ Full |
| RTX 30xx | Ampere | sm_86 (8.6) | ✓ Full |
| RTX 40xx | Ada Lovelace | sm_89 (8.9) | ✓ Full |
| H100 | Hopper | sm_90 (9.0) | ✓ Full (PyTorch max) |
| **RTX 50xx** | **Blackwell** | **sm_120 (12.0)** | **△ Core only** |

**Key Limitation:** PyTorch 2.9.0 can compile custom CUDA extensions up to sm_90 only. For sm_120 GPUs (RTX 50xx), custom extensions must be avoided or replaced with pure PyTorch implementations.

### Installation Command

**For CUDA 12.8 (Recommended for RTX 50xx):**

```bash
conda activate rtx5080_3dgs

pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

**Download size:** ~900MB for torch, ~8MB for torchvision, ~2MB for torchaudio
**Installation time:** 2-5 minutes depending on network speed

**Alternative CUDA versions:**

```bash
# CUDA 12.1 (if you have older CUDA toolkit)
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (for older GPUs)
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu118
```

### Verification

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Compute capability: {torch.cuda.get_device_capability(0)}')"
```

**Expected output:**
```
PyTorch: 2.9.0+cu128
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA GeForce RTX 5080
Compute capability: (12, 0)
```

### CUDA Version Mismatch Bypass (RTX 50xx ONLY)

If system CUDA (12.9) differs from PyTorch CUDA (12.8), you need to bypass the version check:

**File:** `<conda_env>/lib/python3.10/site-packages/torch/utils/cpp_extension.py`

**Find lines ~519-521:**
```python
# BYPASS: CUDA version check disabled for RTX 5080/5090 compatibility
# if cuda_ver.major != torch_cuda_version.major:
#     raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
```

**Manual edit:**
```bash
# Find the file
TORCH_EXT_FILE=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'utils', 'cpp_extension.py'))")

# Backup original
cp "$TORCH_EXT_FILE" "$TORCH_EXT_FILE.backup"

# Edit with your preferred editor
nano "$TORCH_EXT_FILE"
# Or: vim "$TORCH_EXT_FILE"
# Or: code "$TORCH_EXT_FILE"

# Find the if statement checking cuda_ver.major and comment it out
```

**⚠️ Important:** This modification gets overwritten when you upgrade PyTorch. You'll need to reapply after upgrades.

---

## Python Dependencies

### Core Dependencies

Install all required Python packages:

```bash
conda activate rtx5080_3dgs

pip install \
    numpy \
    scipy \
    pillow \
    plyfile \
    tqdm \
    lpips \
    pyyaml \
    requests \
    websockets
```

### Dependency Details

**Complete package list with versions (tested):**

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 2.1.2 | Numerical operations |
| scipy | 1.15.3 | Scientific computing |
| pillow | 11.3.0 | Image processing |
| plyfile | 1.1 | Point cloud I/O |
| tqdm | 4.67.1 | Progress bars |
| lpips | 0.1.4 | Perceptual loss |
| PyYAML | 6.0.3 | Config file parsing |
| requests | 2.32.5 | HTTP requests |
| websockets | 15.0.1 | Web viewer support |

### Automatic Installation via requirements.txt

**Option 1:** If repository has `requirements.txt`:
```bash
pip install -r requirements.txt
```

**Option 2:** Create requirements.txt yourself:
```bash
cat > requirements.txt << 'EOF'
numpy>=1.21.0
scipy>=1.7.0
pillow>=8.0.0
plyfile>=0.7.0
tqdm>=4.60.0
lpips>=0.1.4
pyyaml>=5.4.0
requests>=2.25.0
websockets>=10.0
EOF

pip install -r requirements.txt
```

---

## CUDA Submodule Compilation

This is the **most critical** part for RTX 50xx GPUs.

### Environment Variables (MUST SET BEFORE COMPILATION)

```bash
# Set CUDA architecture targets
export TORCH_CUDA_ARCH_LIST="8.6;9.0"
export CUDAARCHS="90"

# Make permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export TORCH_CUDA_ARCH_LIST="8.6;9.0"' >> ~/.bashrc
echo 'export CUDAARCHS="90"' >> ~/.bashrc
source ~/.bashrc
```

**Why these values?**
- `8.6`: Ampere architecture (RTX 30xx) - widely compatible
- `9.0`: Hopper architecture (H100) - PyTorch 2.9.0 maximum
- **NOT** `12.0`: PyTorch 2.9.0 cannot compile for Blackwell sm_120

**Result:** CUDA code compiles for sm_90 and runs in backward-compatibility mode on sm_120.

### Standard Submodules (Work with sm_90 compilation)

#### 1. diff-gaussian-rasterization

```bash
cd /path/to/taming-3dgs

TORCH_CUDA_ARCH_LIST="8.6;9.0" \
    pip install --no-cache-dir --no-deps \
    -e submodules/diff-gaussian-rasterization
```

**Compilation time:** 1-3 minutes
**Expected output:** `Successfully installed diff_gaussian_rasterization-0.1.0`

#### 2. simple-knn

```bash
TORCH_CUDA_ARCH_LIST="8.6;9.0" \
    pip install --no-cache-dir --no-deps \
    -e submodules/simple-knn
```

**Compilation time:** 30-60 seconds
**Expected output:** `Successfully installed simple_knn-0.1.0`

### Special Case: fused-ssim (RTX 50xx)

**Problem:** Custom CUDA kernels in fused-ssim require sm_120 support, which PyTorch 2.9.0 doesn't have.

**Solution:** We've replaced the CUDA implementation with pure PyTorch (already done in this repo).

**If starting fresh:**

```bash
# Don't compile the CUDA version - it won't work on sm_120
# Instead, use the pure PyTorch version in submodules/fused-ssim/

# The __init__.py has been modified to use F.conv2d instead of custom CUDA
# No compilation needed - it's pure Python!

# Just install in development mode
pip install -e submodules/fused-ssim
```

**Verification that it's using pure PyTorch:**
```bash
grep "import torch.nn.functional as F" submodules/fused-ssim/fused_ssim/__init__.py
# Should find the import statement
```

### Parallel Compilation (Faster)

Compile all submodules simultaneously:

```bash
cd /path/to/taming-3dgs

export TORCH_CUDA_ARCH_LIST="8.6;9.0"

# Launch all compilations in background
(cd submodules/diff-gaussian-rasterization && \
    TORCH_CUDA_ARCH_LIST="8.6;9.0" pip install --no-cache-dir --no-deps -e .) &

(cd submodules/simple-knn && \
    TORCH_CUDA_ARCH_LIST="8.6;9.0" pip install --no-cache-dir --no-deps -e .) &

(cd submodules/fused-ssim && \
    pip install --no-cache-dir --no-deps -e .) &

# Wait for all to complete
wait

echo "All submodules compiled!"
```

### Common Compilation Issues

**Issue 1: `nvcc not found`**
```bash
# Solution: Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export PATH=/opt/cuda/bin:$PATH  # Alternative location
```

**Issue 2: `cuda_runtime.h: No such file or directory`**
```bash
# Solution: Add CUDA includes
export CPATH=/usr/local/cuda/include:$CPATH
export C_INCLUDE_PATH=/usr/local/cuda/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/usr/local/cuda/include:$CPLUS_INCLUDE_PATH
```

**Issue 3: `undefined reference to cudaXXX`**
```bash
# Solution: Add CUDA libraries
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
```

**Issue 4: `no kernel image available` at runtime**
- Check you set `TORCH_CUDA_ARCH_LIST="8.6;9.0"` **before** compilation
- Recompile with `--force-reinstall` flag
- For RTX 50xx: Ensure fused-ssim uses pure PyTorch version

---

## Environment Variables

### Required Variables

Create a file `~/.3dgs_env` with these settings:

```bash
# CUDA paths
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# CUDA compilation targets (CRITICAL for RTX 50xx)
export TORCH_CUDA_ARCH_LIST="8.6;9.0"
export CUDAARCHS="90"

# Optional: Performance tuning
export CUDA_LAUNCH_BLOCKING=0  # Set to 1 for debugging
export TORCH_USE_CUDA_DSA=0    # Set to 1 for detailed CUDA errors
```

**Load automatically:**
```bash
echo 'source ~/.3dgs_env' >> ~/.bashrc
source ~/.bashrc
```

### Per-Session Variables

If you don't want permanent environment variables:

```bash
# Activate conda environment
conda activate rtx5080_3dgs

# Set session variables
export TORCH_CUDA_ARCH_LIST="8.6;9.0"
export PATH=/usr/local/cuda/bin:$PATH

# Now compile or run training
```

---

## Verification

### Complete System Test

Run this comprehensive test script:

```bash
#!/bin/bash

echo "=== System Information ==="
uname -a
echo ""

echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo ""

echo "=== CUDA Toolkit ==="
nvcc --version | grep release
echo ""

echo "=== Python Environment ==="
which python
python --version
echo ""

echo "=== PyTorch Installation ==="
python << 'PYEOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
PYEOF
echo ""

echo "=== CUDA Submodules ==="
python << 'PYEOF'
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings
    print("✓ diff_gaussian_rasterization imported successfully")
except Exception as e:
    print(f"✗ diff_gaussian_rasterization failed: {e}")

try:
    from simple_knn._C import distCUDA2
    print("✓ simple_knn imported successfully")
except Exception as e:
    print(f"✗ simple_knn failed: {e}")

try:
    import fused_ssim
    print("✓ fused_ssim imported successfully")
except Exception as e:
    print(f"✗ fused_ssim failed: {e}")
PYEOF
echo ""

echo "=== CUDA Runtime Test ==="
python << 'PYEOF'
import torch

# Test basic CUDA operations
try:
    x = torch.rand(1000, 1000, device='cuda')
    y = torch.rand(1000, 1000, device='cuda')
    z = torch.matmul(x, y)
    print(f"✓ CUDA matmul test passed (result sum: {z.sum().item():.2f})")
except Exception as e:
    print(f"✗ CUDA matmul test failed: {e}")

# Test SSIM
try:
    import fused_ssim
    img1 = torch.rand(1, 3, 64, 64, device='cuda')
    img2 = torch.rand(1, 3, 64, 64, device='cuda')
    ssim = fused_ssim.fused_ssim(img1, img2)
    print(f"✓ SSIM test passed (value: {ssim.item():.6f})")
except Exception as e:
    print(f"✗ SSIM test failed: {e}")
PYEOF
echo ""

echo "=== All tests complete ==="
```

**Save as `verify_setup.sh` and run:**
```bash
chmod +x verify_setup.sh
./verify_setup.sh
```

### Expected Output

All tests should show ✓:
```
✓ diff_gaussian_rasterization imported successfully
✓ simple_knn imported successfully
✓ fused_ssim imported successfully
✓ CUDA matmul test passed
✓ SSIM test passed
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. "CUDA out of memory" during training

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**

A. **Reduce batch size / image resolution:**
```bash
# Use lower resolution images
python train.py -s data/scene -i images_8 ...  # instead of images_2

# Reduce budget
python train.py ... --budget 0.5 --mode multiplier  # instead of 2.0
```

B. **Use CPU for data loading:**
```bash
python train.py ... --data_device cpu
```

C. **Close GPU-using applications:**
```bash
# Check GPU usage
nvidia-smi

# Close browsers, monitors, etc. before training
pkill -f firefox
pkill -f chrome
```

D. **Increase densification interval:**
```bash
python train.py ... --densification_interval 2000  # instead of 500
```

#### 2. "no kernel image available for execution"

**Symptoms:**
```
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

**Root Cause:** CUDA extension compiled without sm_120 support

**Solutions:**

A. **Recompile with correct architecture flags:**
```bash
export TORCH_CUDA_ARCH_LIST="8.6;9.0"
pip uninstall diff_gaussian_rasterization simple_knn -y
pip install --no-cache-dir --force-reinstall -e submodules/diff-gaussian-rasterization
pip install --no-cache-dir --force-reinstall -e submodules/simple-knn
```

B. **For fused-ssim:** Use pure PyTorch version (see [CUDA Submodule Compilation](#cuda-submodule-compilation))

#### 3. "undefined symbol" errors

**Symptoms:**
```
ImportError: undefined symbol: _ZNK3c107SymBool14guard_or_falseEPKcl
```

**Root Cause:** PyTorch version mismatch

**Solution:**
```bash
# Ensure PyTorch 2.8+ is installed
pip install --upgrade torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128

# Recompile submodules after PyTorch upgrade
pip uninstall diff_gaussian_rasterization simple_knn fused_ssim -y
# ... recompile as shown above
```

#### 4. Conda environment conflicts

**Symptoms:**
- Packages won't install
- Import errors
- Version conflicts

**Solution: Clean slate:**
```bash
# Remove environment
conda deactivate
conda env remove -n rtx5080_3dgs

# Recreate from scratch
conda create -n rtx5080_3dgs python=3.10 -y
conda activate rtx5080_3dgs

# Follow installation steps again
```

#### 5. NCCL library conflicts

**Symptoms:**
```
ImportError: undefined symbol: ncclGroupSimulateEnd
ImportError: libnccl.so.2: cannot open shared object
```

**Solution:**
```bash
# Remove conflicting NCCL versions
pip list | grep nccl
pip uninstall nvidia-nccl-cu11 -y  # Remove cu11 version

# Install correct version
pip install nvidia-nccl-cu12==2.27.5
```

#### 6. Permission denied during compilation

**Symptoms:**
```
error: could not create '/usr/local/lib/python3.10/...'
```

**Solution:**
```bash
# Don't use sudo with pip!
# Make sure you're in conda environment
conda activate rtx5080_3dgs

# Use --user flag if necessary (not recommended in conda)
pip install --user -e submodules/...

# Or fix conda permissions
sudo chown -R $USER:$USER ~/miniconda3/envs/rtx5080_3dgs
```

---

## Performance Optimization

### Memory Optimization

```bash
# Monitor GPU memory during training
watch -n 1 nvidia-smi

# Enable memory-efficient options
python train.py \
    --data_device cpu \           # Load images on CPU
    --densification_interval 2000 # Less frequent densification
```

### Speed Optimization

```bash
# Use optimized CUDA kernels (automatically enabled)
export TORCH_CUDNN_BENCHMARK=1

# Disable debug mode
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0
```

### Multi-GPU Training

**Note:** Default taming-3dgs doesn't support multi-GPU. For single-GPU optimization:

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Or in Python
python train.py ... --gpu 0
```

---

## Environment Export/Import

### Export Working Environment

```bash
# Conda packages only
conda env export > environment.yml

# All packages (including pip)
conda env export --from-history > environment_minimal.yml

# Pip packages only
pip list --format=freeze > requirements.txt
```

### Import on Another Machine

```bash
# Method 1: From conda export
conda env create -f environment.yml

# Method 2: Manual setup + pip requirements
conda create -n rtx5080_3dgs python=3.10 -y
conda activate rtx5080_3dgs
pip install -r requirements.txt

# Then compile CUDA submodules
export TORCH_CUDA_ARCH_LIST="8.6;9.0"
# ... compilation steps ...
```

---

## Summary Checklist

### Pre-Installation
- [ ] NVIDIA GPU with 8GB+ VRAM
- [ ] NVIDIA driver 525+ installed
- [ ] CUDA toolkit 12.1+ installed
- [ ] 50GB+ free disk space
- [ ] 16GB+ system RAM

### Conda Setup
- [ ] Miniconda/Anaconda installed
- [ ] Created `rtx5080_3dgs` environment
- [ ] Python 3.10 confirmed

### PyTorch Installation
- [ ] PyTorch 2.9.0+cu128 installed
- [ ] CUDA available in PyTorch
- [ ] GPU detected by PyTorch
- [ ] (RTX 50xx only) CUDA version bypass applied

### Dependencies
- [ ] Core Python packages installed (numpy, scipy, etc.)
- [ ] lpips, plyfile, tqdm installed

### CUDA Compilation
- [ ] `TORCH_CUDA_ARCH_LIST="8.6;9.0"` set
- [ ] diff-gaussian-rasterization compiled successfully
- [ ] simple-knn compiled successfully
- [ ] fused-ssim installed (pure PyTorch for RTX 50xx)

### Verification
- [ ] All submodules import without errors
- [ ] CUDA matmul test passes
- [ ] SSIM test passes
- [ ] Training runs without errors

### Final Test
- [ ] Successfully trained for 500 iterations on test dataset
- [ ] Point cloud output generated
- [ ] No CUDA errors or memory failures

---

## Quick Reference Commands

```bash
# Activate environment
conda activate rtx5080_3dgs

# Set CUDA arch (before compilation)
export TORCH_CUDA_ARCH_LIST="8.6;9.0"

# Compile submodules
cd /path/to/taming-3dgs
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/fused-ssim

# Verify installation
python -c "from diff_gaussian_rasterization import GaussianRasterizationSettings; from simple_knn._C import distCUDA2; import fused_ssim; print('✓ All imports successful')"

# Test training
python train.py -s data/bonsai -i images_8 -m ./test --budget 0.5 --mode multiplier --iterations 100 --quiet

# Check GPU
nvidia-smi
```

---

## Additional Resources

### Documentation
- PyTorch: https://pytorch.org/docs/
- CUDA: https://docs.nvidia.com/cuda/
- Conda: https://docs.conda.io/

### Troubleshooting
- PyTorch CUDA Issues: https://pytorch.org/get-started/locally/
- NVIDIA CUDA Forum: https://forums.developer.nvidia.com/

### Related Projects
- Original 3DGS: https://github.com/graphdeco-inria/gaussian-splatting
- Taming 3DGS: Repository-specific documentation

---

**Last Updated:** October 25, 2025
**Tested On:** RTX 5080, Manjaro Linux, PyTorch 2.9.0+cu128
**Status:** ✓ Fully functional for research and production use
