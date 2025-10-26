# Quick Setup Reference - 3DGS on RTX 5080

**One-page quick reference for experienced users**

## TL;DR

```bash
# 1. Install NVIDIA driver + CUDA 12.8+
nvidia-smi  # Verify driver 525+

# 2. Create conda environment
conda create -n rtx5080_3dgs python=3.10 -y
conda activate rtx5080_3dgs

# 3. Install PyTorch
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 4. Install dependencies
pip install numpy scipy pillow plyfile tqdm lpips pyyaml requests websockets

# 5. Set CUDA arch (CRITICAL for RTX 50xx)
export TORCH_CUDA_ARCH_LIST="8.6;9.0"

# 6. Compile submodules
cd /path/to/taming-3dgs
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/fused-ssim  # Pure PyTorch version

# 7. Test
python train.py -s data/bonsai -i images_8 -m ./test --budget 0.5 --mode multiplier --iterations 100
```

## Critical for RTX 50xx (Blackwell sm_120)

### Before Compilation
```bash
export TORCH_CUDA_ARCH_LIST="8.6;9.0"  # Force sm_90 compilation
```

### CUDA Version Bypass
Edit: `<conda_env>/lib/python3.10/site-packages/torch/utils/cpp_extension.py`

Lines ~519-521:
```python
# BYPASS: CUDA version check disabled for RTX 5080 compatibility
# if cuda_ver.major != torch_cuda_version.major:
#     raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
```

### fused-ssim Fix
The fused-ssim in this repo already uses pure PyTorch (no CUDA kernels).
Just `pip install -e submodules/fused-ssim` - no compilation needed.

## Verification

```bash
python << 'EOF'
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings
from simple_knn._C import distCUDA2
import fused_ssim

print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute: {torch.cuda.get_device_capability(0)}")

# Test CUDA
x = torch.rand(100, 100, device='cuda')
print(f"✓ CUDA works: {x.sum().item():.2f}")

# Test SSIM
img1 = torch.rand(1, 3, 64, 64, device='cuda')
img2 = torch.rand(1, 3, 64, 64, device='cuda')
ssim = fused_ssim.fused_ssim(img1, img2)
print(f"✓ SSIM works: {ssim.item():.6f}")
EOF
```

## Common Issues

| Error | Solution |
|-------|----------|
| `no kernel image available` | Recompile with `TORCH_CUDA_ARCH_LIST="8.6;9.0"` |
| `CUDA out of memory` | Use `--data_device cpu` or `images_8` |
| `undefined symbol` | Upgrade to PyTorch 2.9.0, recompile submodules |
| `libnccl.so.2` | `pip uninstall nvidia-nccl-cu11 -y && pip install nvidia-nccl-cu12==2.27.5` |

## Memory-Optimized Training

```bash
python train.py \
    -s data/scene \
    -i images_8 \              # Lowest resolution
    -m ./output \
    --budget 0.3 \             # Minimal Gaussians
    --mode multiplier \
    --data_device cpu \        # Load data on CPU
    --densification_interval 3000 \
    --iterations 500 \
    --test_iterations -1 \
    --quiet
```

## Full Production Training

```bash
python train.py \
    -s data/scene \
    -i images_4 \              # Higher resolution
    -m ./output \
    --budget 15 \              # Standard for outdoor
    --mode multiplier \
    --densification_interval 500 \
    --iterations 30000 \
    --eval
```

## Environment Variables

Add to `~/.bashrc`:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.6;9.0"
export CUDAARCHS="90"
```

## System Requirements

- **GPU:** 8GB+ VRAM (16GB recommended)
- **RAM:** 16GB+ (32GB recommended)
- **Storage:** 50GB+ (for conda env + datasets)
- **Driver:** NVIDIA 525+
- **CUDA:** 12.1+ (12.8 recommended)

## File Locations

- Conda env: `~/miniconda3/envs/rtx5080_3dgs/`
- PyTorch cpp_extension.py: `<env>/lib/python3.10/site-packages/torch/utils/cpp_extension.py`
- Submodules: `<repo>/submodules/{diff-gaussian-rasterization,simple-knn,fused-ssim}/`

## Quick Diagnostics

```bash
# GPU info
nvidia-smi

# CUDA version
nvcc --version

# Python/PyTorch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# GPU memory
watch -n 1 nvidia-smi

# Training progress
tail -f <output_dir>/log.txt
```

---

**Full Guide:** See `SYSTEM_SETUP_GUIDE.md` for detailed explanations
**Training Success Report:** See `RTX5080_TRAINING_SUCCESS.md` for fixes applied
