# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Taming 3DGS is a research implementation for high-quality radiance field reconstruction with limited resources. It improves upon the original 3D Gaussian Splatting (3DGS) by making the densification process deterministic and implementing low-level optimizations for fast convergence.

This is a fork/enhancement of the Inria GRAPHDECO 3D Gaussian Splatting implementation with additional budget-constrained training modes.

## Environment Setup

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate taming_3dgs
```

The project requires:
- Python 3.7.13
- PyTorch 1.12.1
- CUDA 11.6
- Three custom submodules (diff-gaussian-rasterization, simple-knn, fused-ssim)

### RTX 5080/5090 (Blackwell Architecture) Compatibility

**Issue**: RTX 5080 and RTX 5090 GPUs have compute capability sm_120, which is not supported by older PyTorch versions. Additionally, newer CUDA toolkits (12.9) may mismatch with PyTorch's bundled CUDA version.

**Solution**: Follow these steps for RTX 5080/5090 setup:

1. **Set CUDA architecture environment variables first** (forces sm_90 compilation for compatibility):
   ```bash
   # Add to ~/.bashrc or ~/.zshrc:
   export TORCH_CUDA_ARCH_LIST="8.6;9.0"
   export CUDAARCHS="90"

   # Then source the file or restart terminal
   source ~/.bashrc  # or ~/.zshrc
   ```

2. **Upgrade to PyTorch 2.4+** (required for C++ API compatibility):
   ```bash
   conda activate rtx5080_3dgs

   # Upgrade PyTorch to 2.4+ with CUDA 12.1
   pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   # Verify version
   python -c "import torch; print('PyTorch:', torch.__version__)"
   ```

3. **Bypass CUDA version check in PyTorch** (if system CUDA != PyTorch CUDA):

   **IMPORTANT**: This step must be done AFTER upgrading PyTorch, as the upgrade overwrites cpp_extension.py.

   Edit: `<conda_env>/lib/python3.10/site-packages/torch/utils/cpp_extension.py`

   Find lines ~412-413 and comment out the version check:
   ```python
   # BYPASS: CUDA version check disabled for RTX 5080/5090 compatibility
   # if cuda_ver.major != torch_cuda_version.major:
   # raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
   ```

4. **Compile CUDA submodules with forced architecture**:
   ```bash
   cd /path/to/taming-3dgs

   TORCH_CUDA_ARCH_LIST="8.6;9.0" pip install --force-reinstall --no-cache-dir --no-deps -e submodules/diff-gaussian-rasterization
   TORCH_CUDA_ARCH_LIST="8.6;9.0" pip install --force-reinstall --no-cache-dir --no-deps -e submodules/simple-knn
   TORCH_CUDA_ARCH_LIST="8.6;9.0" pip install --force-reinstall --no-cache-dir --no-deps -e submodules/fused-ssim
   ```

5. **Verify installation**:
   ```bash
   python -c "
   from diff_gaussian_rasterization import GaussianRasterizer
   from simple_knn._C import distCUDA2
   import fused_ssim
   print('✓ All CUDA submodules working')
   "
   ```

**Note**: The header fixes (`#include <cstdint>` in `rasterizer_impl.h` and `#include <cfloat>` in `simple_knn.cu`) are already present in this repository's submodules.

**Performance**: Using sm_90 kernels on sm_120 hardware runs in backward-compatibility mode with minor performance impact, but training remains functional.

## Training Commands

### Single Scene Training

Basic training command structure:
```bash
python train.py -s <dataset_path> -m <output_path> [options]
```

Key training flags:
- `-s/--source_path`: Path to dataset (must contain `sparse/` directory for COLMAP data or `transforms_train.json` for Blender format)
- `-m/--model_path`: Output directory for trained model
- `-i/--images`: Image subdirectory name (e.g., `images_2`, `images_4` for downsampled data)
- `--budget`: Final Gaussian count (interpretation depends on `--mode`)
- `--mode`: Either `multiplier` (budget = multiplier × SfM point count) or `final_count` (budget = exact final count)
- `--densification_interval`: Frequency of densification operations (default: 100)
- `--cams`: Number of cameras for computing Gaussian scores (default: 10)
- `--optimizer_type`: Optimizer type (default: "default")
- `--sh_lower`: Enable less-frequent SH updates (once every 16 iterations) for speed
- `--ho_iteration`: Iteration to enable high-opacity Gaussians (default: 15000)
- `--websockets`: Enable web-based viewer during training
- `--port`: Port number for web viewer (requires `--websockets`)
- `--eval`: Create train/test split for evaluation
- `--test_iterations`: Iterations at which to run test evaluation (use -1 to disable)
- `--quiet`: Suppress detailed output

### Budget vs Big Mode Examples

Budget mode (constrained Gaussian count):
```bash
# MipNeRF360 outdoor scenes: 15x multiplier
python train.py -s data/bicycle -i images_4 -m ./eval/bicycle_budget --budget 15 --mode multiplier --densification_interval 500 --eval

# Tanks&Temples indoor scenes: 2x multiplier
python train.py -s data/room -i images_2 -m ./eval/room_budget --budget 2 --mode multiplier --densification_interval 500 --eval
```

Big mode (unconstrained, high-quality):
```bash
# Specify exact final Gaussian count
python train.py -s data/bicycle -i images_4 -m ./eval/bicycle_big --budget 5987095 --mode final_count --densification_interval 100 --eval
```

### Full Evaluation Pipeline

Use `full_eval.py` to run all datasets:
```bash
python full_eval.py \
    -m360 <MipNeRF360_path> \
    -tat <TanksAndTemples_path> \
    -db <DeepBlending_path> \
    --mode budget  # or --mode big
    [--sh_lower] \
    [--dry_run]
```

Flags:
- `--skip_training`: Skip training phase
- `--skip_rendering`: Skip rendering phase
- `--skip_metrics`: Skip metrics computation
- `--dry_run`: Print commands without executing

## Rendering

Render trained models to generate test/train images:
```bash
python render.py -m <model_path> [--iteration <iter>] [--skip_train] [--skip_test]
```

Output structure:
```
<model_path>/
  train/ours_<iteration>/
    renders/  # Rendered images
    gt/       # Ground truth images
  test/ours_<iteration>/
    renders/
    gt/
```

## Metrics Evaluation

Compute PSNR, SSIM, and LPIPS metrics on rendered images:
```bash
python metrics.py -m <model_path>
```

Outputs:
- `<model_path>/results.json`: Average metrics per method
- `<model_path>/per_view.json`: Per-image metrics

## Architecture

### Core Components

**Scene Representation** (`scene/`):
- `gaussian_model.py`: GaussianModel class managing 3D Gaussian primitives with position, opacity, scaling, rotation, and spherical harmonic features
- `__init__.py`: Scene class handling dataset loading (COLMAP or Blender format), camera management, and model initialization
- `dataset_readers.py`: Parsers for COLMAP and Blender dataset formats
- `cameras.py`: Camera representation with intrinsics/extrinsics

**Rendering** (`gaussian_renderer/`):
- Uses custom CUDA rasterizer (in `submodules/diff-gaussian-rasterization`)
- `network_gui.py` and `network_gui_ws.py`: Optional web-based viewer for monitoring training

**Training Pipeline** (`train.py`):
- Iterative optimization with adaptive densification/pruning
- Uses Gaussian importance scoring (computed via `utils/taming_utils.py`) to control primitive count
- Two rendering modes: sigmoid-based opacity (training) vs absolute values (rendering)
- Densification occurs between `densify_from_iter` (500) and `densify_until_iter` (15000) at `densification_interval` frequency

**Key Algorithmic Contribution** (`utils/taming_utils.py`):
- `compute_gaussian_score()`: Computes importance scores for each Gaussian primitive based on:
  - Gaussian properties: gradients, opacity, depth, radii, scale
  - Per-pixel properties: distance accumulation, loss accumulation, blend weights, render counts
  - Edge-weighted photometric loss
- `get_count_array()`: Generates pruning schedule to reach target budget
- Importance scoring uses multiple configurable coefficients (see `score_coefficients` in train.py)

### Training Arguments Structure

Arguments are organized into three parameter groups (`arguments/__init__.py`):

1. **ModelParams**: Dataset paths, image resolution, background color, eval split
2. **PipelineParams**: Rendering configuration (SH conversion, covariance computation)
3. **OptimizationParams**: Learning rates, densification parameters, iteration counts, optimizer selection

### Optimizer Types

The code supports custom optimizers including SparseGaussianAdam (when available from diff-gaussian-rasterization).

### Dataset Format Requirements

**COLMAP format** (preferred):
- Must have `sparse/` directory with COLMAP reconstruction
- Images in subdirectories (e.g., `images/`, `images_2/`, `images_4/`)

**Blender format**:
- Must have `transforms_train.json` and optionally `transforms_test.json`
- Images referenced in JSON files

## Web Viewer Usage

To monitor training in real-time:

1. Forward port to your local machine if training remotely
2. Update port number in `web_viewer/app.js`
3. Start training with `--websockets --port <port_num>`
4. Open `web_viewer/render.html` in browser

## Submodules

Three custom CUDA extensions in `submodules/`:
- `diff-gaussian-rasterization`: Differentiable rasterizer with custom backward passes
- `simple-knn`: Fast k-nearest neighbors for initialization
- `fused-ssim`: Optimized SSIM computation

All are installed as editable pip packages during environment setup.

## Testing Individual Scenes

When testing changes, use a single small scene first:
```bash
python train.py -s data/bonsai -i images_2 -m ./test_output --budget 2 --mode multiplier --quiet --eval
python render.py -m ./test_output
python metrics.py -m ./test_output
```

## Notes

- Training outputs are saved to `<model_path>/point_cloud/iteration_<N>/point_cloud.ply`
- Checkpoints include optimizer state and can be resumed with `--start_checkpoint <path>`
- The importance scoring mechanism is the key differentiator from vanilla 3DGS
- Scene-specific budget values for "big" mode are hardcoded in `full_eval.py`
