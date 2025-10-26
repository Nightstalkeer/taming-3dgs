# Submodules Build Structure Report

**Date:** 2025-10-26
**Author:** Technical Documentation
**Status:** ✅ All submodules successfully compiled and functional

---

## Executive Summary

This document provides a detailed analysis of the three custom CUDA extension submodules used in Taming 3DGS, their build artifacts, compilation status, and the reasons behind different build folder structures.

**Key Finding:** All three submodules (`diff-gaussian-rasterization`, `simple-knn`, `fused-ssim`) are successfully compiled and functional, despite only `fused-ssim` retaining a `build/` directory.

---

## Table of Contents

1. [Overview](#overview)
2. [Submodules Inventory](#submodules-inventory)
3. [Compilation Status](#compilation-status)
4. [Build Artifacts Analysis](#build-artifacts-analysis)
5. [Why Only fused-ssim Has build/ Folder](#why-only-fused-ssim-has-build-folder)
6. [Build Process Explanation](#build-process-explanation)
7. [Verification Commands](#verification-commands)
8. [Troubleshooting](#troubleshooting)
9. [References](#references)

---

## Overview

Taming 3DGS depends on three custom CUDA extensions that must be compiled from source:

1. **diff-gaussian-rasterization** - Differentiable Gaussian rasterizer
2. **simple-knn** - Fast k-nearest neighbors for initialization
3. **fused-ssim** - Optimized SSIM computation for loss calculation

Each submodule is installed as an editable pip package using PyTorch's C++/CUDA extension build system.

---

## Submodules Inventory

### Directory Structure

```
submodules/
├── diff-gaussian-rasterization/    # 2.5 MB compiled extension
│   ├── cuda_rasterizer/            # CUDA kernels (forward, backward, adam)
│   ├── diff_gaussian_rasterization/
│   │   ├── _C.cpython-310-x86_64-linux-gnu.so  # ✅ Compiled extension
│   │   └── __init__.py
│   ├── setup.py
│   └── *.egg-info/
│
├── simple-knn/                     # 2.4 MB compiled extension
│   ├── simple_knn/
│   │   ├── _C.cpython-310-x86_64-linux-gnu.so  # ✅ Compiled extension
│   │   └── .gitkeep
│   ├── simple_knn.cu               # CUDA implementation
│   ├── spatial.cu
│   ├── setup.py
│   └── *.egg-info/
│
└── fused-ssim/                     # 1.5 MB compiled extension
    ├── fused_ssim/
    │   └── fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so  # ✅ Compiled extension
    ├── build/                      # ⚠️ Leftover build artifacts
    │   ├── lib.linux-x86_64-cpython-310/
    │   │   └── fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so
    │   └── temp.linux-x86_64-cpython-310/
    ├── fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so  # ⚠️ Duplicate in root
    ├── ssim.cu
    ├── setup.py
    └── *.egg-info/
```

---

## Compilation Status

### ✅ All Submodules Successfully Compiled

| Submodule | Extension File | Size | Compiled Date | Status |
|-----------|---------------|------|---------------|---------|
| **diff-gaussian-rasterization** | `_C.cpython-310-x86_64-linux-gnu.so` | 2.5 MB | Oct 25 02:43 | ✅ Working |
| **simple-knn** | `_C.cpython-310-x86_64-linux-gnu.so` | 2.4 MB | Oct 25 02:43 | ✅ Working |
| **fused-ssim** | `fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so` | 1.5 MB | Oct 25 03:38 | ✅ Working |

### Compilation Timeline

```
Oct 25 02:43  → diff-gaussian-rasterization compiled
Oct 25 02:43  → simple-knn compiled
Oct 25 03:38  → fused-ssim compiled (~55 minutes later)
```

**Note:** The time gap between compilations (~55 minutes) is a key factor in why `fused-ssim` retained build artifacts.

---

## Build Artifacts Analysis

### Normal Extension Location (Where It Should Be)

Each compiled extension should reside in its package directory:

```python
# Import locations
from diff_gaussian_rasterization import _C
# → submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/_C.so

from simple_knn import _C
# → submodules/simple-knn/simple_knn/_C.so

import fused_ssim_cuda
# → submodules/fused-ssim/fused_ssim/fused_ssim_cuda.so
```

### fused-ssim Anomaly: Duplicate .so Files

The `fused-ssim` submodule has the compiled extension in **THREE locations**:

1. **✅ Correct location:**
   `submodules/fused-ssim/fused_ssim/fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so`

2. **⚠️ Build artifact (temporary):**
   `submodules/fused-ssim/build/lib.linux-x86_64-cpython-310/fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so`

3. **⚠️ Root directory (shouldn't be here):**
   `submodules/fused-ssim/fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so`

**Why this matters:** Only location #1 is actually used by Python imports. Locations #2 and #3 are leftover artifacts that can be safely deleted.

---

## Why Only fused-ssim Has build/ Folder

### The Build Process (setuptools + PyTorch)

When you run `pip install -e .` on a CUDA extension, the build process follows these steps:

```bash
# Step 1: Create build/ directory
mkdir build/temp.linux-x86_64-cpython-310/   # Temporary object files
mkdir build/lib.linux-x86_64-cpython-310/    # Compiled .so files

# Step 2: Compile CUDA/C++ source files
nvcc -c ssim.cu → build/temp.*/ssim.o
g++ -c ext.cpp → build/temp.*/ext.o

# Step 3: Link into shared library
nvcc -shared *.o → build/lib.*/fused_ssim_cuda.so

# Step 4: Copy to package directory
cp build/lib.*/fused_ssim_cuda.so fused_ssim/fused_ssim_cuda.so

# Step 5: Cleanup (OPTIONAL)
rm -rf build/   # ← This step may or may not happen
```

### Why Cleanup Differs

The `build/` folder cleanup depends on several factors:

| Factor | Result |
|--------|--------|
| **Installation flags** | `--no-cache-dir`, `--force-reinstall` affect cleanup |
| **Build success/failure** | Failed builds may leave artifacts |
| **setuptools version** | Newer versions may clean more aggressively |
| **Timing/interruption** | Interrupted builds leave artifacts |
| **Installation method** | `pip install -e .` vs `python setup.py install` |

### Timeline Analysis: Why fused-ssim Kept build/

```
02:43 → diff-gaussian-rasterization compiled
        ├─ build/ created
        ├─ .so compiled
        └─ build/ cleaned up ✅

02:43 → simple-knn compiled
        ├─ build/ created
        ├─ .so compiled
        └─ build/ cleaned up ✅

03:38 → fused-ssim compiled (~55 minutes later)
        ├─ build/ created
        ├─ .so compiled
        └─ build/ NOT cleaned up ⚠️
```

**Likely reason:** The 55-minute gap suggests `fused-ssim` was compiled in a separate session with different flags, possibly:
- Different pip version
- Different installation flags (`--force-reinstall`, `--no-deps`, etc.)
- Manual interruption or error during cleanup phase
- Different Python/setuptools configuration

---

## Build Process Explanation

### setup.py Configuration

All three submodules use similar `setup.py` files with PyTorch's `CUDAExtension`:

```python
# Example from fused-ssim/setup.py
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="fused_ssim",
    packages=['fused_ssim'],
    ext_modules=[
        CUDAExtension(
            name="fused_ssim_cuda",       # Name of the compiled module
            sources=["ssim.cu", "ext.cpp"]  # CUDA and C++ sources
        )
    ],
    cmdclass={'build_ext': BuildExtension}  # Uses PyTorch's custom builder
)
```

### Key Differences Between Submodules

| Submodule | Extension Name | Sources | Special Flags |
|-----------|---------------|---------|---------------|
| **diff-gaussian-rasterization** | `_C` | 5 .cu files + ext.cpp | `-I third_party/glm/` |
| **simple-knn** | `_C` | 2 .cu files + ext.cpp | `/wd4624` (Windows only) |
| **fused-ssim** | `fused_ssim_cuda` | 1 .cu file + ext.cpp | None |

### Installation Commands Used

Based on CLAUDE.md RTX 5080/5090 setup, the submodules were likely installed with:

```bash
# Standard installation (used for diff-gaussian-rasterization and simple-knn)
TORCH_CUDA_ARCH_LIST="8.6;9.0" pip install -e submodules/diff-gaussian-rasterization
TORCH_CUDA_ARCH_LIST="8.6;9.0" pip install -e submodules/simple-knn

# Force reinstall (possibly used for fused-ssim)
TORCH_CUDA_ARCH_LIST="8.6;9.0" pip install --force-reinstall --no-cache-dir --no-deps \
    -e submodules/fused-ssim
```

The `--force-reinstall --no-cache-dir --no-deps` flags may affect cleanup behavior.

---

## Verification Commands

### Check Compilation Status

```bash
# Find all compiled extension files
find submodules/ -name "*.so" -o -name "*.pyd"

# Expected output:
# submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cpython-310-x86_64-linux-gnu.so
# submodules/simple-knn/simple_knn/_C.cpython-310-x86_64-linux-gnu.so
# submodules/fused-ssim/fused_ssim/fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so
# submodules/fused-ssim/build/lib.*/fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so  # duplicate
# submodules/fused-ssim/fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so              # duplicate
```

### Test Import Functionality

```bash
python -c "
from diff_gaussian_rasterization import GaussianRasterizer
from simple_knn._C import distCUDA2
import fused_ssim
print('✅ All CUDA submodules working')
"
```

### Check File Sizes

```bash
ls -lh submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/_C*.so
# -rwxr-xr-x  2508096 Oct 25 02:43  (2.5 MB)

ls -lh submodules/simple-knn/simple_knn/_C*.so
# -rwxr-xr-x  2457208 Oct 25 02:43  (2.4 MB)

ls -lh submodules/fused-ssim/fused_ssim/fused_ssim_cuda*.so
# -rwxr-xr-x  1476760 Oct 25 03:38  (1.5 MB)
```

### Check Installed Packages

```bash
pip list | grep -E "(diff-gaussian|simple-knn|fused-ssim)"

# Expected output:
# diff-gaussian-rasterization  0.0.0  /path/to/submodules/diff-gaussian-rasterization
# fused-ssim                   0.0.0  /path/to/submodules/fused-ssim
# simple-knn                   0.0.0  /path/to/submodules/simple-knn
```

---

## Troubleshooting

### Safe Cleanup: Removing Unnecessary Artifacts

The `build/` folder and duplicate `.so` files can be safely removed:

```bash
# Remove build artifacts from fused-ssim
cd submodules/fused-ssim/
rm -rf build/
rm -f fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so  # root duplicate

# Verify import still works
python -c "import fused_ssim; print('✅ Still works')"
```

**Warning:** Do NOT delete the `.so` file inside the package directory:
```bash
# ❌ DO NOT DELETE THIS
submodules/fused-ssim/fused_ssim/fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so
```

### Rebuilding Extensions

If you need to rebuild any submodule:

```bash
# Clean rebuild with force
TORCH_CUDA_ARCH_LIST="8.6;9.0" pip install --force-reinstall --no-cache-dir --no-deps \
    -e submodules/<submodule-name>

# Standard editable install
TORCH_CUDA_ARCH_LIST="8.6;9.0" pip install -e submodules/<submodule-name>
```

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **ImportError: No module named '_C'** | Missing compiled extension | Reinstall submodule with pip |
| **CUDA mismatch error** | PyTorch CUDA ≠ System CUDA | Set `TORCH_CUDA_ARCH_LIST="8.6;9.0"` |
| **sm_120 not supported** | RTX 5080/5090 compilation fails | Use PyTorch 2.4+, bypass version check |
| **Undefined symbol** | Missing CUDA library | Check `LD_LIBRARY_PATH`, reinstall CUDA toolkit |
| **Multiple .so files** | Confusion about which is used | Only package-directory `.so` is imported |

---

## References

### Related Documentation

- **[CLAUDE.md](../../CLAUDE.md)** - Complete RTX 5080/5090 setup instructions
- **[RTX5080_SETUP_SESSION.md](../rtx5080/RTX5080_SETUP_SESSION.md)** - Initial CUDA extension setup
- **[RTX5080_PYTORCH_UPGRADE_SESSION.md](../rtx5080/RTX5080_PYTORCH_UPGRADE_SESSION.md)** - PyTorch 2.4 upgrade for Blackwell

### Submodule Repositories

- **diff-gaussian-rasterization:** Custom fork with RTX 5080/5090 fixes
- **simple-knn:** Custom fork with `#include <cstdint>` fix
- **fused-ssim:** SSIM loss implementation

### Build System Documentation

- **PyTorch C++ Extensions:** [pytorch.org/tutorials/advanced/cpp_extension.html](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- **setuptools build_ext:** [setuptools.pypa.io](https://setuptools.pypa.io)
- **CUDA Compilation:** [docs.nvidia.com/cuda/cuda-compiler-driver-nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc)

---

## Appendix: Full File Listing

### diff-gaussian-rasterization (2.5 MB)

```
diff-gaussian-rasterization/
├── CMakeLists.txt
├── cuda_rasterizer/
│   ├── auxiliary.h
│   ├── backward.cu
│   ├── backward.h
│   ├── config.h
│   ├── forward.cu
│   ├── forward.h
│   ├── adam.cu
│   └── rasterizer_impl.cu
│   └── rasterizer_impl.h
├── diff_gaussian_rasterization/
│   ├── _C.cpython-310-x86_64-linux-gnu.so  # ✅ 2.5 MB
│   ├── __init__.py
│   └── __pycache__/
├── diff_gaussian_rasterization.egg-info/
├── ext.cpp
├── rasterize_points.cu
├── rasterize_points.h
├── setup.py
└── third_party/glm/
```

### simple-knn (2.4 MB)

```
simple-knn/
├── ext.cpp
├── setup.py
├── simple_knn/
│   ├── _C.cpython-310-x86_64-linux-gnu.so  # ✅ 2.4 MB
│   └── .gitkeep
├── simple_knn.cu
├── simple_knn.h
├── simple_knn.egg-info/
├── spatial.cu
└── spatial.h
```

### fused-ssim (1.5 MB)

```
fused-ssim/
├── build/                                    # ⚠️ Can be deleted
│   ├── lib.linux-x86_64-cpython-310/
│   │   └── fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so  # duplicate
│   └── temp.linux-x86_64-cpython-310/
├── ext.cpp
├── fused_ssim/
│   └── fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so  # ✅ 1.5 MB
├── fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so     # ⚠️ Duplicate, can delete
├── fused_ssim.egg-info/
├── setup.py
├── ssim.cu
└── ssim.h
```

---

## Summary

**Status:** ✅ All three submodules are successfully compiled and functional.

**Key Findings:**
1. All extensions have the correct `.so` file in their package directories
2. Only `fused-ssim` has leftover `build/` artifacts (safe to delete)
3. Build cleanup behavior varies based on installation timing and flags
4. Duplicate `.so` files in `fused-ssim` are harmless but unnecessary

**Recommendation:**
```bash
# Clean up unnecessary artifacts (optional)
rm -rf submodules/fused-ssim/build/
rm -f submodules/fused-ssim/fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so
```

**No action required** - all submodules are working correctly as-is.

---

**Last Updated:** 2025-10-26
**Verified On:** RTX 5080 (16GB VRAM) with PyTorch 2.4.1+cu121