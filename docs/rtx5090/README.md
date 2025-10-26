# RTX 5090 Documentation

**Status:** Placeholder - Awaiting hardware testing

## Overview

This directory will contain documentation specific to NVIDIA GeForce RTX 5090 (32GB VRAM) once hardware becomes available for testing.

## Expected Specifications

- **GPU:** NVIDIA GeForce RTX 5090
- **VRAM:** 32GB GDDR7
- **Architecture:** Blackwell (compute capability sm_120)
- **Expected Big Mode Interval:** 150-200 (between RTX 5080 and A6000)

## Predicted Compatibility

Based on RTX 5080 results, the RTX 5090 should:
- ✅ Run Budget Mode without issues
- ✅ Run Big Mode with interval=150-200 (less aggressive than A6000's 100, more aggressive than RTX 5080's 300)
- ✅ Require same Blackwell architecture fixes (PyTorch 2.4+, TORCH_CUDA_ARCH_LIST="8.6;9.0")
- ✅ Benefit from expandable_segments memory configuration

## Future Documentation

When RTX 5090 hardware is available, this directory will include:
- RTX5090_TRAINING_SCRIPT_VALIDATION.md
- RTX5090_BUDGET_MODE_REPORT.md
- RTX5090_BIG_MODE_REPORT.md
- Optimal densification interval benchmarks
- Memory usage profiles
- Performance comparisons with RTX 5080 and A6000

## Setup Instructions

Until hardware-specific testing is complete, use the RTX 5080 setup instructions from [CLAUDE.md](../../CLAUDE.md) with these modifications:
- Increase densification interval for Big Mode from 300 to 150-200
- Expect better memory headroom (32GB vs 16GB)
- Same PyTorch and CUDA requirements as RTX 5080

---

**Last Updated:** 2025-10-25
**Status:** Awaiting RTX 5090 hardware availability
