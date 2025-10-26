# 3D Gaussian Splatting Training Report

## Overview
This report documents the training, rendering, and evaluation of 3D Gaussian Splatting models on multiple datasets with both budget and high-quality configurations.

## Command Executed
```bash
./train.sh
```

---

## Training Phase

### Budget Models (Fast Training)

| Scene | Training Time | Final Loss | Progress |
|-------|---------------|------------|----------|
| 🚴 bicycle | 05:39 | 0.0922729 | ✅ Complete |
| 🌿 garden | 08:52 | 0.0380270 | ✅ Complete |
| 🏠 stump | 04:12 | 0.0851958 | ✅ Complete |
| 🏠 room | 02:26 | 0.0242826 | ✅ Complete |
| 🍽️ counter | 02:49 | 0.0315194 | ✅ Complete |
| 👨‍🍳 kitchen | 03:25 | 0.0285318 | ✅ Complete |
| 🌳 bonsai | 02:43 | 0.0256222 | ✅ Complete |

**Total Budget Training Time: ~30 minutes**

### High-Quality Models (Extended Training)

| Scene | Training Time | Final Loss | Progress |
|-------|---------------|------------|----------|
| 🚴 bicycle | 27:23 | 0.0460447 | ✅ Complete |
| 🌿 garden | 24:12 | 0.0282364 | ✅ Complete |
| 🏠 stump | 19:00 | 0.0277612 | ✅ Complete |
| 🏠 room | 12:09 | 0.0201634 | ✅ Complete |
| 🍽️ counter | 11:05 | 0.0267880 | ✅ Complete |
| 👨‍🍳 kitchen | 09:29 | 0.0229252 | ✅ Complete |
| 🌳 bonsai | 05:26 | 0.0249579 | ✅ Complete |

**Total High-Quality Training Time: ~1 hour 48 minutes**

---

## Rendering Phase

### Budget Models Rendering
All budget models successfully rendered with training and test camera views:
- **Training cameras**: 109-272 views per scene
- **Test cameras**: 16-39 views per scene
- **Rendering speed**: ~1.5-2.6 it/s

### High-Quality Models Rendering
All high-quality models successfully rendered with similar camera configurations:
- **Training cameras**: 109-272 views per scene
- **Test cameras**: 16-39 views per scene
- **Rendering speed**: ~1.4-2.6 it/s

---

## Evaluation Metrics

### Budget Models Performance

| Scene | SSIM ↑ | PSNR ↑ | LPIPS ↓ |
|-------|--------|--------|---------|
| 🚴 bicycle | 0.7189 | 24.90 | 0.2926 |
| 🌿 garden | 0.8600 | 27.54 | 0.1246 |
| 🏠 stump | 0.7298 | 25.85 | 0.2946 |
| 🏠 room | 0.9099 | 31.52 | 0.2480 |
| 🍽️ counter | 0.9018 | 29.01 | 0.2188 |
| 👨‍🍳 kitchen | 0.9234 | 31.24 | 0.1384 |
| 🌳 bonsai | 0.9385 | 32.25 | 0.2174 |

**Budget Average**: SSIM: 0.8403 | PSNR: 28.90 | LPIPS: 0.2192

### High-Quality Models Performance

| Scene | SSIM ↑ | PSNR ↑ | LPIPS ↓ |
|-------|--------|--------|---------|
| 🚴 bicycle | 0.7777 | 25.48 | 0.1930 |
| 🌿 garden | 0.8736 | 27.86 | 0.0986 |
| 🏠 stump | 0.7711 | 26.60 | 0.2080 |
| 🏠 room | 0.9246 | 32.15 | 0.2086 |
| 🍽️ counter | 0.9134 | 29.52 | 0.1933 |
| 👨‍🍳 kitchen | 0.9319 | 32.12 | 0.1206 |
| 🌳 bonsai | 0.9451 | 32.75 | 0.1992 |

**High-Quality Average**: SSIM: 0.8767 | PSNR: 29.50 | LPIPS: 0.1745

---

## Performance Comparison

### Quality Improvements (High-Quality vs Budget)

| Scene | SSIM Δ | PSNR Δ | LPIPS Δ |
|-------|---------|--------|---------|
| 🚴 bicycle | +0.0588 | +0.58 | -0.0996 |
| 🌿 garden | +0.0136 | +0.32 | -0.0260 |
| 🏠 stump | +0.0413 | +0.75 | -0.0866 |
| 🏠 room | +0.0147 | +0.63 | -0.0394 |
| 🍽️ counter | +0.0116 | +0.51 | -0.0255 |
| 👨‍🍳 kitchen | +0.0085 | +0.88 | -0.0178 |
| 🌳 bonsai | +0.0066 | +0.50 | -0.0182 |

### Best Performing Scenes
- **Highest SSIM**: 🌳 bonsai (0.9451)
- **Highest PSNR**: 🌳 bonsai (32.75)
- **Lowest LPIPS**: 🌿 garden (0.0986)

---

## Summary

✅ **Training Status**: All models trained successfully
✅ **Rendering Status**: All scenes rendered without errors
✅ **Evaluation Status**: Metrics computed for all models

📁 **Output Location**: `./eval/` directory
⏱️ **Total Runtime**: ~2 hours 30 minutes
🎯 **Models Generated**: 14 total (7 budget + 7 high-quality)

---

*Report generated on 2025-10-15*