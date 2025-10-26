# Taming 3DGS Documentation

This directory contains comprehensive documentation organized by category for the Taming 3DGS project.

## Directory Structure

```
docs/
├── architecture/     # Codebase architecture documentation ⭐NEW⭐
├── rtx5080/          # RTX 5080 (16GB VRAM) hardware documentation
├── rtx5090/          # RTX 5090 (32GB VRAM) hardware documentation
├── setup/            # System setup and installation guides
├── training/         # Training reports and results
├── datasets/         # Dataset preparation and status
├── development/      # Development & internals documentation
└── misc/             # Miscellaneous utilities and tools
```

---

## 📁 Architecture Documentation ⭐NEW⭐

- **[CODEBASE_ARCHITECTURE.md](architecture/CODEBASE_ARCHITECTURE.md)** - **Comprehensive Python codebase architecture guide**
  - Complete documentation of all Python files and their functions
  - Detailed explanation of core algorithms (importance scoring, densification)
  - Data flow diagrams for training, rendering, and evaluation pipelines
  - Function-by-function reference guide with parameters and usage examples
  - Key algorithms with step-by-step breakdowns
  - Glossary of technical terms and concepts
  - Perfect for developers who want to understand or modify the code

---

## 📁 RTX 5080 Documentation (16GB VRAM)

### Training Reports
- **[BUDGET_MODE_TRAINING_REPORT.md](rtx5080/BUDGET_MODE_TRAINING_REPORT.md)** - Budget mode results (13/13 datasets, 2h 2m)
- **[BIG_MODE_TRAINING_REPORT.md](rtx5080/BIG_MODE_TRAINING_REPORT.md)** - Big mode results (13/13 datasets, 18.8M Gaussians, 27.66 GB)
- **[BIG_MODE_BUGS_AND_FIXES.md](rtx5080/BIG_MODE_BUGS_AND_FIXES.md)** - Big mode optimization and bug fixes

### Session Histories
- **[SESSION_HISTORY_2025-10-25.md](rtx5080/SESSION_HISTORY_2025-10-25.md)** - Latest session summary (Oct 25, 2025)
- **[RTX5080_SETUP_SESSION.md](rtx5080/RTX5080_SETUP_SESSION.md)** - Initial RTX 5080 setup session
- **[RTX5080_PYTORCH_UPGRADE_SESSION.md](rtx5080/RTX5080_PYTORCH_UPGRADE_SESSION.md)** - PyTorch upgrade for Blackwell
- **[RTX5080_TRAINING_FIX_SESSION.md](rtx5080/RTX5080_TRAINING_FIX_SESSION.md)** - Training bug fix session
- **[RTX5080_TRAINING_SUCCESS.md](rtx5080/RTX5080_TRAINING_SUCCESS.md)** - First successful training

### Validation & Bug Fixes
- **[RTX5080_TRAINING_SCRIPT_VALIDATION.md](rtx5080/RTX5080_TRAINING_SCRIPT_VALIDATION.md)** - Script validation
- **[DENSIFICATION_ALGORITHM_FIX.md](rtx5080/DENSIFICATION_ALGORITHM_FIX.md)** - Budget control algorithm fix
- **[LOOP_BUG_FIX.md](rtx5080/LOOP_BUG_FIX.md)** - Training loop script fix

---

## 📁 RTX 5090 Documentation (32GB VRAM)

- **[README.md](rtx5090/README.md)** - RTX 5090 setup predictions and future documentation plan

**Status:** ⏳ Awaiting hardware availability

---

## 📁 Setup & Installation

- **[QUICK_SETUP.md](setup/QUICK_SETUP.md)** - Quick start setup guide
- **[SETUP_REPORT.md](setup/SETUP_REPORT.md)** - Detailed setup report
- **[SYSTEM_SETUP_GUIDE.md](setup/SYSTEM_SETUP_GUIDE.md)** - Comprehensive system setup guide

---

## 📁 Training Reports

- **[ALL_DATASETS_TRAINING_REPORT.md](training/ALL_DATASETS_TRAINING_REPORT.md)** - Complete datasets training report
- **[FINAL_ALL_DATASETS_REPORT.md](training/FINAL_ALL_DATASETS_REPORT.md)** - Final comprehensive report
- **[TRAINING_COMMANDS_REPORT.md](training/TRAINING_COMMANDS_REPORT.md)** - Training commands reference

---

## 📁 Dataset Documentation

- **[DATASET_STATUS_REPORT.md](datasets/DATASET_STATUS_REPORT.md)** - Dataset preparation status

---

## 📁 Development & Architecture

- **[SUBMODULES_BUILD_STRUCTURE.md](development/SUBMODULES_BUILD_STRUCTURE.md)** - CUDA submodules compilation and build artifacts analysis

---

## 📁 Miscellaneous

- **[RunMarkdown.md](misc/RunMarkdown.md)** - Markdown utilities and tools

---

## Quick Links by Topic

### 🚀 Getting Started
1. [Quick Setup Guide](setup/QUICK_SETUP.md) - Start here for fast setup
2. [System Setup Guide](setup/SYSTEM_SETUP_GUIDE.md) - Comprehensive installation
3. [Dataset Status](datasets/DATASET_STATUS_REPORT.md) - Check dataset availability

### 📖 Understanding the Code
1. [Codebase Architecture](architecture/CODEBASE_ARCHITECTURE.md) - **Complete Python codebase documentation**
2. [Submodules Build Structure](development/SUBMODULES_BUILD_STRUCTURE.md) - CUDA extensions

### 🔧 RTX 5080 Setup & Troubleshooting
1. [RTX 5080 Setup Session](rtx5080/RTX5080_SETUP_SESSION.md) - Initial setup
2. [PyTorch Upgrade Guide](rtx5080/RTX5080_PYTORCH_UPGRADE_SESSION.md) - Blackwell compatibility
3. [Bug Fixes](rtx5080/BIG_MODE_BUGS_AND_FIXES.md) - All known issues and solutions

### 📊 Training Results
1. [Budget Mode Report](rtx5080/BUDGET_MODE_TRAINING_REPORT.md) - Budget mode (13/13 datasets)
2. [Big Mode Report](rtx5080/BIG_MODE_BUGS_AND_FIXES.md) - Big mode optimization
3. [All Datasets Report](training/ALL_DATASETS_TRAINING_REPORT.md) - Complete training results

### 🐛 Bug Fixes & Troubleshooting
1. [Densification Algorithm Fix](rtx5080/DENSIFICATION_ALGORITHM_FIX.md)
2. [Loop Bug Fix](rtx5080/LOOP_BUG_FIX.md)
3. [Big Mode Bugs](rtx5080/BIG_MODE_BUGS_AND_FIXES.md)

---

## Hardware Comparison

| Feature               | RTX A6000      | RTX 5080           | RTX 5090           |
|-----------------------|----------------|--------------------|--------------------|
| **VRAM**              | 48 GB          | 16 GB              | 32 GB              |
| **Architecture**      | Ampere (sm_86) | Blackwell (sm_120) | Blackwell (sm_120) |
| **Big Mode Interval** | 100            | 300                | 150-200 (est.)     |
| **Budget Mode**       | ✅              | ✅                  | ✅ (predicted)      |
| **Big Mode**          | ✅              | ✅ (optimized)      | ✅ (predicted)      |
| **Documentation**     | Original       | ✅ Complete         | ⏳ Pending          |

---

## Training Results Summary

### RTX 5080 Results

#### Budget Mode (Completed ✅)
- **Status:** 13/13 datasets successful
- **Time:** 2h 2m 26s total
- **Output:** 1.93 GB
- **Quality:** Within ±0.2% of target budget
- **Report:** [BUDGET_MODE_TRAINING_REPORT.md](rtx5080/BUDGET_MODE_TRAINING_REPORT.md)

#### Big Mode (Completed ✅)
- **Status:** 13/13 datasets trained and rendered
- **Total Gaussians:** 18.8M (2.43x avg increase over Budget Mode)
- **Total Storage:** 27.66 GB
- **Quality Range:** 1.43x - 14.26x more Gaussians per dataset
- **Report:** [BIG_MODE_TRAINING_REPORT.md](rtx5080/BIG_MODE_TRAINING_REPORT.md)
- **Technical Analysis:** [BIG_MODE_BUGS_AND_FIXES.md](rtx5080/BIG_MODE_BUGS_AND_FIXES.md)

---

## Key Findings & Optimizations

### RTX 5080 Optimizations
1. **Memory Management** - Densification interval: 100 → 300 for 16GB VRAM
2. **Fragmentation Fix** - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
3. **Array Bounds Fix** - Safe indexing for variable densification intervals
4. **Blackwell Support** - PyTorch 2.4+ with TORCH_CUDA_ARCH_LIST="8.6;9.0"

### Critical Bug Fixes
- **Bug #1:** CUDA OOM Error (densification_interval optimization)
- **Bug #2:** Memory Fragmentation (expandable_segments config)
- **Bug #3:** Array Index Out of Bounds (safe min() indexing)

---

## Contributing to Documentation

When adding new documentation:

1. **Choose the appropriate category:**
   - `rtx5080/` - RTX 5080 specific content
   - `rtx5090/` - RTX 5090 specific content
   - `setup/` - Installation and setup guides
   - `training/` - Training results and reports
   - `datasets/` - Dataset information
   - `development/` - Codebase architecture and technical documentation
   - `misc/` - Utilities and other content

2. **Update this README.md** with links to new documents

3. **Follow naming conventions:**
   - Use descriptive names (e.g., `RTX5080_BIG_MODE_REPORT.md`)
   - Include dates for session histories (e.g., `SESSION_HISTORY_2025-10-25.md`)

4. **Update the main README.md** if necessary

---

## External Resources

- **Main Project:** [README.md](../README.md) - Project overview and quick start
- **Claude Code Guide:** [CLAUDE.md](../CLAUDE.md) - Complete setup instructions for Claude Code
- **Original Paper:** [Taming 3DGS Paper](https://arxiv.org/abs/2406.15643)
- **Project Website:** [https://humansensinglab.github.io/taming-3dgs/](https://humansensinglab.github.io/taming-3dgs/)

---

**Last Updated:** 2025-10-26
**Total Documents:** 22 files across 8 categories
**Maintained By:** Claude Code sessions with user vortex
