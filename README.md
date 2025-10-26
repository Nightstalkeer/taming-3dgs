# Taming 3DGS: High-Quality Radiance Fields with Limited Resources
Saswat Subhajyoti Mallick*, Rahul Goel*, Bernhard Kerbl, Francisco Vicente Carrasco, Markus Steinberger, Fernando De La Torre (* indicates equal contribution)

[![arxiv](https://img.shields.io/badge/arxiv-2406.15643-red)](https://arxiv.org/abs/2406.15643)
[![webpage](https://img.shields.io/badge/webpage-green)](https://humansensinglab.github.io/taming-3dgs/)

<p align="center">
    <img src="assets/teaser_taming.jpg" width="700px"/>
</p>

**TL;DR** We improve the densification process to make the primitive count deterministic and implement several low-level optimizations for fast convergence.

## Usage
Clone the repository
```bash
git clone https://github.com/humansensinglab/taming-3dgs.git --recursive
```
Follow the instructions in the [original 3DGS repository](https://github.com/graphdeco-inria/gaussian-splatting) to setup the environment.

<details>
<summary><span style="font-weight: bold;">additional flags for train.py</span></summary>

  #### --cams
  Number of cameras required to compute gaussian scores. Default set to 10.
  #### --budget
  The final number of gaussians to end up with. Can be a float or an integer based on `--mode`.
  #### --mode
  multiplier: the final count of gaussians will be `multiplier` x the initial (SfM) count <br>
  final_count: the final count of gaussians will be set exactly to `final_count`.
  #### --websockets
  Whether to use the web based viewer or not.
  #### --ho_iteration
  High opacity gaussians will be enabled from which iteration. Defaults to 15000 (after densification ends).
  #### --sh_lower
  Whether to enable less-frequent (once every 16 iterations) SH updates to gain speed.
  #### --benchmark_dir
  The location of the folder where the timing results are stored. No time profiling will be done if left blank.
</details>
<br>

## Reproducibility

Our experiments were carried out on a machine with 24 vCPUs, 512GB RAM and 1xNvidia RTX A4500 20G GPU. For reproducing the results reported in our paper, please run the following scripts from the project root:
```bash
# MODE={budget|big}
python full_eval.py \
    -m360 <MipNeRF360 dataset path> \
    -tat <TanksAndTemples dataset path> \
    -db <DeepBlending dataset path> \
    --sh_lower \
    --mode ${MODE}
```

* `MODE=budget`: results on budgeted setting [reported as **Ours** in the paper]
* `MODE=big`   : results on unconstrained setting [reported as **Ours(Big)** in the paper]

Add `--dry_run` to print the commands without executing them. The `train.sh` has a compiled list of scripts for evaluation.

## Web viewer
We provide a basic browser-based renderer to track the training progress. Steps for usage are as follows: <br>
1. Move the [web-viewer](./web_viewer/) folder into the host machine from where you wish to view the training.
2. Assign a port number while spawning a training session using `train.py --port <port_num> --websockets`. If you're running this on a remote server, forward the port to your host system.
3. Maintain the same port number in the [app.js](./web_viewer/app.js) file.
4. Open [render.html](./web_viewer/render.html) in your browser.

## Note
The performance optimizations that are **drop-in replacements** for the original implementation are available under the [rasterizer](https://github.com/humansensinglab/taming-3dgs/tree/rasterizer) branch. All the performance optimizations are released under the MIT License. Please refer to the [Inria repository](https://github.com/graphdeco-inria/gaussian-splatting) for complete instructions.

They have also been integrated with the original [Inria repository](https://github.com/graphdeco-inria/gaussian-splatting) so it can be directly used there.

---

## Consumer GPU Support (RTX 5080/5090)

This repository includes optimizations and documentation for running Taming 3DGS on consumer NVIDIA RTX 50-series GPUs (Blackwell architecture). These GPUs have different memory and compute constraints compared to workstation GPUs like the A6000.

### Quick Start - RTX 5080/5090

1. **Setup Environment** - See [CLAUDE.md](CLAUDE.md) for RTX 5080/5090 setup instructions
2. **Run Training** - Use `train_rtx5080.sh` for optimized training on 16GB consumer GPUs
3. **Check Reports** - See documentation below for detailed results and bug fixes

### Hardware-Specific Documentation

📂 **[Complete Documentation Index](docs/README.md)** - Organized by hardware platform

#### 📁 RTX 5080 (16GB VRAM) - Consumer GPU
- **[RTX5080_TRAINING_SCRIPT_VALIDATION.md](docs/rtx5080/RTX5080_TRAINING_SCRIPT_VALIDATION.md)** - Initial test mode validation
- **[BUDGET_MODE_TRAINING_REPORT.md](docs/rtx5080/BUDGET_MODE_TRAINING_REPORT.md)** - ✅ **COMPLETE PIPELINE**: Training + Rendering + Metrics (13/13 datasets, avg PSNR 25.93 dB)
- **[BIG_MODE_TRAINING_REPORT.md](docs/rtx5080/BIG_MODE_TRAINING_REPORT.md)** - Big mode training results (high-quality, unconstrained)
- **[BIG_MODE_BUGS_AND_FIXES.md](docs/rtx5080/BIG_MODE_BUGS_AND_FIXES.md)** - Big mode optimization and bug fixes
- **[SESSION_HISTORY_2025-10-25.md](docs/rtx5080/SESSION_HISTORY_2025-10-25.md)** - Latest session summary
- **[DENSIFICATION_ALGORITHM_FIX.md](docs/rtx5080/DENSIFICATION_ALGORITHM_FIX.md)** - Budget control algorithm fix
- **[LOOP_BUG_FIX.md](docs/rtx5080/LOOP_BUG_FIX.md)** - Training loop script fix

#### 📁 RTX 5090 (32GB VRAM) - High-End Consumer GPU
*Note: RTX 5090 with 32GB VRAM can likely use settings closer to A6000 (interval=100-200)*
- **[RTX5090 Documentation](docs/rtx5090/README.md)** - Placeholder and predictions (awaiting hardware)

### Key Differences: Consumer vs Workstation GPUs

| Feature | RTX A6000 (48GB) | RTX 5080 (16GB) | RTX 5090 (32GB) |
|---------|------------------|-----------------|-----------------|
| **VRAM** | 48 GB | 16 GB | 32 GB |
| **Architecture** | Ampere (sm_86) | Blackwell (sm_120) | Blackwell (sm_120) |
| **Big Mode Interval** | 100 | 300 (optimized) | 150-200 (estimated) |
| **Budget Mode** | ✅ Works | ✅ Works | ✅ Works |
| **Big Mode** | ✅ Works | ✅ Works (optimized) | ✅ Expected to work |
| **Memory Strategy** | Default | Expandable segments | Expandable segments |

### Training Scripts

- **`train_rtx5080.sh`** - Optimized for RTX 5080 (16GB VRAM)
  - Budget mode: interval=500, multiplier mode
  - Big mode: interval=300, final_count mode
  - Includes memory fragmentation fixes

- **`train.py`** - Core training script (shared across all GPUs)
  - Fixed array bounds checking for variable densification intervals
  - Works with any interval value (100, 200, 300, 400, 500+)

### Known Issues & Fixes (RTX 5080)

#### ✅ Fixed Issues
1. **CUDA OOM Error** - Solved by reducing densification interval from 100 → 300
2. **Memory Fragmentation** - Fixed with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
3. **Array Index Out of Bounds** - Fixed with safe array indexing in train.py:176-184

#### 🔧 Blackwell Architecture Compatibility
- **Issue:** RTX 5080/5090 have compute capability sm_120 (not supported by PyTorch <2.4)
- **Solution:** Use PyTorch 2.4+ with CUDA 12.1+, compile with TORCH_CUDA_ARCH_LIST="8.6;9.0"
- **Details:** See [CLAUDE.md](CLAUDE.md) RTX 5080/5090 section

### Training Results (RTX 5080)

#### Budget Mode (Completed ✅)
- **Status:** 13/13 datasets successful
- **Time:** 2h 2m 26s total
- **Output:** 1.93 GB
- **Quality:** Within ±0.2% of target budget
- **Report:** [BUDGET_MODE_TRAINING_REPORT.md](docs/rtx5080/BUDGET_MODE_TRAINING_REPORT.md)

#### Big Mode (Completed ✅)
- **Status:** 13/13 datasets trained and rendered
- **Total Gaussians:** 18.8M (2.43x avg increase over Budget Mode)
- **Total Storage:** 27.66 GB
- **Quality Range:** 1.43x - 14.26x more Gaussians per dataset
- **Report:** [BIG_MODE_TRAINING_REPORT.md](docs/rtx5080/BIG_MODE_TRAINING_REPORT.md)
- **Technical Analysis:** [BIG_MODE_BUGS_AND_FIXES.md](docs/rtx5080/BIG_MODE_BUGS_AND_FIXES.md)

### File Organization

```
taming-3dgs/
├── README.md                           # This file (main documentation)
├── CLAUDE.md                           # Claude Code project instructions
├── LICENSE.md / LICENSE_ORIGINAL.md   # License files
│
.├── docs/                               # 📁 Documentation directory (22 files)
│   ├── README.md                       # Documentation hub and index
│   │
│   ├── architecture/                   # Codebase architecture - 1 file ⭐NEW⭐
│   │   └── CODEBASE_ARCHITECTURE.md    # Complete Python codebase documentation
│   │
│   ├── rtx5080/                        # RTX 5080 (16GB VRAM) - 10 files
│   │   ├── BUDGET_MODE_TRAINING_REPORT.md
│   │   ├── BIG_MODE_BUGS_AND_FIXES.md
│   │   ├── SESSION_HISTORY_2025-10-25.md
│   │   ├── RTX5080_SETUP_SESSION.md
│   │   ├── RTX5080_PYTORCH_UPGRADE_SESSION.md
│   │   ├── RTX5080_TRAINING_FIX_SESSION.md
│   │   ├── RTX5080_TRAINING_SUCCESS.md
│   │   ├── RTX5080_TRAINING_SCRIPT_VALIDATION.md
│   │   ├── DENSIFICATION_ALGORITHM_FIX.md
│   │   └── LOOP_BUG_FIX.md
│   │
│   ├── rtx5090/                        # RTX 5090 (32GB VRAM) - 1 file
│   │   └── README.md                   # Placeholder & predictions
│   │
│   ├── setup/                          # Setup & installation - 3 files
│   │   ├── QUICK_SETUP.md
│   │   ├── SETUP_REPORT.md
│   │   └── SYSTEM_SETUP_GUIDE.md
│   │
│   ├── training/                       # Training reports - 3 files
│   │   ├── ALL_DATASETS_TRAINING_REPORT.md
│   │   ├── FINAL_ALL_DATASETS_REPORT.md
│   │   └── TRAINING_COMMANDS_REPORT.md
│   │
│   ├── development/                    # Development & internals - 1 file
│   │   └── SUBMODULES_BUILD_STRUCTURE.md
│   │
│   ├── datasets/                       # Dataset documentation - 1 file
│   │   └── DATASET_STATUS_REPORT.md
│   │
│   └── misc/                           # Utilities - 1 file
│       └── RunMarkdown.md
│
├── Training Scripts/
│   ├── train_rtx5080.sh                # RTX 5080 optimized
│   ├── train.py                        # Core training
│   ├── render.py                       # Rendering
│   ├── metrics.py                      # Quality metrics
│   └── full_eval.py                    # Full evaluation pipeline
│
└── Training Logs/
    ├── training_big_mode_fixed.log     # Current Big Mode
    └── training_budget_mode_fixed.log  # Budget Mode (complete)
```

### Getting Help

**📚 Documentation Hub:** [docs/README.md](docs/README.md) - Complete index of all 20 documentation files

#### Quick Links by Category

**🚀 Getting Started:**
- [Quick Setup Guide](docs/setup/QUICK_SETUP.md) - Fast setup for RTX 5080/5090
- [System Setup Guide](docs/setup/SYSTEM_SETUP_GUIDE.md) - Comprehensive installation
- [CLAUDE.md](CLAUDE.md) - Complete RTX 5080/5090 setup instructions

**🔧 RTX 5080 Hardware:**
- [RTX 5080 Setup Session](docs/rtx5080/RTX5080_SETUP_SESSION.md) - Initial setup
- [PyTorch Upgrade](docs/rtx5080/RTX5080_PYTORCH_UPGRADE_SESSION.md) - Blackwell compatibility
- [Bug Fixes](docs/rtx5080/BIG_MODE_BUGS_AND_FIXES.md) - All known issues & solutions

**📊 Training Results:**
- [Budget Mode Report](docs/rtx5080/BUDGET_MODE_TRAINING_REPORT.md) - 13/13 datasets (2h 2m)
- [Big Mode Report](docs/rtx5080/BIG_MODE_BUGS_AND_FIXES.md) - Optimization & validation
- [All Datasets Report](docs/training/ALL_DATASETS_TRAINING_REPORT.md) - Complete results

**🐛 Troubleshooting:**
- [Densification Fix](docs/rtx5080/DENSIFICATION_ALGORITHM_FIX.md) - Budget algorithm
- [Loop Bug Fix](docs/rtx5080/LOOP_BUG_FIX.md) - Training script
- [Big Mode Bugs](docs/rtx5080/BIG_MODE_BUGS_AND_FIXES.md) - Memory & array issues

**📦 Datasets:**
- [Dataset Status](docs/datasets/DATASET_STATUS_REPORT.md) - Dataset availability

**🛠️ Development:**
- [Submodules Build Structure](docs/development/SUBMODULES_BUILD_STRUCTURE.md) - CUDA extensions compilation analysis

**📖 Architecture Documentation:**
- [Codebase Architecture](docs/architecture/CODEBASE_ARCHITECTURE.md) - **Comprehensive guide to understanding the Python codebase**
  - Detailed explanation of all Python files and their functions
  - Data flow diagrams for training, rendering, and evaluation
  - Key algorithms with step-by-step breakdowns
  - Function reference guide with parameters and usage
  - Perfect for developers who want to understand or modify the code

---

## Citation
If you find this repo useful, please cite:
```
@inproceedings{10.1145/3680528.3687694,
    author = {Mallick, Saswat Subhajyoti and Goel, Rahul and Kerbl, Bernhard and Steinberger, Markus and Carrasco, Francisco Vicente and De La Torre, Fernando},
    title = {Taming 3DGS: High-Quality Radiance Fields with Limited Resources},
    year = {2024},
    isbn = {9798400711312},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3680528.3687694},
    doi = {10.1145/3680528.3687694},
    booktitle = {SIGGRAPH Asia 2024 Conference Papers},
    articleno = {2},
    numpages = {11},
    keywords = {Radiance Fields, Gaussian Splatting},
    series = {SA '24}
}

```
