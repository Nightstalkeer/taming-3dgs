# Training Commands Report for train.py

## Basic Command Structure

```bash
python train.py -s <source_data> -m <model_output> [options]
```

## Required Parameters

- `-s, --source_path`: Path to the source dataset (e.g., `data/bicycle`)
- `-m, --model_path`: Path where the trained model will be saved (e.g., `./eval/bicycle_budget`)

## Key Optional Parameters

### Image Resolution
- `-i, --images`: Image folder name (e.g., `images_4`, `images_2`)

### Training Mode and Budget
- `--mode`: Training mode (`multiplier` or `final_count`)
- `--budget`: Number of Gaussians budget (e.g., `15`, `2`, `5987095`)
- `--optimizer_type`: Optimizer type (`default`)

### Training Control
- `--densification_interval`: Interval for densification (e.g., `500`, `100`)
- `--test_iterations`: Test iterations (`-1` for default)

### Output Control
- `--quiet`: Suppress verbose output
- `--eval`: Enable evaluation mode

## Available Datasets

Based on your `data/` directory:

### ✅ Working Datasets (have required structure):
- bicycle (Colmap format with sparse/ directory)
- bonsai (Colmap format with sparse/ directory)
- counter (Colmap format with sparse/ directory)
- garden (Colmap format with sparse/ directory)
- kitchen (Colmap format with sparse/ directory)
- room (Colmap format with sparse/ directory)
- stump (Colmap format with sparse/ directory)

### ❌ Non-working Datasets (missing required files):
- drjohnson (empty directory)
- flowers (empty directory)
- playroom (empty directory)
- treehill (empty directory)
- truck (empty directory)
- train (conflicts with train.py, avoid using)

## Example Commands from train.sh

### Budget Mode Training (Small Models)
```bash
# ✅ Working MipNeRF360 scenes with images_4
python train.py -s data/bicycle -i images_4 -m ./eval/bicycle_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 15 --densification_interval 500 --mode multiplier

python train.py -s data/garden -i images_4 -m ./eval/garden_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 15 --densification_interval 500 --mode multiplier

python train.py -s data/stump -i images_4 -m ./eval/stump_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 15 --densification_interval 500 --mode multiplier

# ❌ These don't work - datasets missing required files:
# python train.py -s data/flowers -i images_4 -m ./eval/flowers_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 15 --densification_interval 500 --mode multiplier
# python train.py -s data/treehill -i images_4 -m ./eval/treehill_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 15 --densification_interval 500 --mode multiplier
```

### Tanks&Temples scenes with images_2
```bash
python train.py -s data/room -i images_2 -m ./eval/room_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 2 --densification_interval 500 --mode multiplier

python train.py -s data/counter -i images_2 -m ./eval/counter_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 2 --densification_interval 500 --mode multiplier

python train.py -s data/kitchen -i images_2 -m ./eval/kitchen_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 2 --densification_interval 500 --mode multiplier

python train.py -s data/bonsai -i images_2 -m ./eval/bonsai_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 2 --densification_interval 500 --mode multiplier
```

### No specific images folder
```bash
python train.py -s data/truck -m ./eval/truck_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 2 --densification_interval 500 --mode multiplier

python train.py -s data/train -m ./eval/train_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 2 --densification_interval 500 --mode multiplier
```

### Larger budget scenes
```bash
python train.py -s data/drjohnson -m ./eval/drjohnson_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 5 --densification_interval 500 --mode multiplier

python train.py -s data/playroom -m ./eval/playroom_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 5 --densification_interval 500 --mode multiplier
```

### Full Count Mode Training (Large Models)
```bash
# High-quality training with final_count mode
python train.py -s data/bicycle -i images_4 -m ./eval/bicycle_big --quiet --eval --test_iterations -1 --optimizer_type default --budget 5987095 --densification_interval 100 --mode final_count

python train.py -s data/flowers -i images_4 -m ./eval/flowers_big --quiet --eval --test_iterations -1 --optimizer_type default --budget 3618411 --densification_interval 100 --mode final_count

python train.py -s data/garden -i images_4 -m ./eval/garden_big --quiet --eval --test_iterations -1 --optimizer_type default --budget 5728191 --densification_interval 100 --mode final_count
```

## Training Modes Explanation

### Multiplier Mode
- Uses small budgets (2, 5, 15)
- Faster training
- Good for quick experiments
- Uses `--mode multiplier`

### Final Count Mode
- Uses large budgets (millions of Gaussians)
- Higher quality results
- Longer training time
- Uses `--mode final_count`

## Quick Start Commands

### Simple training (recommended for testing):
```bash
python train.py -s data/bicycle -i images_4 -m ./eval/test_output --quiet --eval --test_iterations -1 --optimizer_type default --budget 15 --densification_interval 500 --mode multiplier
```

### High-quality training:
```bash
python train.py -s data/bicycle -i images_4 -m ./eval/test_output_hq --quiet --eval --test_iterations -1 --optimizer_type default --budget 5987095 --densification_interval 100 --mode final_count
```

## Post-Training Commands

After training, you can render and evaluate:

```bash
# Render results
python render.py -m ./eval/bicycle_budget

# Compute metrics
python metrics.py -m ./eval/bicycle_budget
```

## Dependencies Status
✅ All required submodules are installed:
- diff-gaussian-rasterization
- simple-knn
- fused-ssim
- websockets

## Notes
- Training typically runs for 30,000 iterations
- Loss should decrease over time (good training shows loss going from ~0.25 to ~0.08)
- Output models are saved to the specified `-m` directory
- Use `--quiet` to reduce output verbosity