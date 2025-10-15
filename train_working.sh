#!/bin/bash

# Train Working Datasets Only - No Failures Expected
# This script contains only datasets that have the required structure (sparse/ directory)
# Created to avoid failures from missing datasets in original train.sh

echo "=== Starting Training Phase - Budget Mode ==="

# ✅ Working MipNeRF360 scenes with images_4 (budget=15)
echo "Training bicycle..."
python train.py -s data/bicycle -i images_4 -m ./eval/bicycle_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 15 --densification_interval 500 --mode multiplier

echo "Training garden..."
python train.py -s data/garden -i images_4 -m ./eval/garden_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 15 --densification_interval 500 --mode multiplier

echo "Training stump..."
python train.py -s data/stump -i images_4 -m ./eval/stump_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 15 --densification_interval 500 --mode multiplier

# ✅ Working Tanks&Temples scenes with images_2 (budget=2)
echo "Training room..."
python train.py -s data/room -i images_2 -m ./eval/room_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 2 --densification_interval 500 --mode multiplier

echo "Training counter..."
python train.py -s data/counter -i images_2 -m ./eval/counter_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 2 --densification_interval 500 --mode multiplier

echo "Training kitchen..."
python train.py -s data/kitchen -i images_2 -m ./eval/kitchen_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 2 --densification_interval 500 --mode multiplier

echo "Training bonsai..."
python train.py -s data/bonsai -i images_2 -m ./eval/bonsai_budget --quiet --eval --test_iterations -1 --optimizer_type default --budget 2 --densification_interval 500 --mode multiplier

echo "=== Starting Rendering Phase ==="

# Render results for budget models
python render.py -m ./eval/bicycle_budget
python render.py -m ./eval/garden_budget
python render.py -m ./eval/stump_budget
python render.py -m ./eval/room_budget
python render.py -m ./eval/counter_budget
python render.py -m ./eval/kitchen_budget
python render.py -m ./eval/bonsai_budget

echo "=== Starting Metrics Phase ==="

# Compute metrics for budget models
python metrics.py -m ./eval/bicycle_budget
python metrics.py -m ./eval/garden_budget
python metrics.py -m ./eval/stump_budget
python metrics.py -m ./eval/room_budget
python metrics.py -m ./eval/counter_budget
python metrics.py -m ./eval/kitchen_budget
python metrics.py -m ./eval/bonsai_budget

echo "=== Starting High-Quality Training Phase - Final Count Mode ==="

# ✅ High-quality training with final_count mode for working datasets
echo "Training bicycle (high quality)..."
python train.py -s data/bicycle -i images_4 -m ./eval/bicycle_big --quiet --eval --test_iterations -1 --optimizer_type default --budget 5987095 --densification_interval 100 --mode final_count

echo "Training garden (high quality)..."
python train.py -s data/garden -i images_4 -m ./eval/garden_big --quiet --eval --test_iterations -1 --optimizer_type default --budget 5728191 --densification_interval 100 --mode final_count

echo "Training stump (high quality)..."
python train.py -s data/stump -i images_4 -m ./eval/stump_big --quiet --eval --test_iterations -1 --optimizer_type default --budget 4867429 --densification_interval 100 --mode final_count

echo "Training room (high quality)..."
python train.py -s data/room -i images_2 -m ./eval/room_big --quiet --eval --test_iterations -1 --optimizer_type default --budget 1548960 --densification_interval 100 --mode final_count

echo "Training counter (high quality)..."
python train.py -s data/counter -i images_2 -m ./eval/counter_big --quiet --eval --test_iterations -1 --optimizer_type default --budget 1190919 --densification_interval 100 --mode final_count

echo "Training kitchen (high quality)..."
python train.py -s data/kitchen -i images_2 -m ./eval/kitchen_big --quiet --eval --test_iterations -1 --optimizer_type default --budget 1803735 --densification_interval 100 --mode final_count

echo "Training bonsai (high quality)..."
python train.py -s data/bonsai -i images_2 -m ./eval/bonsai_big --quiet --eval --test_iterations -1 --optimizer_type default --budget 1252367 --densification_interval 100 --mode final_count

echo "=== Starting High-Quality Rendering Phase ==="

# Render results for high-quality models
python render.py -m ./eval/bicycle_big
python render.py -m ./eval/garden_big
python render.py -m ./eval/stump_big
python render.py -m ./eval/room_big
python render.py -m ./eval/counter_big
python render.py -m ./eval/kitchen_big
python render.py -m ./eval/bonsai_big

echo "=== Starting High-Quality Metrics Phase ==="

# Compute metrics for high-quality models
python metrics.py -m ./eval/bicycle_big
python metrics.py -m ./eval/garden_big
python metrics.py -m ./eval/stump_big
python metrics.py -m ./eval/room_big
python metrics.py -m ./eval/counter_big
python metrics.py -m ./eval/kitchen_big
python metrics.py -m ./eval/bonsai_big

echo "=== All Training, Rendering, and Metrics Complete! ==="
echo "Working datasets: bicycle, garden, stump, room, counter, kitchen, bonsai"
echo "Results saved in ./eval/ directory"