#!/bin/bash
cd "/home/vortex/Computer Vision/3DGS research/taming-3dgs"

for dataset in bonsai counter drjohnson flowers garden kitchen playroom room stump train treehill truck; do
    echo "=== Training $dataset ==="
    python train.py -s "data/$dataset" -i images_8 -m "dataset_test_results/$dataset" \
        --budget 0.3 --mode multiplier --iterations 500 --densification_interval 3000 \
        --data_device cpu --test_iterations -1 --quiet

    if [ -f "dataset_test_results/$dataset/point_cloud/iteration_500/point_cloud.ply" ]; then
        size=$(du -h "dataset_test_results/$dataset/point_cloud/iteration_500/point_cloud.ply" | cut -f1)
        echo "✓ $dataset completed successfully ($size)"
    else
        echo "✗ $dataset failed"
    fi
    echo ""
done
