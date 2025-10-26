#!/bin/bash
cd "/home/vortex/Computer Vision/3DGS research/taming-3dgs"

for dataset in drjohnson playroom train truck; do
    echo "=== $dataset ==="
    ls -la "data/$dataset/" | grep -E "^d" | grep "images"
    echo ""
done
