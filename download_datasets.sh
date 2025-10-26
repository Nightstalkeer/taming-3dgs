#!/bin/bash

# Download datasets for Taming 3DGS training
# This script downloads MipNeRF360, Tanks&Temples, and DeepBlending datasets

set -e  # Exit on error

echo "=========================================="
echo "Taming 3DGS Dataset Download Script"
echo "=========================================="
echo ""

# Create data directory
mkdir -p data
cd data

echo "Dataset Information:"
echo "  1. MipNeRF360: ~12 GB (7 scenes)"
echo "  2. Tanks&Temples + DeepBlending: ~651 MB (6 scenes)"
echo "  Total download size: ~13 GB"
echo ""
read -p "Do you want to proceed with the download? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled."
    exit 0
fi

echo ""
echo "=========================================="
echo "Downloading MipNeRF360 Dataset (~12 GB)"
echo "=========================================="
echo "This may take 10-30 minutes depending on your connection..."
echo ""

if [ ! -f "360_v2.zip" ]; then
    wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
    echo "✓ MipNeRF360 downloaded"
else
    echo "✓ MipNeRF360 already downloaded, skipping..."
fi

echo ""
echo "Extracting MipNeRF360 dataset..."
unzip -q 360_v2.zip
echo "✓ MipNeRF360 extracted"

# Move scenes to data directory
for scene in bicycle bonsai counter garden kitchen room stump; do
    if [ -d "360_v2/$scene" ]; then
        mv "360_v2/$scene" ./
        echo "  ✓ $scene"
    fi
done

# Cleanup
rm -rf 360_v2
echo "✓ MipNeRF360 setup complete"

echo ""
echo "=========================================="
echo "Downloading Tanks&Temples + DeepBlending"
echo "=========================================="
echo "Size: ~651 MB"
echo ""

if [ ! -f "tandt_db.zip" ]; then
    wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
    echo "✓ Tanks&Temples + DeepBlending downloaded"
else
    echo "✓ Tanks&Temples + DeepBlending already downloaded, skipping..."
fi

echo ""
echo "Extracting Tanks&Temples + DeepBlending..."
unzip -q tandt_db.zip
echo "✓ Extracted"

# Move scenes to data directory
for scene in truck train; do
    if [ -d "tandt/$scene" ]; then
        mv "tandt/$scene" ./
        echo "  ✓ $scene (Tanks&Temples)"
    fi
done

for scene in drjohnson playroom; do
    if [ -d "db/$scene" ]; then
        mv "db/$scene" ./
        echo "  ✓ $scene (DeepBlending)"
    fi
done

# Cleanup
rm -rf tandt db
echo "✓ Tanks&Temples + DeepBlending setup complete"

cd ..

echo ""
echo "=========================================="
echo "Verifying Dataset Structure"
echo "=========================================="
echo ""

# Verify datasets
WORKING=0
BROKEN=0

for scene in bicycle garden stump bonsai counter kitchen room truck train drjohnson playroom; do
    if [ -d "data/$scene/sparse" ] || [ -f "data/$scene/transforms_train.json" ]; then
        echo "✓ $scene - Ready"
        ((WORKING++))
    else
        echo "✗ $scene - Missing or incomplete"
        ((BROKEN++))
    fi
done

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Dataset Status:"
echo "  ✓ Working: $WORKING scenes"
echo "  ✗ Missing: $BROKEN scenes"
echo ""
echo "Note: flowers and treehill are not publicly available"
echo "      (require author permission from MipNeRF360 paper)"
echo ""
echo "You can now run training with:"
echo "  conda activate 3dgs"
echo "  ./train_rtx5080.sh"
echo ""
echo "Or test a single scene:"
echo "  python train.py -s data/bicycle -i images_4 -m ./test_output --budget 15 --mode multiplier --eval"
echo ""
