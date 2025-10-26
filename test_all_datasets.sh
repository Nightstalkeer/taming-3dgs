#!/bin/bash

# Comprehensive Dataset Training Test Script
# Tests all 13 datasets with memory-optimized settings
# Created: October 25, 2025

set -e  # Exit on error

# Configuration
BASE_DIR="/home/vortex/Computer Vision/3DGS research/taming-3dgs"
DATA_DIR="$BASE_DIR/data"
OUTPUT_DIR="$BASE_DIR/dataset_test_results"
LOG_DIR="$OUTPUT_DIR/logs"
SUMMARY_FILE="$OUTPUT_DIR/training_summary.txt"

# Training parameters (memory-optimized)
RESOLUTION="images_8"
BUDGET="0.3"
MODE="multiplier"
ITERATIONS="500"
DENSIFICATION_INTERVAL="3000"
DATA_DEVICE="cpu"

# Datasets to test
DATASETS=(
    "bicycle"
    "bonsai"
    "counter"
    "drjohnson"
    "flowers"
    "garden"
    "kitchen"
    "playroom"
    "room"
    "stump"
    "train"
    "treehill"
    "truck"
)

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Initialize summary file
echo "=== 3D Gaussian Splatting - All Datasets Training Test ===" > "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "Configuration:" >> "$SUMMARY_FILE"
echo "  Resolution: $RESOLUTION" >> "$SUMMARY_FILE"
echo "  Budget: $BUDGET (multiplier mode)" >> "$SUMMARY_FILE"
echo "  Iterations: $ITERATIONS" >> "$SUMMARY_FILE"
echo "  Densification Interval: $DENSIFICATION_INTERVAL" >> "$SUMMARY_FILE"
echo "  Data Device: $DATA_DEVICE" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Results:" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Function to format time
format_time() {
    local seconds=$1
    printf "%02d:%02d:%02d" $((seconds/3600)) $((seconds%3600/60)) $((seconds%60))
}

# Function to train a single dataset
train_dataset() {
    local dataset=$1
    local dataset_path="$DATA_DIR/$dataset"
    local output_path="$OUTPUT_DIR/$dataset"
    local log_file="$LOG_DIR/${dataset}.log"

    echo -e "${BLUE}[$dataset]${NC} Starting training..."

    # Check if dataset exists
    if [ ! -d "$dataset_path" ]; then
        echo -e "${RED}[$dataset]${NC} Dataset not found at $dataset_path"
        echo "❌ $dataset: FAILED (dataset not found)" >> "$SUMMARY_FILE"
        return 1
    fi

    # Clean output directory
    rm -rf "$output_path"

    # Record start time
    local start_time=$(date +%s)

    # Run training
    if python "$BASE_DIR/train.py" \
        -s "$dataset_path" \
        -i "$RESOLUTION" \
        -m "$output_path" \
        --budget "$BUDGET" \
        --mode "$MODE" \
        --iterations "$ITERATIONS" \
        --densification_interval "$DENSIFICATION_INTERVAL" \
        --data_device "$DATA_DEVICE" \
        --test_iterations -1 \
        --quiet \
        > "$log_file" 2>&1; then

        # Record end time
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local formatted_time=$(format_time $duration)

        # Check if output was generated
        if [ -f "$output_path/point_cloud/iteration_$ITERATIONS/point_cloud.ply" ]; then
            local file_size=$(du -h "$output_path/point_cloud/iteration_$ITERATIONS/point_cloud.ply" | cut -f1)
            local final_loss=$(grep -oP 'Loss=\K[0-9.]+' "$log_file" | tail -1)

            echo -e "${GREEN}[$dataset]${NC} ✓ Success! (${formatted_time}, ${file_size}, loss=${final_loss})"
            echo "✓ $dataset: SUCCESS (time: ${formatted_time}, size: ${file_size}, final_loss: ${final_loss})" >> "$SUMMARY_FILE"
            return 0
        else
            echo -e "${RED}[$dataset]${NC} ✗ Failed: No output file generated"
            echo "❌ $dataset: FAILED (no output file, time: ${formatted_time})" >> "$SUMMARY_FILE"
            return 1
        fi
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local formatted_time=$(format_time $duration)

        # Extract error message
        local error_msg=$(tail -5 "$log_file" | grep -E "Error|Exception" | head -1)

        echo -e "${RED}[$dataset]${NC} ✗ Training failed (${formatted_time})"
        echo "   Error: ${error_msg:0:80}"
        echo "❌ $dataset: FAILED (error after ${formatted_time})" >> "$SUMMARY_FILE"
        echo "   Error: $error_msg" >> "$SUMMARY_FILE"
        return 1
    fi
}

# Main execution
echo "============================================"
echo "  3DGS Training Test - All Datasets"
echo "============================================"
echo ""
echo "Total datasets: ${#DATASETS[@]}"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"
echo ""
echo "Starting training tests..."
echo ""

# Track statistics
total_datasets=${#DATASETS[@]}
successful=0
failed=0
overall_start=$(date +%s)

# Train each dataset
for dataset in "${DATASETS[@]}"; do
    if train_dataset "$dataset"; then
        ((successful++))
    else
        ((failed++))
    fi
    echo ""
done

# Calculate overall time
overall_end=$(date +%s)
overall_duration=$((overall_end - overall_start))
overall_formatted=$(format_time $overall_duration)

# Print summary
echo "============================================"
echo "  Training Complete!"
echo "============================================"
echo ""
echo "Total Datasets: $total_datasets"
echo -e "${GREEN}Successful: $successful${NC}"
echo -e "${RED}Failed: $failed${NC}"
echo "Total Time: $overall_formatted"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Summary: $SUMMARY_FILE"
echo "Logs: $LOG_DIR"
echo ""

# Add summary statistics to file
echo "" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "Summary Statistics:" >> "$SUMMARY_FILE"
echo "  Total Datasets: $total_datasets" >> "$SUMMARY_FILE"
echo "  Successful: $successful" >> "$SUMMARY_FILE"
echo "  Failed: $failed" >> "$SUMMARY_FILE"
echo "  Success Rate: $(awk "BEGIN {printf \"%.1f%%\", ($successful/$total_datasets)*100}")" >> "$SUMMARY_FILE"
echo "  Total Time: $overall_formatted" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# GPU info
echo "GPU Information:" >> "$SUMMARY_FILE"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Display summary file
cat "$SUMMARY_FILE"

# Exit with appropriate code
if [ $failed -eq 0 ]; then
    echo -e "${GREEN}All datasets trained successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}Some datasets failed. Check logs in $LOG_DIR${NC}"
    exit 1
fi
