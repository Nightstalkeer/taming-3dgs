#!/bin/bash
# ============================================================================
# RTX 5080 (Blackwell sm_120) Optimized Training Script
# ============================================================================
# Created: October 25, 2025
# GPU: NVIDIA GeForce RTX 5080 (16GB VRAM, sm_120)
# PyTorch: 2.9.0+cu128
# Python: 3.10.19
#
# This script is optimized for RTX 5080 with:
# - Memory-efficient settings (--data_device cpu)
# - Pure PyTorch SSIM (no CUDA kernel issues)
# - Batch k-NN processing (automatic fallback)
# - sm_90 compatibility mode for custom CUDA extensions
#
# Usage:
#   ./train_rtx5080.sh [MODE] [PHASE]
#
# Modes:
#   test     - Quick test (500 iterations, low resolution, budget=0.3)
#   budget   - Budget mode (30k iterations, standard budgets)
#   big      - High quality (30k iterations, final_count mode)
#
# Phases (optional, for budget/big modes):
#   train    - Only training phase
#   render   - Only rendering phase
#   metrics  - Only metrics phase
#   all      - All phases (default)
#
# Examples:
#   ./train_rtx5080.sh test              # Quick test all datasets
#   ./train_rtx5080.sh budget train      # Budget mode training only
#   ./train_rtx5080.sh big all           # Full high-quality pipeline
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

MODE="${1:-budget}"
PHASE="${2:-all}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "${CYAN}============================================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}============================================================================${NC}"
}

print_section() {
    echo -e "\n${BLUE}===== $1 =====${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

print_header "RTX 5080 Training Script - Pre-flight Checks"

# Check conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    print_error "No conda environment activated"
    echo "Please activate the environment: conda activate rtx5080_3dgs"
    exit 1
else
    print_success "Conda environment: $CONDA_DEFAULT_ENV"
fi

# Check CUDA architecture environment variable
if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
    print_warning "TORCH_CUDA_ARCH_LIST not set, setting it now..."
    export TORCH_CUDA_ARCH_LIST="8.6;9.0"
    print_info "Set TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
else
    print_success "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
fi

# Check Python version
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
print_info "Python version: $PYTHON_VERSION"

# Check PyTorch version
PYTORCH_VERSION=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "NOT FOUND")
if [[ "$PYTORCH_VERSION" == "NOT FOUND" ]]; then
    print_error "PyTorch not found!"
    exit 1
else
    print_success "PyTorch version: $PYTORCH_VERSION"
fi

# Check GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "NOT FOUND")
if [[ "$GPU_NAME" == "NOT FOUND" ]]; then
    print_error "No GPU detected!"
    exit 1
else
    print_success "GPU: $GPU_NAME"
fi

# Check GPU memory
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo "0")
print_info "GPU Memory: ${GPU_MEMORY}MB"

# Verify custom CUDA modules
print_info "Verifying custom CUDA modules..."
python -c "
from diff_gaussian_rasterization import GaussianRasterizationSettings
from simple_knn._C import distCUDA2
import fused_ssim
print('✓ All CUDA modules loaded successfully')
" 2>/dev/null && print_success "All CUDA modules verified" || {
    print_error "CUDA module verification failed!"
    echo "Please recompile submodules:"
    echo "  TORCH_CUDA_ARCH_LIST=\"8.6;9.0\" pip install -e submodules/diff-gaussian-rasterization"
    echo "  TORCH_CUDA_ARCH_LIST=\"8.6;9.0\" pip install -e submodules/simple-knn"
    exit 1
}

# Check data directory
if [ ! -d "data" ]; then
    print_error "data/ directory not found!"
    exit 1
fi

print_success "All pre-flight checks passed!"

# ============================================================================
# Dataset Configuration
# ============================================================================

# Datasets organized by type and available resolutions
# Format: "dataset_name:resolution:budget_multiplier:scene_type"

MIPNERF360_DATASETS=(
    "bicycle:images_4:15:outdoor"
    "flowers:images_4:15:outdoor"
    "garden:images_4:15:outdoor"
    "stump:images_4:15:outdoor"
    "treehill:images_4:15:outdoor"
    "truck:images:15:outdoor"  # Only has images/
)

TANKS_TEMPLES_DATASETS=(
    "counter:images_2:2:indoor"
    "kitchen:images_2:2:indoor"
    "room:images_2:2:indoor"
    "playroom:images:2:indoor"  # Only has images/
    "train:images:2:indoor"     # Only has images/
)

BLENDER_DATASETS=(
    "bonsai:images_2:2:synthetic"
)

CUSTOM_DATASETS=(
    "drjohnson:images:5:custom"  # Only has images/
)

# Combine all datasets
ALL_DATASETS=(
    "${MIPNERF360_DATASETS[@]}"
    "${TANKS_TEMPLES_DATASETS[@]}"
    "${BLENDER_DATASETS[@]}"
    "${CUSTOM_DATASETS[@]}"
)

# Debug: Show dataset count
print_info "DEBUG: Total datasets in ALL_DATASETS: ${#ALL_DATASETS[@]}"
for i in "${!ALL_DATASETS[@]}"; do
    echo "  Dataset[$i]: ${ALL_DATASETS[$i]}"
done
echo ""

# Big mode final counts (from original train.sh)
declare -A BIG_MODE_COUNTS=(
    ["bicycle"]="5987095"
    ["flowers"]="3618411"
    ["garden"]="5728191"
    ["stump"]="4867429"
    ["treehill"]="3770257"
    ["truck"]="2584171"
    ["counter"]="1190919"
    ["kitchen"]="1803735"
    ["room"]="1548960"
    ["playroom"]="2326100"
    ["train"]="1085480"
    ["bonsai"]="1252367"
    ["drjohnson"]="3273600"
)

# ============================================================================
# Training Functions
# ============================================================================

train_dataset() {
    local dataset_name=$1
    local resolution=$2
    local budget=$3
    local mode=$4
    local output_dir=$5
    local iterations=$6
    local densify_interval=$7
    local extra_args=$8

    local dataset_path="data/${dataset_name}"

    # Check if dataset exists
    if [ ! -d "$dataset_path" ]; then
        print_error "Dataset not found: $dataset_path"
        return 1
    fi

    # Check if resolution directory exists
    if [ "$resolution" != "images" ] && [ ! -d "$dataset_path/$resolution" ]; then
        print_warning "$dataset_name: $resolution not found, falling back to images/"
        resolution="images"
    fi

    print_info "Training $dataset_name (${resolution}, budget=${budget}, mode=${mode})"

    # Build command
    local cmd="python train.py"
    cmd="$cmd -s $dataset_path"
    cmd="$cmd -i $resolution"
    cmd="$cmd -m $output_dir"
    cmd="$cmd --budget $budget"
    cmd="$cmd --mode $mode"
    cmd="$cmd --iterations $iterations"
    cmd="$cmd --densification_interval $densify_interval"
    cmd="$cmd --data_device cpu"  # Critical for RTX 5080 memory efficiency
    cmd="$cmd --optimizer_type default"
    cmd="$cmd $extra_args"

    # Execute with timing
    local start_time=$(date +%s)

    if eval "$cmd"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "$dataset_name completed in ${duration}s"

        # Check output file
        if [ -f "$output_dir/point_cloud/iteration_${iterations}/point_cloud.ply" ]; then
            local size=$(du -h "$output_dir/point_cloud/iteration_${iterations}/point_cloud.ply" | cut -f1)
            print_info "Output size: $size"
        fi

        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_error "$dataset_name failed after ${duration}s"
        return 1
    fi
}

render_dataset() {
    local model_path=$1
    local dataset_name=$2

    if [ ! -d "$model_path" ]; then
        print_warning "Model not found: $model_path (skipping)"
        return 1
    fi

    print_info "Rendering $dataset_name"

    if python render.py -m "$model_path"; then
        print_success "$dataset_name rendered"
        return 0
    else
        print_error "$dataset_name render failed"
        return 1
    fi
}

compute_metrics() {
    local model_path=$1
    local dataset_name=$2

    if [ ! -d "$model_path" ]; then
        print_warning "Model not found: $model_path (skipping)"
        return 1
    fi

    print_info "Computing metrics for $dataset_name"

    if python metrics.py -m "$model_path"; then
        print_success "$dataset_name metrics computed"

        # Display results if available
        if [ -f "$model_path/results.json" ]; then
            print_info "Results:"
            python -c "import json; data=json.load(open('$model_path/results.json')); print('  PSNR: {:.2f} | SSIM: {:.4f} | LPIPS: {:.4f}'.format(data.get('PSNR', 0), data.get('SSIM', 0), data.get('LPIPS', 0)))" 2>/dev/null || true
        fi

        return 0
    else
        print_error "$dataset_name metrics failed"
        return 1
    fi
}

# ============================================================================
# Test Mode (Quick Validation)
# ============================================================================

run_test_mode() {
    print_header "TEST MODE - Quick Validation (500 iterations)"

    print_info "Configuration:"
    echo "  - Resolution: images_8 (or images if unavailable)"
    echo "  - Budget: 0.3 (multiplier)"
    echo "  - Iterations: 500"
    echo "  - Densification interval: 3000"
    echo "  - Data device: CPU"
    echo "  - Test iterations: disabled (-1)"
    echo ""

    local success_count=0
    local fail_count=0
    local total_count=${#ALL_DATASETS[@]}

    mkdir -p dataset_test_results

    print_section "Training All Datasets (Test Mode)"

    for dataset_info in "${ALL_DATASETS[@]}"; do
        IFS=':' read -r name resolution budget scene_type <<< "$dataset_info"

        # Use images_8 for test mode if available, otherwise original resolution
        local test_resolution="images_8"
        if [ ! -d "data/${name}/${test_resolution}" ]; then
            test_resolution="images"
        fi

        if train_dataset "$name" "$test_resolution" "0.3" "multiplier" \
            "dataset_test_results/$name" "500" "3000" \
            "--test_iterations -1 --quiet"; then
            success_count=$((success_count + 1))
        else
            fail_count=$((fail_count + 1))
        fi

        echo ""
    done

    print_section "Test Mode Summary"
    print_success "Successful: $success_count/$total_count"
    if [ $fail_count -gt 0 ]; then
        print_error "Failed: $fail_count/$total_count"
    fi

    echo ""
    print_info "Test outputs saved to: dataset_test_results/"
}

# ============================================================================
# Budget Mode (Standard Quality)
# ============================================================================

run_budget_mode() {
    print_header "BUDGET MODE - Standard Quality Training"

    print_info "Configuration:"
    echo "  - MipNeRF360 outdoor: images_4, budget=15"
    echo "  - Tanks&Temples indoor: images_2, budget=2"
    echo "  - Blender synthetic: images_2, budget=2"
    echo "  - Custom datasets: images, budget=5"
    echo "  - Iterations: 30000"
    echo "  - Densification interval: 500"
    echo "  - Data device: CPU"
    echo ""

    mkdir -p eval

    local success_count=0
    local fail_count=0
    local total_count=${#ALL_DATASETS[@]}

    # Training phase
    if [[ "$PHASE" == "train" || "$PHASE" == "all" ]]; then
        print_section "Training Phase - Budget Mode"

        print_info "DEBUG: About to loop through ${#ALL_DATASETS[@]} datasets"
        local iteration_count=0
        for dataset_info in "${ALL_DATASETS[@]}"; do
            iteration_count=$((iteration_count + 1))
            print_info "DEBUG: Loop iteration $iteration_count: processing '$dataset_info'"
            IFS=':' read -r name resolution budget scene_type <<< "$dataset_info"
            print_info "DEBUG: Parsed: name=$name, resolution=$resolution, budget=$budget, type=$scene_type"

            if train_dataset "$name" "$resolution" "$budget" "multiplier" \
                "eval/${name}_budget" "30000" "500" \
                "--eval --test_iterations 7000 30000"; then
                success_count=$((success_count + 1))
                print_success "DEBUG: Dataset $name succeeded (count: $success_count)"
            else
                fail_count=$((fail_count + 1))
                print_error "DEBUG: Dataset $name failed (count: $fail_count)"
            fi

            echo ""
            print_info "DEBUG: Loop iteration $iteration_count completed, continuing..."
        done

        print_info "DEBUG: Loop completed after $iteration_count iterations"
        print_info "Training: $success_count/$total_count successful"
    fi

    # Rendering phase
    if [[ "$PHASE" == "render" || "$PHASE" == "all" ]]; then
        print_section "Rendering Phase - Budget Mode"

        for dataset_info in "${ALL_DATASETS[@]}"; do
            IFS=':' read -r name resolution budget scene_type <<< "$dataset_info"
            render_dataset "eval/${name}_budget" "$name"
            echo ""
        done
    fi

    # Metrics phase
    if [[ "$PHASE" == "metrics" || "$PHASE" == "all" ]]; then
        print_section "Metrics Phase - Budget Mode"

        for dataset_info in "${ALL_DATASETS[@]}"; do
            IFS=':' read -r name resolution budget scene_type <<< "$dataset_info"
            compute_metrics "eval/${name}_budget" "$name"
            echo ""
        done
    fi

    print_section "Budget Mode Complete"
    if [ $fail_count -gt 0 ]; then
        print_warning "Some datasets failed during training"
    fi
    print_info "Results saved to: eval/*_budget/"
}

# ============================================================================
# Big Mode (High Quality)
# ============================================================================

run_big_mode() {
    print_header "BIG MODE - High Quality Training (Final Count)"

    # RTX 5080 optimization: Use relaxed densification interval
    local densify_interval=300  # 300 for RTX 5080 (vs 100 for A6000)

    print_info "Configuration:"
    echo "  - Mode: final_count (exact Gaussian count targets)"
    echo "  - Iterations: 30000"
    echo "  - Densification interval: $densify_interval (RTX 5080 optimized)"
    echo "  - Data device: CPU"
    echo "  - Memory: Expandable segments enabled"
    echo ""

    # Enable PyTorch CUDA memory optimization
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    mkdir -p eval

    local success_count=0
    local fail_count=0
    local total_count=${#ALL_DATASETS[@]}

    # Training phase
    if [[ "$PHASE" == "train" || "$PHASE" == "all" ]]; then
        print_section "Training Phase - Big Mode"

        for dataset_info in "${ALL_DATASETS[@]}"; do
            IFS=':' read -r name resolution budget scene_type <<< "$dataset_info"

            local final_count=${BIG_MODE_COUNTS[$name]}
            if [ -z "$final_count" ]; then
                print_warning "$name: No final count defined, skipping"
                continue
            fi

            if train_dataset "$name" "$resolution" "$final_count" "final_count" \
                "eval/${name}_big" "30000" "$densify_interval" \
                "--eval --test_iterations 7000 30000"; then
                success_count=$((success_count + 1))
            else
                fail_count=$((fail_count + 1))
            fi

            echo ""
        done

        print_info "Training: $success_count/$total_count successful"
    fi

    # Rendering phase
    if [[ "$PHASE" == "render" || "$PHASE" == "all" ]]; then
        print_section "Rendering Phase - Big Mode"

        for dataset_info in "${ALL_DATASETS[@]}"; do
            IFS=':' read -r name resolution budget scene_type <<< "$dataset_info"
            render_dataset "eval/${name}_big" "$name"
            echo ""
        done
    fi

    # Metrics phase
    if [[ "$PHASE" == "metrics" || "$PHASE" == "all" ]]; then
        print_section "Metrics Phase - Big Mode"

        for dataset_info in "${ALL_DATASETS[@]}"; do
            IFS=':' read -r name resolution budget scene_type <<< "$dataset_info"
            compute_metrics "eval/${name}_big" "$name"
            echo ""
        done
    fi

    print_section "Big Mode Complete"
    if [ $fail_count -gt 0 ]; then
        print_warning "Some datasets failed during training"
    fi
    print_info "Results saved to: eval/*_big/"
}

# ============================================================================
# Main Execution
# ============================================================================

print_info "Mode: $MODE"
print_info "Phase: $PHASE"
echo ""

START_TIME=$(date +%s)

case "$MODE" in
    test)
        run_test_mode
        ;;
    budget)
        run_budget_mode
        ;;
    big)
        run_big_mode
        ;;
    *)
        print_error "Invalid mode: $MODE"
        echo "Valid modes: test, budget, big"
        exit 1
        ;;
esac

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

print_header "All Processing Complete"
print_success "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"

# GPU status
print_info "Final GPU status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader || true

echo ""
print_info "Script finished successfully!"
