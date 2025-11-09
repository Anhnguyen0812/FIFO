#!/bin/bash

# =============================================================================
# FIFO Model Evaluation Script - Evaluate on ALL Datasets
# =============================================================================
# This script evaluates the trained FIFO model on all available datasets:
# - Foggy Zurich (FZ)
# - Foggy Driving (FD) 
# - Foggy Driving Dense (FDD)
# - Cityscapes Lindau (optional)
#
# Usage: bash evaluate_all_datasets.sh
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}  FIFO Model Evaluation - All Datasets${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# =============================================================================
# Configuration
# =============================================================================

# Model checkpoint
MODEL_PATH="/home/anhngp/Documents/1/fifo/fast_training10000.pth"
MODEL_NAME="FIFO_10K"

# Project directory
PROJECT_DIR="/home/anhngp/Documents/1/fifo"
cd "$PROJECT_DIR"

# =============================================================================
# Check prerequisites
# =============================================================================

echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}❌ Error: Model not found at $MODEL_PATH${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Model found: $MODEL_PATH${NC}"

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${YELLOW}⚠ Warning: No conda environment detected${NC}"
    echo -e "${YELLOW}  Attempting to activate 'fifo' environment...${NC}"
    source ~/anaconda3/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh
    conda activate fifo || echo -e "${RED}❌ Could not activate conda environment${NC}"
else
    echo -e "${GREEN}✓ Conda environment active: $CONDA_DEFAULT_ENV${NC}"
fi

# Check Python packages
python -c "import torch; import numpy; import PIL" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Required Python packages found${NC}"
else
    echo -e "${RED}❌ Error: Missing required Python packages${NC}"
    exit 1
fi

echo ""

# =============================================================================
# Dataset paths - UPDATE THESE PATHS!
# =============================================================================

# Foggy Zurich dataset
FZ_DATA_DIR="/home/anhngp/Documents/datasets/Foggy_Zurich"
FZ_DATA_LIST="./lists_file_names/leftImg8bit_testall_filenames.txt"

# Foggy Driving dataset
FD_DATA_DIR="/home/anhngp/Documents/datasets/Foggy_Driving"
FD_DATA_LIST="./lists_file_names/leftImg8bit_testfine_filenames.txt"

# Foggy Driving Dense dataset
FDD_DATA_DIR="/home/anhngp/Documents/datasets/Foggy_Driving_Dense"
FDD_DATA_LIST="./lists_file_names/leftImg8bit_testdense_filenames.txt"

# Cityscapes Lindau (optional)
CLINDAU_DATA_DIR="/home/anhngp/Documents/datasets/Cityscapes"
CLINDAU_DATA_LIST="./dataset/cityscapes_list/clear_lindau.txt"

# =============================================================================
# Function to evaluate on a dataset
# =============================================================================

evaluate_dataset() {
    local dataset_name=$1
    local data_dir=$2
    local data_list=$3
    local save_dir=$4
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Evaluating on: $dataset_name${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Check if dataset exists
    if [ ! -d "$data_dir" ]; then
        echo -e "${YELLOW}⚠ Dataset not found: $data_dir${NC}"
        echo -e "${YELLOW}  Skipping $dataset_name...${NC}"
        echo ""
        return 1
    fi
    
    # Check if data list exists
    if [ ! -f "$data_list" ]; then
        echo -e "${YELLOW}⚠ Data list not found: $data_list${NC}"
        echo -e "${YELLOW}  Skipping $dataset_name...${NC}"
        echo ""
        return 1
    fi
    
    echo -e "${GREEN}✓ Dataset found: $data_dir${NC}"
    echo -e "${GREEN}✓ Data list found: $data_list${NC}"
    echo ""
    
    # Create save directory
    mkdir -p "$save_dir"
    
    # Run evaluation
    echo -e "${YELLOW}Running evaluation...${NC}"
    python evaluate.py \
        --restore-from "$MODEL_PATH" \
        --data-dir-eval "$data_dir" \
        --data-list-eval "$data_list" \
        --save-dir-result "$save_dir" \
        --file-name "$MODEL_NAME"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Evaluation completed for $dataset_name${NC}"
        
        # Show results if available
        if [ -f "$save_dir/miou.txt" ]; then
            echo -e "${GREEN}Results:${NC}"
            cat "$save_dir/miou.txt"
        fi
    else
        echo -e "${RED}❌ Evaluation failed for $dataset_name${NC}"
    fi
    
    echo ""
}

# =============================================================================
# Run evaluations
# =============================================================================

echo -e "${BLUE}Starting evaluations...${NC}"
echo -e "${YELLOW}Model: $MODEL_PATH${NC}"
echo -e "${YELLOW}This may take 10-30 minutes depending on dataset size...${NC}"
echo ""

# Track which datasets were evaluated
evaluated=0
failed=0

# Evaluate Foggy Zurich
if evaluate_dataset "Foggy Zurich (FZ)" "$FZ_DATA_DIR" "$FZ_DATA_LIST" "./result_FZ/$MODEL_NAME"; then
    ((evaluated++))
else
    ((failed++))
fi

# Evaluate Foggy Driving
if evaluate_dataset "Foggy Driving (FD)" "$FD_DATA_DIR" "$FD_DATA_LIST" "./result_FD/$MODEL_NAME"; then
    ((evaluated++))
else
    ((failed++))
fi

# Evaluate Foggy Driving Dense
if evaluate_dataset "Foggy Driving Dense (FDD)" "$FDD_DATA_DIR" "$FDD_DATA_LIST" "./result_FDD/$MODEL_NAME"; then
    ((evaluated++))
else
    ((failed++))
fi

# Evaluate Cityscapes Lindau (optional)
if evaluate_dataset "Cityscapes Lindau" "$CLINDAU_DATA_DIR" "$CLINDAU_DATA_LIST" "./result_Clindau/$MODEL_NAME"; then
    ((evaluated++))
else
    ((failed++))
fi

# =============================================================================
# Summary
# =============================================================================

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}  Evaluation Summary${NC}"
echo -e "${BLUE}=====================================${NC}"
echo -e "${GREEN}Datasets evaluated: $evaluated${NC}"
echo -e "${YELLOW}Datasets skipped: $failed${NC}"
echo ""

if [ $evaluated -gt 0 ]; then
    echo -e "${GREEN}✓ Evaluation completed!${NC}"
    echo ""
    echo -e "${BLUE}Results saved in:${NC}"
    [ -d "./result_FZ/$MODEL_NAME" ] && echo -e "  - ./result_FZ/$MODEL_NAME/"
    [ -d "./result_FD/$MODEL_NAME" ] && echo -e "  - ./result_FD/$MODEL_NAME/"
    [ -d "./result_FDD/$MODEL_NAME" ] && echo -e "  - ./result_FDD/$MODEL_NAME/"
    [ -d "./result_Clindau/$MODEL_NAME" ] && echo -e "  - ./result_Clindau/$MODEL_NAME/"
    echo ""
    
    # Show all mIoU results
    echo -e "${BLUE}mIoU Results:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    for result_dir in ./result_*/$MODEL_NAME; do
        if [ -d "$result_dir" ] && [ -f "$result_dir/miou.txt" ]; then
            dataset_name=$(basename $(dirname "$result_dir"))
            echo -e "${YELLOW}$dataset_name:${NC}"
            cat "$result_dir/miou.txt" | grep -i "miou\|mean"
            echo ""
        fi
    done
else
    echo -e "${RED}❌ No datasets were evaluated${NC}"
    echo -e "${YELLOW}Please check dataset paths in the script${NC}"
fi

echo -e "${BLUE}=====================================${NC}"
