#!/bin/bash

# =============================================================================
# FIFO Model Evaluation Script - Evaluate on ALL Datasets
# =============================================================================
# Run evaluation on all available foggy datasets
# Usage: bash evaluate_all.sh
# =============================================================================

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}  FIFO Evaluation - All Datasets${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# =============================================================================
# Configuration
# =============================================================================

MODEL_PATH="/home/anhngp/Documents/1/fifo/fast_training10000.pth"
FILE_NAME="FIFO_10K_model"

# Check model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}❌ Model not found: $MODEL_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Model: $MODEL_PATH${NC}"
echo -e "${GREEN}✓ Output folder: $FILE_NAME${NC}"
echo ""

# =============================================================================
# Dataset Paths - UPDATE THESE!
# =============================================================================

# Foggy Zurich
FZ_AVAILABLE=false
if [ -d "/home/anhngp/Documents/datasets/Foggy_Zurich" ]; then
    FZ_AVAILABLE=true
    echo -e "${GREEN}✓ Foggy Zurich dataset found${NC}"
else
    echo -e "${YELLOW}⚠ Foggy Zurich not found, skipping...${NC}"
fi

# Foggy Driving  
FD_AVAILABLE=false
if [ -d "/home/anhngp/Documents/datasets/Foggy_Driving" ]; then
    FD_AVAILABLE=true
    echo -e "${GREEN}✓ Foggy Driving dataset found${NC}"
else
    echo -e "${YELLOW}⚠ Foggy Driving not found, skipping...${NC}"
fi

# Foggy Driving Dense
FDD_AVAILABLE=false
if [ -d "/home/anhngp/Documents/datasets/Foggy_Driving_Dense" ]; then
    FDD_AVAILABLE=true
    echo -e "${GREEN}✓ Foggy Driving Dense dataset found${NC}"
else
    echo -e "${YELLOW}⚠ Foggy Driving Dense not found, skipping...${NC}"
fi

echo ""

# =============================================================================
# Evaluate Foggy Zurich
# =============================================================================

if [ "$FZ_AVAILABLE" = true ]; then
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Evaluating on Foggy Zurich (FZ)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    python evaluate.py \
        --file-name "$FILE_NAME" \
        --restore-from "$MODEL_PATH"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Foggy Zurich evaluation completed${NC}"
        if [ -f "./result_FZ/$FILE_NAME/miou.txt" ]; then
            echo -e "${YELLOW}Results:${NC}"
            cat "./result_FZ/$FILE_NAME/miou.txt"
        fi
    else
        echo -e "${RED}❌ Foggy Zurich evaluation failed${NC}"
    fi
    echo ""
fi

# =============================================================================
# Evaluate Foggy Driving
# =============================================================================

if [ "$FD_AVAILABLE" = true ]; then
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Evaluating on Foggy Driving (FD)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    python evaluate.py \
        --file-name "$FILE_NAME" \
        --restore-from "$MODEL_PATH"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Foggy Driving evaluation completed${NC}"
        if [ -f "./result_FD/$FILE_NAME/miou.txt" ]; then
            echo -e "${YELLOW}Results:${NC}"
            cat "./result_FD/$FILE_NAME/miou.txt"
        fi
    else
        echo -e "${RED}❌ Foggy Driving evaluation failed${NC}"
    fi
    echo ""
fi

# =============================================================================
# Evaluate Foggy Driving Dense
# =============================================================================

if [ "$FDD_AVAILABLE" = true ]; then
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Evaluating on Foggy Driving Dense (FDD)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    python evaluate.py \
        --file-name "$FILE_NAME" \
        --restore-from "$MODEL_PATH"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Foggy Driving Dense evaluation completed${NC}"
        if [ -f "./result_FDD/$FILE_NAME/miou.txt" ]; then
            echo -e "${YELLOW}Results:${NC}"
            cat "./result_FDD/$FILE_NAME/miou.txt"
        fi
    else
        echo -e "${RED}❌ Foggy Driving Dense evaluation failed${NC}"
    fi
    echo ""
fi

# =============================================================================
# Summary
# =============================================================================

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}  Summary${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""
echo -e "${YELLOW}Results saved in:${NC}"
[ -d "./result_FZ/$FILE_NAME" ] && echo -e "  - ./result_FZ/$FILE_NAME/"
[ -d "./result_FD/$FILE_NAME" ] && echo -e "  - ./result_FD/$FILE_NAME/"
[ -d "./result_FDD/$FILE_NAME" ] && echo -e "  - ./result_FDD/$FILE_NAME/"
echo ""

echo -e "${BLUE}All mIoU Results:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for result_dir in ./result_*/$FILE_NAME; do
    if [ -d "$result_dir" ]; then
        dataset=$(basename $(dirname "$result_dir"))
        echo -e "${YELLOW}$dataset:${NC}"
        if [ -f "$result_dir/miou.txt" ]; then
            cat "$result_dir/miou.txt" | head -1
        else
            echo "  No results yet"
        fi
        echo ""
    fi
done

echo -e "${GREEN}✓ All evaluations completed!${NC}"
