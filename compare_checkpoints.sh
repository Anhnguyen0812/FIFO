#!/bin/bash

# Script to evaluate all checkpoints and compare results

set -e

MODEL_DIR="/home/anhngp/Documents/1/fifo"

echo "======================================"
echo "  Evaluating All Checkpoints"
echo "======================================"
echo ""

# Checkpoint 1: FIFO 5000 (full model)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Evaluating: fast_training-11-09-00-25_FIFO5000.pth"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python evaluate.py \
    --file-name 'FIFO_5K_full' \
    --restore-from "$MODEL_DIR/fast_training-11-09-00-25_FIFO5000.pth"
echo ""

# Checkpoint 2: FogPassFilter 5000 (only FogPassFilter)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. Evaluating: fast_training-11-09-00-25_fogpassfilter_5000.pth"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "(Note: This is only FogPassFilter weights, may not work directly)"
python evaluate.py \
    --file-name 'FIFO_5K_fogpass' \
    --restore-from "$MODEL_DIR/fast_training-11-09-00-25_fogpassfilter_5000.pth" || echo "⚠️ FogPassFilter-only checkpoint failed (expected)"
echo ""

# Checkpoint 3: Training 10000 (latest)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. Evaluating: fast_training10000.pth"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python evaluate.py \
    --file-name 'FIFO_10K' \
    --restore-from "$MODEL_DIR/fast_training10000.pth"
echo ""

# Summary
echo "======================================"
echo "  Results Summary"
echo "======================================"
echo ""

echo "FIFO 5K Full Model:"
if [ -f "./result_FZ/FIFO_5K_full/miou.txt" ]; then
    cat "./result_FZ/FIFO_5K_full/miou.txt" | head -1
else
    echo "  No results"
fi
echo ""

echo "FIFO 10K Model:"
if [ -f "./result_FZ/FIFO_10K/miou.txt" ]; then
    cat "./result_FZ/FIFO_10K/miou.txt" | head -1
else
    echo "  No results"
fi
echo ""

echo "======================================"
