#!/bin/bash

# ============================================================================
# FIFO COMPLETE TRAINING SCRIPT - 2 STAGES
# ============================================================================
# This script runs full training: Stage 1 (FogPassFilter) + Stage 2 (Full Model)
# Expected result: mIoU ~40-43%
# ============================================================================

set -e  # Exit on error

echo "======================================"
echo "  FIFO Full Training - 2 Stages"
echo "======================================"
echo ""

# ============================================================================
# STAGE 1: Train FogPassFilter (0 → 10K steps)
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STAGE 1: FogPassFilter Training"
echo "Steps: 0 → 10,000"
echo "Expected time: ~1.8 hours on P100"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python main.py \
    --file-name "CS_scenes_" \
    --modeltrain fogpass \
    --num-steps 10000 \
    --num-steps-stop 10000 \
    --batch-size 4 \
    --save-pred-every 5000 \
    --snapshot-dir ./snapshots \
    --gpu 0

echo ""
echo "✅ Stage 1 completed!"
echo "Checkpoints saved:"
ls -lh ./snapshots/*5000*.pth 2>/dev/null || echo "  (No 5K checkpoint)"
ls -lh ./snapshots/*10000*.pth 2>/dev/null || echo "  (No 10K checkpoint)"
ls -lh ./snapshots/CS_scenes_10000.pth 2>/dev/null || echo "  ⚠️  Final checkpoint not found!"
echo ""

# Check if Stage 1 checkpoint exists
if [ ! -f "./snapshots/CS_scenes_10000.pth" ]; then
    echo "❌ Error: Stage 1 checkpoint not found!"
    echo "   Expected: ./snapshots/CS_scenes_10000.pth"
    echo "   Cannot continue to Stage 2"
    exit 1
fi

# ============================================================================
# STAGE 2: Train Full Model (10K → 20K steps)
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STAGE 2: Full Model Training"
echo "Steps: 10,000 → 20,000"
echo "Expected time: ~1.8 hours on P100"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📂 Loading checkpoint: ./snapshots/CS_scenes_10000.pth"

python main.py \
    --file-name "CS_scenes_" \
    --modeltrain train \
    --restore-from ./snapshots/CS_scenes_10000.pth \
    --restore-from-fogpass ./snapshots/CS_scenes_10000.pth \
    --num-steps 20000 \
    --num-steps-stop 20000 \
    --batch-size 4 \
    --save-pred-every 2000 \
    --snapshot-dir ./snapshots \
    --gpu 0

echo ""
echo "✅ Stage 2 completed!"
echo "Final checkpoint:"
ls -lh ./snapshots/CS_scenes_20000.pth 2>/dev/null || echo "  ⚠️  Final checkpoint not found!"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "======================================"
echo "  Training Complete!"
echo "======================================"
echo ""
echo "Checkpoints created:"
echo "  - Stage 1: ./snapshots/CS_scenes_10000.pth"
echo "  - Stage 2: ./snapshots/CS_scenes_20000.pth"
echo ""
echo "Next steps:"
echo "  1. Copy final model for evaluation:"
echo "     cp ./snapshots/CS_scenes_20000.pth ./FIFO_20K_final.pth"
echo ""
echo "  2. Evaluate on local machine:"
echo "     python evaluate_cpu.py --file-name 'FIFO_20K' --restore-from ./FIFO_20K_final.pth"
echo ""
echo "Expected mIoU: ~40-43%"
echo "======================================"
