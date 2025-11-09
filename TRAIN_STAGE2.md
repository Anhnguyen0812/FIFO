"""
Training Stage 2 - Full Model Training
=======================================
Resume from Stage 1 checkpoint to train full segmentation model

Current situation:
- You have 3 checkpoints from Stage 1 (fogpass mode)
- All only trained FogPassFilter (no segmentation learning)
- Result: 6% mIoU (model hasn't learned segmentation)

Solution: Train Stage 2 with mode='train'
"""

# =============================================================================
# STAGE 2 TRAINING - ON KAGGLE
# =============================================================================

# Copy this to a new Kaggle cell:

# Resume from Stage 1 checkpoint
python main.py \
    --mode train \
    --restore-from ./snapshots/fast_training-11-09-00-25_FIFO5000.pth \
    --restore-from-fogpass ./snapshots/fast_training-11-09-00-25_fogpassfilter_5000.pth \
    --num-steps 20000 \
    --num-steps-stop 20000 \
    --batch-size 4

# =============================================================================
# WHAT THIS WILL DO
# =============================================================================

# Stage 2 (train mode):
# - Load model from FIFO5000.pth
# - Load FogPassFilter from fogpassfilter_5000.pth
# - FREEZE FogPassFilter (no more training)
# - TRAIN model for segmentation
# - Train for 15K more steps (5K → 20K)
# - Expected time: ~3 hours on P100
# - Expected mIoU: ~40-43%

# =============================================================================
# CHECKPOINTS WILL BE SAVED
# =============================================================================

# Every 2000 steps:
# - snapshots/fast_training-XX-XX-XX-XX_FIFO7000.pth
# - snapshots/fast_training-XX-XX-XX-XX_FIFO9000.pth
# - snapshots/fast_training-XX-XX-XX-XX_FIFO11000.pth
# - ...
# - snapshots/fast_training-XX-XX-XX-XX_FIFO20000.pth (FINAL)

# Final checkpoint at 20K:
# - snapshots/CS_scenes_20000.pth

# =============================================================================
# VERIFY STAGE 2 IS RUNNING
# =============================================================================

# Look for these logs in Kaggle output:
# ✅ "fsm loss: ..."
# ✅ "SF_loss_seg: ..."  
# ✅ "CW_loss_seg: ..."
# ✅ "consistency loss: ..."
# ✅ "total_loss: ..."

# If you see these → Stage 2 is running correctly!
# If you only see "fpf loss" → Still in Stage 1!

# =============================================================================
# AFTER TRAINING COMPLETES
# =============================================================================

# 1. Download checkpoint:
!cp ./snapshots/CS_scenes_20000.pth /kaggle/working/

# 2. Save Version → Output → Download

# 3. Evaluate on local:
python evaluate_cpu.py \
    --file-name 'FIFO_20K' \
    --restore-from ./CS_scenes_20000.pth

# Expected result: mIoU ~40-43% (10x better than 6%!)
