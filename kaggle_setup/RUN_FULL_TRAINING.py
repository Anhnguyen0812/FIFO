# ============================================================================
# FIFO COMPLETE TRAINING - 2 STAGES
# ============================================================================
# Run this in Kaggle to train full FIFO model with proper segmentation
# Expected result: mIoU ~40-43% (instead of 6%)
# ============================================================================

# CELL 1: Setup
import os
os.chdir('/kaggle/working/fifo')

# CELL 2: Generate dataset lists (if not done)
!python generate_dataset_lists.py
!python generate_realfog_list.py

# ============================================================================
# STAGE 1: Train FogPassFilter (0 → 10K steps)
# ============================================================================
# CELL 3: Stage 1 Training
print("=" * 60)
print("STAGE 1: Training FogPassFilter (0 → 10K steps)")
print("Expected time: ~1.8 hours on P100")
print("=" * 60)

!python main.py \
    --mode fogpass \
    --num-steps 10000 \
    --num-steps-stop 10000 \
    --batch-size 4 \
    --save-pred-every 5000

print("\n✅ Stage 1 completed!")
print("Checkpoints saved:")
!ls -lh ./snapshots/*5000*.pth
!ls -lh ./snapshots/*10000*.pth

# ============================================================================
# STAGE 2: Train Full Model (10K → 20K steps)
# ============================================================================
# CELL 4: Stage 2 Training
print("=" * 60)
print("STAGE 2: Training Full Model (10K → 20K steps)")
print("Expected time: ~1.8 hours on P100")
print("=" * 60)

# Find the latest FIFO checkpoint from Stage 1
import glob
fifo_checkpoints = sorted(glob.glob('./snapshots/*FIFO10000*.pth'))
fogpass_checkpoints = sorted(glob.glob('./snapshots/*fogpassfilter_10000*.pth'))

if not fifo_checkpoints:
    print("❌ Error: No FIFO checkpoint found at 10000 steps!")
    print("Looking for any checkpoint with 'FIFO' in name...")
    fifo_checkpoints = sorted(glob.glob('./snapshots/*FIFO*.pth'))
    print(f"Found: {fifo_checkpoints}")

if not fogpass_checkpoints:
    print("❌ Error: No FogPassFilter checkpoint found!")
    print("Will use FIFO checkpoint for both")
    fogpass_checkpoints = fifo_checkpoints

restore_from = fifo_checkpoints[-1] if fifo_checkpoints else './snapshots/CS_scenes_10000.pth'
restore_fogpass = fogpass_checkpoints[-1] if fogpass_checkpoints else restore_from

print(f"📂 Loading model from: {restore_from}")
print(f"📂 Loading FogPassFilter from: {restore_fogpass}")

!python main.py \
    --mode train \
    --restore-from {restore_from} \
    --restore-from-fogpass {restore_fogpass} \
    --num-steps 20000 \
    --num-steps-stop 20000 \
    --batch-size 4 \
    --save-pred-every 2000

print("\n✅ Stage 2 completed!")
print("Final checkpoint saved:")
!ls -lh ./snapshots/CS_scenes_20000.pth

# ============================================================================
# PREPARE FOR DOWNLOAD
# ============================================================================
# CELL 5: Copy to output for download
print("=" * 60)
print("Preparing checkpoint for download...")
print("=" * 60)

!mkdir -p /kaggle/working/trained_models
!cp ./snapshots/CS_scenes_20000.pth /kaggle/working/trained_models/FIFO_20K_final.pth

# Also copy intermediate checkpoints
!cp ./snapshots/*FIFO20000*.pth /kaggle/working/trained_models/ 2>/dev/null || true
!cp ./snapshots/*FIFO15000*.pth /kaggle/working/trained_models/ 2>/dev/null || true

print("✅ Files ready for download:")
!ls -lh /kaggle/working/trained_models/

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print("Next steps:")
print("1. Click 'Save Version' → 'Save & Run All'")
print("2. Go to 'Output' tab")
print("3. Download 'FIFO_20K_final.pth'")
print("4. Evaluate on local machine:")
print("   python evaluate_cpu.py --file-name 'FIFO_20K' --restore-from ./FIFO_20K_final.pth")
print("\nExpected mIoU: ~40-43% (10x better than current 6%!)")
print("=" * 60)

# ============================================================================
# VERIFY TRAINING LOGS
# ============================================================================
# CELL 6: Check if Stage 2 ran correctly
print("\n" + "=" * 60)
print("VERIFICATION: Check if Stage 2 trained segmentation")
print("=" * 60)

# Check wandb logs
wandb_dirs = glob.glob('./wandb/offline-run-*')
if wandb_dirs:
    latest_run = sorted(wandb_dirs)[-1]
    print(f"📊 Latest W&B run: {latest_run}")
    print("\n✅ If you saw these losses during training, Stage 2 ran correctly:")
    print("   - SF_loss_seg")
    print("   - CW_loss_seg")
    print("   - fsm loss")
    print("   - consistency loss")
    print("   - total_loss")
    print("\n❌ If you only saw 'fpf loss', Stage 2 did NOT run!")
else:
    print("⚠️ No W&B logs found")

print("\n💾 Final checkpoint size should be ~700 MB")
!ls -lh ./snapshots/CS_scenes_20000.pth | awk '{print "Actual size: " $5}'
