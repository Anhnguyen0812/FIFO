# ============================================================================
# KAGGLE CELLS - COPY TỪNG CELL VÀO KAGGLE NOTEBOOK
# ============================================================================

# ============================================================================
# CELL 1: Setup Environment
# ============================================================================
import os
os.chdir('/kaggle/working/fifo')
!pwd

# ============================================================================
# CELL 2: Generate Dataset Lists
# ============================================================================
!python kaggle_setup/generate_dataset_lists.py
!python kaggle_setup/generate_realfog_list.py

# Verify lists created
!ls -lh dataset/cityscapes_list/*.txt
!ls -lh realfog_all_filenames.txt

# ============================================================================
# CELL 3: STAGE 1 - Train FogPassFilter (0 → 10K steps)
# ============================================================================
print("=" * 70)
print("STAGE 1: Training FogPassFilter (0 → 10K steps)")
print("Mode: fogpass")
print("Expected time: ~1.8 hours on P100")
print("=" * 70)

!python main.py \
    --file-name 'FIFO_model' \
    --modeltrain fogpass \
    --num-steps 10000 \
    --num-steps-stop 10000 \
    --batch-size 4 \
    --save-pred-every 5000 \
    --snapshot-dir './snapshots/' \
    --gpu 0

print("\n✅ Stage 1 completed!")

# ============================================================================
# CELL 4: Check Stage 1 Checkpoints
# ============================================================================
print("Checkpoints from Stage 1:")
!ls -lh ./snapshots/*.pth

# ============================================================================
# CELL 5: STAGE 2 - Train Full Model (10K → 20K steps)
# ============================================================================
print("=" * 70)
print("STAGE 2: Training Full Segmentation Model (10K → 20K steps)")
print("Mode: train")
print("Expected time: ~1.8 hours on P100")
print("=" * 70)

!python main.py \
    --file-name 'FIFO_model' \
    --modeltrain train \
    --restore-from './snapshots/FIFO_model10000.pth' \
    --restore-from-fogpass './snapshots/FIFO_model10000.pth' \
    --num-steps 20000 \
    --num-steps-stop 20000 \
    --batch-size 4 \
    --save-pred-every 2000 \
    --snapshot-dir './snapshots/' \
    --gpu 0

print("\n✅ Stage 2 completed!")

# ============================================================================
# CELL 6: Check Final Checkpoints
# ============================================================================
print("Final checkpoints:")
!ls -lh ./snapshots/*.pth

print("\nFinal model:")
!ls -lh ./snapshots/FIFO_model20000.pth

# ============================================================================
# CELL 7: Prepare for Download
# ============================================================================
# Copy final model to working directory for easy download
!cp ./snapshots/FIFO_model20000.pth /kaggle/working/FIFO_20K_final.pth

print("✅ Model ready for download at: /kaggle/working/FIFO_20K_final.pth")
print("File size:")
!ls -lh /kaggle/working/FIFO_20K_final.pth

# ============================================================================
# CELL 8: Verify Training Success
# ============================================================================
print("=" * 70)
print("VERIFICATION")
print("=" * 70)

import torch

# Load and check checkpoint
checkpoint = torch.load('./snapshots/FIFO_model20000.pth', map_location='cpu')
print(f"Checkpoint keys: {checkpoint.keys()}")
print(f"Contains state_dict: {'state_dict' in checkpoint}")
print(f"Contains fogpass1_state_dict: {'fogpass1_state_dict' in checkpoint}")
print(f"Contains fogpass2_state_dict: {'fogpass2_state_dict' in checkpoint}")
print(f"Training iteration: {checkpoint.get('train_iter', 'N/A')}")

print("\n✅ If all keys present, checkpoint is valid!")
print("\n📥 Next: Click 'Save Version' → Go to 'Output' → Download FIFO_20K_final.pth")
print("📊 Then evaluate on local: python evaluate_cpu.py --file-name 'FIFO_20K' --restore-from ./FIFO_20K_final.pth")
print("🎯 Expected mIoU: ~40-43% (instead of 6%!)")
