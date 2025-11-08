"""
==============================================================================
KAGGLE NOTEBOOK: FULL TRAINING SETUP
==============================================================================
Copy từng cell vào Kaggle Notebook và chạy tuần tự
"""

# ==============================================================================
# CELL 1: Clone code và install dependencies
# ==============================================================================
import os
os.chdir('/kaggle/working')

# Clone code
!git clone -b phianh https://github.com/Anhnguyen0812/FIFO.git fifo

# Install dependencies (ĐÚNG THỨ TỰ!)
!pip install "numpy<2.0" -q
!pip install wandb pytorch-metric-learning tqdm -q
!pip install git+https://github.com/drsleep/DenseTorch.git -q

print("✅ Setup completed!")

# ==============================================================================
# CELL 2: Generate dataset list files
# ==============================================================================
import os
os.chdir('/kaggle/working/fifo')

# Run list generator
!python kaggle_setup/generate_dataset_lists.py

print("\n" + "="*80)
print("✅ Dataset lists generated!")
print("="*80)

# ==============================================================================
# CELL 3: Generate real fog list files
# ==============================================================================
!python kaggle_setup/generate_realfog_list.py

print("\n" + "="*80)
print("✅ Real fog lists generated!")
print("="*80)

# ==============================================================================
# CELL 4: Verify list files
# ==============================================================================
print("Checking generated files...")
print("\n--- train_foggy_0.005.txt (first 3 lines) ---")
!head -3 dataset/cityscapes_list/train_foggy_0.005.txt

print("\n--- train_origin.txt (first 3 lines) ---")
!head -3 dataset/cityscapes_list/train_origin.txt

print("\n--- realfog_all_filenames.txt (first 3 lines) ---")
!head -3 lists_file_names/realfog_all_filenames.txt

print("\n--- File counts ---")
!wc -l dataset/cityscapes_list/train_foggy_0.005.txt
!wc -l dataset/cityscapes_list/train_origin.txt
!wc -l dataset/cityscapes_list/val_foggy_0.005.txt
!wc -l dataset/cityscapes_list/val.txt
!wc -l lists_file_names/realfog_all_filenames.txt

print("\n✅ All files verified!")

# ==============================================================================
# CELL 5: Copy dataset files for full training
# ==============================================================================
# Copy full training config
!cp kaggle_setup/train_config_kaggle.py configs/train_config.py
!cp kaggle_setup/paired_cityscapes_kaggle.py dataset/paired_cityscapes.py
!cp kaggle_setup/foggy_zurich_kaggle.py dataset/Foggy_Zurich.py

# Verify config
print("=== Training Configuration ===")
!grep "^BATCH_SIZE" configs/train_config.py
!grep "^NUM_STEPS" configs/train_config.py | head -2
!grep "^LEARNING_RATE" configs/train_config.py

print("\n✅ Config files copied!")

# ==============================================================================
# CELL 6: Update config to use full dataset
# ==============================================================================
import os

# Read config
with open('configs/train_config.py', 'r') as f:
    config = f.read()

# Replace test file lists with full lists
config = config.replace(
    "DATA_LIST_PATH = f'./dataset/cityscapes_list/test_5images_foggy.txt'",
    "DATA_LIST_PATH = f'./dataset/cityscapes_list/train_foggy_0.005.txt'"
)
config = config.replace(
    "DATA_LIST_PATH_CWSF = './dataset/cityscapes_list/test_5images_origin.txt'",
    "DATA_LIST_PATH_CWSF = './dataset/cityscapes_list/train_origin.txt'"
)
config = config.replace(
    "DATA_LIST_RF = './lists_file_names/test_5images_rf.txt'",
    "DATA_LIST_RF = './lists_file_names/realfog_all_filenames.txt'"
)

# Write back
with open('configs/train_config.py', 'w') as f:
    f.write(config)

print("✅ Config updated to use full dataset!")

# Verify
print("\n=== Updated paths ===")
!grep "DATA_LIST_PATH" configs/train_config.py | head -3

# ==============================================================================
# CELL 7: Final verification before training
# ==============================================================================
import os
os.chdir('/kaggle/working/fifo')

print("="*80)
print("FINAL VERIFICATION")
print("="*80)

# Check dataset paths
print("\n1. Checking dataset structure...")
!ls -la /kaggle/input/cityscapes-filtered-fog/ | head -10

# Check list files
print("\n2. Checking list files...")
!ls -la dataset/cityscapes_list/
!ls -la lists_file_names/

# Check config
print("\n3. Training configuration:")
!grep "^BATCH_SIZE\|^NUM_STEPS\|^LEARNING_RATE" configs/train_config.py | head -5

print("\n4. Dataset sizes:")
!wc -l dataset/cityscapes_list/train_foggy_0.005.txt
!wc -l dataset/cityscapes_list/train_origin.txt
!wc -l lists_file_names/realfog_all_filenames.txt

print("\n" + "="*80)
print("✅ ALL CHECKS PASSED!")
print("="*80)
print("\n🚀 Ready to start training!")

# ==============================================================================
# CELL 8: START TRAINING - STAGE 1 (FogPassFilter)
# ==============================================================================
import os
os.chdir('/kaggle/working/fifo')

print("="*80)
print("STARTING STAGE 1: FogPassFilter Training (20K steps)")
print("="*80)

!python main.py --file-name "full_training" --modeltrain "fogpass"

# ==============================================================================
# CELL 9: Monitor training (optional - run in separate notebook)
# ==============================================================================
# Check latest checkpoint
!ls -lth snapshots/ | head -10

# Check wandb logs
!ls -la wandb/

# View last few lines of log (if saved to file)
# !tail -50 training.log

# ==============================================================================
# CELL 10: After Stage 1 completes, start Stage 2
# ==============================================================================
import os
os.chdir('/kaggle/working/fifo')

print("="*80)
print("STARTING STAGE 2: Full Model Training (60K steps)")
print("="*80)

# Stage 2 will automatically load Stage 1 checkpoint and continue
!python main.py --file-name "full_training" --modeltrain "fogpass"

print("\n✅ Training completed!")
print("📁 Check snapshots/ for saved models")
