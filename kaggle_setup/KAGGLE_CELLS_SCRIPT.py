"""
KAGGLE CELL SCRIPT - COPY VÀO KAGGLE NOTEBOOK
Run từng cell theo thứ tự
"""

# ============================================================
# CELL 1: Clone Code
# ============================================================
!git clone -b phianh https://github.com/Anhnguyen0812/FIFO.git /kaggle/working/fifo
%cd /kaggle/working/fifo
!git status


# ============================================================
# CELL 2: Verify Setup
# ============================================================
!chmod +x kaggle_setup/verify_setup.sh
!bash kaggle_setup/verify_setup.sh


# ============================================================
# CELL 3: Install Dependencies (QUAN TRỌNG - FIX NUMPY + DENSETORCH)
# ============================================================
!pip install "numpy<2.0" -q
!pip install wandb pytorch-metric-learning tqdm -q
!pip install git+https://github.com/drsleep/DenseTorch.git -q
print("\n✓ All dependencies installed!")


# ============================================================
# CELL 4: Verify Imports
# ============================================================
import torch
import densetorch as dt
import wandb
import pytorch_metric_learning
from tqdm import tqdm

print("✓ PyTorch:", torch.__version__)
print("✓ CUDA available:", torch.cuda.is_available())
print("✓ DenseTorch imported successfully")
print("✓ Wandb imported successfully")
print("✓ pytorch-metric-learning imported successfully")
print("✓ tqdm imported successfully")


# ============================================================
# CELL 5: Check Dataset
# ============================================================
import os

dataset_path = '/kaggle/input/cityscapes-filtered-fog'
if os.path.exists(dataset_path):
    print(f"✓ Dataset found at: {dataset_path}\n")
    for item in os.listdir(dataset_path):
        print(f"  📁 {item}/")
else:
    print(f"✗ Dataset NOT found!")
    print("Available:")
    !ls /kaggle/input/


# ============================================================
# CELL 6: Copy Config for TEST
# ============================================================
!cp kaggle_setup/train_config_kaggle_test.py configs/train_config.py
!head -20 configs/train_config.py


# ============================================================
# CELL 7: Setup Wandb
# ============================================================
import os
os.environ['WANDB_MODE'] = 'offline'
print("✓ Wandb offline mode")


# ============================================================
# CELL 8: Create Directories
# ============================================================
!mkdir -p /kaggle/working/snapshots/FIFO_test
!mkdir -p /kaggle/working/results


# ============================================================
# CELL 9: RUN TEST TRAINING (50 steps, ~5-10 min)
# ============================================================
!python main.py \
    --file-name "test_5images" \
    --modeltrain "fogpass" \
    --batch-size 1 \
    --num-steps 50 \
    --num-steps-stop 50 \
    --save-pred-every 10 \
    --gpu 0


# ============================================================
# CELL 10: Check Results
# ============================================================
import glob
import os

checkpoints = sorted(glob.glob('/kaggle/working/snapshots/FIFO_test/*.pth'))
print(f"Found {len(checkpoints)} checkpoint(s):\n")

for ckpt in checkpoints:
    name = os.path.basename(ckpt)
    size = os.path.getsize(ckpt) / (1024**2)
    print(f"  ✓ {name} ({size:.2f} MB)")

if checkpoints:
    print("\n✅ TEST SUCCESSFUL!")
else:
    print("\n⚠️ No checkpoints found!")


# ============================================================
# CELL 11: Validate Checkpoint
# ============================================================
import torch

if checkpoints:
    latest = checkpoints[-1]
    checkpoint = torch.load(latest, map_location='cpu')
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    print(f"Training iter: {checkpoint.get('train_iter', 'N/A')}")
    print("✓ Checkpoint valid!")
