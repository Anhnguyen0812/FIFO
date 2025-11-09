"""
CORRECT Training Config for Stage 2
====================================
Train full segmentation model (not just FogPassFilter)
"""

from dataset.paired_cityscapes import Pairedcityscapes
from dataset.foggy_zurich import foggyzurichDataSet
from torchvision import transforms
from torch.utils import data
import os
import torch

# ============================================================================
# DATASET PATHS - Kaggle
# ============================================================================
DATA_DIRECTORY = '/kaggle/input/cityscapes-filtered-fog'
DATA_LIST_PATH = './dataset/cityscapes_list/train_foggy_0.005.txt'
DATA_LIST_PATH_IM = './dataset/cityscapes_list/train_origin.txt'
DATA_LIST_PATH_RF = 'realfog_all_filenames.txt'

# ============================================================================
# CRITICAL: STAGE 2 CONFIGURATION
# ============================================================================
# This is Stage 2 - FULL MODEL TRAINING
TRAINING_MODE = 'train'  # ← MUST BE 'train', NOT 'fogpass'!

# ============================================================================
# RESUME FROM STAGE 1 CHECKPOINTS
# ============================================================================
# Load model + FogPassFilter from Stage 1
RESTORE_FROM = './snapshots/fast_training-11-09-00-25_FIFO5000.pth'
RESTORE_FROM_fogpass = './snapshots/fast_training-11-09-00-25_fogpassfilter_5000.pth'

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
INPUT_SIZE = '640,640'
NUM_CLASSES = 19
BATCH_SIZE = 4  # REQUIRED - hardcoded in fogpassfilter.py

# ============================================================================
# TRAINING STEPS - STAGE 2
# ============================================================================
# Continue from 5K to 20K (train for 15K more steps)
NUM_STEPS = 20000  # Loop until this step
NUM_STEPS_STOP = 20000  # Stop at this step

# ============================================================================
# LEARNING RATE
# ============================================================================
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
POWER = 0.9
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# ============================================================================
# CHECKPOINT & LOGGING
# ============================================================================
SAVE_PRED_EVERY = 2000  # Save every 2K steps
SNAPSHOT_DIR = './snapshots/'
LOG_DIR = './log'

# ============================================================================
# LOSS WEIGHTS
# ============================================================================
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET = 0.001
LAMBDA_FSM = 1.0  # Fog Style Matching loss weight
LAMBDA_CON = 0.1  # Consistency loss weight
LAMBDA_REAL_FOG = 1

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
random_mirror = True
random_crop = False

# ============================================================================
# GPU & MISC
# ============================================================================
GPU = 0
NUM_WORKERS = 4
RANDOM_SEED = 1234

# ============================================================================
# USAGE IN KAGGLE NOTEBOOK
# ============================================================================
"""
STEP 1: Upload this config to Kaggle

STEP 2: In Kaggle cell, run:

import sys
sys.path.insert(0, '/kaggle/working/fifo')

# Override train_config with Stage 2 settings
from configs.train_config import get_arguments
args = get_arguments()

# CRITICAL: Set mode to 'train'
args.modeltrain = 'train'  # ← THIS IS THE KEY!

# Set checkpoints
args.restore_from = './snapshots/fast_training-11-09-00-25_FIFO5000.pth'
args.restore_from_fogpass = './snapshots/fast_training-11-09-00-25_fogpassfilter_5000.pth'

# Set steps
args.num_steps = 20000
args.num_steps_stop = 20000

# Run training
!python main.py \\
    --mode train \\
    --restore-from ./snapshots/fast_training-11-09-00-25_FIFO5000.pth \\
    --restore-from-fogpass ./snapshots/fast_training-11-09-00-25_fogpassfilter_5000.pth \\
    --num-steps 20000 \\
    --num-steps-stop 20000

STEP 3: Verify Stage 2 is running

You should see in logs:
✅ "fsm loss: X.XX"
✅ "SF_loss_seg: X.XX"
✅ "CW_loss_seg: X.XX"
✅ "consistency loss: X.XX"
✅ "total_loss: X.XX"

If you only see snapshot messages → Still in Stage 1!

STEP 4: After training completes

Download checkpoint:
!cp ./snapshots/CS_scenes_20000.pth /kaggle/working/

Expected result after evaluation:
- mIoU: ~40-43% (instead of 6%!)
"""
