"""
FIFO Training Configuration - RESUME from Step 10000 to 12000
==============================================================
This config resumes training from step 10000 checkpoint to complete 12000 steps.

Use this when:
- Training stopped at step 10000 (Stage 1 complete)
- Need to continue Stage 2 training (10K → 12K)
- Have checkpoint: CS_scenes_10000.pth
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
# RESUME CONFIGURATION - CRITICAL!
# ============================================================================
# MUST set to resume from step 10000
RESTORE_FROM = './snapshots/CS_scenes_10000.pth'  # Path to checkpoint
RESTORE_FROM_fogpass = './snapshots/CS_scenes_10000.pth'  # Same checkpoint

# Training will resume from step 10000
START_ITER = 10000  # Will be detected automatically from checkpoint name

# ============================================================================
# TRAINING MODE - Stage 2
# ============================================================================
# MUST use 'train' mode for Stage 2 (full model training)
TRAINING_MODE = 'train'  # NOT 'fogpass'

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
INPUT_SIZE = '640,640'
NUM_CLASSES = 19
BATCH_SIZE = 4  # DO NOT CHANGE

# ============================================================================
# TRAINING STEPS - RESUME CONFIGURATION
# ============================================================================
# Loop will run from 10000 to 12000
NUM_STEPS = 12000  # MUST be >= target step (12000)
NUM_STEPS_STOP = 12000  # Target final step

# This means: train for 2000 more steps (10K → 12K)

# ============================================================================
# LEARNING RATE - Same as original
# ============================================================================
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
POWER = 0.9
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# ============================================================================
# CHECKPOINT & LOGGING
# ============================================================================
SAVE_PRED_EVERY = 2000  # Will save at step 12000
SNAPSHOT_DIR = './snapshots/'  # Same directory as before
LOG_DIR = './log'

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
LAMBDA_REAL_FOG = 1
random_mirror = True
random_crop = False

# ============================================================================
# TRAINING SETUP
# ============================================================================
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET = 0.001

GPU = 0
NUM_WORKERS = 4
RANDOM_SEED = 1234

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def get_training_augmentation():
    """Get training data augmentation pipeline"""
    train_transform = [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
    return transforms.Compose(train_transform)

def get_training_dataset():
    """Create training dataset with paired foggy/clear images"""
    transform = get_training_augmentation()
    
    train_dataset = Pairedcityscapes(
        DATA_DIRECTORY,
        DATA_LIST_PATH,
        DATA_LIST_PATH_IM,
        crop_size=tuple(map(int, INPUT_SIZE.split(','))),
        mean=(0.485, 0.456, 0.406),
        mirror=random_mirror
    )
    
    return train_dataset

def get_real_fog_dataset():
    """Create real fog dataset (Foggy Zurich)"""
    transform = get_training_augmentation()
    
    real_fog_dataset = foggyzurichDataSet(
        DATA_DIRECTORY,
        DATA_LIST_PATH_RF,
        crop_size=tuple(map(int, INPUT_SIZE.split(','))),
        mean=(0.485, 0.456, 0.406)
    )
    
    return real_fog_dataset

def create_data_loader(dataset, batch_size=BATCH_SIZE):
    """Create data loader with specified batch size"""
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
"""
RESUME TRAINING STEPS:

1. Make sure checkpoint exists:
   Check that ./snapshots/CS_scenes_10000.pth exists

2. Copy this config to Kaggle notebook:
   Upload train_config_kaggle_resume.py

3. In Kaggle notebook cell, run:
   ```python
   # Import resume config
   import sys
   sys.path.insert(0, '/kaggle/working/fifo/kaggle_setup')
   from train_config_kaggle_resume import *
   
   # Update train_config.py to use these settings
   import configs.train_config as cfg
   cfg.RESTORE_FROM = RESTORE_FROM
   cfg.NUM_STEPS = NUM_STEPS
   cfg.NUM_STEPS_STOP = NUM_STEPS_STOP
   cfg.TRAINING_MODE = TRAINING_MODE
   ```

4. OR simpler: Run main.py with arguments:
   ```bash
   !python main.py \
       --restore-from ./snapshots/CS_scenes_10000.pth \
       --num-steps 12000 \
       --num-steps-stop 12000 \
       --mode train
   ```

5. Check that training resumes from step 10000:
   - Progress bar should show: 10000/12000
   - Only 2000 steps remaining
   - ETA: ~22 minutes

EXPECTED RESULTS:
- Resume from: Step 10000 (FogPassFilter trained)
- Train to: Step 12000 (Full model)
- Time needed: ~22 minutes
- Checkpoint saved: CS_scenes_12000.pth
- Mode: 'train' (full model, not just FogPassFilter)

VERIFICATION:
After starting, you should see:
- "Loading checkpoint from ./snapshots/CS_scenes_10000.pth"
- Progress: 10000/12000 (83%)
- ETA: ~22 minutes
- "save model .." at step 12000

TROUBLESHOOTING:
- If "checkpoint not found": Check path ./snapshots/CS_scenes_10000.pth
- If starts from step 0: RESTORE_FROM not set correctly
- If only trains FogPassFilter: Change TRAINING_MODE to 'train'
"""
