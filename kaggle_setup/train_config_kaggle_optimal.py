"""
FIFO Training Configuration - OPTIMAL for Filtered Dataset (~500-800 images)
=============================================================================
This config is optimized for the filtered Cityscapes dataset to prevent overfitting
while achieving good performance.

Dataset size: ~500-800 images (filtered)
Steps: 20,000 (optimal balance)
Training time: ~4 hours on P100
Expected mIoU: ~40-43% on test sets
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
DATA_LIST_PATH_RF = 'realfog_all_filenames.txt'  # Generated from generate_realfog_list.py

# ============================================================================
# TRAINING MODE
# ============================================================================
# Stage 1: 'fogpass' - Train FogPassFilter only (conv1, res1) - 0 to 10K steps
# Stage 2: 'train' - Train full model (all layers) - 10K to 20K steps
TRAINING_MODE = 'fogpass'  # Start with fogpass, will switch to 'train' at 10K

# ============================================================================
# MODEL CONFIGURATION - OPTIMAL FOR FILTERED DATASET
# ============================================================================
# CRITICAL: Must use 'without_pretraining' (not 'no_model')
RESTORE_FROM = 'without_pretraining'

# Model architecture
INPUT_SIZE = '640,640'  # Input image size
NUM_CLASSES = 19  # Cityscapes has 19 classes

# BATCH SIZE = 4 (REQUIRED - hardcoded in fogpassfilter.py)
BATCH_SIZE = 4  # DO NOT CHANGE - fogpassfilter uses indices [0,1,2,3]

# ============================================================================
# TRAINING STEPS - OPTIMIZED FOR ~500-800 IMAGES
# ============================================================================
# With 500-800 images:
# - Each epoch = 125-200 iterations
# - 20K steps = 100-160 epochs (reasonable, prevents overfitting)
# - 10K steps for Stage 1 (FogPassFilter)
# - 10K steps for Stage 2 (Full model)

NUM_STEPS = 10000  # Stage 1: FogPassFilter training
NUM_STEPS_STOP = 20000  # Stage 2: Total steps (10K + 10K)

# ============================================================================
# LEARNING RATE - Keep default for stability
# ============================================================================
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
POWER = 0.9
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# ============================================================================
# CHECKPOINT & LOGGING - More frequent for shorter training
# ============================================================================
SAVE_PRED_EVERY = 2000  # Save checkpoint every 2K steps (10 checkpoints total)
SNAPSHOT_DIR = './snapshots_optimal/'  # Changed directory name
LOG_DIR = './log_optimal'

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# Real fog dataset weight (for Stage 2 training)
LAMBDA_REAL_FOG = 1

# Data augmentation (helps prevent overfitting with small dataset)
random_mirror = True  # Random horizontal flip
random_crop = False   # Disabled for foggy images

# ============================================================================
# TRAINING SETUP
# ============================================================================
# Losses configuration
LAMBDA_SEG = 0.1  # Segmentation loss weight
LAMBDA_ADV_TARGET = 0.001  # Adversarial loss weight

# GPU settings
GPU = 0
NUM_WORKERS = 4

# Deterministic training
RANDOM_SEED = 1234

# ============================================================================
# OPTIMIZER - No changes needed
# ============================================================================
# Using SGD with momentum for main model
# Using Adam for discriminator

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
    
    # Paired Cityscapes dataset (foggy + clear)
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
    
    # Real fog dataset
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
TRAINING STEPS:

1. Generate dataset list files (if not done):
   !python generate_dataset_lists.py
   !python generate_realfog_list.py

2. Stage 1 - Train FogPassFilter (0 to 10K steps):
   !python main.py --mode fogpass

3. Stage 2 - Train full model (10K to 20K steps):
   !python main.py --mode train

Or run both stages continuously:
   !python main.py --mode fogpass  # Will auto-switch to 'train' at 10K steps

EXPECTED RESULTS:
- Training time: ~4 hours on P100 GPU (~1.4 it/s)
- Stage 1 (FogPassFilter): 10K steps = ~2 hours
- Stage 2 (Full model): 10K steps = ~2 hours
- Expected mIoU on test sets: ~40-43%
- Checkpoints saved every 2K steps in ./snapshots_optimal/
- Risk of overfitting: LOW (optimal balance)

MONITORING:
- Watch training/validation loss gap
- If validation loss increases while training loss decreases → overfitting
- Can use wandb for real-time monitoring

WHY 20K STEPS?
- With ~500-800 images, 20K steps = 100-160 epochs
- This is optimal balance between learning and not overfitting
- 60K steps would be 300-480 epochs → severe overfitting!

CHECKPOINTS:
- Step 2K: snapshots_optimal/CS_scenes_2000.pth
- Step 4K: snapshots_optimal/CS_scenes_4000.pth
- ...
- Step 20K: snapshots_optimal/CS_scenes_20000.pth (final model)
"""
