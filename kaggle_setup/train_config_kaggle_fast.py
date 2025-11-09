"""
Config cho FAST training (2 giờ) trên Kaggle
Giảm steps để train nhanh, phù hợp cho testing/demo
"""
import argparse
import numpy as np

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
BETA = 0.005
BATCH_SIZE = 4  # Keep at 4 for code compatibility
ITER_SIZE = 1
NUM_WORKERS = 4

# Kaggle paths
KAGGLE_DATA_ROOT = '/kaggle/input/cityscapes-filtered-fog'

# Dataset paths
DATA_DIRECTORY = KAGGLE_DATA_ROOT
DATA_LIST_PATH = f'./dataset/cityscapes_list/train_foggy_{BETA}.txt'
DATA_CITY_PATH = './dataset/cityscapes_list/clear_lindau.txt'
INPUT_SIZE = '960,480'  # Smaller size to fit batch=4 in GPU memory
DATA_DIRECTORY_CWSF = f'{KAGGLE_DATA_ROOT}/leftImg8bit_filtered/leftImg8bit_data'
DATA_LIST_PATH_CWSF = './dataset/cityscapes_list/train_origin.txt'
DATA_LIST_RF = './lists_file_names/realfog_all_filenames.txt'  # Use generated file without RGB/ prefix
DATA_DIR = KAGGLE_DATA_ROOT
INPUT_SIZE_RF = '900,506'  # Smaller size to fit batch=4 in GPU memory
NUM_CLASSES = 19 

# Training params - FAST MODE (2 hours @ 1.47 it/s = ~10,600 steps)
NUM_STEPS = 12000  # Total steps
NUM_STEPS_STOP = 10000  # Stop early for safety (2 hours)
RANDOM_SEED = 1234
RESTORE_FROM = 'without_pretraining'
RESTORE_FROM_fogpass = 'without_pretraining'
SAVE_PRED_EVERY = 2000  # Save every 2K steps instead of 5K

# Snapshots
SNAPSHOT_DIR = '/kaggle/working/snapshots/FIFO_model'   

# Learning rates
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
POWER = 0.9
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# Loss weights
LAMBDA_FSM = {'layer0':0.5, 'layer1':0.5}
LAMBDA_CON = {'conv1':0.01, 'res1':0.01}

# GPU
GPU = 0
SET = 'train'

def get_arguments():
    parser = argparse.ArgumentParser(description="FIFO Training - Fast Mode")
    
    # Data params
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY)
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH)
    parser.add_argument("--data-city-list", type=str, default=DATA_CITY_PATH)
    parser.add_argument("--data-list-rf", type=str, default=DATA_LIST_RF)
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE)
    parser.add_argument("--input-size-rf", type=str, default=INPUT_SIZE_RF)
    parser.add_argument("--data-dir-cwsf", type=str, default=DATA_DIRECTORY_CWSF)
    parser.add_argument("--data-list-cwsf", type=str, default=DATA_LIST_PATH_CWSF)
    parser.add_argument("--data-dir-rf", type=str, default=DATA_DIR)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    
    # Training params
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM)
    parser.add_argument("--restore-from-fogpass", type=str, default=RESTORE_FROM_fogpass)
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY)
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--gpu", type=int, default=GPU)
    parser.add_argument("--set", type=str, default=SET)
    
    # Loss params
    parser.add_argument("--lambda-fsm", type=dict, default=LAMBDA_FSM)
    parser.add_argument("--lambda-con", type=dict, default=LAMBDA_CON)
    
    # Required args
    parser.add_argument("--file-name", type=str, required=True)
    parser.add_argument("--modeltrain", type=str, required=True)
    
    return parser.parse_args()
