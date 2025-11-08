"""
Config cho training FIFO trên Kaggle
Sử dụng với dataset đã upload trên Kaggle
"""
import argparse
import numpy as np

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
BETA = 0.005
BATCH_SIZE = 8  # For T4 x2 (4 per GPU). Use 4 for single GPU.
ITER_SIZE = 1
NUM_WORKERS = 4

# Đường dẫn Kaggle (mặc định dataset được mount tại /kaggle/input/)
# Thay 'your-dataset-name' bằng tên dataset bạn đã upload trên Kaggle
KAGGLE_DATA_ROOT = '/kaggle/input/cityscapes-filtered-fog'

# Đường dẫn các thư mục data
DATA_DIRECTORY = KAGGLE_DATA_ROOT
DATA_LIST_PATH = f'./dataset/cityscapes_list/train_foggy_{BETA}.txt'
DATA_CITY_PATH = './dataset/cityscapes_list/clear_lindau.txt'
INPUT_SIZE = '2048,1024'
DATA_DIRECTORY_CWSF = f'{KAGGLE_DATA_ROOT}/leftImg8bit_filtered/leftImg8bit_data'
DATA_LIST_PATH_CWSF = './dataset/cityscapes_list/train_origin.txt'
DATA_LIST_RF = './lists_file_names/realfog_all_filenames.txt'  # Use generated file without RGB/ prefix
DATA_DIR = KAGGLE_DATA_ROOT
INPUT_SIZE_RF = '1920,1080'
NUM_CLASSES = 19 

# Training params - full training
NUM_STEPS = 100000 
NUM_STEPS_STOP = 60000  # early stopping
RANDOM_SEED = 1234
RESTORE_FROM = 'without_pretraining'
RESTORE_FROM_fogpass = 'without_pretraining'
SAVE_PRED_EVERY = 5000

# Đường dẫn lưu snapshots - Kaggle working directory
SNAPSHOT_DIR = '/kaggle/working/snapshots/FIFO_model'   

SET = 'train'

def get_arguments():
    parser = argparse.ArgumentParser(description="FIFO framework on Kaggle")

    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Iteration size")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="Number of workers for dataloader")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Root directory of foggy cityscapes data")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="List of foggy images")
    parser.add_argument("--data-city-list", type=str, default=DATA_CITY_PATH,
                        help="List of city images")
    parser.add_argument("--data-list-rf", type=str, default=DATA_LIST_RF,
                        help="List of real fog images")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Input size")
    parser.add_argument("--input-size-rf", type=str, default=INPUT_SIZE_RF,
                        help="Input size for real fog")
    parser.add_argument("--data-dir-cwsf", type=str, default=DATA_DIRECTORY_CWSF,
                        help="Root directory of clear weather cityscapes")
    parser.add_argument("--data-list-cwsf", type=str, default=DATA_LIST_PATH_CWSF,
                        help="List of clear weather images")
    parser.add_argument("--data-dir-rf", type=str, default=DATA_DIR,
                        help="Root directory of real fog data")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Early stopping iteration")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Path to restore model")
    parser.add_argument("--restore-from-fogpass", type=str, default=RESTORE_FROM_fogpass,
                        help="Path to restore fogpass model")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Directory to save snapshots")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID")
    parser.add_argument("--set", type=str, default=SET,
                        help="Training or validation set")
    parser.add_argument("--lambda-fsm", type=float, default=0.0000001,
                        help="Lambda for FSM loss")
    parser.add_argument("--lambda-con", type=float, default=0.0001,
                        help="Lambda for consistency loss")
    parser.add_argument("--file-name", type=str, required=True,
                        help="Experiment name")
    parser.add_argument("--modeltrain", type=str, required=True,
                        help="Mode: 'train' or 'fogpass'")
    
    return parser.parse_args()

args = get_arguments()
