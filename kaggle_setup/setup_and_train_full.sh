#!/bin/bash
# Script setup môi trường và train FIFO trên Kaggle - FULL TRAINING

echo "=========================================="
echo "FIFO Training Setup on Kaggle - FULL MODE"
echo "=========================================="

# Tạo thư mục lưu snapshots
mkdir -p /kaggle/working/snapshots/FIFO_model
mkdir -p /kaggle/working/results

# Kiểm tra GPU
echo "Checking GPU availability..."
nvidia-smi

# Kiểm tra cấu trúc dataset
echo "Checking dataset structure..."
ls -la /kaggle/input/

# Kiểm tra dataset cityscapes
if [ -d "/kaggle/input/cityscapes-filtered-fog" ]; then
    echo "Dataset found!"
    ls -la /kaggle/input/cityscapes-filtered-fog/
else
    echo "WARNING: Dataset not found at /kaggle/input/cityscapes-filtered-fog"
    echo "Please check your Kaggle dataset name and update the path"
fi

# Copy config cho full training
echo "Copying full training config..."
cp /kaggle/working/fifo/kaggle_setup/train_config_kaggle.py /kaggle/working/fifo/configs/train_config.py

# Install dependencies
echo "Installing dependencies..."
pip install wandb pytorch-metric-learning tqdm

# Login wandb (cần thay YOUR_WANDB_KEY)
echo "Setting up Wandb..."
# Uncomment và thay YOUR_WANDB_KEY bằng API key của bạn
# wandb login YOUR_WANDB_KEY

# Hoặc sử dụng wandb offline mode nếu không dùng wandb
export WANDB_MODE=offline

echo "=========================================="
echo "Starting FULL training..."
echo "=========================================="

# Stage 1: Train FogPassFilter (20000 steps)
echo "Stage 1: Training FogPassFilter..."
cd /kaggle/working/fifo
python main.py \
    --file-name "fifo_full_fogpass" \
    --modeltrain "fogpass" \
    --batch-size 4 \
    --num-steps 20000 \
    --num-steps-stop 20000 \
    --save-pred-every 5000 \
    --gpu 0

# Stage 2: Train full model (60000 steps)
echo "Stage 2: Training full model..."
# Tìm checkpoint fogpass mới nhất
FOGPASS_CHECKPOINT=$(ls -t /kaggle/working/snapshots/FIFO_model/*fogpassfilter*.pth | head -1)
echo "Using FogPassFilter checkpoint: $FOGPASS_CHECKPOINT"

python main.py \
    --file-name "fifo_full_train" \
    --modeltrain "train" \
    --batch-size 4 \
    --num-steps 60000 \
    --num-steps-stop 60000 \
    --save-pred-every 5000 \
    --restore-from-fogpass "$FOGPASS_CHECKPOINT" \
    --gpu 0

echo "=========================================="
echo "Full training completed!"
echo "Check results at: /kaggle/working/snapshots/FIFO_model/"
echo "=========================================="
