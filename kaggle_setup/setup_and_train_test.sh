#!/bin/bash
# Script setup môi trường và train FIFO trên Kaggle - TEST với 5 ảnh

echo "=========================================="
echo "FIFO Training Setup on Kaggle - TEST MODE"
echo "=========================================="

# Tạo thư mục lưu snapshots
mkdir -p /kaggle/working/snapshots/FIFO_test
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

# Copy config cho test
echo "Copying test config..."
cp /kaggle/working/fifo/kaggle_setup/train_config_kaggle_test.py /kaggle/working/fifo/configs/train_config.py

# Install dependencies
echo "Installing dependencies..."
pip install "numpy<2.0"
pip install wandb pytorch-metric-learning tqdm
pip install git+https://github.com/drsleep/DenseTorch.git

# Login wandb (cần thay YOUR_WANDB_KEY)
echo "Setting up Wandb..."
# Uncomment và thay YOUR_WANDB_KEY bằng API key của bạn
# wandb login YOUR_WANDB_KEY

# Hoặc sử dụng wandb offline mode
export WANDB_MODE=offline

# Kiểm tra file list có đúng không
echo "Checking data list files..."
cat /kaggle/working/fifo/dataset/cityscapes_list/test_5images_foggy.txt
cat /kaggle/working/fifo/dataset/cityscapes_list/test_5images_origin.txt

echo "=========================================="
echo "Starting TEST training with 5 images..."
echo "=========================================="

# Chạy training test
cd /kaggle/working/fifo
python main.py \
    --file-name "test_5images" \
    --modeltrain "fogpass" \
    --batch-size 1 \
    --num-steps 50 \
    --num-steps-stop 50 \
    --save-pred-every 10 \
    --gpu 0

echo "=========================================="
echo "Test training completed!"
echo "Check results at: /kaggle/working/snapshots/FIFO_test/"
echo "=========================================="
