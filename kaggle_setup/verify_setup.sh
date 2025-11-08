#!/bin/bash
# Script kiểm tra setup trên Kaggle

echo "======================================"
echo "FIFO Kaggle Setup Verification"
echo "======================================"

# Kiểm tra GPU
echo ""
echo "1. Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo "✓ GPU available"
else
    echo "✗ GPU not found!"
fi

# Kiểm tra Python và PyTorch
echo ""
echo "2. Checking Python and PyTorch..."
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# Kiểm tra thư viện cần thiết
echo ""
echo "3. Checking required libraries..."
REQUIRED_LIBS=("torch" "torchvision" "numpy" "PIL" "tqdm" "wandb" "pytorch_metric_learning")

for lib in "${REQUIRED_LIBS[@]}"; do
    if python -c "import $lib" 2>/dev/null; then
        echo "✓ $lib installed"
    else
        echo "✗ $lib not found!"
    fi
done

# Kiểm tra dataset
echo ""
echo "4. Checking dataset structure..."
DATASET_ROOT="/kaggle/input/cityscapes-filtered-fog"

if [ -d "$DATASET_ROOT" ]; then
    echo "✓ Dataset root found: $DATASET_ROOT"
    
    # Kiểm tra các thư mục chính
    FOLDERS=("foggy_filtered" "gtFine_filtered" "leftImg8bit_filtered" "realfog_filtered_2gb")
    for folder in "${FOLDERS[@]}"; do
        if [ -d "$DATASET_ROOT/$folder" ]; then
            echo "  ✓ $folder"
        else
            echo "  ✗ $folder NOT FOUND!"
        fi
    done
else
    echo "✗ Dataset not found at $DATASET_ROOT"
    echo "  Please check:"
    echo "  1. Dataset has been uploaded to Kaggle"
    echo "  2. Dataset has been added to this notebook"
    echo "  3. Dataset name matches the path in config"
fi

# Kiểm tra code structure
echo ""
echo "5. Checking code structure..."
CODE_ROOT="/kaggle/working/fifo"

if [ -d "$CODE_ROOT" ]; then
    echo "✓ Code root found: $CODE_ROOT"
    
    # Kiểm tra các file quan trọng
    FILES=("main.py" "configs/train_config.py" "model/refinenetlw.py" "model/fogpassfilter.py")
    for file in "${FILES[@]}"; do
        if [ -f "$CODE_ROOT/$file" ]; then
            echo "  ✓ $file"
        else
            echo "  ✗ $file NOT FOUND!"
        fi
    done
    
    # Kiểm tra kaggle_setup
    if [ -d "$CODE_ROOT/kaggle_setup" ]; then
        echo "  ✓ kaggle_setup directory"
    else
        echo "  ✗ kaggle_setup directory NOT FOUND!"
    fi
else
    echo "✗ Code not found at $CODE_ROOT"
    echo "  Please upload code to /kaggle/working/fifo"
fi

# Kiểm tra list files
echo ""
echo "6. Checking data list files..."
LIST_FILES=(
    "dataset/cityscapes_list/test_5images_foggy.txt"
    "dataset/cityscapes_list/test_5images_origin.txt"
    "lists_file_names/test_5images_rf.txt"
    "lists_file_names/RGB_sum_filenames.txt"
)

for file in "${LIST_FILES[@]}"; do
    if [ -f "$CODE_ROOT/$file" ]; then
        lines=$(wc -l < "$CODE_ROOT/$file")
        echo "  ✓ $file ($lines lines)"
    else
        echo "  ✗ $file NOT FOUND!"
    fi
done

# Kiểm tra thư mục output
echo ""
echo "7. Checking output directories..."
OUTPUT_DIRS=(
    "/kaggle/working/snapshots"
    "/kaggle/working/results"
)

for dir in "${OUTPUT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir exists"
    else
        echo "  ℹ Creating $dir..."
        mkdir -p "$dir"
    fi
done

# Tổng kết
echo ""
echo "======================================"
echo "Verification completed!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. If all checks passed: Run training script"
echo "2. If any check failed: Fix the issue before training"
echo "3. For test: bash kaggle_setup/setup_and_train_test.sh"
echo "4. For full: bash kaggle_setup/setup_and_train_full.sh"
echo ""
