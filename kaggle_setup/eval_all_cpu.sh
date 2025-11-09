#!/bin/bash
# Quick evaluation script for CPU

MODEL_PATH="snapshots/full_training_FIFO60000.pth"
FILE_NAME="FIFO_model"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    echo "Please download from Kaggle first!"
    exit 1
fi

echo "✅ Model found: $MODEL_PATH"
echo "🖥️  Running on CPU (will take ~1 hour for all datasets)"
echo ""

# Foggy Zurich
if [ -d "$HOME/data/foggy_zurich" ]; then
    echo "=== Evaluating Foggy Zurich ==="
    python evaluate.py \
        --restore-from $MODEL_PATH \
        --data-dir-eval $HOME/data/foggy_zurich \
        --data-list-eval lists_file_names/leftImg8bit_testall_filenames.txt \
        --file-name $FILE_NAME
    echo "✅ FZ Done!"
else
    echo "⏭️  Skipping FZ (dataset not found)"
fi

# Foggy Driving
if [ -d "$HOME/data/foggy_driving" ]; then
    echo ""
    echo "=== Evaluating Foggy Driving ==="
    python evaluate.py \
        --restore-from $MODEL_PATH \
        --data-dir-eval $HOME/data/foggy_driving \
        --data-list-eval lists_file_names/foggy_driving_filenames.txt \
        --file-name $FILE_NAME
    echo "✅ FD Done!"
else
    echo "⏭️  Skipping FD (dataset not found)"
fi

# Foggy Driving Dense
if [ -d "$HOME/data/foggy_driving_dense" ]; then
    echo ""
    echo "=== Evaluating Foggy Driving Dense ==="
    python evaluate.py \
        --restore-from $MODEL_PATH \
        --data-dir-eval $HOME/data/foggy_driving_dense \
        --data-list-eval lists_file_names/foggy_driving_dense_filenames.txt \
        --file-name $FILE_NAME
    echo "✅ FDD Done!"
else
    echo "⏭️  Skipping FDD (dataset not found)"
fi

echo ""
echo "========================================="
echo "✅ ALL EVALUATIONS COMPLETE!"
echo "========================================="
echo "Check results in:"
echo "  - result_FZ/$FILE_NAME/"
echo "  - result_FD/$FILE_NAME/"
echo "  - result_FDD/$FILE_NAME/"
