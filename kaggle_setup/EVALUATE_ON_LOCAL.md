# 🎯 EVALUATE MODEL ON LOCAL MACHINE

Hướng dẫn đánh giá model đã train trên Kaggle, chạy trên máy local của bạn.

---

## 📋 Prerequisites

1. ✅ Model đã train xong trên Kaggle
2. ✅ Download model checkpoint về local
3. ✅ Dataset Foggy Zurich/Cityscapes trên local
4. ✅ Python environment với dependencies

---

## 📦 Step 1: Setup Environment

### Clone code từ GitHub

```bash
cd ~/Documents/1/fifo  # Hoặc thư mục bạn muốn
git pull origin phianh  # Pull code mới nhất
```

### Install dependencies

```bash
pip install torch torchvision
pip install numpy pillow tqdm
pip install matplotlib opencv-python
pip install git+https://github.com/drsleep/DenseTorch.git
```

---

## 💾 Step 2: Download Model từ Kaggle

### Option A: Download qua Kaggle UI

1. Vào Kaggle notebook đã train
2. Tìm file model: `/kaggle/working/snapshots/FIFO_model/full_training_FIFO60000.pth`
3. Right-click → Download
4. Copy vào thư mục local: `~/Documents/1/fifo/snapshots/`

### Option B: Download qua Kaggle API

```bash
# Install Kaggle API
pip install kaggle

# Setup API credentials (one-time)
# Download kaggle.json from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download notebook output files
kaggle kernels output <your-username>/<notebook-name> -p ~/Documents/1/fifo/snapshots/
```

---

## 📁 Step 3: Chuẩn bị Dataset

### Structure cần có trên local:

```
~/data/foggy_zurich/
├── RGB/
│   ├── test/
│   │   ├── *.png
│   └── val/
│       ├── *.png
└── gt/
    └── test/
        ├── *_gt_labelTrainIds.png
```

Hoặc Cityscapes:

```
~/data/cityscapes/
├── leftImg8bit/
│   └── val/
│       ├── frankfurt/*.png
│       ├── lindau/*.png
│       └── munster/*.png
└── gtFine/
    └── val/
        ├── frankfurt/*_gtFine_labelIds.png
        ├── lindau/*_gtFine_labelIds.png
        └── munster/*_gtFine_labelIds.png
```

---

## 🚀 Step 4: Run Evaluation

### A. Evaluate trên Foggy Zurich Test Set

```bash
cd ~/Documents/1/fifo

python evaluate.py \
    --restore-from snapshots/full_training_FIFO60000.pth \
    --data-dir ~/data/foggy_zurich \
    --data-list lists_file_names/leftImg8bit_testall_filenames.txt \
    --gpu 0
```

### B. Evaluate trên Cityscapes Foggy Val

```bash
python evaluate.py \
    --restore-from snapshots/full_training_FIFO60000.pth \
    --data-dir /path/to/cityscapes \
    --data-list dataset/cityscapes_list/val_foggy_0.005.txt \
    --gpu 0
```

### C. Evaluate trên Cityscapes Clear (Lindau)

```bash
python evaluate.py \
    --restore-from snapshots/full_training_FIFO60000.pth \
    --data-dir /path/to/cityscapes \
    --data-list dataset/cityscapes_list/clear_lindau.txt \
    --gpu 0
```

---

## 📊 Expected Output

```
===========================================
Test Foggy Zurich
===========================================
IoU: [0.95, 0.81, 0.89, ...]
Mean IoU: 0.68

Per-class results:
  0: road        - IoU: 0.95
  1: sidewalk    - IoU: 0.81
  2: building    - IoU: 0.89
  ...
  18: bicycle    - IoU: 0.72

===========================================
Results saved to: result/FIFO_model/
```

---

## 🎨 Step 5: Visualize Results (Optional)

### Tạo script visualize predictions

```python
# visualize_results.py
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model.refinenetlw import rf_lw101

# Load model
model = rf_lw101(num_classes=19)
checkpoint = torch.load('snapshots/full_training_FIFO60000.pth', 
                       map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Load and predict on single image
img = Image.open('test_image.png').convert('RGB')
# ... preprocessing ...
with torch.no_grad():
    output = model(img_tensor)
    pred = output.argmax(1).cpu().numpy()[0]

# Visualize
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img)
plt.title('Input')
plt.subplot(132)
plt.imshow(pred)
plt.title('Prediction')
plt.subplot(133)
plt.imshow(gt)
plt.title('Ground Truth')
plt.show()
```

---

## 🛠️ Troubleshooting

### Error: CUDA out of memory

```bash
# Dùng CPU thay vì GPU
python evaluate.py \
    --restore-from snapshots/model.pth \
    --data-dir ~/data \
    --gpu -1  # CPU mode
```

### Error: Module not found

```bash
# Install missing dependencies
pip install pillow numpy torch torchvision
pip install git+https://github.com/drsleep/DenseTorch.git
```

### Error: File not found

Kiểm tra paths:
```bash
# Check model exists
ls -lh snapshots/*.pth

# Check data exists
ls ~/data/foggy_zurich/RGB/test/ | head

# Check list files
cat lists_file_names/leftImg8bit_testall_filenames.txt | head
```

---

## 📝 Evaluate với nhiều scales (Better accuracy)

```bash
# Multi-scale evaluation (slower but more accurate)
python evaluate.py \
    --restore-from snapshots/full_training_FIFO60000.pth \
    --data-dir ~/data/foggy_zurich \
    --data-list lists_file_names/leftImg8bit_testall_filenames.txt \
    --gpu 0 \
    --scales 0.5,0.75,1.0,1.25,1.5
```

---

## 📊 Compare với Baseline

```bash
# Evaluate baseline model
python evaluate.py \
    --restore-from Cityscapes_pretrained_model.pth \
    --data-dir ~/data/foggy_zurich \
    --data-list lists_file_names/leftImg8bit_testall_filenames.txt \
    --gpu 0

# Evaluate FIFO model
python evaluate.py \
    --restore-from snapshots/full_training_FIFO60000.pth \
    --data-dir ~/data/foggy_zurich \
    --data-list lists_file_names/leftImg8bit_testall_filenames.txt \
    --gpu 0

# Compare results
python compare_models.py \
    --fifo-model snapshots/full_training_FIFO60000.pth \
    --baseline-model Cityscapes_pretrained_model.pth \
    --data-dir ~/data/foggy_zurich
```

---

## 🎯 Quick Evaluation Script

Tạo file `quick_eval.sh`:

```bash
#!/bin/bash

MODEL_PATH="snapshots/full_training_FIFO60000.pth"
DATA_DIR=~/data/foggy_zurich
GPU=0

echo "Evaluating FIFO model..."
python evaluate.py \
    --restore-from $MODEL_PATH \
    --data-dir $DATA_DIR \
    --data-list lists_file_names/leftImg8bit_testall_filenames.txt \
    --gpu $GPU

echo "Done! Check result/FIFO_model/ for outputs"
```

Chạy:
```bash
chmod +x quick_eval.sh
./quick_eval.sh
```

---

## ✅ Success Checklist

- [ ] Code pulled từ GitHub (branch phianh)
- [ ] Dependencies installed
- [ ] Model downloaded từ Kaggle về local
- [ ] Dataset có sẵn trên local
- [ ] Chạy evaluate.py thành công
- [ ] Thấy IoU scores và Mean IoU
- [ ] Results saved trong `result/FIFO_model/`

---

## 📚 Additional Resources

- `evaluate.py` - Main evaluation script
- `compute_iou.py` - Compute IoU metrics
- `compare_models.py` - Compare FIFO vs Baseline
- `inference_single_image.py` - Test on single image

---

**Chúc đánh giá model thành công!** 🎉
