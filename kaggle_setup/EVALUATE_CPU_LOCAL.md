# 🎯 EVALUATE ON LOCAL (CPU) - QUICK GUIDE

Hướng dẫn nhanh evaluate model trên máy local **không có GPU**.

---

## 📦 Step 1: Download Model từ Kaggle

### Trong Kaggle (sau khi train xong):

```python
# List checkpoints
!ls -lh /kaggle/working/snapshots/FIFO_model/

# Download link
from IPython.display import FileLink
FileLink('/kaggle/working/snapshots/FIFO_model/full_training_FIFO60000.pth')
```

Click download, copy vào `~/Documents/1/fifo/snapshots/`

---

## 🚀 Step 2: Run Evaluation

### A. Foggy Zurich (FZ)

```bash
cd ~/Documents/1/fifo

python evaluate.py \
    --restore-from snapshots/full_training_FIFO60000.pth \
    --data-dir-eval /path/to/foggy_zurich \
    --data-list-eval lists_file_names/leftImg8bit_testall_filenames.txt \
    --file-name FIFO_model
```

### B. Foggy Driving (FD)

```bash
python evaluate.py \
    --restore-from snapshots/full_training_FIFO60000.pth \
    --data-dir-eval /path/to/foggy_driving \
    --data-list-eval lists_file_names/foggy_driving_filenames.txt \
    --file-name FIFO_model
```

### C. Foggy Driving Dense (FDD)

```bash
python evaluate.py \
    --restore-from snapshots/full_training_FIFO60000.pth \
    --data-dir-eval /path/to/foggy_driving_dense \
    --data-list-eval lists_file_names/foggy_driving_dense_filenames.txt \
    --file-name FIFO_model
```

---

## ⏱️ Thời gian chạy (CPU)

- **FZ** (~40 images): ~5-10 phút
- **FD** (~100 images): ~15-20 phút  
- **FDD** (~300 images): ~45-60 phút

CPU chậm hơn GPU ~10-20x, nhưng vẫn OK cho evaluation!

---

## 📊 Kết quả

```
result_FZ/FIFO_model/
├── predictions/
│   ├── *.png
└── metrics.txt  # IoU scores

result_FD/FIFO_model/
result_FDD/FIFO_model/
result_Clindau/FIFO_model/
```

---

## 🔧 Nếu thiếu dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pillow tqdm matplotlib
```

---

## ⚡ Script nhanh (evaluate tất cả)

Tạo `eval_all.sh`:

```bash
#!/bin/bash

MODEL="snapshots/full_training_FIFO60000.pth"

echo "=== Evaluating Foggy Zurich ==="
python evaluate.py \
    --restore-from $MODEL \
    --data-dir-eval ~/data/foggy_zurich \
    --data-list-eval lists_file_names/leftImg8bit_testall_filenames.txt \
    --file-name FIFO_model

echo "=== Evaluating Foggy Driving ==="
python evaluate.py \
    --restore-from $MODEL \
    --data-dir-eval ~/data/foggy_driving \
    --data-list-eval lists_file_names/foggy_driving_filenames.txt \
    --file-name FIFO_model

echo "=== Done! Check result_* folders ==="
```

Chạy:
```bash
chmod +x eval_all.sh
./eval_all.sh
```

---

## ✅ Checklist

- [ ] Model downloaded từ Kaggle
- [ ] Model trong `snapshots/full_training_FIFO60000.pth`
- [ ] Dataset FZ, FD, FDD có sẵn trên local
- [ ] Chạy `evaluate.py` (sẽ tự dùng CPU)
- [ ] Đợi ~1 giờ (cho tất cả datasets)
- [ ] Check results trong `result_FZ/`, `result_FD/`, `result_FDD/`

---

**Evaluate.py tự động detect CPU và chạy được!** ✅
