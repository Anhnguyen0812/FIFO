# FIFO Training on Kaggle - Quick Start

## 📋 TÓM TẮT NHANH

### Cấu trúc Dataset trên Kaggle
```
/kaggle/input/cityscapes-filtered-fog/
├── foggy_filtered/foggy_data/leftImg8bit_foggy/
├── gtFine_filtered/gtFine_data/gtFine/
├── leftImg8bit_filtered/leftImg8bit_data/leftImg8bit/
└── realfog_filtered_2gb/RGB/
```

### Các File Quan Trọng

#### 1. Config Files
- `train_config_kaggle.py` - Config cho full training
- `train_config_kaggle_test.py` - Config cho test với 5 ảnh

#### 2. Dataset Files  
- `paired_cityscapes_kaggle.py` - Dataset class cho paired images
- `foggy_zurich_kaggle.py` - Dataset class cho real fog

#### 3. Scripts
- `setup_and_train_test.sh` - Chạy test với 5 ảnh
- `setup_and_train_full.sh` - Chạy full training
- `verify_setup.sh` - Kiểm tra setup

#### 4. Main Script
- `main_kaggle.py` - Script training chính cho Kaggle

---

## 🚀 CÁCH SỬ DỤNG NHANH

### Bước 1: Upload Dataset
1. Upload dataset lên Kaggle với tên: `cityscapes-filtered-fog`
2. Hoặc update `KAGGLE_DATA_ROOT` trong config files

### Bước 2: Setup Code
```bash
# Trong Kaggle Notebook cell đầu tiên
!git clone YOUR_REPO_URL /kaggle/working/fifo
# Hoặc upload zip file và giải nén
```

### Bước 3: Test với 5 ảnh
```bash
# Cell mới
!bash /kaggle/working/fifo/kaggle_setup/verify_setup.sh
!bash /kaggle/working/fifo/kaggle_setup/setup_and_train_test.sh
```

### Bước 4: Full Training (nếu test OK)
```bash
# Tạo notebook mới với GPU T4 x2
!bash /kaggle/working/fifo/kaggle_setup/setup_and_train_full.sh
```

---

## 📝 CHI TIẾT TỪNG BƯỚC

### TEST với 5 ảnh (5-10 phút)

```python
# Cell 1: Clone code
!git clone YOUR_REPO /kaggle/working/fifo
%cd /kaggle/working/fifo

# Cell 2: Verify setup
!bash kaggle_setup/verify_setup.sh

# Cell 3: Install dependencies
!pip install -r kaggle_setup/requirements.txt -q

# Cell 4: Copy test config
!cp kaggle_setup/train_config_kaggle_test.py configs/train_config.py

# Cell 5: Setup wandb offline
import os
os.environ['WANDB_MODE'] = 'offline'

# Cell 6: Run test
!python main.py \
    --file-name "test_5img" \
    --modeltrain "fogpass" \
    --batch-size 1 \
    --num-steps 50 \
    --num-steps-stop 50 \
    --gpu 0

# Cell 7: Check output
!ls -lh /kaggle/working/snapshots/FIFO_test/
```

### FULL TRAINING (16-24 giờ)

**Stage 1: Train FogPassFilter**
```python
# Cell 1-5: Giống như test

# Cell 6: Copy full config
!cp kaggle_setup/train_config_kaggle.py configs/train_config.py

# Cell 7: Train FogPassFilter
!python main.py \
    --file-name "fifo_fogpass" \
    --modeltrain "fogpass" \
    --batch-size 4 \
    --num-steps 20000 \
    --num-steps-stop 20000 \
    --save-pred-every 5000 \
    --gpu 0
```

**Stage 2: Train Full Model**
```python
# Cell 8: Find checkpoint
import glob
ckpt = sorted(glob.glob('/kaggle/working/snapshots/FIFO_model/*fogpass*.pth'))[-1]
print(f"Using checkpoint: {ckpt}")

# Cell 9: Train full model
!python main.py \
    --file-name "fifo_full" \
    --modeltrain "train" \
    --batch-size 4 \
    --num-steps 60000 \
    --num-steps-stop 60000 \
    --save-pred-every 5000 \
    --restore-from-fogpass {ckpt} \
    --gpu 0
```

---

## 🔧 CẤU HÌNH KAGGLE

### Cho Test
- **GPU**: Tesla P100 hoặc T4 (đủ)
- **Memory**: 13-16GB (đủ)
- **Time**: 5-10 phút

### Cho Full Training
- **GPU**: T4 x2 (khuyến nghị) hoặc T4 single
- **Memory**: 16GB+ RAM
- **Time**: 16-24 giờ
- **Persistence**: Files only (để giữ checkpoints)
- **Internet**: ON (nếu dùng wandb online)

---

## 📊 THÔNG SỐ TRAINING

### Test Configuration
```python
BATCH_SIZE = 1
NUM_STEPS = 50
NUM_WORKERS = 2
SAVE_EVERY = 10
```

### Full Configuration
```python
BATCH_SIZE = 4
NUM_STEPS = 100000
NUM_STEPS_STOP = 60000
NUM_WORKERS = 4
SAVE_EVERY = 5000
```

### Loss Weights
```python
LAMBDA_FSM = 0.0000001
LAMBDA_CON = 0.0001
```

---

## 📁 OUTPUT FILES

### Checkpoints Location
```
/kaggle/working/snapshots/FIFO_model/
├── [name]_fogpassfilter_5000.pth
├── [name]_FIFO5000.pth
├── [name]_FIFO10000.pth
└── ...
```

### Checkpoint Format
```python
{
    'state_dict': model_state,
    'fogpass1_state_dict': fp1_state,
    'fogpass2_state_dict': fp2_state,
    'train_iter': iteration,
    'args': training_args
}
```

---

## 🐛 TROUBLESHOOTING

### GPU không hoạt động
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())  # Should be 1 or 2
```

### Dataset không tìm thấy
1. Check: Dataset đã add vào notebook chưa?
2. Check: Tên dataset khớp với config?
3. Run: `!ls /kaggle/input/`

### Out of Memory
- Giảm batch_size: 4 → 2 → 1
- Giảm num_workers: 4 → 2
- Giảm crop_size trong dataset code

### Import Error
```bash
!pip install pytorch-metric-learning wandb tqdm -q
```

### Wandb Error
```python
import os
os.environ['WANDB_MODE'] = 'offline'
```

---

## 💾 LƯU KẾT QUẢ

### Cách 1: Commit Notebook
1. Click "Save Version"
2. Chọn "Save & Run All"
3. Sau khi xong, vào Output tab
4. Download .pth files

### Cách 2: Copy sang Dataset
```bash
!mkdir -p /kaggle/working/fifo_output
!cp /kaggle/working/snapshots/FIFO_model/*.pth /kaggle/working/fifo_output/
# Commit notebook → output thành dataset
```

### Cách 3: Resume Training
```python
import glob
latest = sorted(glob.glob('/kaggle/working/snapshots/FIFO_model/*FIFO*.pth'))[-1]
# Use --restore-from and --restore-from-fogpass with latest
```

---

## ⏱️ TIMELINE

| Stage | Steps | Time (T4) | Checkpoints |
|-------|-------|-----------|-------------|
| Test | 50 | 5-10 min | 5 |
| FogPass | 20K | 4-6 hours | 4 |
| Full | 60K | 12-18 hours | 12 |
| **Total** | **80K** | **16-24 hours** | **16** |

---

## 📞 SUPPORT

Nếu gặp lỗi:
1. Run `verify_setup.sh` để check
2. Đọc error message trong cell output
3. Check file HUONG_DAN_KAGGLE.md để biết chi tiết

---

## ✅ CHECKLIST

### Trước khi Test
- [ ] Dataset uploaded & added
- [ ] Code uploaded to /kaggle/working/fifo
- [ ] GPU enabled
- [ ] Dependencies installed

### Trước khi Full Training
- [ ] Test passed
- [ ] GPU T4 x2 selected
- [ ] Persistence enabled
- [ ] Enough time quota (check Kaggle limit)

---

**Good luck! 🎉**
