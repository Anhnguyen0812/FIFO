# 🚀 HƯỚNG DẪN CHẠY FIFO TRÊN KAGGLE - TỪNG BƯỚC CHI TIẾT

Repository: https://github.com/Anhnguyen0812/FIFO/tree/phianh

---

## 📋 BƯỚC 1: CHUẨN BỊ DATASET TRÊN KAGGLE

### 1.1 Upload Dataset
1. Truy cập https://www.kaggle.com/
2. Đăng nhập tài khoản
3. Click **Datasets** → **New Dataset**
4. Upload thư mục `cityscapes-filtered-fog` với cấu trúc:
   ```
   cityscapes-filtered-fog/
   ├── foggy_filtered/
   │   └── foggy_data/
   │       └── leftImg8bit_foggy/
   │           ├── train/
   │           └── val/
   ├── gtFine_filtered/
   │   └── gtFine_data/
   │       └── gtFine/
   │           ├── train/
   │           └── val/
   ├── leftImg8bit_filtered/
   │   └── leftImg8bit_data/
   │       └── leftImg8bit/
   │           ├── train/
   │           └── val/
   └── realfog_filtered_2gb/
       └── RGB/
   ```

5. **Tên dataset**: `cityscapes-filtered-fog` (quan trọng!)
6. Set **Public** hoặc **Private**
7. Click **Create**

### 1.2 Xác nhận Dataset đã upload
Sau khi upload xong, dataset sẽ có URL dạng:
```
https://www.kaggle.com/datasets/YOUR_USERNAME/cityscapes-filtered-fog
```

---

## 📋 BƯỚC 2: TẠO KAGGLE NOTEBOOK CHO TEST

### 2.1 Tạo Notebook mới
1. Vào **Code** → **New Notebook**
2. Đặt tên: `FIFO-Test-5Images`
3. **Settings** (góc phải):
   - **Accelerator**: GPU P100 hoặc T4
   - **Internet**: ON
   - **Persistence**: Files only

### 2.2 Add Dataset vào Notebook
1. Panel bên phải, click **+ Add Data**
2. Tìm dataset: `cityscapes-filtered-fog`
3. Click **Add**

---

## 📋 BƯỚC 3: CHẠY TEST VỚI 5 ẢNH (CELLS TRONG KAGGLE NOTEBOOK)

### Cell 1: Clone code từ GitHub
```python
!git clone -b phianh https://github.com/Anhnguyen0812/FIFO.git /kaggle/working/fifo
%cd /kaggle/working/fifo
!git status
```

**Kỳ vọng output**: 
```
Cloning into '/kaggle/working/fifo'...
On branch phianh
```

---

### Cell 2: Kiểm tra cấu trúc
```bash
!ls -la /kaggle/working/fifo/
!ls -la /kaggle/working/fifo/kaggle_setup/
```

**Kỳ vọng**: Thấy các thư mục main.py, configs/, model/, kaggle_setup/, etc.

---

### Cell 3: Verify setup
```bash
!chmod +x /kaggle/working/fifo/kaggle_setup/verify_setup.sh
!bash /kaggle/working/fifo/kaggle_setup/verify_setup.sh
```

**Kỳ vọng**: Các check đều có ✓ (tick xanh)

---

### Cell 4: Kiểm tra dataset path
```python
import os

# Kiểm tra dataset có tồn tại không
dataset_path = '/kaggle/input/cityscapes-filtered-fog'
if os.path.exists(dataset_path):
    print(f"✓ Dataset found at: {dataset_path}")
    
    # List các thư mục chính
    for item in os.listdir(dataset_path):
        full_path = os.path.join(dataset_path, item)
        print(f"  - {item}/")
else:
    print(f"✗ Dataset NOT found at: {dataset_path}")
    print(f"\nAvailable datasets:")
    !ls -la /kaggle/input/
```

**Nếu dataset path khác**, update trong file config:
```python
# Nếu path khác, update biến này
KAGGLE_DATA_ROOT = '/kaggle/input/YOUR-DATASET-NAME'
```

---

### Cell 5: Install dependencies
```bash
# Fix NumPy version conflict
!pip install "numpy<2.0" -q
!pip install wandb pytorch-metric-learning tqdm -q
!pip install git+https://github.com/drsleep/DenseTorch.git -q
```

**Kỳ vọng**: Install thành công, không có ERROR

---

### Cell 6: Copy config cho test
```bash
!cp /kaggle/working/fifo/kaggle_setup/train_config_kaggle_test.py /kaggle/working/fifo/configs/train_config.py
```

---

### Cell 7: Setup Wandb (offline mode)
```python
import os
os.environ['WANDB_MODE'] = 'offline'
print("Wandb set to offline mode")
```

**Lưu ý**: Nếu muốn dùng Wandb online:
```python
import wandb
wandb.login(key='YOUR_WANDB_API_KEY')
```

---

### Cell 8: Tạo thư mục output
```bash
!mkdir -p /kaggle/working/snapshots/FIFO_test
!mkdir -p /kaggle/working/results
!ls -la /kaggle/working/
```

---

### Cell 9: 🚀 CHẠY TEST TRAINING (50 STEPS)
```bash
%cd /kaggle/working/fifo

!python main.py \
    --file-name "test_5images" \
    --modeltrain "fogpass" \
    --batch-size 1 \
    --num-steps 50 \
    --num-steps-stop 50 \
    --save-pred-every 10 \
    --gpu 0
```

**Thời gian**: ~5-10 phút

**Kỳ vọng output**:
```
Loading datasets...
Datasets loaded successfully!
Starting training for 50 steps...
  0%|          | 0/50 [00:00<?, ?it/s]
...
taking snapshot ...
Training completed!
```

---

### Cell 10: Kiểm tra kết quả
```python
import glob
import os

# Check snapshots
snapshot_dir = '/kaggle/working/snapshots/FIFO_test'
checkpoints = glob.glob(f'{snapshot_dir}/*.pth')

print(f"Found {len(checkpoints)} checkpoint(s):")
for ckpt in sorted(checkpoints):
    size = os.path.getsize(ckpt) / (1024**2)  # MB
    print(f"  - {os.path.basename(ckpt)} ({size:.2f} MB)")

# Load và kiểm tra checkpoint
if checkpoints:
    import torch
    latest_ckpt = sorted(checkpoints)[-1]
    print(f"\nLoading checkpoint: {latest_ckpt}")
    
    checkpoint = torch.load(latest_ckpt, map_location='cpu')
    print(f"Keys in checkpoint: {list(checkpoint.keys())}")
    print(f"Training iteration: {checkpoint.get('train_iter', 'N/A')}")
    print("✓ Checkpoint valid!")
```

**Kỳ vọng**: Thấy các file .pth và load thành công

---

## ✅ NẾU TEST THÀNH CÔNG → CHUYỂN SANG FULL TRAINING

Nếu Cell 9 chạy thành công không lỗi, tiếp tục với Full Training!

---

## 📋 BƯỚC 4: CHẠY FULL TRAINING (NOTEBOOK MỚI)

### 4.1 Tạo Notebook mới cho Full Training
1. **Save Version** notebook test (để backup)
2. Tạo notebook mới: `FIFO-Full-Training`
3. **Settings** (QUAN TRỌNG):
   - **Accelerator**: **GPU T4 x2** (khuyến nghị)
   - **Internet**: ON
   - **Persistence**: **Files only** (để giữ checkpoints)

### 4.2 Add Dataset
- Add dataset `cityscapes-filtered-fog` (giống test)

---

### FULL TRAINING CELLS

### Cell 1: Clone code
```python
!git clone -b phianh https://github.com/Anhnguyen0812/FIFO.git /kaggle/working/fifo
%cd /kaggle/working/fifo
```

### Cell 2: Install dependencies
```bash
# Fix NumPy version conflict
!pip install "numpy<2.0" -q
!pip install wandb pytorch-metric-learning tqdm -q
!pip install git+https://github.com/drsleep/DenseTorch.git -q
```

### Cell 3: Copy config FULL
```bash
!cp /kaggle/working/fifo/kaggle_setup/train_config_kaggle.py /kaggle/working/fifo/configs/train_config.py
```

### Cell 4: Setup Wandb
```python
import os
os.environ['WANDB_MODE'] = 'offline'
```

### Cell 5: Tạo thư mục
```bash
!mkdir -p /kaggle/working/snapshots/FIFO_model
!mkdir -p /kaggle/working/results
```

### Cell 6: Kiểm tra GPU
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

**Kỳ vọng**: GPU count = 2 (nếu chọn T4 x2)

---

### Cell 7: 🚀 STAGE 1 - Train FogPassFilter (20,000 steps)
```bash
%cd /kaggle/working/fifo

!python main.py \
    --file-name "fifo_fogpass_stage1" \
    --modeltrain "fogpass" \
    --batch-size 4 \
    --num-steps 20000 \
    --num-steps-stop 20000 \
    --save-pred-every 5000 \
    --gpu 0
```

**Thời gian**: ~4-6 giờ
**Checkpoint**: Tạo file `*_fogpassfilter_5000.pth`

---

### Cell 8: Tìm checkpoint FogPassFilter
```python
import glob

checkpoint_dir = '/kaggle/working/snapshots/FIFO_model'
fogpass_checkpoints = sorted(glob.glob(f'{checkpoint_dir}/*fogpassfilter*.pth'))

if fogpass_checkpoints:
    latest_fogpass = fogpass_checkpoints[-1]
    print(f"✓ Found FogPassFilter checkpoint: {latest_fogpass}")
    print(f"Will use this for Stage 2")
else:
    print("✗ No FogPassFilter checkpoint found!")
    print("Please check Stage 1 completed successfully")
```

---

### Cell 9: 🚀 STAGE 2 - Train Full Model (60,000 steps)
```python
import glob

# Tìm checkpoint
fogpass_ckpt = sorted(glob.glob('/kaggle/working/snapshots/FIFO_model/*fogpassfilter*.pth'))[-1]
print(f"Using checkpoint: {fogpass_ckpt}")

# Chạy training
!python main.py \
    --file-name "fifo_full_stage2" \
    --modeltrain "train" \
    --batch-size 4 \
    --num-steps 60000 \
    --num-steps-stop 60000 \
    --save-pred-every 5000 \
    --restore-from-fogpass {fogpass_ckpt} \
    --gpu 0
```

**Thời gian**: ~12-18 giờ
**Checkpoints**: Lưu mỗi 5000 iterations

---

### Cell 10: Kiểm tra tất cả checkpoints
```python
import glob
import os

checkpoint_dir = '/kaggle/working/snapshots/FIFO_model'
all_checkpoints = sorted(glob.glob(f'{checkpoint_dir}/*.pth'))

print(f"Total checkpoints: {len(all_checkpoints)}\n")

for ckpt in all_checkpoints:
    name = os.path.basename(ckpt)
    size = os.path.getsize(ckpt) / (1024**2)  # MB
    print(f"  {name}")
    print(f"    Size: {size:.2f} MB")
    print()
```

---

## 💾 BƯỚC 5: LƯU KẾT QUẢ

### Cách 1: Commit Notebook (Khuyến nghị)
1. Click **Save Version** ở góc trên
2. Chọn **Save & Run All (Commit)**
3. Đợi notebook chạy xong
4. Vào **Output** tab
5. Download các file `.pth`

### Cách 2: Copy vào thư mục output
```python
# Cell mới
!mkdir -p /kaggle/working/fifo_checkpoints
!cp /kaggle/working/snapshots/FIFO_model/*.pth /kaggle/working/fifo_checkpoints/
!ls -lh /kaggle/working/fifo_checkpoints/
```

Sau đó commit notebook, output sẽ thành dataset có thể download.

---

## 🔄 TIẾP TỤC TRAINING (RESUME)

Nếu notebook timeout hoặc muốn tiếp tục:

### Cell mới: Resume Training
```python
import glob
import torch

# Tìm checkpoint mới nhất
all_ckpts = sorted(glob.glob('/kaggle/working/snapshots/FIFO_model/*FIFO*.pth'))
if all_ckpts:
    latest_ckpt = all_ckpts[-1]
    print(f"Resuming from: {latest_ckpt}")
    
    # Load để check iteration
    ckpt_data = torch.load(latest_ckpt, map_location='cpu')
    current_iter = ckpt_data.get('train_iter', 0)
    print(f"Current iteration: {current_iter}")
    print(f"Will continue to: 60000")
    
    # Resume training
    !python main.py \
        --file-name "fifo_resume" \
        --modeltrain "train" \
        --batch-size 4 \
        --num-steps 60000 \
        --num-steps-stop 60000 \
        --save-pred-every 5000 \
        --restore-from {latest_ckpt} \
        --restore-from-fogpass {latest_ckpt} \
        --gpu 0
else:
    print("No checkpoint found to resume!")
```

---

## 🐛 TROUBLESHOOTING

### Lỗi: "Dataset not found"
```python
# Kiểm tra tên dataset
!ls /kaggle/input/

# Nếu tên khác, update config
# Sửa file: configs/train_config.py
# Dòng: KAGGLE_DATA_ROOT = '/kaggle/input/TEN-MOI'
```

### Lỗi: "ModuleNotFoundError"
```bash
!pip install wandb pytorch-metric-learning tqdm -q
!pip install git+https://github.com/drsleep/DenseTorch.git -q
```

### Lỗi: "CUDA out of memory"
```python
# Giảm batch_size
# Thay --batch-size 4 thành --batch-size 2 hoặc 1
```

### Lỗi: "No such file or directory" cho test_5images
```bash
# Kiểm tra file tồn tại
!cat /kaggle/working/fifo/dataset/cityscapes_list/test_5images_foggy.txt
!cat /kaggle/working/fifo/dataset/cityscapes_list/test_5images_origin.txt
```

### Checkpoint không tải được
```python
import torch
ckpt_path = "PATH_TO_CHECKPOINT.pth"
try:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print("✓ Checkpoint loaded")
    print(f"Keys: {list(ckpt.keys())}")
except Exception as e:
    print(f"✗ Error: {e}")
```

---

## 📊 MONITOR TRAINING

### Xem logs trong Notebook
- Output hiển thị real-time trong cell
- Progress bar từ tqdm
- Loss values được in ra

### Kiểm tra GPU usage
```bash
# Cell riêng, chạy song song
!watch -n 5 nvidia-smi
```

### Check tiến trình
```python
import glob
checkpoints = glob.glob('/kaggle/working/snapshots/FIFO_model/*FIFO*.pth')
print(f"Checkpoints saved: {len(checkpoints)}")
for ckpt in sorted(checkpoints)[-3:]:  # 3 mới nhất
    print(f"  - {os.path.basename(ckpt)}")
```

---

## ⏱️ THỜI GIAN ƯỚC TÍNH

| Stage | Steps | GPU | Time |
|-------|-------|-----|------|
| **Test** | 50 | T4 | 5-10 min |
| **Stage 1 (FogPass)** | 20K | T4 | 4-6 hours |
| **Stage 2 (Full)** | 60K | T4 | 12-18 hours |
| **Total Full Training** | 80K | T4 | **16-24 hours** |

**Kaggle Limits**:
- GPU T4 x2: 30 hours/week
- Có thể chia thành nhiều session

---

## ✅ CHECKLIST TRƯỚC KHI CHẠY

### Test (5 ảnh):
- [ ] Dataset `cityscapes-filtered-fog` đã upload
- [ ] Dataset đã add vào notebook
- [ ] Code clone từ branch `phianh`
- [ ] GPU đã enable
- [ ] Dependencies installed
- [ ] Verify setup passed

### Full Training:
- [ ] Test chạy thành công
- [ ] Notebook mới với GPU T4 x2
- [ ] Persistence: Files only
- [ ] Đủ quota (check Kaggle settings)
- [ ] Config file đã copy đúng

---

## 🎯 KẾT QUẢ MONG ĐỢI

Sau khi hoàn thành, bạn sẽ có:
1. **~16 checkpoint files** (.pth)
2. **Final model**: iteration 60000
3. **File size**: mỗi checkpoint ~500-800MB
4. **Model trained**: Segmentation + FogPassFilter

---

## 📞 HỖ TRỢ

Nếu gặp lỗi:
1. Copy error message đầy đủ
2. Check cell output
3. Run verify_setup.sh
4. Check Kaggle logs

**Repository**: https://github.com/Anhnguyen0812/FIFO/tree/phianh

---

**Chúc bạn training thành công! 🚀🎉**
