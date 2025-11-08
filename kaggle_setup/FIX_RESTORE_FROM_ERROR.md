# 🔧 FIX: FileNotFoundError: 'no_model'

## ❌ LỖI

```
FileNotFoundError: [Errno 2] No such file or directory: 'no_model'
```

## 🎯 NGUYÊN NHÂN

Config file có `RESTORE_FROM = 'no_model'` nhưng code trong `main.py` check với `'without_pretraining'`

## ✅ ĐÃ SỬA

Đã cập nhật các config files:
- `train_config_kaggle_test.py`
- `train_config_kaggle.py`

Từ:
```python
RESTORE_FROM = 'no_model'
RESTORE_FROM_fogpass = 'no_model'
```

Thành:
```python
RESTORE_FROM = 'without_pretraining'
RESTORE_FROM_fogpass = 'without_pretraining'
```

## 🚀 BÂY GIỜ CHẠY LẠI

### Pull code mới từ GitHub:

```bash
# Trong Kaggle, xóa thư mục cũ và clone lại
!rm -rf /kaggle/working/fifo
!git clone -b phianh https://github.com/Anhnguyen0812/FIFO.git /kaggle/working/fifo
%cd /kaggle/working/fifo
```

### Hoặc nếu đã clone, pull update:

```bash
%cd /kaggle/working/fifo
!git pull origin phianh
```

### Rồi chạy đầy đủ:

```bash
# Cell 1: Clone/Pull code
!git clone -b phianh https://github.com/Anhnguyen0812/FIFO.git /kaggle/working/fifo
%cd /kaggle/working/fifo

# Cell 2: Install
!pip install "numpy<2.0" -q
!pip install wandb pytorch-metric-learning tqdm -q
!pip install git+https://github.com/drsleep/DenseTorch.git -q

# Cell 3: Copy config (config đã được fix)
!cp kaggle_setup/train_config_kaggle_test.py configs/train_config.py

# Cell 4: Wandb
import os
os.environ['WANDB_MODE'] = 'offline'

# Cell 5: Tạo thư mục
!mkdir -p /kaggle/working/snapshots/FIFO_test

# Cell 6: CHẠY
!python main.py --file-name "test" --modeltrain "fogpass" \
    --batch-size 1 --num-steps 50 --num-steps-stop 50 --gpu 0
```

---

## 📝 LƯU Ý

Lỗi này do mismatch giữa:
- Code trong `main.py` line 23: `RESTORE_FROM = 'without_pretraining'`
- Config files: `RESTORE_FROM = 'no_model'`

Đã được fix trong config files mới nhất trên GitHub branch `phianh`.

---

**Push code lên GitHub và pull lại trong Kaggle!** 🚀
