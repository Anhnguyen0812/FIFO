# 🔧 FIX: ModuleNotFoundError: No module named 'densetorch'

## ❌ LỖI GẶP PHẢI
```
ModuleNotFoundError: No module named 'densetorch'
```

## ✅ GIẢI PHÁP

### Trong Kaggle Notebook, chạy cell sau TRƯỚC khi train:

```bash
!pip install wandb pytorch-metric-learning tqdm -q
!pip install git+https://github.com/drsleep/DenseTorch.git -q
```

---

## 📋 CẬP NHẬT HƯỚNG DẪN ĐẦY ĐỦ

### Cell 5: Install dependencies (CẬP NHẬT)
```bash
!pip install wandb pytorch-metric-learning tqdm -q
!pip install git+https://github.com/drsleep/DenseTorch.git -q
```

**Thời gian**: ~2-3 phút

**Kỳ vọng output**:
```
Successfully installed wandb-...
Successfully installed pytorch-metric-learning-...
Successfully installed tqdm-...
Successfully installed densetorch-...
```

---

## 🚀 THỨ TỰ CELLS ĐÚNG

### 1. Clone Code
```bash
!git clone -b phianh https://github.com/Anhnguyen0812/FIFO.git /kaggle/working/fifo
%cd /kaggle/working/fifo
```

### 2. Verify
```bash
!bash kaggle_setup/verify_setup.sh
```

### 3. Install (QUAN TRỌNG - THÊM DENSETORCH)
```bash
!pip install wandb pytorch-metric-learning tqdm -q
!pip install git+https://github.com/drsleep/DenseTorch.git -q
```

### 4. Kiểm tra install thành công
```python
# Verify all imports work
try:
    import wandb
    import pytorch_metric_learning
    import tqdm
    import densetorch as dt
    print("✓ All dependencies installed successfully!")
except ImportError as e:
    print(f"✗ Import error: {e}")
```

### 5. Config
```bash
!cp kaggle_setup/train_config_kaggle_test.py configs/train_config.py
```

### 6. Wandb
```python
import os
os.environ['WANDB_MODE'] = 'offline'
```

### 7. Tạo thư mục
```bash
!mkdir -p /kaggle/working/snapshots/FIFO_test
```

### 8. CHẠY TRAINING
```bash
!python main.py --file-name "test" --modeltrain "fogpass" \
    --batch-size 1 --num-steps 50 --num-steps-stop 50 --gpu 0
```

---

## 🔍 KIỂM TRA DENSETORCH ĐÃ INSTALL CHƯA

```python
import sys

try:
    import densetorch as dt
    print(f"✓ DenseTorch installed at: {dt.__file__}")
    print(f"✓ Version: {dt.__version__ if hasattr(dt, '__version__') else 'N/A'}")
except ImportError:
    print("✗ DenseTorch NOT installed!")
    print("\nInstall with:")
    print("!pip install git+https://github.com/drsleep/DenseTorch.git")
```

---

## 📦 REQUIREMENTS ĐẦY ĐỦ

File: `requirements.txt` (đã cập nhật)
```
torch>=1.7.0
torchvision>=0.8.0
numpy>=1.19.0
pillow>=8.0.0
tqdm>=4.50.0
wandb>=0.10.0
pytorch-metric-learning>=0.9.0
matplotlib>=3.3.0
packaging>=20.0
git+https://github.com/drsleep/DenseTorch.git
```

---

## 🐛 NẾU VẪN LỖI

### Thử install từng bước:

```bash
# Step 1: Basic dependencies
!pip install numpy pillow matplotlib packaging

# Step 2: PyTorch (nếu chưa có)
!pip install torch torchvision

# Step 3: Training dependencies
!pip install wandb pytorch-metric-learning tqdm

# Step 4: DenseTorch (QUAN TRỌNG)
!pip install git+https://github.com/drsleep/DenseTorch.git

# Step 5: Verify
!python -c "import densetorch; print('✓ DenseTorch OK')"
```

### Check Python path:
```python
import sys
print("Python executable:", sys.executable)
print("\nPython path:")
for p in sys.path:
    print(f"  {p}")
```

---

## 📝 LƯU Ý

- **DenseTorch** là thư viện bắt buộc cho FIFO
- Được sử dụng trong `utils/optimisers.py` và `utils/network.py`
- Phải install từ GitHub vì không có trên PyPI
- Install mất ~1-2 phút

---

## ✅ CHECKLIST

- [ ] Clone code từ GitHub
- [ ] Run verify_setup.sh
- [ ] Install wandb, pytorch-metric-learning, tqdm
- [ ] **Install DenseTorch** ← QUAN TRỌNG
- [ ] Verify import densetorch OK
- [ ] Copy config file
- [ ] Chạy training

---

**Fix đã được cập nhật vào tất cả files hướng dẫn!**

Các file đã cập nhật:
- ✓ KAGGLE_NOTEBOOK_SETUP.md
- ✓ requirements.txt
- ✓ setup_and_train_test.sh
- ✓ setup_and_train_full.sh
- ✓ SUMMARY.md

**Bây giờ chạy lại với lệnh install mới!** 🚀
