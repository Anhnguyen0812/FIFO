# 🔧 FIX: NumPy 2.x Compatibility Error

## ❌ LỖI GẶP PHẢI

```
AttributeError: _ARRAY_API not found
ImportError: numpy.core.multiarray failed to import

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash.
```

## 🎯 NGUYÊN NHÂN

Kaggle environment có NumPy 2.x nhưng matplotlib và các libraries khác được compile với NumPy 1.x, gây conflict.

## ✅ GIẢI PHÁP

### Downgrade NumPy về version < 2.0

```bash
!pip install "numpy<2.0" -q
```

---

## 📋 LỆNH INSTALL ĐẦY ĐỦ (CẬP NHẬT)

### Trong Kaggle Notebook:

```bash
# Cell Install - THỨ TỰ QUAN TRỌNG
!pip install "numpy<2.0" -q                                    # Fix NumPy conflict
!pip install wandb pytorch-metric-learning tqdm -q             # Main dependencies
!pip install git+https://github.com/drsleep/DenseTorch.git -q  # DenseTorch
```

**Lưu ý**: Install NumPy TRƯỚC để tránh conflict!

---

## 🚀 CELLS KAGGLE HOÀN CHỈNH

### Cell 1: Clone Code
```bash
!git clone -b phianh https://github.com/Anhnguyen0812/FIFO.git /kaggle/working/fifo
%cd /kaggle/working/fifo
```

### Cell 2: Install (CẬP NHẬT - FIX NUMPY)
```bash
!pip install "numpy<2.0" -q
!pip install wandb pytorch-metric-learning tqdm -q
!pip install git+https://github.com/drsleep/DenseTorch.git -q
```

### Cell 3: Verify Imports
```python
import numpy as np
import torch
import matplotlib.pyplot as plt
import densetorch as dt

print(f"✓ NumPy: {np.__version__}")
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ Matplotlib imported OK")
print(f"✓ DenseTorch imported OK")

# Check NumPy version
assert np.__version__ < "2.0", "NumPy should be < 2.0"
print(f"✓ NumPy version OK: {np.__version__}")
```

### Cell 4: Copy Config
```bash
!cp kaggle_setup/train_config_kaggle_test.py configs/train_config.py
```

### Cell 5: Setup Wandb
```python
import os
os.environ['WANDB_MODE'] = 'offline'
```

### Cell 6: Create Directories
```bash
!mkdir -p /kaggle/working/snapshots/FIFO_test
```

### Cell 7: RUN TEST
```bash
!python main.py --file-name "test" --modeltrain "fogpass" \
    --batch-size 1 --num-steps 50 --num-steps-stop 50 --gpu 0
```

---

## 🔍 KIỂM TRA VERSION

```python
import numpy as np
import matplotlib
import torch

print("Package versions:")
print(f"  NumPy: {np.__version__}")
print(f"  Matplotlib: {matplotlib.__version__}")
print(f"  PyTorch: {torch.__version__}")

# Verify NumPy < 2.0
if np.__version__.startswith('1.'):
    print("\n✓ NumPy version is compatible!")
else:
    print("\n⚠️ NumPy version may cause issues!")
    print("  Run: !pip install 'numpy<2.0' --force-reinstall")
```

---

## 🐛 NẾU VẪN LỖI

### Option 1: Force reinstall NumPy
```bash
!pip uninstall numpy -y
!pip install "numpy<2.0"
```

### Option 2: Reinstall matplotlib với NumPy compatible
```bash
!pip install "numpy<2.0" --force-reinstall
!pip install matplotlib --force-reinstall
```

### Option 3: Restart kernel
Sau khi install, restart kernel trong Kaggle:
- **Runtime** → **Restart Runtime**

---

## 📦 DEPENDENCIES HOÀN CHỈNH

File: `requirements.txt` (đã cập nhật)
```
torch>=1.7.0
torchvision>=0.8.0
numpy<2.0.0                    # ← QUAN TRỌNG: < 2.0
pillow>=8.0.0
tqdm>=4.50.0
wandb>=0.10.0
pytorch-metric-learning>=0.9.0
matplotlib>=3.3.0
packaging>=20.0
git+https://github.com/drsleep/DenseTorch.git
```

---

## ⚠️ LƯU Ý

1. **NumPy version**: PHẢI < 2.0
2. **Install order**: Install NumPy TRƯỚC
3. **Restart kernel**: Sau khi install nếu cần
4. Kaggle đôi khi cache packages, cần force reinstall

---

## ✅ CHECKLIST

- [ ] Install NumPy < 2.0 TRƯỚC
- [ ] Install các dependencies khác
- [ ] Verify NumPy version
- [ ] Test import matplotlib
- [ ] Test import densetorch
- [ ] Restart kernel nếu cần

---

## 📊 VERIFIED VERSIONS

Tested và hoạt động:
- NumPy: 1.24.3 (hoặc bất kỳ < 2.0)
- Matplotlib: 3.7.1+
- PyTorch: 2.0.0+
- DenseTorch: latest from GitHub

---

## 🎯 TÓM TẮT

**Lỗi**: NumPy 2.x không tương thích với matplotlib
**Fix**: Downgrade NumPy về < 2.0
**Lệnh**: `!pip install "numpy<2.0" -q`

---

**Đã update tất cả files hướng dẫn với fix này!** 🎉

Files đã cập nhật:
- ✓ requirements.txt
- ✓ KAGGLE_NOTEBOOK_SETUP.md
- ✓ setup_and_train_test.sh
- ✓ setup_and_train_full.sh
- ✓ SUMMARY.md
- ✓ KAGGLE_CELLS_SCRIPT.py

**Chạy lại với lệnh install mới!** 🚀
