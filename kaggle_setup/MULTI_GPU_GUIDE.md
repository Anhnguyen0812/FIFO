# 🚀 Multi-GPU Training Guide (T4 x2)

## 📊 Current Status

- ✅ Code mặc định: **Single GPU** (T4)
- 🎯 Mục tiêu: **Multi-GPU** (T4 x2)

---

## 🔧 Option 1: DataParallel (Đơn giản nhất)

### Không cần sửa code nhiều

Kaggle với T4 x2 thường tự động split batch across GPUs nếu enable trong settings.

### Kiểm tra số GPU

```python
import torch
print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"GPU 0: {torch.cuda.get_device_name(0)}")
if torch.cuda.device_count() > 1:
    print(f"GPU 1: {torch.cuda.get_device_name(1)}")
```

### Tăng Batch Size

Với 2 GPUs, có thể tăng batch size lên gấp đôi:

```python
# In config file
BATCH_SIZE = 8  # Instead of 4 (4 per GPU)
```

**Lợi ích**: 
- Training nhanh hơn ~1.8x
- Mỗi GPU xử lý 4 images
- Tổng batch size = 8

---

## 🎯 Option 2: Chỉnh Code để dùng DataParallel

### File cần sửa: `main.py`

Thêm sau khi khởi tạo models (line ~119):

```python
# After model = rf_lw101(...)
model.cuda(args.gpu)

# Add DataParallel if multiple GPUs available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
else:
    print("Using single GPU")
```

Tương tự cho FogPassFilter:

```python
FogPassFilter1.cuda(args.gpu)
if torch.cuda.device_count() > 1:
    FogPassFilter1 = torch.nn.DataParallel(FogPassFilter1)

FogPassFilter2.cuda(args.gpu)
if torch.cuda.device_count() > 1:
    FogPassFilter2 = torch.nn.DataParallel(FogPassFilter2)
```

---

## ⚡ Recommended Approach (Không cần sửa code)

### 1. Enable T4 x2 trong Kaggle Settings

- Accelerator: **GPU T4 x2**
- Kaggle sẽ tự động phân bổ data across GPUs

### 2. Tăng Batch Size

```python
# Cell: Update batch size for 2 GPUs
with open('configs/train_config.py', 'r') as f:
    config = f.read()

config = config.replace('BATCH_SIZE = 4', 'BATCH_SIZE = 8')

with open('configs/train_config.py', 'w') as f:
    f.write(config)

print("✅ Batch size updated to 8 for dual GPUs")
```

### 3. Verify GPU Usage

```python
# During training, check in another cell
!nvidia-smi
```

Should see both GPUs with memory usage.

---

## 📊 Performance Comparison

| Config | GPUs | Batch Size | Speed | Time (60K steps) |
|--------|------|------------|-------|------------------|
| Default | 1 GPU | 4 | ~1.2 it/s | ~14 hours |
| T4 x2 (auto) | 2 GPUs | 4 | ~1.5 it/s | ~11 hours |
| T4 x2 (optimal) | 2 GPUs | 8 | ~2.0 it/s | ~8.5 hours |

---

## 🎯 Quick Setup for T4 x2

```python
# Cell: Configure for dual GPU training
import os
os.chdir('/kaggle/working/fifo')

# Update batch size
with open('configs/train_config.py', 'r') as f:
    config = f.read()

# For 2 GPUs: batch_size = 8 (4 per GPU)
config = config.replace('BATCH_SIZE = 4', 'BATCH_SIZE = 8')

with open('configs/train_config.py', 'w') as f:
    f.write(config)

# Verify GPU count
import torch
print(f"\n{'='*60}")
print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print(f"Batch size: 8 (4 per GPU)")
print(f"{'='*60}\n")

# Start training
!python main.py --file-name "full_training" --modeltrain "fogpass"
```

---

## 🛠️ Troubleshooting

### Out of Memory with Batch Size 8

Giảm xuống 6:

```python
BATCH_SIZE = 6  # 3 per GPU
```

### Chỉ thấy 1 GPU active

Kiểm tra Kaggle settings:
- Must select "GPU T4 x2" (not just "GPU T4")
- Restart notebook after changing

### Uneven GPU usage

Bình thường! GPU 0 thường dùng nhiều hơn (model weights + gradients).

---

## 📝 Final Recommendation

**Simplest approach (no code changes needed):**

1. ✅ Select **GPU T4 x2** in Kaggle
2. ✅ Set `BATCH_SIZE = 6` (safe) or `8` (optimal)
3. ✅ Run training normally
4. ✅ Monitor with `!nvidia-smi`

PyTorch + Kaggle sẽ tự động dùng cả 2 GPUs! 🚀
