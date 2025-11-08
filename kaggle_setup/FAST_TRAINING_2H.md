# ⚡ FAST TRAINING MODE (2 giờ)

## 🎯 Mục tiêu: Training trong 2 giờ

**Cách 1: Giảm số steps**
- Original: 60,000 steps (~11 giờ @ 1.47 it/s)
- Fast mode: **10,000 steps (~2 giờ @ 1.47 it/s)**

**Trade-off**: Model sẽ kém chính xác hơn, nhưng có thể test/demo

---

## 🚀 Option 1: Dùng Fast Config

### Trong Kaggle:

```python
# Cell: Copy fast config
!cp kaggle_setup/train_config_kaggle_fast.py configs/train_config.py
!cp kaggle_setup/paired_cityscapes_kaggle.py dataset/paired_cityscapes.py
!cp kaggle_setup/foggy_zurich_kaggle.py dataset/Foggy_Zurich.py

# Verify
!grep "NUM_STEPS" configs/train_config.py | head -2

# Train (should complete in ~2 hours)
%cd /kaggle/working/fifo
!python main.py --file-name "fast_training" --modeltrain "fogpass"
```

### Checkpoints:
- Step 2,000: ~17 phút
- Step 4,000: ~34 phút
- Step 6,000: ~51 phút
- Step 8,000: ~68 phút
- **Step 10,000: ~113 phút (< 2 giờ)** ✅

---

## 🚀 Option 2: Override trong command

Không cần đổi config, override trực tiếp:

```python
%cd /kaggle/working/fifo
!python main.py \
    --file-name "fast_training" \
    --modeltrain "fogpass" \
    --num-steps 12000 \
    --num-steps-stop 10000 \
    --save-pred-every 2000
```

---

## 🚀 Option 3: Tăng tốc training (Nếu có thể)

### A. Tăng batch size (nếu GPU memory đủ)

```python
# Test với batch_size=6 hoặc 8
!python main.py \
    --file-name "fast_training" \
    --modeltrain "fogpass" \
    --batch-size 6 \
    --num-steps-stop 10000
```

**Tốc độ**: Có thể tăng lên ~1.8-2.0 it/s → train nhanh hơn

### B. Giảm kích thước ảnh

Sửa trong config:
```python
INPUT_SIZE = '1024,512'  # Thay vì '2048,1024'
INPUT_SIZE_RF = '960,540'  # Thay vì '1920,1080'
```

**Tốc độ**: Có thể tăng lên ~2.5 it/s → train trong ~1.5 giờ

---

## 📊 So sánh các options:

| Option | Steps | Batch Size | Image Size | Time | Accuracy |
|--------|-------|------------|------------|------|----------|
| Original | 60K | 4 | 2048x1024 | 11h | 100% |
| **Fast** | 10K | 4 | 2048x1024 | **2h** | ~70% |
| Fast + Large Batch | 10K | 8 | 2048x1024 | 1.5h | ~70% |
| Fast + Small Image | 10K | 4 | 1024x512 | 1h | ~60% |

---

## ⚠️ Lưu ý:

1. **Model chưa converge**: 10K steps không đủ cho model tốt nhất
2. **Chỉ dùng cho**:
   - Testing pipeline
   - Demo
   - Kiểm tra code works
3. **Để có model tốt**: Cần ít nhất 40-50K steps

---

## 🎯 Khuyến nghị:

**Nếu chỉ có 2 giờ**:
1. Dùng **train_config_kaggle_fast.py** (10K steps)
2. Hoặc override: `--num-steps-stop 10000`
3. Accept accuracy thấp hơn để đổi lấy tốc độ

**Nếu muốn model tốt**:
1. Train 11 giờ với 60K steps (recommended)
2. Hoặc chia làm 2 sessions (mỗi session 6 giờ)

---

## 🚀 Quick Start (2h training):

```bash
# Clone
cd /kaggle/working
git clone -b phianh https://github.com/Anhnguyen0812/FIFO.git fifo

# Install
pip install "numpy<2.0" wandb pytorch-metric-learning tqdm -q
pip install git+https://github.com/drsleep/DenseTorch.git -q

# Generate lists
cd fifo
python kaggle_setup/generate_dataset_lists.py
python kaggle_setup/generate_realfog_list.py

# Use fast config
cp kaggle_setup/train_config_kaggle_fast.py configs/train_config.py
cp kaggle_setup/paired_cityscapes_kaggle.py dataset/paired_cityscapes.py
cp kaggle_setup/foggy_zurich_kaggle.py dataset/Foggy_Zurich.py

# Train (2 hours)
python main.py --file-name "fast_2h" --modeltrain "fogpass"
```

✅ **Hoàn tất trong ~2 giờ!**
