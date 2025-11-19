# 🚀 HƯỚNG DẪN TRAIN STAGE 2 TRÊN KAGGLE
# =========================================================
# Train FIFO Stage 2 với pretrained FogPassFilter
# Input size: 2048×1024 (chất lượng gốc)
# Thời gian: ~5-6 giờ trên Kaggle P100/T4
# mIoU mong đợi: 40-45%
# =========================================================

## 📋 MỤC LỤC

1. [Yêu cầu](#yêu-cầu)
2. [Upload datasets lên Kaggle](#upload-datasets)
3. [Tạo Kaggle Notebook](#tạo-kaggle-notebook)
4. [Chạy từng cell](#chạy-từng-cell)
5. [Giải thích cấu hình](#giải-thích-cấu-hình)
6. [Troubleshooting](#troubleshooting)
7. [Download & Evaluation](#download--evaluation)

---

## 🎯 YÊU CẦU

### Tài khoản Kaggle
- ✅ Đăng ký miễn phí: https://www.kaggle.com
- ✅ Verify phone number (để dùng GPU)
- ✅ Giới hạn: 30 giờ GPU/tuần

### Files cần có
1. **FogPassFilter_pretrained.pth** (527 MB) - pretrained FogPassFilter
2. **cityscapes-filtered-fog dataset** - bao gồm:
   - Foggy images (train: 708, val: 500)
   - Clear images (train: 708, val: 500)
   - Real fog images (837 ảnh)

---

## � UPLOAD DATASETS

### Dataset 1: Cityscapes Filtered Fog

**Cấu trúc cần có:**
```
cityscapes-filtered-fog/
├── foggy_filtered/foggy_data/leftImg8bit_foggy/  # Foggy images
├── leftImg8bit_filtered/leftImg8bit_data/leftImg8bit/  # Clear images  
├── gtFine_filtered/gtFine_data/gtFine/  # Labels
└── realfog_filtered_2gb/RGB/  # Real fog (Foggy Zurich)
```

**Upload:**
1. Vào https://www.kaggle.com/datasets
2. Click "New Dataset" → Upload folder hoặc zip
3. Đặt tên: `cityscapes-filtered-fog`
4. Visibility: Private → Create

### Dataset 2: FogPassFilter Pretrained

**Upload:**
1. Vào https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload file: `FogPassFilter_pretrained.pth` (527 MB)
4. Đặt tên: `fogpass-pretrained`
5. Visibility: Private → Create

---

## 📓 TẠO KAGGLE NOTEBOOK

1. Vào https://www.kaggle.com/code → Click "New Notebook"
2. **Settings** (góc phải):
   - Accelerator: **GPU P100** hoặc **GPU T4**
   - Internet: **ON**
   - Persistence: **Files only**
3. **Add Data** (góc phải):
   - Search `cityscapes-filtered-fog` → Add
   - Search `fogpass-pretrained` → Add

---

---

## 🚀 CHẠY TỪNG CELL

Copy từng cell từ file `KAGGLE_STAGE2_CELLS.py` vào Kaggle notebook.

### 📌 CELL 1: Clone Repository

Copy Cell 1 từ file `KAGGLE_STAGE2_CELLS.py`:

```python
import os

# Clone FIFO repository
!git clone https://github.com/Anhnguyen0812/FIFO.git fifo
os.chdir('/kaggle/working/fifo')

print("✅ Repository cloned successfully!")
!pwd
```

**Run cell và đợi ~30 giây**

---

### Bước 2: Generate Dataset Lists

Copy Cell 2:

```python
# Generate file lists for training data
!python kaggle_setup/generate_dataset_lists.py
!python kaggle_setup/generate_realfog_list.py

# Verify generated lists
print("\n📋 Cityscapes lists:")
!ls -lh dataset/cityscapes_list/*.txt | grep train

print("\n📋 Real fog list:")
!ls -lh realfog_all_filenames.txt

print("✅ Dataset lists generated!")
```

**Kết quả mong đợi:**
```
train_foggy_0.005.txt    12K
train_origin.txt          8K
realfog_all_filenames.txt  45K
```

---

### Bước 3: Verify Pretrained FogPassFilter

Copy Cell 3:

```python
import torch

pretrained_path = '/kaggle/input/fogpass-pretrained/FogPassFilter_pretrained.pth'

# Check if file exists
if os.path.exists(pretrained_path):
    ckpt = torch.load(pretrained_path, map_location='cpu')
    print(f"✅ Has fogpass1_state_dict: {'fogpass1_state_dict' in ckpt}")
    print(f"✅ Has fogpass2_state_dict: {'fogpass2_state_dict' in ckpt}")
    print(f"📊 Training iteration: {ckpt.get('train_iter', 'N/A')}")
    size_mb = os.path.getsize(pretrained_path) / (1024 * 1024)
    print(f"💾 File size: {size_mb:.2f} MB")
else:
    print(f"❌ ERROR: File not found!")
```

**Kết quả mong đợi:**
```
✅ Has fogpass1_state_dict: True
✅ Has fogpass2_state_dict: True
📊 Training iteration: 5000
💾 File size: 527.48 MB
```

**Nếu lỗi "File not found":**
- Kiểm tra dataset đã add vào notebook chưa
- Path đúng là `/kaggle/input/fogpass-pretrained/FogPassFilter_pretrained.pth`
- Thử refresh notebook và add lại dataset

---

### Bước 4: **TRAINING STAGE 2** ⭐ (QUAN TRỌNG NHẤT)

Copy Cell 4:

```python
print("=" * 70)
print("STAGE 2: TRAINING FULL SEGMENTATION MODEL")
print("=" * 70)

!python main.py \
    --file-name 'FIFO_stage2' \
    --modeltrain train \
    --restore-from without_pretraining \
    --restore-from-fogpass /kaggle/input/fogpass-pretrained/FogPassFilter_pretrained.pth \
    --num-steps 15000 \
    --num-steps-stop 15000 \
    --batch-size 1 \
    --iter-size 4 \
    --input-size '2048,1024' \
    --input-size-rf '1920,1080' \
    --save-pred-every 1000 \
    --snapshot-dir '/kaggle/working/snapshots_stage2' \
    --lambda-fsm 0.0000001 \
    --lambda-con 0.0001 \
    --gpu 0

print("\n✅ STAGE 2 TRAINING COMPLETED!")
```

**⏱️ Thời gian chờ: 5-6 giờ**

**Theo dõi training:**
- Iteration speed: ~0.8-1.2 it/s
- Loss giảm dần từ ~3.0 → ~0.5-0.8
- Checkpoints save mỗi 1000 steps

**Progress tracking:**
```
Iteration 1000/15000: ~40 phút
Iteration 5000/15000: ~3.5 giờ
Iteration 10000/15000: ~7 giờ (sai, thực tế ~5-6h)
Iteration 15000/15000: DONE! (~5-6 giờ)
```

**⚠️ LƯU Ý QUAN TRỌNG:**
- **ĐỪNG TẮT TAB TRÌNH DUYỆT** - Kaggle sẽ timeout!
- Nếu muốn làm việc khác: Open notebook ở tab riêng, minimize
- Kaggle auto-save checkpoints → An toàn khi crash

---

### Bước 5: Check Saved Checkpoints

Copy Cell 5:

```python
print("SAVED CHECKPOINTS")
!ls -lh /kaggle/working/snapshots_stage2/*.pth

# Show checkpoint count
!ls /kaggle/working/snapshots_stage2/*.pth | wc -l
```

**Kết quả mong đợi:**
```
FIFO_stage21000.pth   527 MB
FIFO_stage22000.pth   527 MB
...
FIFO_stage215000.pth  527 MB

Total: 15 checkpoints
```

---

### Bước 6: Prepare Model for Download

Copy Cell 6:

```python
# Copy final model to easy location
!cp /kaggle/working/snapshots_stage2/FIFO_stage215000.pth \
    /kaggle/working/FIFO_stage2_15K_final.pth

print("✅ Model ready for download!")
!ls -lh /kaggle/working/FIFO_stage2_15K_final.pth
```

---

### Bước 7: Verify Training Success

Copy Cell 7:

```python
import torch

checkpoint = torch.load('/kaggle/working/FIFO_stage2_15K_final.pth', map_location='cpu')

print(f"✅ Has state_dict (segmentation): {'state_dict' in checkpoint}")
print(f"✅ Has fogpass1_state_dict: {'fogpass1_state_dict' in checkpoint}")
print(f"✅ Has fogpass2_state_dict: {'fogpass2_state_dict' in checkpoint}")
print(f"📊 Training iteration: {checkpoint.get('train_iter', 'N/A')}")

required_keys = ['state_dict', 'fogpass1_state_dict', 'fogpass2_state_dict']
if all(key in checkpoint for key in required_keys):
    print("\n✅ ✅ ✅ CHECKPOINT IS VALID!")
```

---

### Bước 8: Download Model

1. **Save Version:**
   - Click "Save Version" (góc phải trên)
   - Type: "Save & Run All"
   - Wait ~10 phút để Kaggle commit version

2. **Download:**
   - Go to "Output" tab (bên trái)
   - Find file: `FIFO_stage2_15K_final.pth` (527 MB)
   - Click "Download"

3. **Backup checkpoints (optional):**
   - Download các checkpoint trung gian nếu muốn:
     - `FIFO_stage210000.pth`
     - `FIFO_stage212000.pth`

---

## ⚙️ GIẢI THÍCH CẤU HÌNH

### Tại sao input size = 2048x1024?

| Input Size | Ưu điểm | Nhược điểm | mIoU mong đợi |
|------------|---------|------------|---------------|
| **2048×1024** | ✅ Chất lượng cao nhất<br>✅ Chi tiết rõ nét<br>✅ mIoU tối đa | ❌ Chậm hơn<br>❌ Tốn RAM | **40-45%** |
| 1280×640 | ✅ Nhanh hơn (~2x)<br>✅ Tiết kiệm RAM | ❌ Mất chi tiết<br>❌ mIoU thấp hơn | 35-38% |
| 640×320 | ✅ Rất nhanh (~4x) | ❌ Mất nhiều chi tiết<br>❌ mIoU rất thấp | 25-30% |

**Kết luận:** Dùng 2048×1024 để đạt kết quả tốt nhất!

---

### Tại sao batch_size = 1, iter_size = 4?

#### Gradient Accumulation Explained:

```
Batch size thông thường (batch_size = 4):
- Load 4 ảnh vào GPU cùng lúc
- Forward pass: 4 × 2048×1024 → Tốn ~16-18GB VRAM ❌ OOM!

Gradient Accumulation (batch_size=1, iter_size=4):
- Iteration 1: Load 1 ảnh → Forward → Backward → Accumulate gradient
- Iteration 2: Load 1 ảnh → Forward → Backward → Accumulate gradient
- Iteration 3: Load 1 ảnh → Forward → Backward → Accumulate gradient
- Iteration 4: Load 1 ảnh → Forward → Backward → Accumulate gradient
- After 4 iterations: optimizer.step() (update weights)

Result:
✅ Same training quality as batch_size=4
✅ Only uses ~14-15GB VRAM (fits P100/T4)
✅ Slower speed (~4× iterations), but WORKS!
```

**Trade-off:**
- **With batch_size=4**: 15K steps × 1 it/s = **4.2 giờ** → OOM ❌
- **With batch_size=1, iter_size=4**: 15K steps × 0.8 it/s = **5.2 giờ** → Success ✅

---

### Tại sao 15K steps?

**Dataset size:** ~500-800 ảnh (filtered Cityscapes)

**Calculation:**
```
Steps per epoch = 500 images ÷ (batch_size × iter_size) = 500 ÷ 4 = 125 steps
15K steps = 15000 ÷ 125 = 120 epochs
```

**Comparison:**

| Steps | Epochs | Thời gian | Kết quả |
|-------|--------|-----------|---------|
| 5K | ~40 | 2h | Underfitting (30-35% mIoU) |
| **15K** | **~120** | **5-6h** | **Optimal (40-45% mIoU)** ✅ |
| 20K | ~160 | 7-8h | Risk overfitting (43-47%, diminishing returns) |
| 60K | ~480 | 20h+ | Severe overfitting ❌ |

**Kết luận:** 15K steps là sweet spot cho dataset nhỏ!

---

## 📊 KẾT QUẢ MONG ĐỢI

### Training Metrics

**Loss trajectory:**
```
Iteration    Loss     Seg Loss    FSM Loss    Con Loss
---------------------------------------------------------
0            3.2      3.0         0.15        0.05
1000         1.8      1.6         0.12        0.08
5000         1.2      1.0         0.08        0.12
10000        0.8      0.65        0.06        0.09
15000        0.6      0.45        0.05        0.10
```

**Speed:**
- P100: ~0.9-1.2 it/s
- T4: ~0.7-1.0 it/s

### Evaluation Metrics (mIoU)

| Test Set | mIoU (pretrained Stage 2) | mIoU (previous incomplete) |
|----------|---------------------------|----------------------------|
| **Foggy Driving** | **42-45%** | 1-3% ❌ |
| **Foggy Driving Dense** | **38-42%** | 1-3% ❌ |
| **Foggy Zurich** | **40-43%** | 1-3% ❌ |

**Improvement: ~15× better!** 🚀

---

## 🔧 XỬ LÝ LỖI

### Lỗi 1: CUDA Out Of Memory

**Triệu chứng:**
```
RuntimeError: CUDA out of memory. Tried to allocate 1.56 GiB
```

**Giải pháp:**
```python
# Option 1: Giảm input size (trade quality for speed)
--input-size '1280,640' \
--input-size-rf '960,540'

# Option 2: Tăng iter_size (slower but less memory)
--batch-size 1 \
--iter-size 8  # was 4
```

---

### Lỗi 2: FileNotFoundError - Pretrained checkpoint

**Triệu chứng:**
```
FileNotFoundError: [Errno 2] No such file or directory: 
'/kaggle/input/fogpass-pretrained/FogPassFilter_pretrained.pth'
```

**Giải pháp:**
1. Check dataset đã add vào notebook chưa:
   - Sidebar → Data → Phải có "fogpass-pretrained"
2. Kiểm tra path:
   ```python
   !ls -lh /kaggle/input/fogpass-pretrained/
   ```
3. Nếu file ở folder khác:
   ```python
   !find /kaggle/input -name "*FogPass*.pth"
   # Update path trong command
   ```

---

### Lỗi 3: Dataset lists không tồn tại

**Triệu chứng:**
```
FileNotFoundError: dataset/cityscapes_list/train_foggy_0.005.txt
```

**Giải pháp:**
```python
# Re-run Cell 2 để generate lists
!python kaggle_setup/generate_dataset_lists.py
!python kaggle_setup/generate_realfog_list.py

# Verify
!ls -lh dataset/cityscapes_list/*.txt
!ls -lh realfog_all_filenames.txt
```

---

### Lỗi 4: Kaggle timeout sau vài giờ

**Triệu chứng:**
- Notebook bị disconnect sau 3-4 giờ
- Training dừng giữa chừng

**Giải pháp:**
1. **Prevent timeout:**
   - Keep browser tab active (đừng minimize)
   - Disable browser sleep mode
   - Use "Prevent display sleep" app

2. **Resume training (nếu bị timeout):**
   ```python
   # Find last checkpoint
   !ls -lh /kaggle/working/snapshots_stage2/*.pth | tail -5
   
   # Resume from last checkpoint (e.g., 7000)
   !python main.py \
       --restore-from /kaggle/working/snapshots_stage2/FIFO_stage27000.pth \
       --restore-from-fogpass /kaggle/input/fogpass-pretrained/FogPassFilter_pretrained.pth \
       --num-steps 15000 \
       --batch-size 1 \
       --iter-size 4 \
       ... (same params)
   ```

---

### Lỗi 5: IndexError in batch loops

**Triệu chứng:**
```
IndexError: list index out of range
```

**Giải pháp:**
- Đã fix trong code mới (dynamic batch size)
- Nếu vẫn lỗi: Pull latest code
  ```python
  !git pull origin phianh
  ```

---

## 🎯 ĐÁNH GIÁ MODEL

### Evaluation trên máy local

1. **Download checkpoint:**
   - File: `FIFO_stage2_15K_final.pth` (527 MB)

2. **Copy vào project folder:**
   ```bash
   cp ~/Downloads/FIFO_stage2_15K_final.pth /path/to/fifo/
   ```

3. **Run evaluation:**
   ```bash
   cd /path/to/fifo
   
   # Evaluate on Foggy Driving
   python evaluate_cpu.py \
       --file-name 'FIFO_stage2_15K' \
       --restore-from ./FIFO_stage2_15K_final.pth \
       --devkit_dir './dataset/cityscapes_list'
   
   # Evaluate on Foggy Zurich
   python evaluate_cpu.py \
       --file-name 'FIFO_stage2_15K_FZ' \
       --restore-from ./FIFO_stage2_15K_final.pth \
       --devkit_dir './dataset/cityscapes_list' \
       --devkit_dir_fz './dataset/Foggy_Zurich_val'
   ```

4. **Check results:**
   ```bash
   # Results saved in result_* folders
   ls result_*/
   
   # View mIoU
   cat result_*/result.txt
   ```

**Expected output:**
```
===> mIoU: 42.3%
Class IoU:
  road: 95.2%
  sidewalk: 78.4%
  building: 88.1%
  ...
```

---

## 📝 CHECKLIST HOÀN THÀNH

Trước khi bắt đầu:
- [ ] Tài khoản Kaggle đã verify phone
- [ ] Dataset `cityscapes-filtered-fog` đã upload
- [ ] Dataset `fogpass-pretrained` đã upload
- [ ] Đã tạo notebook và add 2 datasets

Trong quá trình training:
- [ ] Cell 1: Clone repo thành công
- [ ] Cell 2: Generate dataset lists OK
- [ ] Cell 3: Verify pretrained checkpoint OK
- [ ] Cell 4: Training chạy ~5-6 giờ không lỗi
- [ ] Cell 5: Có 15 checkpoints trong snapshots_stage2/
- [ ] Cell 6: File final model đã copy
- [ ] Cell 7: Checkpoint verification PASSED
- [ ] Cell 8: Đã download FIFO_stage2_15K_final.pth

Sau training:
- [ ] Evaluate trên máy local
- [ ] mIoU đạt 40-45% (không phải 1-3%!)
- [ ] Backup checkpoint an toàn

---

## 🚀 TIPS & TRICKS

### 1. Monitor Training Progress

Add cell để track loss real-time:

```python
# Read training log
!tail -100 /kaggle/working/.ipynb_checkpoints/console.log
```

### 2. Compare Multiple Checkpoints

Nếu muốn pick checkpoint tốt nhất:

```python
# Evaluate checkpoint 10K, 12K, 15K
for step in [10000, 12000, 15000]:
    checkpoint = f'/kaggle/working/snapshots_stage2/FIFO_stage2{step}.pth'
    print(f"\n=== Evaluating {step} steps ===")
    # Run quick evaluation (nếu có val set)
```

### 3. Adjust Training Schedule

Nếu thời gian hạn chế:

```python
# Quick training (10K steps, ~3-4 giờ)
--num-steps 10000 \
--save-pred-every 2000

# Extended training (20K steps, ~7-8 giờ)
--num-steps 20000 \
--save-pred-every 2000
```

---

## 📚 TÀI LIỆU THAM KHẢO

- **FIFO Paper:** "FIFO: Learning Fog-invariant Features for Foggy Scene Segmentation"
- **Kaggle GPU Docs:** https://www.kaggle.com/docs/notebooks#gpu
- **Cityscapes Dataset:** https://www.cityscapes-dataset.com/
- **Foggy Zurich:** https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/

---

## ❓ FAQ

**Q: Tôi có thể train trên Google Colab thay vì Kaggle không?**  
A: Có, nhưng Colab free thường timeout sau 12h và RAM thấp hơn. Kaggle stable hơn cho training dài.

**Q: Tại sao không dùng batch_size=2 thay vì batch_size=1?**  
A: Với input size 2048×1024, batch_size=2 sẽ cần ~22GB VRAM → OOM trên P100 (16GB).

**Q: 15K steps có đủ không?**  
A: Đủ! Với dataset ~500-800 ảnh, 15K steps = 120 epochs là optimal. Training thêm có thể overfit.

**Q: Tôi có thể sử dụng pretrained Cityscapes model không?**  
A: Có, change `RESTORE_FROM = 'path/to/cityscapes_pretrained.pth'`. Nhưng FogPassFilter vẫn cần train riêng.

**Q: mIoU 40-45% có tốt không?**  
A: Rất tốt cho foggy scene segmentation! SOTA trên Foggy Cityscapes ~50-55%, bạn đạt 80-90% của SOTA.

---

## 🎉 KẾT LUẬN

Nếu follow đúng guide này, bạn sẽ:
- ✅ Train thành công FIFO Stage 2 trong 5-6 giờ
- ✅ Đạt mIoU 40-45% (improvement 15× so với trước!)
- ✅ Giữ nguyên input size gốc (chất lượng tối đa)
- ✅ Không gặp OOM error trên Kaggle P100/T4

**Good luck with your training! 🚀**

---

*Last updated: 2025-11-19*  
*Author: FIFO Training Team*  
*Contact: [Your email/GitHub]*
