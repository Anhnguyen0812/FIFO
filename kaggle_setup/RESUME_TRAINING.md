# Resume Training từ Step 10000 → 12000

Training đã dừng ở step 10000. Đây là hướng dẫn resume để hoàn thành 12000 steps.

## 🎯 Tình huống

- ✅ Training hoàn thành: Step 10000 (Stage 1 - FogPassFilter)
- ✅ Checkpoint saved: `CS_scenes_10000.pth`
- ⏳ Cần tiếp tục: Step 10000 → 12000 (Stage 2 - Full model)
- ⏰ Thời gian cần: ~22 phút

## 🚀 Cách 1: Chạy lệnh trực tiếp (ĐƠN GIẢN NHẤT)

Trong Kaggle notebook, tạo cell mới và chạy:

```python
!python main.py \
    --restore-from ./snapshots/CS_scenes_10000.pth \
    --num-steps 12000 \
    --num-steps-stop 12000 \
    --mode train
```

**Giải thích:**
- `--restore-from`: Load checkpoint từ step 10000
- `--num-steps`: Loop chạy đến 12000 (không phải 10000)
- `--num-steps-stop`: Dừng ở 12000
- `--mode train`: Stage 2 (full model), không phải `fogpass`

## 🔧 Cách 2: Sử dụng config file

### Bước 1: Upload file config

Upload `train_config_kaggle_resume.py` vào Kaggle

### Bước 2: Tạo cell mới trong notebook

```python
import sys
sys.path.insert(0, '/kaggle/working/fifo')

# Import arguments
from configs.train_config import get_arguments
args = get_arguments()

# Override with resume settings
args.restore_from = './snapshots/CS_scenes_10000.pth'
args.restore_from_fogpass = './snapshots/CS_scenes_10000.pth'
args.num_steps = 12000
args.num_steps_stop = 12000
args.modeltrain = 'train'  # Stage 2: full model

# Run training
import main
```

### Bước 3: Chạy cell

Training sẽ tự động resume từ step 10000

## ✅ Xác nhận training đang chạy đúng

Bạn sẽ thấy:

```
Loading checkpoint from ./snapshots/CS_scenes_10000.pth
83%|█████████████████████████████▉      | 10000/12000 [00:00<22:00,  1.5it/s]
```

**Các dấu hiệu đúng:**
- ✅ Progress bar bắt đầu từ 10000 (không phải 0)
- ✅ Total steps: 12000
- ✅ ETA: ~22 phút
- ✅ Mode: `train` (không phải `fogpass`)

## 🔴 Các lỗi thường gặp

### Lỗi 1: FileNotFoundError: CS_scenes_10000.pth

**Nguyên nhân:** Checkpoint không tồn tại hoặc đường dẫn sai

**Giải pháp:**
```python
# Check checkpoint exists
!ls -lh ./snapshots/CS_scenes_10000.pth

# If not found, check all checkpoints
!ls -lh ./snapshots/
```

### Lỗi 2: Training starts from step 0

**Nguyên nhân:** `--restore-from` không được set

**Giải pháp:** Đảm bảo argument `--restore-from` được truyền vào

### Lỗi 3: Training stops at 10000 again

**Nguyên nhân:** `NUM_STEPS = 10000` chưa được đổi thành 12000

**Giải pháp:** Set `--num-steps 12000` (không phải 10000)

## 📊 Kết quả mong đợi

Sau khi hoàn thành:

```
 100%|████████████████████████████████████| 12000/12000 [22:00<00:00,  1.5it/s]
save model ..
Checkpoint saved: ./snapshots/CS_scenes_12000.pth
```

**Checkpoints bạn sẽ có:**
- ✅ `CS_scenes_5000.pth` - Mid-training Stage 1
- ✅ `CS_scenes_10000.pth` - Stage 1 complete (FogPassFilter)
- ✅ `CS_scenes_12000.pth` - **FINAL MODEL** (Stage 1 + Stage 2)

## 📥 Download về local

Sau khi training xong:

```python
from IPython.display import FileLink

# Download final model
FileLink('./snapshots/CS_scenes_12000.pth')

# Or download all checkpoints
!zip -r snapshots.zip ./snapshots/
FileLink('snapshots.zip')
```

## 🧪 Evaluate trên local

Sau khi download về local:

```bash
cd /home/anhngp/Documents/1/fifo

# Copy model vào thư mục chính
cp /path/to/downloaded/CS_scenes_12000.pth ./FIFO_12K_model.pth

# Run evaluation
bash kaggle_setup/eval_all_cpu.sh
```

## 🎯 Dự đoán mIoU

- **Step 10000** (Stage 1 only): ~32-35%
- **Step 12000** (Stage 1 + Stage 2): ~37-40%
- **Improvement**: +5-7% từ Stage 2

## 💡 Tips

1. **Monitor GPU usage:** Đảm bảo GPU đang được sử dụng
2. **Check mode:** Phải là `train`, không phải `fogpass`
3. **Save logs:** Enable wandb để track training
4. **Backup checkpoint:** Download CS_scenes_10000.pth phòng khi cần train lại

## ⏭️ (Optional) Train thêm đến 20K steps

Nếu muốn train thêm để đạt mIoU cao hơn (~40-43%):

```python
!python main.py \
    --restore-from ./snapshots/CS_scenes_12000.pth \
    --num-steps 20000 \
    --num-steps-stop 20000 \
    --mode train
```

Thời gian cần: thêm ~1.5 giờ (12K → 20K = 8000 steps)
