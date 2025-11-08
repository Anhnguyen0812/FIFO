# HƯỚNG DẪN SETUP VÀ CHẠY FIFO TRÊN KAGGLE

## Mục Lục
1. [Chuẩn bị Dataset trên Kaggle](#1-chuẩn-bị-dataset-trên-kaggle)
2. [Setup Code trên Kaggle](#2-setup-code-trên-kaggle)
3. [Test với 5 ảnh](#3-test-với-5-ảnh)
4. [Train đầy đủ với GPU T4 x2](#4-train-đầy-đủ-với-gpu-t4-x2)
5. [Giám sát và tải kết quả](#5-giám-sát-và-tải-kết-quả)

---

## 1. Chuẩn bị Dataset trên Kaggle

### Bước 1.1: Upload Dataset
1. Truy cập Kaggle.com và đăng nhập
2. Vào **Datasets** → **New Dataset**
3. Upload thư mục `cityscapes-filtered-fog` với cấu trúc:
   ```
   cityscapes-filtered-fog/
   ├── foggy_filtered/foggy_data/leftImg8bit_foggy/
   ├── gtFine_filtered/gtFine_data/gtFine/
   ├── leftImg8bit_filtered/leftImg8bit_data/leftImg8bit/
   └── realfog_filtered_2gb/RGB/
   ```

4. Đặt tên dataset: `cityscapes-filtered-fog` (hoặc tên khác, nhớ update lại trong config)
5. Đợi Kaggle xử lý và public dataset

### Bước 1.2: Kiểm tra đường dẫn
Sau khi upload, dataset sẽ có đường dẫn:
```
/kaggle/input/cityscapes-filtered-fog/
```

**LƯU Ý**: Nếu tên dataset của bạn khác, cần update trong file:
- `kaggle_setup/train_config_kaggle.py`
- `kaggle_setup/train_config_kaggle_test.py`

Thay đổi dòng:
```python
KAGGLE_DATA_ROOT = '/kaggle/input/cityscapes-filtered-fog'
```
thành tên dataset của bạn.

---

## 2. Setup Code trên Kaggle

### Bước 2.1: Tạo Kaggle Notebook mới
1. Vào **Code** → **New Notebook**
2. Chọn **GPU** (cho test) hoặc **GPU T4 x2** (cho full training)
3. Đặt tên notebook: `FIFO-Training`

### Bước 2.2: Upload code lên Kaggle
Có 2 cách:

**Cách 1: Upload từ GitHub (Khuyến nghị)**
```bash
# Cell 1: Clone repository
!git clone https://github.com/your-username/fifo.git /kaggle/working/fifo
%cd /kaggle/working/fifo
```

**Cách 2: Upload trực tiếp**
1. Zip toàn bộ thư mục `fifo`
2. Upload vào Kaggle Notebook
3. Giải nén:
```bash
!unzip fifo.zip -d /kaggle/working/
%cd /kaggle/working/fifo
```

### Bước 2.3: Add dataset vào notebook
1. Trong Kaggle Notebook, click **Add Data** ở panel bên phải
2. Tìm và add dataset `cityscapes-filtered-fog` (hoặc tên bạn đã đặt)
3. Dataset sẽ tự động mount vào `/kaggle/input/`

---

## 3. Test với 5 ảnh

### Bước 3.1: Cấu hình Test
File config đã được tạo sẵn tại: `kaggle_setup/train_config_kaggle_test.py`

Thông số test:
- Batch size: 1
- Num steps: 50
- Save every: 10 iterations
- Dataset: chỉ 5 ảnh (định nghĩa trong `test_5images_*.txt`)

### Bước 3.2: Chạy Test
Trong Kaggle Notebook, tạo cell mới và chạy:

```bash
# Cell 2: Setup và chạy test
!bash /kaggle/working/fifo/kaggle_setup/setup_and_train_test.sh
```

Hoặc chạy từng bước:

```bash
# Cell 2: Install dependencies
!pip install wandb pytorch-metric-learning tqdm -q

# Cell 3: Setup config cho test
!cp /kaggle/working/fifo/kaggle_setup/train_config_kaggle_test.py /kaggle/working/fifo/configs/train_config.py

# Cell 4: Setup Wandb (optional - offline mode)
import os
os.environ['WANDB_MODE'] = 'offline'

# Cell 5: Kiểm tra dataset
!ls -la /kaggle/input/cityscapes-filtered-fog/

# Cell 6: Chạy test training
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

### Bước 3.3: Kiểm tra kết quả test
```bash
# Cell 7: Xem kết quả
!ls -la /kaggle/working/snapshots/FIFO_test/
```

Nếu test thành công, bạn sẽ thấy:
- Checkpoint files (.pth)
- Training logs
- Không có lỗi

---

## 4. Train đầy đủ với GPU T4 x2

### Bước 4.1: Tạo Notebook mới cho Full Training
1. **Tạo notebook mới** hoặc **Save Version** của notebook test
2. **Quan trọng**: Chọn **GPU T4 x2** trong Settings
3. Set **Internet: ON** (nếu cần wandb online)
4. Set **Persistence: Files only** để giữ checkpoints

### Bước 4.2: Cấu hình Full Training
File config: `kaggle_setup/train_config_kaggle.py`

Thông số full:
- Batch size: 4
- Num steps: 100,000 (có thể giảm nếu cần)
- Early stopping: 60,000
- Save every: 5,000 iterations

### Bước 4.3: Chạy Full Training

**Option 1: Sử dụng script tự động (Khuyến nghị)**
```bash
# Cell: Full training với script
!bash /kaggle/working/fifo/kaggle_setup/setup_and_train_full.sh
```

**Option 2: Chạy từng stage thủ công**

```bash
# Cell 1: Install dependencies
!pip install wandb pytorch-metric-learning tqdm -q

# Cell 2: Setup config
!cp /kaggle/working/fifo/kaggle_setup/train_config_kaggle.py /kaggle/working/fifo/configs/train_config.py

# Cell 3: Setup Wandb
# Cách 1: Offline mode
import os
os.environ['WANDB_MODE'] = 'offline'

# Cách 2: Login wandb (nếu cần online tracking)
# !wandb login YOUR_API_KEY

# Cell 4: Tạo thư mục snapshots
!mkdir -p /kaggle/working/snapshots/FIFO_model

# Cell 5: Stage 1 - Train FogPassFilter (20,000 steps)
%cd /kaggle/working/fifo
!python main.py \
    --file-name "fifo_fogpass_stage1" \
    --modeltrain "fogpass" \
    --batch-size 4 \
    --num-steps 20000 \
    --num-steps-stop 20000 \
    --save-pred-every 5000 \
    --gpu 0

# Cell 6: Tìm checkpoint FogPassFilter mới nhất
import glob
checkpoints = sorted(glob.glob('/kaggle/working/snapshots/FIFO_model/*fogpassfilter*.pth'))
latest_checkpoint = checkpoints[-1] if checkpoints else None
print(f"Latest FogPass checkpoint: {latest_checkpoint}")

# Cell 7: Stage 2 - Train Full Model (60,000 steps)
!python main.py \
    --file-name "fifo_full_stage2" \
    --modeltrain "train" \
    --batch-size 4 \
    --num-steps 60000 \
    --num-steps-stop 60000 \
    --save-pred-every 5000 \
    --restore-from-fogpass "{latest_checkpoint}" \
    --gpu 0
```

### Bước 4.4: Training với Multi-GPU (T4 x2)
Kaggle T4 x2 cung cấp 2 GPU nhưng code hiện tại chỉ dùng 1 GPU.

Để sử dụng 2 GPU, cần wrap model với DataParallel. Thêm cell:

```python
# Cell: Modify main.py để dùng multi-GPU
# Tìm dòng: model.cuda(args.gpu)
# Thay bằng:
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs")
#     model = nn.DataParallel(model)
# model.cuda()
```

**LƯU Ý**: Với batch_size=4 và 1 GPU T4 (16GB) đã đủ. Nếu muốn tăng batch_size lên 8, mới cần 2 GPU.

---

## 5. Giám sát và tải kết quả

### Bước 5.1: Giám sát quá trình training

**Xem logs trong Kaggle:**
- Output hiển thị trực tiếp trong cell
- Progress bar từ tqdm
- Loss values được log

**Sử dụng Wandb (nếu đã setup):**
```python
# Trong cell khác để xem logs
import wandb
wandb.init(project='FIFO-Kaggle', resume='allow')
```

**Kiểm tra checkpoints:**
```bash
!ls -lh /kaggle/working/snapshots/FIFO_model/
```

### Bước 5.2: Tải checkpoints về local

**Cách 1: Commit Notebook**
1. Click **Save Version** → **Save & Run All**
2. Sau khi chạy xong, vào **Output** tab
3. Download các file .pth

**Cách 2: Copy sang Kaggle Dataset**
```bash
# Tạo dataset mới từ output
!mkdir -p /kaggle/working/fifo_checkpoints
!cp /kaggle/working/snapshots/FIFO_model/*.pth /kaggle/working/fifo_checkpoints/
```
Sau đó commit notebook, output sẽ thành dataset mới có thể tải về.

**Cách 3: Upload lên Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
!cp /kaggle/working/snapshots/FIFO_model/*.pth /content/drive/MyDrive/
```

### Bước 5.3: Tiếp tục training (Resume)
Nếu notebook bị timeout hoặc muốn tiếp tục:

```bash
# Tìm checkpoint cuối cùng
import glob
checkpoints = sorted(glob.glob('/kaggle/working/snapshots/FIFO_model/*FIFO*.pth'))
latest_checkpoint = checkpoints[-1]
print(f"Resume from: {latest_checkpoint}")

# Resume training
!python main.py \
    --file-name "fifo_resume" \
    --modeltrain "train" \
    --batch-size 4 \
    --num-steps 100000 \
    --num-steps-stop 100000 \
    --save-pred-every 5000 \
    --restore-from "{latest_checkpoint}" \
    --restore-from-fogpass "{latest_checkpoint}" \
    --gpu 0
```

---

## Troubleshooting

### Lỗi: "Dataset not found"
**Giải pháp:**
1. Kiểm tra tên dataset trong Kaggle
2. Update `KAGGLE_DATA_ROOT` trong config files
3. Đảm bảo đã Add Data vào notebook

### Lỗi: "Out of memory"
**Giải pháp:**
1. Giảm batch_size từ 4 xuống 2 hoặc 1
2. Giảm crop_size trong dataset (hiện tại 600)
3. Sử dụng GPU T4 x2

### Lỗi: "Module not found"
**Giải pháp:**
```bash
!pip install wandb pytorch-metric-learning tqdm -q
```

### Training quá chậm
**Giải pháp:**
1. Kiểm tra đang dùng GPU: `!nvidia-smi`
2. Giảm num_workers trong dataloader
3. Giảm số iterations (num_steps)

### Wandb không hoạt động
**Giải pháp:**
```python
import os
os.environ['WANDB_MODE'] = 'offline'
```

---

## Checklist trước khi chạy

### Test với 5 ảnh:
- [ ] Dataset đã upload lên Kaggle
- [ ] Code đã upload/clone vào /kaggle/working/fifo
- [ ] Dataset đã được Add vào notebook
- [ ] GPU đã được enable
- [ ] Đã chạy script test thành công

### Full Training:
- [ ] Test với 5 ảnh đã OK
- [ ] Đã chọn GPU T4 x2 (hoặc T4)
- [ ] Đã set Persistence: Files only
- [ ] Đã tạo thư mục snapshots
- [ ] Đã setup Wandb (nếu cần)
- [ ] Đã kiểm tra đường dẫn dataset
- [ ] Sẵn sàng commit để lưu checkpoints

---

## Thời gian ước tính

**Test với 5 ảnh (50 steps):**
- ~5-10 phút

**Stage 1 - FogPassFilter (20,000 steps):**
- ~4-6 giờ với T4
- Batch size 4: ~1.5-2s/iteration

**Stage 2 - Full Model (60,000 steps):**
- ~12-18 giờ với T4
- Batch size 4: ~2-2.5s/iteration

**Tổng full training:**
- ~16-24 giờ

**Kaggle limit:** 
- GPU T4 x2: 30 giờ/tuần
- Nên chia thành nhiều sessions và commit thường xuyên

---

## Liên hệ & Support

Nếu gặp vấn đề:
1. Kiểm tra Output logs
2. Xem file README.md trong repo
3. Kiểm tra các file trong kaggle_setup/

Good luck với training! 🚀
