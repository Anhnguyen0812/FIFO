# Kaggle Setup Files - FIFO Training

📦 Thư mục này chứa tất cả files cần thiết để chạy FIFO trên Kaggle

## 📁 Cấu trúc Files

```
kaggle_setup/
├── README.md                           # File này
├── KAGGLE_NOTEBOOK_SETUP.md           # ⭐ HƯỚNG DẪN CHÍNH - ĐỌC ĐẦU TIÊN
├── HUONG_DAN_KAGGLE.md                # Hướng dẫn chi tiết tiếng Việt
├── QUICKSTART.md                      # Quick reference
│
├── train_config_kaggle.py             # Config cho full training
├── train_config_kaggle_test.py        # Config cho test 5 ảnh
│
├── paired_cityscapes_kaggle.py        # Dataset class cho paired images
├── foggy_zurich_kaggle.py             # Dataset class cho real fog
├── main_kaggle.py                     # Main training script (backup)
│
├── setup_and_train_test.sh            # Script auto test
├── setup_and_train_full.sh            # Script auto full training
├── verify_setup.sh                    # Script kiểm tra setup
│
└── requirements.txt                   # Dependencies
```

---

## 🚀 BẮT ĐẦU NHANH

### Bước 1: Đọc hướng dẫn
👉 **ĐỌC FILE NÀY TRƯỚC**: `KAGGLE_NOTEBOOK_SETUP.md`

### Bước 2: Upload dataset lên Kaggle
- Tên dataset: `cityscapes-filtered-fog`
- Theo cấu trúc trong `dataset_structure(1).txt`

### Bước 3: Tạo Kaggle Notebook
- Chọn GPU (T4 hoặc T4 x2)
- Add dataset vào notebook

### Bước 4: Clone code trong Kaggle
```bash
!git clone -b phianh https://github.com/Anhnguyen0812/FIFO.git /kaggle/working/fifo
```

### Bước 5: Chạy test hoặc full training
Xem chi tiết trong `KAGGLE_NOTEBOOK_SETUP.md`

---

## 📚 Tài liệu

| File | Mô tả | Khi nào dùng |
|------|-------|--------------|
| **KAGGLE_NOTEBOOK_SETUP.md** | Hướng dẫn từng cell Kaggle | ⭐ ĐỌC ĐẦU TIÊN |
| HUONG_DAN_KAGGLE.md | Chi tiết về config, troubleshooting | Khi cần hiểu sâu hơn |
| QUICKSTART.md | Reference nhanh | Khi đã quen |

---

## 🔧 Config Files

### train_config_kaggle_test.py
**Dùng cho**: Test với 5 ảnh
- Batch size: 1
- Steps: 50
- Thời gian: ~5-10 phút

### train_config_kaggle.py
**Dùng cho**: Full training
- Batch size: 4
- Steps: 100,000 (stop at 60,000)
- Thời gian: ~16-24 giờ

---

## 📦 Dataset Classes

### paired_cityscapes_kaggle.py
- Load foggy + clear weather paired images
- Với labels (gtFine)
- Dùng cho supervised training

### foggy_zurich_kaggle.py
- Load real fog images
- Không có labels
- Dùng cho domain adaptation

---

## 🛠️ Scripts

### verify_setup.sh
```bash
!bash kaggle_setup/verify_setup.sh
```
Kiểm tra:
- GPU available
- Dataset structure
- Code structure
- Dependencies

### setup_and_train_test.sh
```bash
!bash kaggle_setup/setup_and_train_test.sh
```
Tự động:
1. Install dependencies
2. Copy config
3. Chạy test 50 steps

### setup_and_train_full.sh
```bash
!bash kaggle_setup/setup_and_train_full.sh
```
Tự động:
1. Stage 1: Train FogPassFilter
2. Stage 2: Train full model

---

## 💡 Workflow Khuyến Nghị

```
1. Upload dataset lên Kaggle
   ↓
2. Tạo notebook TEST với GPU T4
   ↓
3. Clone code từ GitHub
   ↓
4. Chạy verify_setup.sh
   ↓
5. Chạy test với 5 ảnh (50 steps)
   ↓
6. Nếu OK → Tạo notebook mới
   ↓
7. Chọn GPU T4 x2
   ↓
8. Chạy full training
   ↓
9. Commit để lưu checkpoints
```

---

## 🎯 Requirements

### Phần cứng
- **Test**: GPU T4 (13-16GB VRAM)
- **Full**: GPU T4 x2 (khuyến nghị)

### Thời gian Kaggle
- Test: ~10 phút
- Full: ~16-24 giờ
- Limit: 30 giờ/tuần (T4 x2)

### Dataset
- Size: ~2-3GB
- Upload time: ~30-60 phút (tùy mạng)

---

## ⚠️ Lưu ý quan trọng

1. **Dataset name**: Phải là `cityscapes-filtered-fog` hoặc update trong config
2. **Branch**: Clone từ branch `phianh`
3. **GPU**: Enable GPU trong Kaggle settings
4. **Persistence**: Chọn "Files only" để giữ checkpoints
5. **Commit**: Commit notebook thường xuyên để backup

---

## 🐛 Common Issues

### Dataset not found
```python
# Check path
!ls /kaggle/input/

# Update nếu cần
KAGGLE_DATA_ROOT = '/kaggle/input/YOUR-DATASET-NAME'
```

### Module not found
```bash
!pip install wandb pytorch-metric-learning tqdm -q
```

### Out of memory
- Giảm batch_size: 4 → 2 → 1
- Giảm num_workers: 4 → 2

### Import error từ kaggle_setup
Files trong `kaggle_setup/` được import trong main code, đảm bảo:
- Code đã clone đầy đủ
- Đang ở đúng directory: `/kaggle/working/fifo`

---

## 📊 Expected Output

### Sau Test (50 steps)
```
/kaggle/working/snapshots/FIFO_test/
└── test_5images-{date}_fogpassfilter_10.pth
└── test_5images-{date}_fogpassfilter_20.pth
└── ...
```

### Sau Full Training (60K steps)
```
/kaggle/working/snapshots/FIFO_model/
├── fifo_fogpass_stage1-{date}_fogpassfilter_5000.pth
├── fifo_full_stage2-{date}_FIFO5000.pth
├── fifo_full_stage2-{date}_FIFO10000.pth
├── ...
└── fifo_full_stage2-{date}_FIFO60000.pth
```

---

## 🔗 Links

- **GitHub Repo**: https://github.com/Anhnguyen0812/FIFO/tree/phianh
- **Kaggle**: https://www.kaggle.com/
- **Wandb** (optional): https://wandb.ai/

---

## 📞 Support

Nếu gặp vấn đề:
1. Đọc `KAGGLE_NOTEBOOK_SETUP.md` - Section Troubleshooting
2. Check output logs trong Kaggle cell
3. Run `verify_setup.sh`
4. Check GitHub issues

---

**Happy Training! 🎉**
