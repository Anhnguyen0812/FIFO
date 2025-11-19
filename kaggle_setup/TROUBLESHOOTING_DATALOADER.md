# TROUBLESHOOTING: FileNotFoundError trong DataLoader
# ===================================================

## 🔴 LỖI

```
FileNotFoundError: Caught FileNotFoundError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/kaggle/working/fifo/dataset/paired_cityscapes.py", line 116, in __getitem__
    src_image = Image.open(datafiles["src_img"]).convert('RGB')
```

## 🔍 NGUYÊN NHÂN

DataLoader không tìm thấy file ảnh vì:
1. Path trong dataset list file không khớp với cấu trúc thư mục thực tế
2. `DATA_DIRECTORY` hoặc `DATA_DIRECTORY_CWSF` không đúng

## ✅ GIẢI PHÁP

### Bước 1: Chạy Cell 2.5 để debug (đã thêm vào KAGGLE_STAGE2_CELLS.py)

Cell này sẽ:
- Kiểm tra cấu trúc thư mục Kaggle input
- Xem sample paths từ list files
- Test xem file ảnh thực sự nằm ở đâu

### Bước 2: Dựa vào output của Cell 2.5, update command

#### Trường hợp 1: Nếu ảnh ở `/kaggle/input/cityscapes-filtered-fog/leftImg8bit_filtered/leftImg8bit_foggy_trainvaltest/`

```python
!python main.py \
--file-name 'FIFO_stage2' \
--modeltrain train \
--restore-from without_pretraining \
--restore-from-fogpass /kaggle/input/fogpass-pretrained/FogPassFilter_pretrained.pth \
--num-steps 15000 \
--batch-size 1 \
--iter-size 4 \
--input-size '2048,1024' \
--input-size-rf '1920,1080' \
--data-dir '/kaggle/input/cityscapes-filtered-fog/leftImg8bit_filtered/leftImg8bit_foggy_trainvaltest' \
--data-dir-rf '/kaggle/input/cityscapes-filtered-fog' \
--data-list './dataset/cityscapes_list/train_foggy_0.005.txt' \
--data-list-rf './lists_file_names/realfog_all_filenames.txt' \
--data-list-cwsf './dataset/cityscapes_list/train_origin.txt' \
--data-dir-cwsf '/kaggle/input/cityscapes-filtered-fog/leftImg8bit_filtered/leftImg8bit_data' \
--save-pred-every 1000 \
--snapshot-dir '/kaggle/working/snapshots_stage2' \
--gpu 0
```

#### Trường hợp 2: Nếu list file chứa path tuyệt đối (không cần prefix)

Sửa `generate_dataset_lists.py` để tạo absolute paths:

```python
# Instead of:
relative/path/to/image.png

# Generate:
/kaggle/input/cityscapes-filtered-fog/leftImg8bit_filtered/leftImg8bit_foggy_trainvaltest/relative/path/to/image.png
```

### Bước 3: Common Fixes

#### Fix 1: Update DATA_DIRECTORY trong command

Thay:
```
--data-dir '/kaggle/input/cityscapes-filtered-fog'
```

Bằng path chính xác tìm được từ Cell 2.5.

#### Fix 2: Regenerate list files với absolute paths

Modify `kaggle_setup/generate_dataset_lists.py`:

```python
# Add base_path when writing to file
base_path = '/kaggle/input/cityscapes-filtered-fog/leftImg8bit_filtered/leftImg8bit_foggy_trainvaltest'
with open(output_file, 'w') as f:
    for img in images:
        f.write(f'{base_path}/{img}\n')  # Absolute path
```

## 📋 DEBUG CHECKLIST

Chạy từng lệnh này trong Kaggle notebook để tìm path đúng:

```python
# 1. Check base structure
!ls -lh /kaggle/input/cityscapes-filtered-fog/

# 2. Find where leftImg8bit folders are
!find /kaggle/input/cityscapes-filtered-fog -name "leftImg8bit*" -type d

# 3. Check first line of list file
!head -1 dataset/cityscapes_list/train_foggy_0.005.txt

# 4. Try to construct full path
import os
list_entry = "bochum/bochum_000000_000313_leftImg8bit_foggy_beta_0.005.png"  # From step 3

# Try these combinations
paths_to_try = [
    f"/kaggle/input/cityscapes-filtered-fog/{list_entry}",
    f"/kaggle/input/cityscapes-filtered-fog/leftImg8bit_filtered/{list_entry}",
    f"/kaggle/input/cityscapes-filtered-fog/leftImg8bit_filtered/leftImg8bit_foggy_trainvaltest/{list_entry}",
]

for p in paths_to_try:
    print(f"{'✅' if os.path.exists(p) else '❌'} {p}")

# 5. Once you find the correct base path, use it in --data-dir
```

## 🎯 EXPECTED OUTPUT FROM CELL 2.5

Nếu paths đúng, bạn sẽ thấy:

```
✅ /kaggle/input/cityscapes-filtered-fog/leftImg8bit_filtered/leftImg8bit_foggy_trainvaltest/bochum/bochum_000000_000313_leftImg8bit_foggy_beta_0.005.png
```

Copy path này (bỏ phần sau `leftImg8bit_foggy_trainvaltest/`) và dùng làm `--data-dir`.

## 🚀 QUICK FIX

Nếu không muốn debug, regenerate lists với absolute paths:

```python
# Cell 2 - Modified version
!python -c "
import os
base = '/kaggle/input/cityscapes-filtered-fog/leftImg8bit_filtered/leftImg8bit_foggy_trainvaltest'

# Find structure
for root, dirs, files in os.walk(base):
    for file in files:
        if 'foggy_beta_0.005' in file:
            full_path = os.path.join(root, file)
            print(full_path)
            break
    break
"
```

Sau đó update training command với base path tìm được.

---

*Last updated: 2025-11-19*  
*For more help: Check STAGE2_TRAINING_GUIDE.md*
