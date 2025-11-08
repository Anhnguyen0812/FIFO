# 🎯 TÓM TẮT: CHẠY FIFO TRÊN KAGGLE

## Repository
```
https://github.com/Anhnguyen0812/FIFO/tree/phianh
```

## 🚀 CELLS CHẠY TRONG KAGGLE (Copy & Paste)

### Cell 1: Clone Code
```bash
!git clone -b phianh https://github.com/Anhnguyen0812/FIFO.git /kaggle/working/fifo
%cd /kaggle/working/fifo
```

### Cell 2: Install
```bash
!pip install "numpy<2.0" -q
!pip install wandb pytorch-metric-learning tqdm -q
!pip install git+https://github.com/drsleep/DenseTorch.git -q
```

### Cell 3: Verify
```bash
!bash kaggle_setup/verify_setup.sh
```

### Cell 4: Config Test
```bash
!cp kaggle_setup/train_config_kaggle_test.py configs/train_config.py
```

### Cell 5: Wandb Offline
```python
import os
os.environ['WANDB_MODE'] = 'offline'
```

### Cell 6: TEST (10 phút)
```bash
!python main.py --file-name "test" --modeltrain "fogpass" --batch-size 1 --num-steps 50 --num-steps-stop 50 --gpu 0
```

---

## 📊 Nếu TEST OK → FULL TRAINING

### Cell 1-3: Giống test

### Cell 4: Config Full
```bash
!cp kaggle_setup/train_config_kaggle.py configs/train_config.py
```

### Cell 5: Wandb (giống test)

### Cell 6: Stage 1 (4-6h)
```bash
!python main.py --file-name "fogpass" --modeltrain "fogpass" --batch-size 4 --num-steps 20000 --num-steps-stop 20000 --save-pred-every 5000 --gpu 0
```

### Cell 7: Tìm checkpoint
```python
import glob
ckpt = sorted(glob.glob('/kaggle/working/snapshots/FIFO_model/*fogpass*.pth'))[-1]
print(ckpt)
```

### Cell 8: Stage 2 (12-18h)
```bash
!python main.py --file-name "full" --modeltrain "train" --batch-size 4 --num-steps 60000 --num-steps-stop 60000 --save-pred-every 5000 --restore-from-fogpass {checkpoint_từ_cell_7} --gpu 0
```

---

## ⚙️ SETTINGS KAGGLE

### Test:
- GPU: T4
- Internet: ON
- Time: 10 phút

### Full:
- GPU: **T4 x2**
- Internet: ON  
- Persistence: **Files only**
- Time: 16-24h

---

## 📁 DATASET

Tên: `cityscapes-filtered-fog`
Path: `/kaggle/input/cityscapes-filtered-fog`

Cấu trúc:
```
cityscapes-filtered-fog/
├── foggy_filtered/foggy_data/leftImg8bit_foggy/
├── gtFine_filtered/gtFine_data/gtFine/
├── leftImg8bit_filtered/leftImg8bit_data/leftImg8bit/
└── realfog_filtered_2gb/RGB/
```

---

## 📖 ĐỌC THÊM

- **Chi tiết**: `kaggle_setup/KAGGLE_NOTEBOOK_SETUP.md`
- **Quick**: `kaggle_setup/QUICKSTART.md`
- **Template**: `kaggle_setup/FIFO_Kaggle_Test_Template.ipynb`

---

## ✅ CHECKLIST

- [ ] Dataset uploaded: `cityscapes-filtered-fog`
- [ ] Dataset added to notebook
- [ ] GPU enabled (T4 cho test, T4 x2 cho full)
- [ ] Internet ON
- [ ] Clone code từ branch `phianh`

---

**Good luck! 🎉**
