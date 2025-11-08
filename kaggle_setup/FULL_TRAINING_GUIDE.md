# 🚀 FULL TRAINING ON KAGGLE - COMPLETE GUIDE

## 📋 Prerequisites

- ✅ Dataset uploaded: `/kaggle/input/cityscapes-filtered-fog/`
- ✅ Kaggle GPU: T4 x2 (recommended)
- ✅ Code on GitHub branch: `phianh`

---

## 🎯 Quick Start (3 Steps)

### Step 1: Setup & Generate Lists

```python
# Cell 1: Clone and install
import os
os.chdir('/kaggle/working')
!git clone -b phianh https://github.com/Anhnguyen0812/FIFO.git fifo

!pip install "numpy<2.0" -q
!pip install wandb pytorch-metric-learning tqdm -q
!pip install git+https://github.com/drsleep/DenseTorch.git -q
```

```python
# Cell 2: Generate dataset lists
os.chdir('/kaggle/working/fifo')
!python kaggle_setup/generate_dataset_lists.py
!python kaggle_setup/generate_realfog_list.py
```

### Step 2: Copy Config Files

```python
# Cell 3: Copy and update config
!cp kaggle_setup/train_config_kaggle.py configs/train_config.py
!cp kaggle_setup/paired_cityscapes_kaggle.py dataset/paired_cityscapes.py
!cp kaggle_setup/foggy_zurich_kaggle.py dataset/Foggy_Zurich.py

# Update to use full dataset
import re
with open('configs/train_config.py', 'r') as f:
    config = f.read()

config = config.replace('test_5images_foggy.txt', 'train_foggy_0.005.txt')
config = config.replace('test_5images_origin.txt', 'train_origin.txt')
config = config.replace('test_5images_rf.txt', 'realfog_all_filenames.txt')

with open('configs/train_config.py', 'w') as f:
    f.write(config)

print("✅ Config updated!")
```

### Step 3: Start Training

```python
# Cell 4: Train!
os.chdir('/kaggle/working/fifo')
!python main.py --file-name "full_training" --modeltrain "fogpass"
```

---

## 📊 Training Details

### Stage 1: FogPassFilter (20K steps)
- **Purpose**: Train fog-aware feature modules
- **Duration**: ~5-6 hours (T4 x2)
- **Checkpoint**: Saved at step 20,000

### Stage 2: Full Model (60K steps)
- **Purpose**: Full semantic segmentation training
- **Duration**: ~15-18 hours (T4 x2)
- **Checkpoints**: Saved every 5,000 steps

### Total Training Time: ~24 hours

---

## 📁 Generated Files

After running `generate_dataset_lists.py`:

```
dataset/cityscapes_list/
├── train_foggy_0.005.txt    # ~2,976 foggy training images
├── train_origin.txt          # ~708 clear training images  
├── val_foggy_0.005.txt       # ~500 foggy val images
├── val.txt                   # ~500 clear val images
├── label_val.txt             # ~500 val labels
├── clear_lindau.txt          # ~59 Lindau clear images
└── label_lindau.txt          # ~59 Lindau labels
```

After running `generate_realfog_list.py`:

```
lists_file_names/
├── realfog_all_filenames.txt # ~837 real fog images
└── test_5images_rf.txt       # 5 images for testing
```

---

## ⚙️ Configuration

### Full Training Config (`train_config_kaggle.py`)

```python
BATCH_SIZE = 4
NUM_STEPS = 100000
NUM_STEPS_STOP = 60000
SAVE_PRED_EVERY = 5000
LEARNING_RATE = 2.5e-4
```

### Dataset Sizes

| Dataset | Type | Count |
|---------|------|-------|
| Train Foggy | Synthetic | ~2,976 |
| Train Clear | Original | ~708 |
| Real Fog | Real-world | ~837 |
| Val Foggy | Validation | ~500 |
| Val Clear | Validation | ~500 |

---

## 🔍 Monitoring Training

### Check progress

```python
# View training iterations
# Progress bar will show: X/100000 [time, speed]

# Check saved models
!ls -lth /kaggle/working/fifo/snapshots/ | head -10
```

### Expected outputs

```
snapshots/
├── FIFO_20000.pth           # Stage 1 checkpoint
├── FIFO_25000.pth           # Stage 2 checkpoints
├── FIFO_30000.pth
├── ...
└── FIFO_60000.pth           # Final model
```

---

## 🛠️ Troubleshooting

### Out of Memory (OOM)

Reduce batch size in config:
```python
BATCH_SIZE = 2  # Instead of 4
```

### Dataset not found

Verify dataset structure:
```python
!ls -la /kaggle/input/cityscapes-filtered-fog/
```

Should see:
- `foggy_filtered/`
- `leftImg8bit_filtered/`
- `gtFine_filtered/`
- `realfog_filtered_2gb/`

### Training slow

Check GPU:
```python
!nvidia-smi
```

Make sure using T4 x2 in Kaggle settings.

---

## 📝 Full Training Checklist

- [ ] Clone code from GitHub (branch: phianh)
- [ ] Install dependencies (NumPy < 2.0, DenseTorch, etc.)
- [ ] Generate dataset list files
- [ ] Generate real fog list files
- [ ] Copy config files
- [ ] Update config to use full dataset
- [ ] Verify all paths and file counts
- [ ] Start training with `python main.py --file-name "full_training" --modeltrain "fogpass"`
- [ ] Monitor progress and checkpoints
- [ ] Wait ~24 hours for completion
- [ ] Download final model (FIFO_60000.pth)

---

## 🎯 After Training

### Evaluate model

```python
!python evaluate.py \
    --restore-from snapshots/FIFO_60000.pth \
    --data-dir /kaggle/input/cityscapes-filtered-fog \
    --gpu 0
```

### Download model

```python
# In Kaggle, right-click file and download
# Or use Kaggle API
from IPython.display import FileLink
FileLink('/kaggle/working/fifo/snapshots/FIFO_60000.pth')
```

---

## 📚 Additional Scripts

All helper scripts in `kaggle_setup/`:

- `generate_dataset_lists.py` - Generate Cityscapes lists
- `generate_realfog_list.py` - Generate real fog lists
- `FULL_TRAINING_CELLS.py` - Copy-paste cells for notebook
- `train_config_kaggle.py` - Full training config
- `paired_cityscapes_kaggle.py` - Dataset loader (paired)
- `foggy_zurich_kaggle.py` - Dataset loader (real fog)

---

## ✅ Success Criteria

Training is successful when you see:

1. ✅ No file path errors
2. ✅ Progress bar advancing (X/100000)
3. ✅ Models saving every 5,000 steps
4. ✅ Training completes at step 60,000
5. ✅ Final model: `FIFO_60000.pth` created

---

**Good luck with training! 🚀**
