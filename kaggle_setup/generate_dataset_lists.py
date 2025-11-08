"""
Script để tạo list files từ dataset trên Kaggle
Chạy trên Kaggle: python generate_dataset_lists.py
"""
import os
import glob

# Kaggle dataset root
DATA_ROOT = '/kaggle/input/cityscapes-filtered-fog'

# Output directory
OUTPUT_DIR = '/kaggle/working/fifo/dataset/cityscapes_list'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("GENERATING DATASET LIST FILES")
print("=" * 80)

# ============================================================================
# 1. Train Foggy Images (beta 0.005)
# ============================================================================
print("\n1. Generating train_foggy_0.005.txt...")
foggy_train_path = os.path.join(DATA_ROOT, 'foggy_filtered/foggy_data/leftImg8bit_foggy/train')
foggy_train_files = []

for city in os.listdir(foggy_train_path):
    city_path = os.path.join(foggy_train_path, city)
    if os.path.isdir(city_path):
        files = glob.glob(os.path.join(city_path, '*_beta_0.005.png'))
        for f in sorted(files):
            filename = os.path.basename(f)
            foggy_train_files.append(f"{city}/{filename}")

with open(os.path.join(OUTPUT_DIR, 'train_foggy_0.005.txt'), 'w') as f:
    f.write('\n'.join(foggy_train_files))
print(f"   ✓ Created: {len(foggy_train_files)} files")

# ============================================================================
# 2. Val Foggy Images (beta 0.005)
# ============================================================================
print("\n2. Generating val_foggy_0.005.txt...")
foggy_val_path = os.path.join(DATA_ROOT, 'foggy_filtered/foggy_data/leftImg8bit_foggy/val')
foggy_val_files = []

for city in os.listdir(foggy_val_path):
    city_path = os.path.join(foggy_val_path, city)
    if os.path.isdir(city_path):
        files = glob.glob(os.path.join(city_path, '*_beta_0.005.png'))
        for f in sorted(files):
            filename = os.path.basename(f)
            foggy_val_files.append(f"{city}/{filename}")

with open(os.path.join(OUTPUT_DIR, 'val_foggy_0.005.txt'), 'w') as f:
    f.write('\n'.join(foggy_val_files))
print(f"   ✓ Created: {len(foggy_val_files)} files")

# ============================================================================
# 3. Train Clear Images
# ============================================================================
print("\n3. Generating train_origin.txt...")
clear_train_path = os.path.join(DATA_ROOT, 'leftImg8bit_filtered/leftImg8bit_data/leftImg8bit/train')
clear_train_files = []

for city in os.listdir(clear_train_path):
    city_path = os.path.join(clear_train_path, city)
    if os.path.isdir(city_path):
        files = glob.glob(os.path.join(city_path, '*_leftImg8bit.png'))
        for f in sorted(files):
            filename = os.path.basename(f)
            clear_train_files.append(f"{city}/{filename}")

with open(os.path.join(OUTPUT_DIR, 'train_origin.txt'), 'w') as f:
    f.write('\n'.join(clear_train_files))
print(f"   ✓ Created: {len(clear_train_files)} files")

# ============================================================================
# 4. Val Clear Images
# ============================================================================
print("\n4. Generating val.txt...")
clear_val_path = os.path.join(DATA_ROOT, 'leftImg8bit_filtered/leftImg8bit_data/leftImg8bit/val')
clear_val_files = []

for city in os.listdir(clear_val_path):
    city_path = os.path.join(clear_val_path, city)
    if os.path.isdir(city_path):
        files = glob.glob(os.path.join(city_path, '*_leftImg8bit.png'))
        for f in sorted(files):
            filename = os.path.basename(f)
            clear_val_files.append(f"{city}/{filename}")

with open(os.path.join(OUTPUT_DIR, 'val.txt'), 'w') as f:
    f.write('\n'.join(clear_val_files))
print(f"   ✓ Created: {len(clear_val_files)} files")

# ============================================================================
# 5. Val Labels
# ============================================================================
print("\n5. Generating label_val.txt...")
label_val_path = os.path.join(DATA_ROOT, 'gtFine_filtered/gtFine_data/gtFine/val')
label_val_files = []

for city in os.listdir(label_val_path):
    city_path = os.path.join(label_val_path, city)
    if os.path.isdir(city_path):
        files = glob.glob(os.path.join(city_path, '*_gtFine_labelIds.png'))
        for f in sorted(files):
            filename = os.path.basename(f)
            label_val_files.append(f"{city}/{filename}")

with open(os.path.join(OUTPUT_DIR, 'label_val.txt'), 'w') as f:
    f.write('\n'.join(label_val_files))
print(f"   ✓ Created: {len(label_val_files)} files")

# ============================================================================
# 6. Lindau Clear Images
# ============================================================================
print("\n6. Generating clear_lindau.txt...")
lindau_path = os.path.join(DATA_ROOT, 'leftImg8bit_filtered/leftImg8bit_data/leftImg8bit/val/lindau')
lindau_files = []

if os.path.exists(lindau_path):
    files = glob.glob(os.path.join(lindau_path, '*_leftImg8bit.png'))
    for f in sorted(files):
        filename = os.path.basename(f)
        lindau_files.append(f"lindau/{filename}")

with open(os.path.join(OUTPUT_DIR, 'clear_lindau.txt'), 'w') as f:
    f.write('\n'.join(lindau_files))
print(f"   ✓ Created: {len(lindau_files)} files")

# ============================================================================
# 7. Lindau Labels
# ============================================================================
print("\n7. Generating label_lindau.txt...")
lindau_label_path = os.path.join(DATA_ROOT, 'gtFine_filtered/gtFine_data/gtFine/val/lindau')
lindau_label_files = []

if os.path.exists(lindau_label_path):
    files = glob.glob(os.path.join(lindau_label_path, '*_gtFine_labelIds.png'))
    for f in sorted(files):
        filename = os.path.basename(f)
        lindau_label_files.append(f"lindau/{filename}")

with open(os.path.join(OUTPUT_DIR, 'label_lindau.txt'), 'w') as f:
    f.write('\n'.join(lindau_label_files))
print(f"   ✓ Created: {len(lindau_label_files)} files")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Train Foggy (beta 0.005): {len(foggy_train_files)} files")
print(f"Val Foggy (beta 0.005):   {len(foggy_val_files)} files")
print(f"Train Clear:              {len(clear_train_files)} files")
print(f"Val Clear:                {len(clear_val_files)} files")
print(f"Val Labels:               {len(label_val_files)} files")
print(f"Lindau Clear:             {len(lindau_files)} files")
print(f"Lindau Labels:            {len(lindau_label_files)} files")
print("\n✅ All list files generated successfully!")
print(f"📁 Output directory: {OUTPUT_DIR}")
print("=" * 80)

# ============================================================================
# Show samples
# ============================================================================
print("\n📝 Sample files from each list:")
print("\n--- train_foggy_0.005.txt (first 3) ---")
for line in foggy_train_files[:3]:
    print(f"  {line}")

print("\n--- train_origin.txt (first 3) ---")
for line in clear_train_files[:3]:
    print(f"  {line}")

print("\n--- val.txt (first 3) ---")
for line in clear_val_files[:3]:
    print(f"  {line}")
