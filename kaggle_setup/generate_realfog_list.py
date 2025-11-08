"""
Script để tạo list files cho Real Fog images
Chạy trên Kaggle: python generate_realfog_list.py
"""
import os
import glob

# Kaggle dataset root
DATA_ROOT = '/kaggle/input/cityscapes-filtered-fog'
REALFOG_PATH = os.path.join(DATA_ROOT, 'realfog_filtered_2gb/RGB')

# Output directory
OUTPUT_DIR = '/kaggle/working/fifo/lists_file_names'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("GENERATING REAL FOG LIST FILES")
print("=" * 80)

# ============================================================================
# Get all RGB images
# ============================================================================
print(f"\nScanning: {REALFOG_PATH}")
rgb_files = glob.glob(os.path.join(REALFOG_PATH, '*.png'))
rgb_files = sorted(rgb_files)

print(f"Found {len(rgb_files)} real fog images")

# ============================================================================
# Create list with just filenames (no RGB/ prefix)
# ============================================================================
rf_filenames = []
for f in rgb_files:
    filename = os.path.basename(f)
    rf_filenames.append(filename)

# Save full list
full_list_path = os.path.join(OUTPUT_DIR, 'realfog_all_filenames.txt')
with open(full_list_path, 'w') as f:
    f.write('\n'.join(rf_filenames))
print(f"\n✓ Created: {full_list_path}")
print(f"  Total: {len(rf_filenames)} files")

# ============================================================================
# Create test subset (first 5 images)
# ============================================================================
test_5_files = rf_filenames[:5]
test_5_path = os.path.join(OUTPUT_DIR, 'test_5images_rf.txt')
with open(test_5_path, 'w') as f:
    f.write('\n'.join(test_5_files))
print(f"\n✓ Created: {test_5_path}")
print(f"  Total: {len(test_5_files)} files")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total Real Fog images: {len(rf_filenames)}")
print(f"Test subset (5 images): {len(test_5_files)}")
print(f"\n📁 Output directory: {OUTPUT_DIR}")
print("=" * 80)

# ============================================================================
# Show samples
# ============================================================================
print("\n📝 Sample files (first 10):")
for i, filename in enumerate(rf_filenames[:10], 1):
    print(f"  {i:2d}. {filename}")

print("\n✅ Real fog list files generated successfully!")
