"""
Kaggle Cell: Setup cho Multi-GPU Training (T4 x2)
Copy cell này vào Kaggle notebook
"""

# ==============================================================================
# CELL: Configure Multi-GPU Training
# ==============================================================================
import os
import torch

os.chdir('/kaggle/working/fifo')

print("="*80)
print("MULTI-GPU TRAINING SETUP")
print("="*80)

# 1. Check GPU availability
num_gpus = torch.cuda.device_count()
print(f"\n📊 GPU Information:")
print(f"   Available GPUs: {num_gpus}")

if num_gpus == 0:
    print("   ❌ No GPU detected! Check Kaggle settings.")
    exit(1)

for i in range(num_gpus):
    gpu_name = torch.cuda.get_device_name(i)
    memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"   GPU {i}: {gpu_name} ({memory:.1f} GB)")

# 2. Determine optimal batch size
if num_gpus >= 2:
    recommended_batch = 8  # 4 per GPU
    print(f"\n✅ Multi-GPU detected! Recommended batch size: {recommended_batch}")
else:
    recommended_batch = 4
    print(f"\n⚠️ Single GPU detected. Batch size: {recommended_batch}")

# 3. Update config file
print(f"\n📝 Updating training configuration...")

with open('configs/train_config.py', 'r') as f:
    config = f.read()

# Update batch size
import re
config = re.sub(r'BATCH_SIZE\s*=\s*\d+', f'BATCH_SIZE = {recommended_batch}', config)

with open('configs/train_config.py', 'w') as f:
    f.write(config)

print(f"   ✓ Batch size set to: {recommended_batch}")

# 4. Verify configuration
print(f"\n📋 Final Configuration:")
with open('configs/train_config.py', 'r') as f:
    for line in f:
        if 'BATCH_SIZE' in line and not line.strip().startswith('#'):
            print(f"   {line.strip()}")
            break

# 5. Performance estimates
if num_gpus >= 2:
    estimated_time = 8.5  # hours for 60K steps
    speedup = "~1.8x faster"
else:
    estimated_time = 14
    speedup = "baseline"

print(f"\n⏱️ Estimated Training Time:")
print(f"   Stage 1 (20K steps): ~{estimated_time * 20/60:.1f} hours")
print(f"   Stage 2 (60K steps): ~{estimated_time:.1f} hours")
print(f"   Total: ~{estimated_time * 1.33:.1f} hours")
print(f"   Speedup: {speedup}")

print("\n" + "="*80)
print("✅ MULTI-GPU SETUP COMPLETE!")
print("="*80)
print("\n🚀 Ready to start training with optimized settings!")
print("\nNext: Run training cell with:")
print("   !python main.py --file-name 'full_training' --modeltrain 'fogpass'")
print("\n💡 Monitor GPU usage during training with: !nvidia-smi")
