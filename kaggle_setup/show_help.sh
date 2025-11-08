#!/bin/bash
# Quick help cho Kaggle setup

cat << 'EOF'

╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║                   🚀 FIFO KAGGLE TRAINING GUIDE 🚀                   ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝

📦 Repository: https://github.com/Anhnguyen0812/FIFO/tree/phianh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 QUICK START - KAGGLE CELLS

1️⃣  Clone code
   !git clone -b phianh https://github.com/Anhnguyen0812/FIFO.git /kaggle/working/fifo
   %cd /kaggle/working/fifo

2️⃣  Verify setup
   !bash kaggle_setup/verify_setup.sh

3️⃣  Install dependencies
   !pip install wandb pytorch-metric-learning tqdm -q

4️⃣  Test với 5 ảnh (10 phút)
   !cp kaggle_setup/train_config_kaggle_test.py configs/train_config.py
   !python main.py --file-name "test" --modeltrain "fogpass" \
       --batch-size 1 --num-steps 50 --num-steps-stop 50 --gpu 0

5️⃣  Full training (16-24 giờ) - Nếu test OK
   !cp kaggle_setup/train_config_kaggle.py configs/train_config.py
   !bash kaggle_setup/setup_and_train_full.sh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 DOCUMENTATION

   ⭐ MAIN GUIDE: kaggle_setup/KAGGLE_NOTEBOOK_SETUP.md
   
   📖 Detailed:   kaggle_setup/HUONG_DAN_KAGGLE.md
   ⚡ Quick Ref:  kaggle_setup/QUICKSTART.md
   📝 README:     kaggle_setup/README.md

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ CHECKLIST

   Dataset:
   □ Upload dataset: cityscapes-filtered-fog
   □ Add dataset to notebook
   □ Check path: /kaggle/input/cityscapes-filtered-fog

   Notebook Settings:
   □ GPU: T4 hoặc T4 x2 (cho full training)
   □ Internet: ON
   □ Persistence: Files only

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 EXPECTED RESULTS

   Test (50 steps):
   - Time: 5-10 minutes
   - Checkpoints: 5 files
   - Size: ~500MB each

   Full Training (60K steps):
   - Stage 1: 4-6 hours (FogPassFilter)
   - Stage 2: 12-18 hours (Full model)
   - Checkpoints: ~16 files
   - Total size: ~8-10GB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🐛 COMMON ISSUES

   Dataset not found:
   !ls /kaggle/input/
   → Update KAGGLE_DATA_ROOT in config

   Module not found:
   !pip install wandb pytorch-metric-learning tqdm -q

   Out of memory:
   → Reduce batch-size: 4 → 2 → 1

   Checkpoint error:
   → Check file exists: !ls /kaggle/working/snapshots/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 TIPS

   • Chạy test trước khi full training
   • Commit notebook thường xuyên
   • Monitor GPU: !nvidia-smi
   • Check logs real-time trong cell output
   • Save checkpoints mỗi 5000 steps

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📞 NEED HELP?

   1. Read: kaggle_setup/KAGGLE_NOTEBOOK_SETUP.md
   2. Run: !bash kaggle_setup/verify_setup.sh
   3. Check: Cell output for detailed errors
   4. GitHub: https://github.com/Anhnguyen0812/FIFO/tree/phianh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Good luck! 🎉 Happy training! 🚀

EOF
