#!/bin/bash

# Evaluate checkpoint 1: FIFO 5K
echo "Evaluating FIFO 5K..."
python evaluate.py --file-name 'FIFO_5K' --restore-from /home/anhngp/Documents/1/fifo/fast_training-11-09-00-25_FIFO5000.pth

echo ""
echo "Evaluating FIFO 10K..."
python evaluate.py --file-name 'FIFO_10K' --restore-from /home/anhngp/Documents/1/fifo/fast_training10000.pth

echo ""
echo "======================================"
echo "Results:"
echo "======================================"
echo "FIFO 5K:"
cat ./result_FZ/FIFO_5K/miou.txt 2>/dev/null || echo "No results"
echo ""
echo "FIFO 10K:"
cat ./result_FZ/FIFO_10K/miou.txt 2>/dev/null || echo "No results"
