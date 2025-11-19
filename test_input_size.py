#!/usr/bin/env python
"""Test script to verify input size is correctly applied to datasets"""

import sys
import torch
from dataset.paired_cityscapes import Pairedcityscapes
from dataset.Foggy_Zurich import foggyzurichDataSet
import numpy as np

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def test_dataset_sizes():
    print("=" * 60)
    print("Testing Dataset Input Sizes")
    print("=" * 60)
    
    # Test with different input sizes
    test_configs = [
        ((600, 600), "Default (600x600)"),
        ((1024, 512), "Small (1024x512)"),
        ((2048, 1024), "Full (2048x1024)"),
    ]
    
    for crop_size, desc in test_configs:
        print(f"\n{desc} - crop_size={crop_size}")
        print("-" * 60)
        
        try:
            # Test Pairedcityscapes
            dataset = Pairedcityscapes(
                src_root="./dummy",
                trg_root="./dummy",
                src_list_path="./lists_file_names/test_5images_rf.txt",
                trg_list_path="./lists_file_names/test_5images_rf.txt",
                max_iters=10,
                crop_size=crop_size,
                mean=IMG_MEAN,
                set='train'
            )
            print(f"  ✓ Pairedcityscapes: crop_size set to {dataset.crop_size}")
            
            # Test foggyzurichDataSet
            dataset_rf = foggyzurichDataSet(
                root="./dummy",
                list_path="./lists_file_names/test_5images_rf.txt",
                max_iters=10,
                crop_size=crop_size,
                mean=IMG_MEAN,
                set='train'
            )
            print(f"  ✓ foggyzurichDataSet: crop_size set to {dataset_rf.crop_size}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_dataset_sizes()
