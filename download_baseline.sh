#!/bin/bash
# Script để download baseline model từ Google Drive

echo "Downloading Cityscapes pretrained baseline model..."
echo "URL: https://drive.google.com/file/d/1IKBXXVhYfc6n5Pw23g7HsH_QzqOG03c6/view?usp=sharing"
echo ""
echo "Cách download:"
echo "1. Mở link trên trong browser"
echo "2. Click 'Download' để tải file"
echo "3. Lưu file vào thư mục này với tên: Cityscapes_pretrained_model.pth"
echo ""
echo "Hoặc sử dụng gdown (nếu đã cài):"
echo "  pip install gdown"
echo "  gdown https://drive.google.com/uc?id=1IKBXXVhYfc6n5Pw23g7HsH_QzqOG03c6 -O Cityscapes_pretrained_model.pth"
echo ""

# Thử download bằng gdown nếu có
if command -v gdown &> /dev/null; then
    echo "Found gdown, attempting to download..."
    gdown https://drive.google.com/uc?id=1IKBXXVhYfc6n5Pw23g7HsH_QzqOG03c6 -O Cityscapes_pretrained_model.pth
    
    if [ -f "Cityscapes_pretrained_model.pth" ]; then
        echo "✓ Download successful!"
        echo "File saved: Cityscapes_pretrained_model.pth"
    else
        echo "✗ Download failed. Please download manually from the URL above."
    fi
else
    echo "gdown not found. Install it with: pip install gdown"
    echo "Or download manually from the URL above."
fi
