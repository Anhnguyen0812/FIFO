import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from packaging import version

from model.refinenetlw import rf_lw101

# Constants
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
NUM_CLASSES = 19

# Color palette for visualization
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    """Convert segmentation mask to color image"""
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def load_image(image_path, size=None):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    
    # Store original size
    original_size = img.size[::-1]  # (height, width)
    
    # Resize if size is provided
    if size is not None:
        img = img.resize((size[1], size[0]), Image.BICUBIC)
    
    # Convert to numpy and subtract mean
    img_np = np.array(img, dtype=np.float32)
    img_np -= IMG_MEAN
    
    # Transpose to CHW format
    img_np = img_np.transpose((2, 0, 1))
    
    # Add batch dimension
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)
    
    return img_tensor, original_size

def run_inference(args):
    """Run inference on a single image"""
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = rf_lw101(num_classes=NUM_CLASSES)
    
    checkpoint = torch.load(args.model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # Load and prepare image
    print(f"Loading image: {args.input_image}")
    
    # Multi-scale inference if enabled
    if args.multi_scale:
        scales = [1.0, 0.8, 0.6]
        outputs = []
        
        for scale in scales:
            if scale < 1.0:
                h = int(args.height * scale)
                w = int(args.width * scale)
            else:
                h, w = args.height, args.width
            
            img_tensor, original_size = load_image(args.input_image, size=(h, w))
            img_tensor = img_tensor.to(device)
            
            with torch.no_grad():
                _, _, _, _, _, output = model(Variable(img_tensor))
                
                # Interpolate to original size
                if version.parse(torch.__version__) >= version.parse('0.4.0'):
                    interp = nn.Upsample(size=original_size, mode='bilinear', align_corners=True)
                else:
                    interp = nn.Upsample(size=original_size, mode='bilinear')
                
                output = interp(output)
                outputs.append(output)
        
        # Average multi-scale outputs
        output = torch.stack(outputs).mean(dim=0)
    else:
        # Single-scale inference
        img_tensor, original_size = load_image(args.input_image, size=(args.height, args.width))
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            _, _, _, _, _, output = model(Variable(img_tensor))
            
            # Interpolate to original size
            if version.parse(torch.__version__) >= version.parse('0.4.0'):
                interp = nn.Upsample(size=original_size, mode='bilinear', align_corners=True)
            else:
                interp = nn.Upsample(size=original_size, mode='bilinear')
            
            output = interp(output)
    
    # Process output
    output = output.cpu().numpy()
    output = output[0].transpose(1, 2, 0)
    output_mask = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
    
    # Save results
    output_dir = os.path.dirname(args.output_image) if os.path.dirname(args.output_image) else '.'
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save raw segmentation mask
    output_img = Image.fromarray(output_mask)
    output_img.save(args.output_image)
    print(f"Saved segmentation mask to: {args.output_image}")
    
    # Save colorized version
    if args.save_color:
        output_col = colorize_mask(output_mask)
        color_path = args.output_image.replace('.png', '_color.png')
        output_col.save(color_path)
        print(f"Saved colorized segmentation to: {color_path}")
    
    print("Inference completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Run inference on a single PNG image")
    parser.add_argument("--input-image", type=str, required=True,
                        help="Path to input PNG image")
    parser.add_argument("--output-image", type=str, default="output_segmentation.png",
                        help="Path to save output segmentation mask")
    parser.add_argument("--model-path", type=str, default="FIFO_final_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--height", type=int, default=1080,
                        help="Input height for model (default: 1080)")
    parser.add_argument("--width", type=int, default=1920,
                        help="Input width for model (default: 1920)")
    parser.add_argument("--multi-scale", action="store_true",
                        help="Use multi-scale inference (slower but more accurate)")
    parser.add_argument("--save-color", action="store_true", default=True,
                        help="Save colorized segmentation visualization")
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU instead of GPU")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_image):
        print(f"Error: Input image not found: {args.input_image}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found: {args.model_path}")
        return
    
    run_inference(args)

if __name__ == "__main__":
    main()
