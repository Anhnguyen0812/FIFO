import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from packaging import version
import matplotlib.pyplot as plt

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

# Class names for Cityscapes
class_names = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 
    'motorcycle', 'bicycle'
]

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
    
    return img_tensor, original_size, Image.open(image_path).convert('RGB')

def run_inference_single_model(model, image_tensor, original_size, device, multi_scale=False, scales=[1.0, 0.8, 0.6]):
    """Run inference with a single model"""
    
    if multi_scale:
        outputs = []
        for scale in scales:
            h, w = image_tensor.shape[2], image_tensor.shape[3]
            if scale < 1.0:
                h_scaled = int(h * scale)
                w_scaled = int(w * scale)
                img_scaled = nn.functional.interpolate(image_tensor, size=(h_scaled, w_scaled), 
                                                       mode='bilinear', align_corners=True)
            else:
                img_scaled = image_tensor
            
            img_scaled = img_scaled.to(device)
            
            with torch.no_grad():
                _, _, _, _, _, output = model(Variable(img_scaled))
                
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
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            _, _, _, _, _, output = model(Variable(image_tensor))
            
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
    
    return output_mask

def create_comparison_visualization(original_img, fifo_mask, baseline_mask, save_path):
    """Create side-by-side comparison visualization"""
    
    # Colorize masks
    fifo_colored = colorize_mask(fifo_mask)
    baseline_colored = colorize_mask(baseline_mask)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # FIFO model result
    axes[0, 1].imshow(fifo_colored)
    axes[0, 1].set_title('FIFO Model', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Baseline model result
    axes[1, 0].imshow(baseline_colored)
    axes[1, 0].set_title('Baseline Model', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Difference map
    diff_mask = (fifo_mask != baseline_mask).astype(np.uint8) * 255
    axes[1, 1].imshow(diff_mask, cmap='hot')
    axes[1, 1].set_title('Difference Map (Red = Different Predictions)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison visualization saved to: {save_path}")

def compare_models(args):
    """Compare FIFO model and baseline ResNet-101"""
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Load FIFO model (fine-tuned for foggy scenes)
    print(f"\nLoading FIFO model from: {args.fifo_model_path}")
    fifo_model = rf_lw101(num_classes=NUM_CLASSES)
    checkpoint = torch.load(args.fifo_model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        fifo_model.load_state_dict(checkpoint['state_dict'])
    else:
        fifo_model.load_state_dict(checkpoint)
    fifo_model.eval()
    fifo_model.to(device)
    print("✓ FIFO model loaded successfully")
    
    # Load baseline model
    if args.baseline_model_path and os.path.exists(args.baseline_model_path):
        print(f"\nLoading baseline model from: {args.baseline_model_path}")
        baseline_model = rf_lw101(num_classes=NUM_CLASSES)
        baseline_checkpoint = torch.load(args.baseline_model_path, map_location='cpu')
        if 'state_dict' in baseline_checkpoint:
            baseline_model.load_state_dict(baseline_checkpoint['state_dict'])
        else:
            baseline_model.load_state_dict(baseline_checkpoint)
        print("✓ Baseline model loaded successfully")
    else:
        print(f"\nLoading baseline: RefineNet-LW101 pretrained on ImageNet only")
        print("⚠️  WARNING: This model only has ImageNet weights, NOT trained for segmentation!")
        print("⚠️  Results will be poor - for proper comparison, provide a baseline checkpoint with --baseline-model-path")
        baseline_model = rf_lw101(num_classes=NUM_CLASSES, imagenet=True, pretrained=False)
        print("✓ Baseline model initialized (ImageNet pretrained encoder only)")
    
    baseline_model.eval()
    baseline_model.to(device)

    
    # Load and prepare image
    print(f"\nLoading image: {args.input_image}")
    img_tensor, original_size, original_img = load_image(args.input_image, size=(args.height, args.width))
    
    # Run inference with FIFO model
    print("\nRunning inference with FIFO model...")
    fifo_mask = run_inference_single_model(fifo_model, img_tensor, original_size, device, 
                                           multi_scale=args.multi_scale)
    
    # Run inference with baseline model
    print("Running inference with baseline ResNet-101 model...")
    baseline_mask = run_inference_single_model(baseline_model, img_tensor, original_size, device, 
                                               multi_scale=args.multi_scale)
    
    # Calculate statistics
    total_pixels = fifo_mask.size
    diff_pixels = np.sum(fifo_mask != baseline_mask)
    agreement_percent = (1 - diff_pixels / total_pixels) * 100
    
    print(f"\n{'='*60}")
    print(f"COMPARISON STATISTICS")
    print(f"{'='*60}")
    print(f"Image size: {original_size[1]} x {original_size[0]} pixels")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Different predictions: {diff_pixels:,} pixels ({diff_pixels/total_pixels*100:.2f}%)")
    print(f"Agreement: {agreement_percent:.2f}%")
    print(f"{'='*60}\n")
    
    # Create output directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save individual results
    base_name = os.path.splitext(os.path.basename(args.input_image))[0]
    
    # Save FIFO results
    fifo_mask_img = Image.fromarray(fifo_mask)
    fifo_mask_colored = colorize_mask(fifo_mask)
    fifo_mask_img.save(os.path.join(output_dir, f'{base_name}_fifo_mask.png'))
    fifo_mask_colored.save(os.path.join(output_dir, f'{base_name}_fifo_colored.png'))
    
    # Save baseline results
    baseline_mask_img = Image.fromarray(baseline_mask)
    baseline_mask_colored = colorize_mask(baseline_mask)
    baseline_mask_img.save(os.path.join(output_dir, f'{base_name}_baseline_mask.png'))
    baseline_mask_colored.save(os.path.join(output_dir, f'{base_name}_baseline_colored.png'))
    
    # Create and save comparison visualization
    comparison_path = os.path.join(output_dir, f'{base_name}_comparison.png')
    create_comparison_visualization(original_img, fifo_mask, baseline_mask, comparison_path)
    
    # Save difference map
    diff_mask = (fifo_mask != baseline_mask).astype(np.uint8) * 255
    diff_img = Image.fromarray(diff_mask)
    diff_img.save(os.path.join(output_dir, f'{base_name}_difference.png'))
    
    print(f"Results saved to: {output_dir}/")
    print(f"  - {base_name}_fifo_colored.png")
    print(f"  - {base_name}_baseline_colored.png")
    print(f"  - {base_name}_comparison.png")
    print(f"  - {base_name}_difference.png")
    
    print("\n✓ Comparison completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Compare FIFO model vs Baseline model")
    parser.add_argument("--input-image", type=str, required=True,
                        help="Path to input image (PNG/JPG)")
    parser.add_argument("--fifo-model-path", type=str, default="FIFO_final_model.pth",
                        help="Path to FIFO model checkpoint")
    parser.add_argument("--baseline-model-path", type=str, default=None,
                        help="Path to baseline model checkpoint (optional, if not provided will use ImageNet-only weights)")
    parser.add_argument("--output-dir", type=str, default="comparison_results",
                        help="Directory to save comparison results")
    parser.add_argument("--height", type=int, default=1080,
                        help="Input height for model (default: 1080)")
    parser.add_argument("--width", type=int, default=1920,
                        help="Input width for model (default: 1920)")
    parser.add_argument("--multi-scale", action="store_true",
                        help="Use multi-scale inference (slower but more accurate)")
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU instead of GPU")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_image):
        print(f"Error: Input image not found: {args.input_image}")
        return
    
    if not os.path.exists(args.fifo_model_path):
        print(f"Error: FIFO model checkpoint not found: {args.fifo_model_path}")
        return
    
    compare_models(args)

if __name__ == "__main__":
    main()
