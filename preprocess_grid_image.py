"""
Preprocess images with grid lines for better detection
Usage: python preprocess_grid_image.py --input Test_1.jpg --output Test_1_clean.png
"""

import cv2
import numpy as np
import argparse

def remove_grid_lines(image):
    """Remove horizontal and vertical grid lines from image."""
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply threshold
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Combine lines
    grid_mask = cv2.add(horizontal_lines, vertical_lines)
    
    # Remove lines from original image
    result = image.copy()
    result[grid_mask == 255] = 255  # Set grid pixels to white
    
    return result

def enhance_digits(image):
    """Enhance digit contrast."""
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    
    # Denoise
    denoised = cv2.medianBlur(binary, 3)
    
    return denoised

def main():
    parser = argparse.ArgumentParser(description='Preprocess grid images')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--remove-grid', action='store_true', help='Remove grid lines')
    parser.add_argument('--enhance', action='store_true', help='Enhance digits')
    
    args = parser.parse_args()
    
    # Load image
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not load {args.input}")
        return
    
    print(f"Processing {args.input}...")
    result = image.copy()
    
    # Remove grid lines
    if args.remove_grid:
        print("  - Removing grid lines...")
        result = remove_grid_lines(result)
    
    # Enhance digits
    if args.enhance:
        print("  - Enhancing digits...")
        result = enhance_digits(result)
    
    # Save result
    cv2.imwrite(args.output, result)
    print(f"✅ Saved to {args.output}")
    
    # Show stats
    h, w = result.shape[:2] if len(result.shape) == 2 else result.shape[:2]
    print(f"   Image size: {w}×{h}")

if __name__ == '__main__':
    main()

