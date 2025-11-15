"""
End-to-End Pipeline: Detection → Classification → Visualization
Usage: python pipeline.py --image test_scene.png --model unified_model_19classes_best.pth
"""

import os
import argparse
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image

from detect_objects import TraditionalDetector, HybridDetector

# ============================================================================
# Unified Pipeline
# ============================================================================

class UnifiedPipeline:
    """
    Complete pipeline: Detection → Classification → Visualization
    """
    
    def __init__(self, model_path, label_mapping_path, device='cuda',
                 detector_type='traditional', craft_path=None, target_classes='all'):
        """
        Args:
            model_path: Path to classifier model (.pth)
            label_mapping_path: Path to label mapping JSON
            device: 'cuda' or 'cpu'
            detector_type: 'traditional', 'craft', or 'hybrid'
            craft_path: Path to CRAFT weights (if using CRAFT)
            target_classes: 'digits', 'shapes', or 'all'
        """
        self.device = device
        self.target_classes = target_classes
        
        # Load label mapping
        with open(label_mapping_path, 'r') as f:
            self.label_mapping = json.load(f)
            # Convert string keys to int
            self.label_mapping = {int(k): v for k, v in self.label_mapping.items()}
        
        print(f"✅ Loaded label mapping: {len(self.label_mapping)} classes")
        
        # Define digit and shape class IDs
        self.digit_classes = set(range(10))  # 0-9
        self.shape_classes = set(range(10, 19))  # 10-18 (Circle to Triangle)
        
        # Validate and print target classes
        if self.target_classes == 'digits':
            print(f"🎯 Target: DIGITS only (0-9)")
        elif self.target_classes == 'shapes':
            print(f"🎯 Target: SHAPES only (Circle, Square, etc.)")
        elif self.target_classes == 'all':
            print(f"🎯 Target: ALL classes (digits + shapes)")
        else:
            raise ValueError(f"Invalid target_classes: {self.target_classes}. Use 'digits', 'shapes', or 'all'.")
        
        # Load classifier
        self.classifier = self._load_classifier(model_path)
        print(f"✅ Loaded classifier from {model_path}")
        
        # Setup detector
        if detector_type == 'traditional':
            self.detector = TraditionalDetector(
                min_area=100,            # Giảm xuống để detect chữ số nhỏ
                max_area=50000,          # Cho phép cả objects lớn
                aspect_ratio_range=(0.3, 3.0)  # Loose hơn cho digits
            )
            print(f"✅ Using Traditional CV detector")
        elif detector_type == 'hybrid':
            self.detector = HybridDetector(craft_path)
            print(f"✅ Using Hybrid detector")
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
        
        # Transform for classification
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),  # Increased from 64 to match training
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_classifier(self, model_path):
        """Load pretrained classifier."""
        # Create model
        model = efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, len(self.label_mapping))
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _should_keep_class(self, class_id):
        """
        Check if a predicted class should be kept based on target_classes filter.
        
        Args:
            class_id: int, predicted class ID (0-18)
            
        Returns:
            bool: True if should keep, False if should filter out
        """
        if self.target_classes == 'all':
            return True
        elif self.target_classes == 'digits':
            return class_id in self.digit_classes
        elif self.target_classes == 'shapes':
            return class_id in self.shape_classes
        else:
            return True  # Default to keeping
    
    def process(self, image, visualize=True):
        """
        Process image: detect → classify → visualize.
        
        Args:
            image: numpy array (H, W) or (H, W, 3)
            visualize: whether to return annotated image
            
        Returns:
            results: dict with keys:
                - 'bboxes': list of (x, y, w, h)
                - 'labels': list of class labels
                - 'confidences': list of confidence scores
                - 'annotated_image': image with boxes (if visualize=True)
        """
        # Step 1: Detection
        bboxes = self.detector.detect(image)
        
        if len(bboxes) == 0:
            return {
                'bboxes': [],
                'labels': [],
                'confidences': [],
                'annotated_image': image if visualize else None
            }
        
        # Step 2: Classification
        filtered_bboxes = []
        labels = []
        confidences = []
        
        self.classifier.eval()
        with torch.no_grad():
            for (x, y, w, h) in bboxes:
                # Crop region
                if len(image.shape) == 3:
                    crop = image[y:y+h, x:x+w]
                else:
                    crop = image[y:y+h, x:x+w]
                
                # Skip if empty
                if crop.size == 0:
                    continue
                
                # Transform and classify
                try:
                    crop_tensor = self.transform(crop).unsqueeze(0).to(self.device)
                    output = self.classifier(crop_tensor)
                    probs = torch.softmax(output, dim=1)
                    
                    conf, pred = probs.max(1)
                    pred_class_id = pred.item()
                    
                    # Filter based on target_classes
                    if self._should_keep_class(pred_class_id):
                        filtered_bboxes.append((x, y, w, h))
                        labels.append(self.label_mapping[pred_class_id])
                        confidences.append(conf.item())
                    
                except Exception as e:
                    print(f"Warning: Failed to classify crop at ({x},{y},{w},{h}): {e}")
                    continue
        
        # Update bboxes to only include filtered ones
        bboxes = filtered_bboxes
        
        # Step 3: Visualize
        annotated_image = None
        if visualize:
            annotated_image = self._visualize(image, bboxes, labels, confidences)
        
        return {
            'bboxes': bboxes,
            'labels': labels,
            'confidences': confidences,
            'annotated_image': annotated_image
        }
    
    def _visualize(self, image, bboxes, labels, confidences):
        """Draw bounding boxes and labels on image."""
        result = image.copy()
        
        # Convert to BGR if grayscale
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        for i, (x, y, w, h) in enumerate(bboxes):
            if i >= len(labels):
                break
            
            # Color coding: green for digits, blue for shapes
            label = labels[i]
            if label.isdigit():
                color = (0, 255, 0)  # Green for digits
            else:
                color = (255, 0, 0)  # Blue for shapes
            
            # Draw rectangle
            cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
            
            # Prepare label text
            conf = confidences[i] if i < len(confidences) else 0.0
            label_text = f"{label} ({conf:.2f})"
            
            # Background for text
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(result, (x, y-text_h-10),
                        (x+text_w, y), color, -1)
            
            # Draw text
            cv2.putText(result, label_text, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result
    
    def process_file(self, image_path, output_path=None):
        """
        Process image file and save results.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output (optional)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Processing {image_path}...")
        
        # Process
        results = self.process(image, visualize=True)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"{'='*60}")
        print(f"Detected {len(results['labels'])} objects:")
        for i, (bbox, label, conf) in enumerate(zip(
            results['bboxes'], results['labels'], results['confidences']
        )):
            x, y, w, h = bbox
            print(f"  {i+1}. {label} (confidence: {conf:.3f}) at ({x}, {y}, {w}, {h})")
        print(f"{'='*60}\n")
        
        # Save output
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{base_name}_result.png"
        
        cv2.imwrite(output_path, results['annotated_image'])
        print(f"✅ Saved result to {output_path}")
        
        # Also save JSON
        json_path = output_path.replace('.png', '.json')
        json_results = {
            'image': image_path,
            'detections': [
                {
                    'bbox': {'x': x, 'y': y, 'w': w, 'h': h},
                    'label': label,
                    'confidence': float(conf)
                }
                for (x, y, w, h), label, conf in zip(
                    results['bboxes'], results['labels'], results['confidences']
                )
            ]
        }
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"✅ Saved JSON to {json_path}")
        
        return results

# ============================================================================
# Synthetic Scene Generator
# ============================================================================

def generate_synthetic_scene(mnist_dir, shapes_dir, mnist_csv,
                            num_objects=5, canvas_size=(800, 600), seed=None):
    """
    Generate synthetic test scene with random digits and shapes.
    
    Args:
        mnist_dir: Directory with MNIST images
        shapes_dir: Directory with shape images
        mnist_csv: Path to MNIST labels CSV
        num_objects: Number of objects to place
        canvas_size: (width, height) of canvas
        seed: Random seed
        
    Returns:
        canvas: numpy array (H, W)
        ground_truth: list of (x, y, w, h, label)
    """
    import pandas as pd
    
    if seed is not None:
        np.random.seed(seed)
    
    # Load data
    mnist_df = pd.read_csv(mnist_csv)
    shape_files = [f for f in os.listdir(shapes_dir) if f.endswith('.png')]
    
    # Create white canvas
    canvas = np.ones((canvas_size[1], canvas_size[0]), dtype=np.uint8) * 255
    ground_truth = []
    
    for _ in range(num_objects):
        # Randomly choose digit or shape
        is_digit = np.random.rand() < 0.5
        
        if is_digit:
            # MNIST digit
            sample = mnist_df.sample(1).iloc[0]
            img_path = os.path.join(mnist_dir, sample['image_name'])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            label = str(sample['label'])
            target_size = np.random.randint(60, 100)
        else:
            # Shape
            sample_file = np.random.choice(shape_files)
            img_path = os.path.join(shapes_dir, sample_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            label = sample_file.split('_')[0]
            target_size = np.random.randint(70, 120)
        
        # Resize
        img_resized = cv2.resize(img, (target_size, target_size))
        
        # Random position
        margin = 50
        x = np.random.randint(margin, canvas_size[0] - target_size - margin)
        y = np.random.randint(margin, canvas_size[1] - target_size - margin)
        
        # Paste (overlay dark objects on white)
        try:
            canvas[y:y+target_size, x:x+target_size] = np.minimum(
                canvas[y:y+target_size, x:x+target_size],
                img_resized
            )
            ground_truth.append((x, y, target_size, target_size, label))
        except:
            continue
    
    return canvas, ground_truth

# ============================================================================
# Main
# ============================================================================

def main(args):
    print("="*60)
    print("UNIFIED PIPELINE - DIGITS & SHAPES RECOGNITION")
    print("="*60 + "\n")
    
    # Create pipeline
    pipeline = UnifiedPipeline(
        model_path=args.model,
        label_mapping_path=args.labels,
        device=args.device,
        detector_type=args.detector,
        target_classes=args.target
    )
    
    print()
    
    if args.generate:
        # Generate synthetic test scene
        print("Generating synthetic test scene...")
        canvas, gt = generate_synthetic_scene(
            mnist_dir='mnist_competition/train',
            shapes_dir='Shapes_Classifier/dataset/output',
            mnist_csv='mnist_competition/train_label.csv',
            num_objects=args.num_objects,
            seed=42
        )
        
        # Save canvas
        test_image_path = 'synthetic_test_scene.png'
        cv2.imwrite(test_image_path, canvas)
        print(f"✅ Generated {test_image_path}")
        print(f"   Ground truth: {[item[4] for item in gt]}\n")
        
        # Process
        results = pipeline.process_file(test_image_path)
        
    elif args.image:
        # Process provided image
        results = pipeline.process_file(args.image, args.output)
    
    else:
        print("Error: Provide either --image or --generate")
        return
    
    print("\n✅ Pipeline completed successfully!")

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Unified Pipeline for Digits & Shapes Recognition'
    )
    
    # Model arguments
    parser.add_argument('--model', type=str,
                       default='unified_model_19classes_best.pth',
                       help='Path to classifier model')
    parser.add_argument('--labels', type=str,
                       default='label_mapping.json',
                       help='Path to label mapping JSON')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device: cuda or cpu')
    
    # Detection arguments
    parser.add_argument('--detector', type=str,
                       default='traditional',
                       choices=['traditional', 'hybrid'],
                       help='Detector type')
    
    # Target classes filter
    parser.add_argument('--target', type=str,
                       default='all',
                       choices=['digits', 'shapes', 'all'],
                       help='Target classes: digits (0-9), shapes (Circle, Square, etc.), or all')
    
    # Input/Output
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--output', type=str, help='Path to output image')
    
    # Synthetic data generation
    parser.add_argument('--generate', action='store_true',
                       help='Generate synthetic test scene')
    parser.add_argument('--num-objects', type=int, default=5,
                       help='Number of objects in synthetic scene')
    
    args = parser.parse_args()
    
    main(args)

