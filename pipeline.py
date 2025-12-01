"""
End-to-End Pipeline: Detection → Classification → Visualization
Usage: 
  - Local: python pipeline.py --image test_scene.png
  - MQTT: python pipeline.py --mqtt --broker-url mqtts://... --username ... --password ...
"""

import os
import argparse
import json
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import paho.mqtt.client as mqtt

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
                    # First, classify once to check if it's a digit or shape
                    crop_tensor = self.transform(crop).unsqueeze(0).to(self.device)
                    output = self.classifier(crop_tensor)
                    probs = torch.softmax(output, dim=1)
                    conf_first, pred_first = probs.max(1)
                    pred_class_id_first = pred_first.item()
                    
                    # Only apply TTA (rotation) for shapes (class_id >= 10)
                    # Digits (0-9) don't need TTA to avoid misclassification
                    if pred_class_id_first >= 10:
                        # It's likely a shape - use TTA with rotations
                        crops_to_test = [crop]
                        
                        # Add slight rotations (±5°, ±10°) for better classification
                        # Especially helps with perfectly aligned squares
                        for angle in [5, -5, 10, -10]:
                            # Rotate image
                            center = (crop.shape[1] // 2, crop.shape[0] // 2)
                            M = cv2.getRotationMatrix2D(center, angle, 1.0)
                            rotated = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]), 
                                                     borderValue=255)
                            crops_to_test.append(rotated)
                        
                        # Classify all variations and take average
                        all_probs = [probs]  # Include first prediction
                        for test_crop in crops_to_test[1:]:  # Skip first (already classified)
                            crop_tensor = self.transform(test_crop).unsqueeze(0).to(self.device)
                            output = self.classifier(crop_tensor)
                            probs_rot = torch.softmax(output, dim=1)
                            all_probs.append(probs_rot)
                        
                        # Average probabilities across all augmentations
                        avg_probs = torch.stack(all_probs).mean(dim=0)
                        conf, pred = avg_probs.max(1)
                        pred_class_id = pred.item()
                        confidence = conf.item()
                    else:
                        # It's a digit - use original prediction without TTA
                        conf, pred = probs.max(1)
                        pred_class_id = pred.item()
                        confidence = conf.item()
                    
                    # Filter based on target_classes
                    if self._should_keep_class(pred_class_id):
                        filtered_bboxes.append((x, y, w, h))
                        labels.append(self.label_mapping[pred_class_id])
                        confidences.append(confidence)
                    
                except Exception as e:
                    print(f"Warning: Failed to classify crop at ({x},{y},{w},{h}): {e}")
                    continue
        
        # Update bboxes to only include filtered ones
        bboxes = filtered_bboxes
        
        # Sort detections: top-to-bottom, left-to-right (reading order)
        if len(bboxes) > 0:
            # Calculate row tolerance based on average height
            avg_height = sum(h for (x, y, w, h) in bboxes) / len(bboxes) if bboxes else 30
            row_tolerance = max(avg_height * 0.5, 20)  # 50% of avg height or min 20px
            
            # Create list of tuples: (y_center, x, bbox, label, confidence)
            detections = list(zip(
                [y + h/2 for (x, y, w, h) in bboxes],  # y center coordinate
                [x for (x, y, w, h) in bboxes],       # x coordinate
                bboxes,
                labels,
                confidences
            ))
            
            # Sort by y first (top to bottom), then by x (left to right)
            # Group objects into rows: objects with similar y are in same row
            sorted_detections = sorted(detections, key=lambda d: (
                int(d[0] / row_tolerance),  # Row group (top to bottom)
                d[1]                         # X coordinate (left to right)
            ))
            
            # Unpack sorted results
            bboxes = [bbox for _, _, bbox, _, _ in sorted_detections]
            labels = [label for _, _, _, label, _ in sorted_detections]
            confidences = [conf for _, _, _, _, conf in sorted_detections]
        
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
                            num_digits=0, num_shapes=0, canvas_size=(800, 600), seed=None):
    """
    Generate synthetic test scene with specified number of digits and shapes.
    Objects are placed without overlapping.
    
    Args:
        mnist_dir: Directory with MNIST images
        shapes_dir: Directory with shape images
        mnist_csv: Path to MNIST labels CSV
        num_digits: Number of digits to place
        num_shapes: Number of shapes to place
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
    
    # Create white canvas (BGR color - 3 channels)
    canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
    ground_truth = []
    placed_objects = []  # List of (x, y, w, h) for overlap checking
    
    def check_overlap(new_x, new_y, new_w, new_h, existing_objects, margin=10):
        """Check if new object overlaps with existing objects."""
        for (ex_x, ex_y, ex_w, ex_h) in existing_objects:
            # Check if rectangles overlap (with margin)
            if not (new_x + new_w + margin < ex_x or 
                   new_x > ex_x + ex_w + margin or
                   new_y + new_h + margin < ex_y or
                   new_y > ex_y + ex_h + margin):
                return True
        return False
    
    def place_object(img, label, is_digit=True, max_attempts=100):
        """Try to place an object without overlapping."""
        target_size = np.random.randint(60, 100) if is_digit else np.random.randint(70, 120)
        
        # Convert to BGR based on type
        if is_digit:
            # Digits: Keep grayscale, just convert to BGR (3 channels)
            if len(img.shape) == 2:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img.copy()
        else:
            # Shapes: Convert grayscale to BGR with random color
            if len(img.shape) == 2:
                # Grayscale image - convert to BGR with random color
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                # Apply random color tint (keep grayscale but add subtle color)
                color = np.random.randint(0, 256, 3)  # Random BGR color
                # Blend: 70% grayscale + 30% color for natural look
                img_bgr = cv2.addWeighted(img_bgr, 0.7, 
                                         np.full_like(img_bgr, color, dtype=np.uint8), 0.3, 0)
            else:
                # Already color image - use as is
                img_bgr = img.copy()
        
        img_resized = cv2.resize(img_bgr, (target_size, target_size))
        
        margin = 50
        for attempt in range(max_attempts):
            x = np.random.randint(margin, canvas_size[0] - target_size - margin)
            y = np.random.randint(margin, canvas_size[1] - target_size - margin)
            
            # Check overlap
            if not check_overlap(x, y, target_size, target_size, placed_objects):
                # Place object (blend with canvas for smooth appearance)
                try:
                    # Use alpha blending for smoother appearance
                    roi = canvas[y:y+target_size, x:x+target_size]
                    mask = (img_resized < 250).any(axis=2)  # Create mask for non-white pixels
                    roi[mask] = img_resized[mask]
                    canvas[y:y+target_size, x:x+target_size] = roi
                    
                    placed_objects.append((x, y, target_size, target_size))
                    ground_truth.append((x, y, target_size, target_size, label))
                    return True
                except:
                    continue
        
        # If couldn't place after max_attempts, skip
        print(f"⚠️ Warning: Could not place {label} after {max_attempts} attempts")
        return False
    
    # Place digits
    for _ in range(num_digits):
        sample = mnist_df.sample(1).iloc[0]
        img_path = os.path.join(mnist_dir, sample['image_name'])
        # Load as grayscale (MNIST images are grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        label = str(sample['label'])
        place_object(img, label, is_digit=True)
    
    # Place shapes
    for _ in range(num_shapes):
        sample_file = np.random.choice(shape_files)
        img_path = os.path.join(shapes_dir, sample_file)
        # Try to load as color first, fallback to grayscale
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        label = sample_file.split('_')[0]
        place_object(img, label, is_digit=False)
    
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
    
    # Auto-detect mode: if no --image and no --generate, use MQTT mode
    use_mqtt = args.mqtt or (not args.image and not args.generate)
    
    # MQTT mode
    if use_mqtt:
        print("🌐 MQTT MODE ENABLED")
        print(f"   Listening on:")
        print(f"     - {args.input_topic} (image processing)")
        print(f"     - {args.create_topic} (image generation)")
        print(f"   Publishing to:")
        print(f"     - {args.create_output_topic} (generated images)")
        print(f"     - {args.output_topic} (processing results)")
        print(f"   Broker: {args.broker_url}")
        print(f"   Waiting for requests...\n")
        
        # Create MQTT pipeline
        mqtt_pipeline = MQTTPipeline(
            pipeline=pipeline,
            broker_url=args.broker_url,
            username=args.mqtt_username,
            password=args.mqtt_password,
            input_topic=args.input_topic,
            output_topic=args.output_topic,
            create_topic=args.create_topic,
            create_output_topic=args.create_output_topic,
            port=args.mqtt_port,
            mnist_dir='mnist_competition/train',
            shapes_dir='Shapes_Classifier/dataset/output',
            mnist_csv='mnist_competition/train_label.csv'
        )
        
        # Start listening (this will run forever until interrupted)
        print("💡 Tip: Press Ctrl+C to stop\n")
        mqtt_pipeline.start()
    
    # Local file processing mode
    elif args.generate:
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
        # This should not happen due to auto-detect, but keep as fallback
        print("Error: Provide either --image, --generate, or use MQTT mode (default)")
        return
    
    print("\n✅ Pipeline completed successfully!")

# ============================================================================
# MQTT Integration
# ============================================================================

class MQTTPipeline:
    """
    MQTT-based pipeline: Listen to image/input, process, send to image/output
    """
    
    def __init__(self, pipeline, broker_url, username=None, password=None,
                 input_topic='image/input', output_topic='image/output',
                 create_topic='image/create', create_output_topic='image/input/create',
                 port=8883,
                 mnist_dir='mnist_competition/train',
                 shapes_dir='Shapes_Classifier/dataset/output',
                 mnist_csv='mnist_competition/train_label.csv'):
        """
        Args:
            pipeline: UnifiedPipeline instance
            broker_url: MQTT broker URL (e.g., 'mqtts://broker.hivemq.cloud')
            username: MQTT username
            password: MQTT password
            input_topic: Topic to listen for input images (processing)
            output_topic: Topic to publish processing results
            create_topic: Topic to listen for image generation requests
            create_output_topic: Topic to publish generated images
            port: MQTT port (8883 for TLS, 1883 for plain)
            mnist_dir: Directory with MNIST images
            shapes_dir: Directory with shape images
            mnist_csv: Path to MNIST labels CSV
        """
        self.pipeline = pipeline
        self.broker_url = broker_url
        self.username = username
        self.password = password
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.create_topic = create_topic
        self.create_output_topic = create_output_topic
        self.port = port
        self.mnist_dir = mnist_dir
        self.shapes_dir = shapes_dir
        self.mnist_csv = mnist_csv
        
        # Extract hostname from URL
        if broker_url.startswith('mqtts://'):
            self.hostname = broker_url.replace('mqtts://', '').split('/')[0]
            self.use_tls = True
        elif broker_url.startswith('mqtt://'):
            self.hostname = broker_url.replace('mqtt://', '').split('/')[0]
            self.use_tls = False
        else:
            self.hostname = broker_url.split('/')[0]
            self.use_tls = False
        
        # Create MQTT client
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        # Set credentials if provided
        if username and password:
            self.client.username_pw_set(username, password)
        
        # Enable TLS if needed
        if self.use_tls:
            self.client.tls_set()
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker."""
        if rc == 0:
            print(f"✅ Connected to MQTT broker: {self.hostname}")
            print(f"📡 Subscribing to topics:")
            print(f"   - {self.input_topic} (image processing)")
            print(f"   - {self.create_topic} (image generation)")
            client.subscribe(self.input_topic)
            client.subscribe(self.create_topic)
        else:
            print(f"❌ Failed to connect to MQTT broker. Return code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from MQTT broker."""
        if rc != 0:
            print(f"⚠️ Unexpected MQTT disconnection. Return code: {rc}")
        else:
            print("✅ Disconnected from MQTT broker")
    
    def _on_message(self, client, userdata, msg):
        """Callback when message received."""
        try:
            print(f"\n📨 Received message on topic: {msg.topic}")
            
            # Handle image generation request
            if msg.topic == self.create_topic:
                self._handle_image_generation(client, msg)
                return
            
            # Handle image processing request
            if msg.topic == self.input_topic:
                self._handle_image_processing(client, msg)
                return
                
        except Exception as e:
            print(f"❌ Error in message handler: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_image_generation(self, client, msg):
        """Handle image generation request from image/create topic."""
        try:
            # Parse JSON message
            message_data = json.loads(msg.payload.decode('utf-8'))
            
            # Extract parameters
            num_digits = message_data.get('numberDigit', 0)
            num_shapes = message_data.get('numberShape', 0)
            
            print(f"🎨 Generating synthetic scene:")
            print(f"   Digits: {num_digits}")
            print(f"   Shapes: {num_shapes}")
            
            if num_digits == 0 and num_shapes == 0:
                print("❌ Error: Both numberDigit and numberShape are 0")
                return
            
            # Generate synthetic scene
            canvas, ground_truth = generate_synthetic_scene(
                mnist_dir=self.mnist_dir,
                shapes_dir=self.shapes_dir,
                mnist_csv=self.mnist_csv,
                num_digits=num_digits,
                num_shapes=num_shapes,
                canvas_size=(800, 600),
                seed=None  # Random seed for variety
            )
            
            print(f"✅ Generated scene with {len(ground_truth)} objects")
            print(f"   Ground truth: {[item[4] for item in ground_truth]}")
            
            # Encode image to base64
            image_base64 = self._encode_image_base64(canvas)
            
            # Count digits and shapes in ground truth
            digits_count = sum(1 for (x, y, w, h, label) in ground_truth if label.isdigit())
            shapes_count = len(ground_truth) - digits_count
            
            # Prepare output message (image + count)
            output_message = {
                'image': image_base64,
                'count': {
                    'digits': digits_count,
                    'shapes': shapes_count,
                    'total': len(ground_truth)
                }
            }
            
            # Publish to image/input/create topic (for FE to receive generated image)
            output_json = json.dumps(output_message)
            client.publish(self.create_output_topic, output_json, qos=1)
            print(f"✅ Published generated image to {self.create_output_topic}")
            print(f"   Count: digits={digits_count}, shapes={shapes_count}, total={len(ground_truth)}\n")
            
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON format: {e}")
        except Exception as e:
            print(f"❌ Error generating image: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_image_processing(self, client, msg):
        """Handle image processing request from image/input topic."""
        try:
            # Parse JSON message
            message_data = json.loads(msg.payload.decode('utf-8'))
            
            # Extract image and label
            if 'image' not in message_data:
                print("❌ Error: 'image' field not found in message")
                return
            
            image_base64 = message_data['image']
            label_raw = message_data.get('label', 2)  # Default to 2 (all)
            
            # Extract count info
            count_info = message_data.get('count', {})
            
            # Check if this is a generated image (has count with digits/shapes > 0)
            # If yes, always use hybrid detector and ignore label
            is_generated_image = False
            if count_info:
                num_digits = count_info.get('digits', 0)
                num_shapes = count_info.get('shapes', 0)
                if num_digits > 0 or num_shapes > 0:
                    is_generated_image = True
                    print(f"🔄 Detected generated image - using HYBRID detector (label ignored)")
                    # For generated images, always use 'all' target (FE already handles filtering)
                    target = 'all'
                else:
                    # Regular uploaded image - use label
                    try:
                        label = int(label_raw)
                    except (ValueError, TypeError):
                        print(f"⚠️ Warning: Invalid label format '{label_raw}', defaulting to 2 (all)")
                        label = 2
                    
                    # Map label: 0=digits, 1=shapes, 2=all
                    label_map = {0: 'digits', 1: 'shapes', 2: 'all'}
                    target = label_map.get(label, 'all')
                    print(f"🎯 Target mode: {target} (label={label})")
            else:
                # No count info - regular uploaded image
                try:
                    label = int(label_raw)
                except (ValueError, TypeError):
                    print(f"⚠️ Warning: Invalid label format '{label_raw}', defaulting to 2 (all)")
                    label = 2
                
                # Map label: 0=digits, 1=shapes, 2=all
                label_map = {0: 'digits', 1: 'shapes', 2: 'all'}
                target = label_map.get(label, 'all')
                print(f"🎯 Target mode: {target} (label={label})")
            
            # Decode base64 to image
            image = self._decode_base64_image(image_base64)
            if image is None:
                print("❌ Error: Failed to decode base64 image")
                return
            
            print(f"✅ Decoded image: {image.shape}")
            
            # Update pipeline target_classes
            self.pipeline.target_classes = target
            
            # Switch to hybrid detector if this is a generated image
            original_detector = None
            if is_generated_image:
                original_detector = self.pipeline.detector
                # Create hybrid detector if not already using it
                if not isinstance(self.pipeline.detector, HybridDetector):
                    craft_path = 'weights/craft_mlt_25k.pth'
                    self.pipeline.detector = HybridDetector(craft_path)
                    print(f"✅ Switched to Hybrid detector for generated image")
            
            # Process image
            print("🔄 Processing image...")
            results = self.pipeline.process(image, visualize=True)
            
            # Restore original detector if we switched
            if is_generated_image and original_detector and not isinstance(original_detector, HybridDetector):
                self.pipeline.detector = original_detector
                print(f"✅ Restored original detector")
            
            print(f"✅ Detected {len(results['labels'])} objects")
            
            # Encode result image to base64
            result_image_base64 = self._encode_image_base64(results['annotated_image'])
            
            # Prepare output message
            output_message = {
                'image': result_image_base64,
                'results': {
                    'detections': [
                        {
                            'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                            'label': label,
                            'confidence': float(conf)
                        }
                        for (x, y, w, h), label, conf in zip(
                            results['bboxes'], results['labels'], results['confidences']
                        )
                    ],
                    'count': len(results['labels'])
                }
            }
            
            # Publish result
            output_json = json.dumps(output_message)
            client.publish(self.output_topic, output_json, qos=1)
            print(f"✅ Published result to {self.output_topic}")
            print(f"   Detections: {len(results['labels'])} objects\n")
            
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON format: {e}")
        except Exception as e:
            print(f"❌ Error processing message: {e}")
            import traceback
            traceback.print_exc()
    
    def _decode_base64_image(self, base64_string):
        """
        Decode base64 string to OpenCV image (numpy array).
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            numpy array (H, W, 3) or (H, W) - OpenCV image format
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            # Convert to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                # Try grayscale
                image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            return image
            
        except Exception as e:
            print(f"❌ Error decoding base64 image: {e}")
            return None
    
    def _encode_image_base64(self, image):
        """
        Encode OpenCV image to base64 string.
        
        Args:
            image: numpy array (H, W, 3) or (H, W) - OpenCV image format
            
        Returns:
            str: Base64 encoded image string
        """
        try:
            # Encode image to PNG
            success, buffer = cv2.imencode('.png', image)
            if not success:
                raise ValueError("Failed to encode image")
            
            # Convert to base64
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return image_base64
            
        except Exception as e:
            print(f"❌ Error encoding image to base64: {e}")
            return ""
    
    def start(self):
        """Start MQTT client and begin listening."""
        try:
            print(f"🔌 Connecting to MQTT broker: {self.hostname}:{self.port}")
            print(f"   TLS: {self.use_tls}")
            print(f"   Username: {self.username if self.username else 'None'}")
            
            self.client.connect(self.hostname, self.port, 60)
            print("🔄 Starting MQTT loop...")
            self.client.loop_forever()
            
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            self.client.disconnect()
        except Exception as e:
            print(f"❌ Error starting MQTT client: {e}")
            import traceback
            traceback.print_exc()
    
    def stop(self):
        """Stop MQTT client."""
        self.client.disconnect()
        print("✅ MQTT client stopped")

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Unified Pipeline for Digits & Shapes Recognition'
    )
    
    # Model arguments
    parser.add_argument('--model', type=str,
                       default='unified_model_19classes_best2.pth',
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
    
    # MQTT arguments (with defaults from environment or hardcoded)
    # Default MQTT configuration
    default_broker_url = os.getenv('MQTT_BROKER_URL', 'mqtts://c35f82397d674292948a051226f10fa6.s1.eu.hivemq.cloud')
    default_username = os.getenv('MQTT_USERNAME', 'server')
    default_password = os.getenv('MQTT_PASSWORD', 'Server123456')
    
    parser.add_argument('--mqtt', action='store_true',
                       help='Enable MQTT mode (listen to image/input, publish to image/output). Default if no --image or --generate')
    parser.add_argument('--broker-url', type=str, default=default_broker_url,
                       help=f'MQTT broker URL (default: {default_broker_url})')
    parser.add_argument('--mqtt-username', type=str, default=default_username,
                       help=f'MQTT username (default: {default_username})')
    parser.add_argument('--mqtt-password', type=str, default=default_password,
                       help='MQTT password (default: from env or hardcoded)')
    parser.add_argument('--input-topic', type=str, default='image/input',
                       help='MQTT topic to listen for input images')
    parser.add_argument('--output-topic', type=str, default='image/output',
                       help='MQTT topic to publish results')
    parser.add_argument('--create-topic', type=str, default='image/create',
                       help='MQTT topic to listen for image generation requests')
    parser.add_argument('--create-output-topic', type=str, default='image/input/create',
                       help='MQTT topic to publish generated images')
    parser.add_argument('--mqtt-port', type=int, default=8883,
                       help='MQTT port (8883 for TLS, 1883 for plain)')
    
    args = parser.parse_args()
    
    main(args)

