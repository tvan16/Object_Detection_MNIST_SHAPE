# 🎯 CRAFT + SHAPES DETECTION - COMPLETE GUIDE

## Hệ thống nhận diện chữ số viết tay và hình học trong ảnh phức tạp

---

## 📋 MỤC LỤC

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Data Preparation](#2-data-preparation)
3. [Classification Model](#3-classification-model)
4. [Detection Module](#4-detection-module)
5. [Pipeline Integration](#5-pipeline-integration)
6. [Testing & Evaluation](#6-testing--evaluation)
7. [Advanced: CRAFT Integration](#7-advanced-craft-integration)
8. [Deployment](#8-deployment)

---

## 1. TỔNG QUAN KIẾN TRÚC

### Pipeline Overview

```
┌─────────────────────────────────────────────────┐
│         INPUT: Complex Scene Image              │
│    (Whiteboard/Paper với digits + shapes)      │
└────────────────┬────────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼──────────────┐   ┌─────▼─────────────────┐
│  MODULE 1:       │   │  MODULE 2:            │
│  Digit Detector  │   │  Shape Detector       │
│  (CRAFT/CV)      │   │  (Traditional CV)     │
└───┬──────────────┘   └─────┬─────────────────┘
    │                        │
    │ Bboxes (digits)        │ Bboxes (shapes)
    │                        │
    └────────┬───────────────┘
             │
     ┌───────▼────────────┐
     │  UNIFIED           │
     │  CLASSIFICATION    │
     │  EfficientNet-B0   │
     │  (19 classes)      │
     └───────┬────────────┘
             │
     ┌───────▼────────────────────┐
     │  OUTPUT:                   │
     │  - Annotated image         │
     │  - JSON: positions+labels  │
     │  - Confidence scores       │
     └────────────────────────────┘
```

### Key Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Detection** | Find regions with digits/shapes | CRAFT (text) + CV (shapes) |
| **Classification** | Identify what each region is | EfficientNet-B0 (19 classes) |
| **Pipeline** | Orchestrate detection → classification | Python orchestration |
| **Visualization** | Draw boxes & labels on image | OpenCV |

---

## 2. DATA PREPARATION

### 2.1 Dataset Overview

**MNIST Dataset:**
- 60,000 chữ số viết tay (0-9)
- Size: 28×28 grayscale
- Location: `mnist_competition/train/`
- Labels: `mnist_competition/train_label.csv`

**Shapes Dataset:**
- 90,000 hình học (9 loại)
- Size: 64×64 grayscale
- Classes: Circle, Heptagon, Hexagon, Nonagon, Octagon, Pentagon, Square, Star, Triangle
- Location: `Shapes_Classifier/dataset/output/`

### 2.2 Unified Label Mapping

```python
# Classes 0-9: Digits
# Classes 10-18: Shapes (alphabetical order)

Label Mapping:
├─ 0-9:   Digits (0,1,2,3,4,5,6,7,8,9)
└─ 10-18: Shapes
    ├─ 10: Circle
    ├─ 11: Heptagon
    ├─ 12: Hexagon
    ├─ 13: Nonagon
    ├─ 14: Octagon
    ├─ 15: Pentagon
    ├─ 16: Square
    ├─ 17: Star
    └─ 18: Triangle
```

### 2.3 Data Processing Strategy

**Why resize 28→64 instead of 64→28?**

| Aspect | 28→64 (✅ GOOD) | 64→28 (❌ BAD) |
|--------|----------------|---------------|
| MNIST digits | Simple, upscale OK | - |
| Shapes (polygons) | - | Lose edge details |
| Information loss | Minimal | 75% pixels lost |
| Model compatibility | Works with EfficientNet | Too small for pretrained |
| Aspect ratio | Maintained | Maintained |

**Result:** All images resize to **64×64** for unified model.

### 2.4 Data Augmentation

```python
Training augmentation:
├─ Random rotation: ±15°
├─ Random translation: ±10%
├─ Resize: 64×64
├─ Grayscale → RGB (3 channels)
└─ Normalize: ImageNet stats

Validation/Test:
├─ Resize: 64×64
├─ Grayscale → RGB
└─ Normalize: ImageNet stats
```

---

## 3. CLASSIFICATION MODEL

### 3.1 Architecture: EfficientNet-B0

**Why EfficientNet-B0?**
- ✅ **Balanced**: Good accuracy with reasonable size (5.3M params)
- ✅ **Pretrained**: Transfer learning from ImageNet
- ✅ **Efficient**: Faster than ResNet/ConvNeXt
- ✅ **Modern**: State-of-the-art architecture

**Model Modifications:**
```python
# Original: EfficientNet-B0 (1000 classes for ImageNet)
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

# Modify classifier head for 19 classes
num_features = model.classifier[1].in_features  # 1280
model.classifier[1] = nn.Linear(1280, 19)
```

### 3.2 Training Configuration

```python
Hyperparameters:
├─ Epochs: 10-15
├─ Batch size: 64
├─ Learning rate: 1e-4 (fine-tuning)
├─ Optimizer: Adam
├─ Scheduler: ReduceLROnPlateau (factor=0.5, patience=2)
├─ Loss: CrossEntropyLoss
└─ Device: CUDA (GPU)

Expected Results:
├─ Training accuracy: ~99.5%
├─ Validation accuracy: ~99.0-99.5%
└─ Training time: ~30-60 minutes (RTX 4050)
```

### 3.3 Input/Output Format

**Input:**
- Shape: `(batch, 3, 64, 64)`
- Type: `torch.FloatTensor`
- Range: Normalized [-2.5, 2.5] (ImageNet normalization)

**Output:**
- Shape: `(batch, 19)`
- Type: `torch.FloatTensor`
- Values: Logits (before softmax)

**Inference:**
```python
# Get probabilities
probs = torch.softmax(outputs, dim=1)

# Get prediction
confidence, predicted_class = probs.max(1)
```

---

## 4. DETECTION MODULE

### 4.1 Approach Comparison

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Traditional CV** | ✅ Fast<br>✅ No training<br>✅ Easy debug | ❌ Clean backgrounds only<br>❌ No overlapping | Whiteboard, paper, clean scenes |
| **CRAFT** | ✅ Robust text detection<br>✅ Complex backgrounds<br>✅ Handles rotation | ❌ Text/digits only<br>❌ Needs pretrained model | Document, natural scenes with text |
| **YOLOv8** | ✅ End-to-end<br>✅ Fast inference<br>✅ Handles overlapping | ❌ Needs training<br>❌ Requires labeled data | Production, complex scenes |

**Recommendation:** Start with **Traditional CV**, upgrade to **CRAFT + YOLOv8** later.

### 4.2 Traditional CV Detection (Baseline)

**Algorithm:**
```
1. Preprocessing:
   ├─ Convert to grayscale
   ├─ Gaussian blur (5×5)
   └─ Adaptive threshold

2. Morphological operations:
   ├─ Close operation (3×3 kernel)
   └─ Fill holes

3. Contour detection:
   ├─ Find external contours
   └─ Filter by area & aspect ratio

4. Bounding box extraction:
   ├─ cv2.boundingRect()
   └─ Return (x, y, w, h)
```

**Parameters:**
```python
min_area = 200      # Minimum object size (pixels²)
max_area = 30000    # Maximum object size
aspect_ratio_range = (0.2, 5.0)  # Reject extreme shapes
```

### 4.3 CRAFT Integration (Advanced)

**CRAFT (Character Region Awareness For Text):**
- Detects text at **character level**
- Returns heatmaps for text regions
- Handles curved text, multi-orientation

**Setup:**
```bash
# Clone CRAFT repository
git clone https://github.com/clovaai/CRAFT-pytorch.git

# Download pretrained model (177MB)
# Link: https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ
# Save as: craft_mlt_25k.pth
```

**Usage:**
```python
from craft import CRAFT
import craft_utils

# Load CRAFT
craft_net = CRAFT()
craft_net.load_state_dict(torch.load('craft_mlt_25k.pth'))
craft_net.eval()

# Detect text regions
bboxes, polys, score_text = craft_utils.getDetBoxes(
    textmap, linkmap, text_threshold, link_threshold, low_text
)
```

---

## 5. PIPELINE INTEGRATION

### 5.1 UnifiedPipeline Class

**Complete workflow:**

```python
class UnifiedPipeline:
    def __init__(self, classifier_model, label_mapping, device):
        self.classifier = classifier_model
        self.label_mapping = label_mapping
        self.detector = TraditionalDetector()
    
    def process(self, image):
        """
        Args:
            image: numpy array (H, W) or (H, W, 3)
        
        Returns:
            {
                'bboxes': [(x,y,w,h), ...],
                'labels': ['3', 'Circle', '7', ...],
                'confidences': [0.98, 0.95, ...],
                'annotated_image': numpy array
            }
        """
        # Step 1: Detect regions
        bboxes = self.detector.detect(image)
        
        # Step 2: Classify each region
        labels = []
        confidences = []
        
        for (x, y, w, h) in bboxes:
            crop = image[y:y+h, x:x+w]
            crop_tensor = self.transform(crop)
            
            output = self.classifier(crop_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = probs.max(1)
            
            labels.append(self.label_mapping[pred.item()])
            confidences.append(conf.item())
        
        # Step 3: Visualize
        annotated = self.visualize(image, bboxes, labels, confidences)
        
        return {
            'bboxes': bboxes,
            'labels': labels,
            'confidences': confidences,
            'annotated_image': annotated
        }
```

### 5.2 Error Handling

**Common issues & solutions:**

| Issue | Cause | Solution |
|-------|-------|----------|
| No detections | Too strict thresholds | Lower `min_area`, adjust threshold |
| False positives | Noise, artifacts | Increase `min_area`, better preprocessing |
| Wrong classification | Low confidence | Check crop quality, retrain model |
| Empty crops | Bad bbox extraction | Add bbox validation |

---

## 6. TESTING & EVALUATION

### 6.1 Synthetic Test Data Generation

**Purpose:** Create complex scenes with known ground truth.

**Algorithm:**
```python
def generate_synthetic_scene(num_objects=5, canvas_size=(800,600)):
    # 1. Create white canvas
    canvas = np.ones((H, W)) * 255
    
    # 2. For each object:
    for i in range(num_objects):
        # Randomly choose digit or shape
        is_digit = random.choice([True, False])
        
        if is_digit:
            img = load_random_mnist_image()
        else:
            img = load_random_shape_image()
        
        # Resize to random size (60-120px)
        img_resized = cv2.resize(img, (size, size))
        
        # Random position (avoid overlap if possible)
        x, y = random_position()
        
        # Paste onto canvas
        canvas[y:y+size, x:x+size] = img_resized
        
        # Record ground truth
        ground_truth.append((x, y, size, size, label))
    
    return canvas, ground_truth
```

### 6.2 Evaluation Metrics

**Per-class metrics:**
```
Precision = TP / (TP + FP)  # Accuracy of predictions
Recall = TP / (TP + FN)     # Coverage of ground truth
F1-Score = 2 * (P * R) / (P + R)
```

**Detection metrics:**
- **IoU (Intersection over Union):** Measure bbox accuracy
- **mAP (mean Average Precision):** Overall detection quality

**Classification metrics:**
- **Top-1 Accuracy:** % of correct predictions
- **Confusion Matrix:** Where model gets confused

---

## 7. ADVANCED: CRAFT INTEGRATION

### 7.1 Hybrid Detection Strategy

**For best results, use hybrid approach:**

```
Input Image
    ↓
┌───────────┴───────────┐
│                       │
├─ Text/Digits?         ├─ Shapes?
│  Use CRAFT            │  Use Traditional CV
│  (robust, accurate)   │  (fast, simple)
│                       │
└───────────┬───────────┘
            │
       Merge bboxes
            │
    Classify each region
            │
         Output
```

**Implementation:**
```python
class HybridDetector:
    def __init__(self, craft_model):
        self.craft = craft_model
        self.cv_detector = TraditionalDetector()
    
    def detect(self, image):
        # CRAFT for text/digits
        text_bboxes = self.craft.detect(image)
        
        # Traditional CV for shapes
        # (mask out text regions first)
        masked_image = self.mask_regions(image, text_bboxes)
        shape_bboxes = self.cv_detector.detect(masked_image)
        
        # Merge & deduplicate
        all_bboxes = self.merge_bboxes(text_bboxes, shape_bboxes)
        
        return all_bboxes
```

### 7.2 CRAFT Fine-tuning (Optional)

**If CRAFT misses digits in your specific domain:**

1. Collect labeled data (100-500 images)
2. Generate weak labels using SynthText
3. Fine-tune CRAFT on your data
4. Evaluate on validation set

---

## 8. DEPLOYMENT

### 8.1 Model Export

**Export to ONNX for production:**
```python
import torch.onnx

dummy_input = torch.randn(1, 3, 64, 64).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "unified_classifier.onnx",
    export_params=True,
    opset_version=12,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

### 8.2 Optimization Techniques

| Technique | Speed Gain | Accuracy Impact |
|-----------|------------|-----------------|
| **Model Quantization** | 2-4× faster | -1% to -2% |
| **TensorRT** | 3-5× faster | Minimal |
| **ONNX Runtime** | 1.5-2× faster | None |
| **Batch inference** | 2-3× faster | None |
| **Mixed precision (FP16)** | 1.5-2× faster | Minimal |

### 8.3 Web API (FastAPI)

```python
from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process
    results = pipeline.process(image)
    
    return {
        "detections": [
            {
                "bbox": bbox,
                "label": label,
                "confidence": conf
            }
            for bbox, label, conf in zip(
                results['bboxes'],
                results['labels'],
                results['confidences']
            )
        ]
    }
```

---

## 📊 EXPECTED RESULTS

### Classification Accuracy

| Dataset | Training Acc | Validation Acc | Test Acc (Expected) |
|---------|--------------|----------------|---------------------|
| MNIST Digits | 99.5% | 99.3% | 99.0-99.3% |
| Shapes | 99.0% | 98.5% | 98.0-98.5% |
| **Unified (19 classes)** | **99.3%** | **99.0%** | **98.5-99.0%** |

### Detection Performance

**Traditional CV (clean background):**
- Precision: ~95%
- Recall: ~90%
- Speed: ~50ms per image

**CRAFT + CV (complex background):**
- Precision: ~97%
- Recall: ~95%
- Speed: ~150ms per image

### End-to-End Performance

**Latency breakdown:**
```
Input Image (800×600)
    ↓
Detection: 50-150ms
    ↓
Preprocessing: 10-20ms
    ↓
Classification: 5-10ms per object
    ↓
Visualization: 10-20ms
    ↓
Total: 100-300ms (3-10 FPS)
```

---

## 🚀 QUICK START COMMANDS

```bash
# 1. Setup environment
pip install torch torchvision opencv-python scikit-learn matplotlib tqdm

# 2. Clone CRAFT (optional)
git clone https://github.com/clovaai/CRAFT-pytorch.git

# 3. Train classifier
python train_unified_classifier.py --epochs 10 --batch-size 64

# 4. Test pipeline
python test_pipeline.py --image test_scene.png

# 5. Run demo
python demo.py --model unified_model_19classes_best.pth --image your_image.png
```

---

## 📁 PROJECT STRUCTURE

```
BTL_XLA/
├── mnist_competition/
│   ├── train/              # 60K MNIST images
│   └── train_label.csv     # MNIST labels
├── Shapes_Classifier/
│   └── dataset/output/     # 90K shape images
├── models/
│   ├── unified_model_19classes_best.pth
│   └── label_mapping.json
├── scripts/
│   ├── train_unified_classifier.py
│   ├── test_pipeline.py
│   ├── generate_synthetic_data.py
│   └── demo.py
├── outputs/
│   ├── training_history.png
│   ├── confusion_matrix_19classes.png
│   └── test_results/
└── CRAFT_SHAPES_GUIDE.md   # This file
```

---

## 🎓 LEARNING OUTCOMES

Sau khi hoàn thành project này, bạn sẽ:

1. ✅ Hiểu cách merge nhiều datasets với label mapping
2. ✅ Biết cách resize images hợp lý (upsample vs downsample)
3. ✅ Sử dụng transfer learning với pretrained models
4. ✅ Implement object detection từ scratch
5. ✅ Tích hợp detection + classification pipeline
6. ✅ Generate synthetic data cho testing
7. ✅ Evaluate model với confusion matrix
8. ✅ Export model cho deployment

---

## 📞 TROUBLESHOOTING

### Issue 1: Low accuracy on shapes
**Cause:** Shapes dataset có nhiều noise hoặc class imbalance  
**Solution:** 
- Check data quality với visualize
- Apply class weights trong loss function
- Increase training epochs

### Issue 2: Detection misses objects
**Cause:** Threshold quá strict hoặc objects quá nhỏ/lớn  
**Solution:**
- Adjust `min_area` và `max_area`
- Visualize intermediate steps (binary, morphology)
- Try different preprocessing

### Issue 3: Slow inference
**Cause:** Model too large hoặc không optimize  
**Solution:**
- Use smaller model (MobileNet instead of EfficientNet)
- Batch inference for multiple objects
- Export to ONNX/TensorRT

---

## 📚 REFERENCES

1. **CRAFT Paper:** [Character Region Awareness for Text Detection (CVPR 2019)](https://arxiv.org/abs/1904.01941)
2. **EfficientNet Paper:** [EfficientNet: Rethinking Model Scaling (ICML 2019)](https://arxiv.org/abs/1905.11946)
3. **CRAFT GitHub:** https://github.com/clovaai/CRAFT-pytorch
4. **PyTorch Transfer Learning:** https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

---

**🎉 Good luck with your project! 🚀**

