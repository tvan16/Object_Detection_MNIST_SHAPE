# 🎯 Unified Digits & Shapes Recognition System

Hệ thống nhận diện chữ số viết tay và hình học trong ảnh sử dụng Deep Learning với kiến trúc Detection + Classification.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Giới thiệu dự án

### Tổng quan

Dự án **Unified Digits & Shapes Recognition System** là một hệ thống nhận diện đối tượng hoàn chỉnh, có khả năng phát hiện và phân loại đồng thời **chữ số viết tay** (0-9) và **hình học** (9 loại) trong cùng một ảnh. Hệ thống sử dụng kiến trúc **hai giai đoạn** (Two-Stage): **Detection** để tìm vị trí các đối tượng, sau đó **Classification** để nhận diện loại của từng đối tượng.

### Mục tiêu

- 🎯 Xây dựng một mô hình thống nhất có thể nhận diện cả chữ số và hình học trong cùng một pipeline
- 🎯 Đạt độ chính xác cao (>99%) trên cả hai loại đối tượng
- 🎯 Tối ưu tốc độ inference để có thể áp dụng trong thực tế
- 🎯 Hỗ trợ nhiều phương pháp detection linh hoạt (Traditional CV, CRAFT, Hybrid)
- 🎯 Tích hợp MQTT để xử lý real-time từ frontend

### Ứng dụng thực tế

- 📝 **Nhận diện chữ số viết tay**: Đọc số từ biểu mẫu, hóa đơn, chứng từ
- 🔷 **Phân loại hình học**: Phân tích hình dạng trong ảnh kỹ thuật, bản vẽ
- 🎓 **Giáo dục**: Hỗ trợ học sinh nhận diện số và hình học
- 🏭 **Tự động hóa**: Xử lý ảnh trong dây chuyền sản xuất
- 📱 **Mobile Apps**: Tích hợp vào ứng dụng di động để nhận diện real-time

## 🏗️ Kiến trúc hệ thống

### Pipeline tổng quan

```
┌─────────────┐
│ Input Image │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│   Stage 1: Object Detection     │
│   ┌──────────────────────────┐  │
│   │ Traditional CV Detector  │  │
│   │ (Contour-based)          │  │
│   └──────────────────────────┘  │
│   ┌──────────────────────────┐  │
│   │ CRAFT Detector           │  │
│   │ (Text/Character)         │  │
│   └──────────────────────────┘  │
│   ┌──────────────────────────┐  │
│   │ Hybrid Detector          │  │
│   │ (CV + CRAFT combined)    │  │
│   └──────────────────────────┘  │
└──────┬──────────────────────────┘
       │ Bounding Boxes (x, y, w, h)
       ▼
┌─────────────────────────────────┐
│   Stage 2: Classification       │
│   ┌──────────────────────────┐  │
│   │ EfficientNet-B0          │  │
│   │ (19 classes)             │  │
│   │ - 10 digits (0-9)        │  │
│   │ - 9 shapes               │  │
│   └──────────────────────────┘  │
└──────┬──────────────────────────┘
       │ Labels + Confidences
       ▼
┌─────────────────────────────────┐
│   Stage 3: Post-processing      │
│   - Filter by target (digits/  │
│     shapes/all)                 │
│   - Sort by reading order       │
│   - Visualize annotations       │
└──────┬──────────────────────────┘
       ▼
┌─────────────────────────────────┐
│   Output:                        │
│   - Annotated Image (PNG)       │
│   - JSON Results                 │
└─────────────────────────────────┘
```

### Các thành phần chính

#### 1. **Detection Module** (`detect_objects.py`)

**Traditional CV Detector:**
- Sử dụng OpenCV để tìm contours
- Preprocessing: Denoising, CLAHE, Illumination correction
- Adaptive thresholding để tách foreground/background
- Filter theo area, aspect ratio để loại bỏ noise

**CRAFT Detector:**
- Deep learning model để detect text/characters
- Pre-trained trên MLT dataset (25k images)
- Tốt cho việc detect chữ số và ký tự

**Hybrid Detector:**
- Kết hợp Traditional CV + CRAFT
- CRAFT detect digits, Traditional CV detect shapes
- Merge và deduplicate kết quả
- Tối ưu cho ảnh có cả digits và shapes

#### 2. **Classification Module** (`train_unified_classifier.py`)

**Model Architecture:**
- **Backbone**: EfficientNet-B0 (pre-trained trên ImageNet)
- **Input**: 128x128 RGB images (grayscale converted)
- **Output**: 19 classes (10 digits + 9 shapes)
- **Augmentation**: Rotation, Affine, Perspective, ColorJitter (balanced để giữ shape edges)

**Training Process:**
- Dataset: ~100,000 images (MNIST + Shapes)
- Epochs: 20
- Optimizer: Adam (lr=1e-4)
- Loss: CrossEntropy
- Validation accuracy: ~99.14%

#### 3. **Pipeline Module** (`pipeline.py`)

**Chức năng:**
- Kết hợp Detection + Classification
- Filter theo target classes (digits/shapes/all)
- Sort detections theo reading order (top-to-bottom, left-to-right)
- Visualize với bounding boxes và labels
- Generate synthetic test images
- MQTT integration cho real-time processing

#### 4. **MQTT Integration**

**Topics:**
- `image/create`: Request generate ảnh synthetic
- `image/input/create`: Response với ảnh đã generate
- `image/input`: Request xử lý ảnh
- `image/output`: Response với kết quả detection

**Flow:**
```
Frontend → image/create → AI generate → image/input/create → Frontend
Frontend → image/input → AI process → image/output → Frontend
```

## 🔬 Công nghệ và phương pháp

### Deep Learning

- **EfficientNet-B0**: CNN architecture tối ưu về accuracy/efficiency
- **Transfer Learning**: Pre-trained trên ImageNet, fine-tune trên custom dataset
- **Data Augmentation**: Tăng diversity của training data

### Computer Vision

- **Contour Detection**: Tìm boundaries của objects
- **Adaptive Thresholding**: Tự động điều chỉnh threshold theo local regions
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **Morphological Operations**: Làm sạch và tách objects

### Text Detection

- **CRAFT**: Character Region Awareness For Text detection
- **Region Proposal**: Tìm regions có khả năng chứa text
- **Link Prediction**: Kết nối các characters thành words

### Preprocessing

- **Denoising**: Loại bỏ noise trong ảnh
- **Contrast Enhancement**: Tăng độ tương phản
- **Illumination Correction**: Chuẩn hóa ánh sáng
- **Normalization**: Chuẩn hóa pixel values

## 📊 Dataset

### MNIST Digits
- **Số lượng**: 60,000 training images
- **Format**: 28x28 grayscale
- **Classes**: 10 (0-9)
- **Source**: MNIST Competition dataset

### Shapes Dataset
- **Số lượng**: ~90,000 images
- **Format**: Various sizes, grayscale
- **Classes**: 9 (Circle, Triangle, Square, Pentagon, Hexagon, Heptagon, Octagon, Nonagon, Star)
- **Generation**: Synthetic với random transformations

### Training Strategy
- **Balanced Sampling**: 67% shapes để balance với MNIST
- **Train/Val Split**: 85/15 với stratification
- **Total Training**: ~100,000 images
- **Total Validation**: ~18,000 images

## 🎯 Tính năng nổi bật

### 1. Unified Classification
- ✅ Một model duy nhất cho 19 classes
- ✅ Không cần separate models cho digits và shapes
- ✅ Dễ maintain và deploy

### 2. Flexible Detection
- ✅ **Traditional CV**: Nhanh, tốt cho shapes
- ✅ **CRAFT**: Tốt cho digits và text
- ✅ **Hybrid**: Tối ưu cho cả hai

### 3. Target Filtering
- ✅ Chỉ detect digits: `--target digits`
- ✅ Chỉ detect shapes: `--target shapes`
- ✅ Detect cả hai: `--target all`

### 4. Synthetic Data Generation
- ✅ Tự động tạo test images
- ✅ Control số lượng digits và shapes
- ✅ Không overlap giữa các objects
- ✅ Ground truth labels

### 5. MQTT Real-time Processing
- ✅ Nhận ảnh từ frontend qua MQTT
- ✅ Xử lý và trả kết quả real-time
- ✅ Base64 encoding cho images
- ✅ JSON format cho results

### 6. Reading Order Sorting
- ✅ Sort detections theo thứ tự đọc tự nhiên
- ✅ Top-to-bottom, left-to-right
- ✅ Group objects vào rows

## 📈 Kết quả và Performance

### Classification Accuracy

| Category | Training | Validation | Notes |
|----------|----------|------------|-------|
| **Overall** | 99.3% | **99.14%** | 19 classes combined |
| **Digits (0-9)** | 99.5% | 99.3% | High accuracy |
| **Shapes** | 99.0% | 98.5% | Good, some confusion Circle/Nonagon |
| **Best Class** | - | 99.90% | Digit "1", Triangle |
| **Worst Class** | - | 94.69% | Nonagon (confused with Circle) |

### Per-Class Performance

**Top Performers:**
- Digit "1": 99.90%
- Digit "8": 99.89%
- Triangle: 99.90%
- Star: 99.70%

**Challenging Classes:**
- Nonagon: 94.69% (confused with Circle ~4%)
- Octagon: 97.96% (confused with Circle ~0.78%)

### Inference Speed

| Component | Time (ms) | Notes |
|-----------|-----------|-------|
| **Detection (Traditional)** | 50-100 | Fast, CPU-friendly |
| **Detection (CRAFT)** | 100-200 | Slower, requires GPU |
| **Detection (Hybrid)** | 150-250 | Combines both |
| **Classification (per object)** | 5-10 | EfficientNet-B0 |
| **Total (5 objects)** | 100-300 | End-to-end |

*Tested on RTX 4050 Laptop GPU*

### Model Size

- **EfficientNet-B0**: ~5.3M parameters
- **Model weights**: ~20MB (.pth file)
- **CRAFT weights**: ~85MB
- **Total**: ~105MB

## 🔄 Flow hoạt động chi tiết

### 1. Training Flow

```
Load Datasets (MNIST + Shapes)
    ↓
Create Label Mapping (0-18)
    ↓
Split Train/Val (85/15)
    ↓
Apply Augmentation
    ↓
Train EfficientNet-B0
    ↓
Validate & Save Best Model
    ↓
Evaluate Performance
```

### 2. Inference Flow

```
Input Image
    ↓
Preprocessing (Denoise, CLAHE, etc.)
    ↓
Detection (Traditional/CRAFT/Hybrid)
    ↓
Crop Bounding Boxes
    ↓
Resize to 128x128
    ↓
Classification (EfficientNet-B0)
    ↓
Filter by Target Classes
    ↓
Sort by Reading Order
    ↓
Visualize & Output JSON
```

### 3. MQTT Flow

```
Frontend → image/create (numberDigit, numberShape)
    ↓
AI: Generate Synthetic Image
    ↓
AI → image/input/create (image base64 + count)
    ↓
Frontend: Display Image
    ↓
User: Click "Process"
    ↓
Frontend → image/input (image base64 + label + count)
    ↓
AI: Detect & Classify (Auto Hybrid if count exists)
    ↓
AI → image/output (image base64 + detections JSON)
    ↓
Frontend: Display Results
```

## 📖 Mô tả chi tiết

Dự án này xây dựng một hệ thống hoàn chỉnh để nhận diện và phân loại:
- **10 chữ số**: 0-9 (từ MNIST dataset)
- **9 hình học**: Circle, Triangle, Square, Pentagon, Hexagon, Heptagon, Octagon, Nonagon, Star

### Pipeline

```
Input Image → Detection (Traditional CV/CRAFT) → Classification (EfficientNet-B0) → Output (Annotated Image + JSON)
```

### Đặc điểm nổi bật

- ✅ **19 classes**: Digits (0-9) + Shapes (9 loại)
- ✅ **Độ chính xác cao**: ~99% validation accuracy
- ✅ **Inference nhanh**: ~100-300ms/ảnh
- ✅ **Linh hoạt**: Hỗ trợ nhiều phương pháp detection
- ✅ **Dễ sử dụng**: API đơn giản và rõ ràng
- ✅ **MQTT Integration**: Real-time processing với frontend
- ✅ **Synthetic Data Generation**: Tự động tạo test images

## 🚀 Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- CUDA 11.8+ (optional, cho GPU acceleration)
- RAM: 8GB+ (16GB recommended)
- Disk space: ~5GB (cho datasets và models)

### Bước 1: Clone repository

```bash
# Clone project từ GitHub
git clone https://github.com/your-username/BTL_XLA.git
cd BTL_XLA
```

### Bước 2: Setup Conda Environment (Khuyến nghị)

```bash
# Tạo conda environment mới
conda create -n btl_xla python=3.10 -y

# Activate environment
conda activate btl_xla

# Cài đặt PyTorch với CUDA (nếu có GPU)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Cài đặt các dependencies còn lại
pip install -r requirements.txt
```

**Hoặc nếu chỉ dùng CPU:**

```bash
conda create -n btl_xla python=3.10 -y
conda activate btl_xla
conda install pytorch torchvision cpuonly -c pytorch -y
pip install -r requirements.txt
```

### Bước 3: Tải và chuẩn bị dữ liệu

#### 3.1. MNIST Dataset

```bash
# Giải nén mnist_competition.zip (nếu có)
unzip mnist_competition.zip

# Hoặc tải từ Kaggle/Google Drive
# Cấu trúc: mnist_competition/train/ và mnist_competition/train_label.csv
```

#### 3.2. Shapes Dataset

```bash
# Giải nén dataset trong Shapes_Classifier
cd Shapes_Classifier
unzip dataset.zip
cd ..
```

#### 3.3. CRAFT Weights (cho Hybrid Detector)

```bash
# Tạo thư mục weights
mkdir weights

# Tải CRAFT weights
wget https://drive.google.com/uc?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ -O weights/craft_mlt_25k.pth

# Hoặc dùng gdown
pip install gdown
gdown https://drive.google.com/uc?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ -O weights/craft_mlt_25k.pth
```

### Chuẩn bị dữ liệu hoàn tất

Đảm bảo cấu trúc thư mục như sau:

```
BTL_XLA/
├── mnist_competition/
│   ├── train/              # 60,000 MNIST images
│   ├── train_label.csv
│   └── public_test/
├── Shapes_Classifier/
│   └── dataset/output/     # 90,000 shape images (Circle, Square, etc.)
├── weights/
│   └── craft_mlt_25k.pth   # CRAFT pretrained weights (~85MB)
├── unified_model_19classes_best.pth  # Trained classifier
└── label_mapping.json
```

## 📚 Hướng dẫn sử dụng

### 1. Training Classifier

Train mô hình EfficientNet-B0 trên 19 classes:

#### Sử dụng Python Script

```bash
python train_unified_classifier.py --epochs 20 --batch-size 64
```

**Tham số:**
- `--epochs`: Số epoch (mặc định: 20)
- `--batch-size`: Batch size (mặc định: 64)
- `--lr`: Learning rate (mặc định: 1e-4)
- `--device`: 'cuda' hoặc 'cpu'

#### Sử dụng Jupyter Notebook

```bash
jupyter notebook train_unified_classifier.ipynb
```

**Output:**
- `unified_model_19classes_best.pth`: Model đã train
- `label_mapping.json`: Ánh xạ class labels
- `training_history.png`: Biểu đồ loss/accuracy

### 2. Pipeline - Inference trên ảnh

#### 2.1. Xử lý ảnh có sẵn (tất cả classes)

```bash
python pipeline.py --image Sample.png --output Sample_result.png
```

#### 2.2. Chỉ nhận diện SHAPES

```bash
python pipeline.py --image Sample.png --target shapes --output Sample_shapes_only.png
```

#### 2.3. Chỉ nhận diện DIGITS

```bash
python pipeline.py --image Sample.png --target digits --output Sample_digits_only.png
```

#### 2.4. Sử dụng Hybrid Detector (CRAFT + Traditional CV)

```bash
python pipeline.py --image Sample.png --detector hybrid --target all
```

#### 2.5. Tạo ảnh test synthetic tự động

```bash
# Tạo ảnh với 5 objects (mặc định)
python pipeline.py --generate

# Tạo ảnh với 10 objects
python pipeline.py --generate --num-objects 10

# Tạo và chỉ detect shapes
python pipeline.py --generate --num-objects 8 --target shapes
```

**Pipeline Output:**
- `*_result.png`: Ảnh được annotate với bounding boxes
- `*_result.json`: Kết quả detection ở định dạng JSON

**Pipeline Arguments:**

| Argument | Choices | Default | Mô tả |
|----------|---------|---------|-------|
| `--image` | path | None | Đường dẫn ảnh input |
| `--output` | path | Auto | Đường dẫn ảnh output |
| `--target` | `digits`, `shapes`, `all` | `all` | Loại objects cần detect |
| `--detector` | `traditional`, `hybrid` | `traditional` | Phương pháp detection |
| `--generate` | flag | False | Tạo ảnh test synthetic |
| `--num-objects` | int | 5 | Số objects trong synthetic scene |
| `--model` | path | `unified_model_19classes_best.pth` | Model weights |
| `--labels` | path | `label_mapping.json` | Label mapping |
| `--device` | `cuda`, `cpu` | Auto | Device để inference |

### 3. Evaluation

Đánh giá hiệu năng model:

```bash
python evaluate_model.py
```

**Output:**
- Per-class accuracy report
- Confusion matrix
- Classification report
- `per_class_performance.csv`

### 4. Sử dụng như một module

```python
from pipeline import UnifiedPipeline

# Khởi tạo pipeline - Detect ALL
pipeline = UnifiedPipeline(
    model_path='unified_model_19classes_best.pth',
    label_mapping_path='label_mapping.json',
    device='cuda',  # hoặc 'cpu'
    detector_type='traditional',  # hoặc 'hybrid'
    target_classes='all'  # 'digits', 'shapes', hoặc 'all'
)

# Xử lý ảnh
results = pipeline.process_file('test_image.png')

# Kết quả
print(f"Detected {len(results['labels'])} objects")
for label, conf in zip(results['labels'], results['confidences']):
    print(f"Class: {label}, Confidence: {conf:.2%}")
```

#### Tạo synthetic data

```python
from pipeline import generate_synthetic_scene
import cv2

# Tạo scene với 10 random objects
canvas, ground_truth = generate_synthetic_scene(
    mnist_dir='mnist_competition/train',
    shapes_dir='Shapes_Classifier/dataset/output',
    mnist_csv='mnist_competition/train_label.csv',
    num_objects=10,
    canvas_size=(800, 600),
    seed=42
)

# Lưu ảnh
cv2.imwrite('my_test_scene.png', canvas)

# In ground truth
print("Ground truth labels:", [item[4] for item in ground_truth])
```

## 📁 Cấu trúc dự án

```
BTL_XLA/
├── mnist_competition/              # MNIST dataset
│   ├── train/                      # Training images
│   ├── public_test/                # Test images
│   └── train_label.csv             # Labels
├── Shapes_Classifier/              # Shapes dataset
│   └── dataset/output/             # Shape images
├── train_unified_classifier.py     # Training script
├── detect_objects.py               # Detection module
├── pipeline.py                     # End-to-end pipeline
├── preprocess_grid_image.py        # Preprocessing utilities
├── unified_model_19classes_best.pth    # Trained model
├── label_mapping.json              # Class mapping
├── requirements.txt                # Dependencies
├── CRAFT_SHAPES_GUIDE.md          # Detailed guide
└── README.md                       # This file
```

## 🎯 Class Mapping

| Class ID | Label | Category |
|----------|-------|----------|
| 0-9 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 | Digits |
| 10 | Circle | Shape |
| 11 | Heptagon | Shape |
| 12 | Hexagon | Shape |
| 13 | Nonagon | Shape |
| 14 | Octagon | Shape |
| 15 | Pentagon | Shape |
| 16 | Square | Shape |
| 17 | Star | Shape |
| 18 | Triangle | Shape |

## 📊 Hiệu năng

### Classification Accuracy

| Dataset | Training | Validation |
|---------|----------|------------|
| MNIST Digits | 99.5% | 99.3% |
| Shapes | 99.0% | 98.5% |
| **Unified (19 classes)** | **99.3%** | **99.0%** |

### Inference Speed

| Component | Time (ms) |
|-----------|-----------|
| Detection | 50-150 |
| Classification | 5-10 per object |
| **Total** | **100-300** |

*Tested on RTX 4050*

## 🔧 Advanced Usage

### Custom Detection Parameters

```python
from detect_objects import TraditionalDetector

detector = TraditionalDetector(
    min_area=200,
    max_area=30000,
    aspect_ratio_range=(0.2, 5.0)
)

bboxes = detector.detect(image)
```

### Synthetic Data Generation

```python
from pipeline import generate_synthetic_scene

canvas, ground_truth = generate_synthetic_scene(
    mnist_dir='mnist_competition/train',
    shapes_dir='Shapes_Classifier/dataset/output',
    mnist_csv='mnist_competition/train_label.csv',
    num_objects=5,
    canvas_size=(800, 600),
    seed=42
)
```

## 🔄 Hướng dẫn Push/Pull với GitHub (Sử dụng Conda)

### Lần đầu push lên GitHub

#### Bước 1: Tạo repository trên GitHub

1. Vào [GitHub](https://github.com)
2. Click **New repository**
3. Đặt tên: `BTL_XLA`
4. Chọn **Public** hoặc **Private**
5. **KHÔNG** chọn "Initialize with README"
6. Click **Create repository**

#### Bước 2: Setup Git local (nếu chưa có)

```bash
# Kiểm tra Git đã cài chưa
git --version

# Nếu chưa có, cài Git
# Windows: Download từ https://git-scm.com/
# Linux: sudo apt install git
# macOS: brew install git

# Config thông tin (chỉ cần 1 lần)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

#### Bước 3: Khởi tạo Git repository

```bash
# Activate conda environment
conda activate btl_xla

# Di chuyển vào thư mục project
cd D:\BTL_XLA

# Khởi tạo Git repository
git init

# Kiểm tra status
git status
```

#### Bước 4: Add files và commit

```bash
# Add tất cả files (theo .gitignore)
git add .

# Kiểm tra những gì sẽ commit
git status

# Commit lần đầu
git commit -m "Initial commit: Unified Digits & Shapes Recognition System"
```

#### Bước 5: Kết nối với GitHub và push

```bash
# Thêm remote repository (thay YOUR_USERNAME bằng username GitHub của bạn)
git remote add origin https://github.com/YOUR_USERNAME/BTL_XLA.git

# Kiểm tra remote
git remote -v

# Push lên GitHub (branch main)
git branch -M main
git push -u origin main
```

**Lưu ý về việc push:**
- Theo `.gitignore`, những thứ SAU sẽ được push:
  - ✅ `craft_repo/` (full folder)
  - ✅ `mnist_competition.zip` (file nén)
  - ✅ `mnist_competition/*.csv` (các file CSV)
  - ✅ `Shapes_Classifier/` (trừ folder `dataset/`)
  - ✅ `weights/craft_mlt_25k.pth`
  - ✅ `unified_model_19classes_best.pth`
  - ✅ Tất cả `.py`, `.ipynb`, `.md`, `requirements.txt`
  - ✅ `Sample.png`, `label_mapping.json`

- Những thứ SAU sẽ KHÔNG push (đã bị ignore):
  - ❌ `mnist_competition/train/` (60,000 ảnh)
  - ❌ `mnist_competition/public_test/` (10,000 ảnh)
  - ❌ `Shapes_Classifier/dataset/` (90,000 ảnh)
  - ❌ `__pycache__/`, `.ipynb_checkpoints/`
  - ❌ `*_result.png`, `*_result.json`
  - ❌ `Test_*.png`, `Test_*.jpg`

### Khi muốn update code (push thay đổi mới)

```bash
# Activate environment
conda activate btl_xla

# Kiểm tra thay đổi
git status

# Add files đã thay đổi
git add .

# Commit với message mô tả
git commit -m "Update: Improved detection accuracy"

# Push lên GitHub
git push origin main
```

### Khi muốn tải code mới (pull từ GitHub)

```bash
# Pull code mới nhất
git pull origin main

# Nếu bị conflict, Git sẽ báo - cần resolve manually
```

### Clone project từ GitHub (cho máy khác)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/BTL_XLA.git
cd BTL_XLA

# Setup conda environment
conda create -n btl_xla python=3.10 -y
conda activate btl_xla

# Cài đặt dependencies
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt

# Giải nén datasets
unzip mnist_competition.zip
cd Shapes_Classifier
unzip dataset.zip
cd ..

# Chạy pipeline
python pipeline.py --generate --num-objects 5
```

### Git Commands thường dùng

```bash
# Xem lịch sử commit
git log --oneline

# Xem thay đổi chưa commit
git diff

# Hủy thay đổi chưa add
git restore filename.py

# Tạo branch mới
git checkout -b feature/new-feature

# Chuyển branch
git checkout main

# Merge branch
git merge feature/new-feature

# Xem tất cả branches
git branch -a
```

## 🐛 Troubleshooting

### Lỗi: Model không load được

```bash
# Kiểm tra PyTorch version
python -c "import torch; print(torch.__version__)"

# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Lỗi: Out of memory

- Giảm batch size: `--batch-size 32`
- Sử dụng CPU: `--device cpu`
- Giảm resolution của ảnh input

### Detection rate thấp

- Điều chỉnh threshold: `min_area=100`
- Thử detector khác: `--detector hybrid`

## 📄 Tài liệu tham khảo

- [CRAFT Paper](https://arxiv.org/abs/1904.01941) - Character Region Awareness For Text detection
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946) - Efficient Convolutional Neural Networks
- [CRAFT GitHub](https://github.com/clovaai/CRAFT-pytorch) - Official CRAFT implementation
- [Detailed Guide](CRAFT_SHAPES_GUIDE.md) - Hướng dẫn chi tiết

## 🤝 Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng:

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📝 License

Dự án này được phát triển cho mục đích giáo dục.

## 👥 Tác giả

- Đồ án môn Xử lý ảnh (Image Processing)
- Trường Đại học...

## 🙏 Acknowledgments

- MNIST Dataset
- Shapes Dataset
- CRAFT-pytorch
- EfficientNet

---

**⭐ Nếu project hữu ích, đừng quên cho một star nhé!**

## 📞 Liên hệ

Nếu có câu hỏi hoặc vấn đề, vui lòng tạo [Issue](../../issues) trên GitHub.

