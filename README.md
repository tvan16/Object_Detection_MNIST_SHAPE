# 🎯 Unified Digits & Shapes Recognition System

Hệ thống nhận diện chữ số viết tay và hình học trong ảnh sử dụng Deep Learning với kiến trúc Detection + Classification.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Mô tả

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

## 🚀 Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- CUDA (optional, cho GPU acceleration)
- RAM: 8GB+ (16GB recommended)

### Cài đặt dependencies

```bash
# Clone repository
git clone <your-repo-url>
cd BTL_XLA

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### Chuẩn bị dữ liệu

Đảm bảo cấu trúc thư mục như sau:

```
BTL_XLA/
├── mnist_competition/
│   ├── train/              # 60,000 MNIST images
│   └── train_label.csv
└── Shapes_Classifier/
    └── dataset/output/     # 90,000 shape images
```

## 📚 Hướng dẫn sử dụng

### 1. Training Classifier

Train mô hình EfficientNet-B0 trên 19 classes:

```bash
python train_unified_classifier.py --epochs 10 --batch-size 64
```

**Tham số:**
- `--epochs`: Số epoch (mặc định: 10)
- `--batch-size`: Batch size (mặc định: 64)
- `--lr`: Learning rate (mặc định: 1e-4)

**Output:**
- `unified_model_19classes_best.pth`: Model đã train
- `label_mapping.json`: Ánh xạ class labels

### 2. Inference trên ảnh

#### Xử lý ảnh có sẵn

```bash
python pipeline.py --image your_image.png --output result.png
```

#### Tạo ảnh test synthetic

```bash
python pipeline.py --generate --num-objects 7
```

**Output:**
- `result.png`: Ảnh được annotate với bounding boxes
- `result.json`: Kết quả detection ở định dạng JSON

### 3. Sử dụng như một module

```python
from pipeline import UnifiedPipeline

# Khởi tạo pipeline
pipeline = UnifiedPipeline(
    model_path='unified_model_19classes_best.pth',
    label_mapping_path='label_mapping.json',
    device='cuda'  # hoặc 'cpu'
)

# Xử lý ảnh
results = pipeline.process_file('test_image.png')

# Kết quả
print(f"Detected {len(results['labels'])} objects")
for label, conf in zip(results['labels'], results['confidences']):
    print(f"Class: {label}, Confidence: {conf:.2%}")
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

