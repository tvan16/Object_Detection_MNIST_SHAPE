# BỘ KHOA HỌC VÀ CÔNG NGHỆ

## HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG

---

# BÁO CÁO BÀI TẬP LỚN

**HỌC PHẦN:** XỬ LÝ ẢNH

**Đề tài:** Nhận dạng chữ số và hình học đơn giản bằng mạng Neural

**Giảng viên:** TS. Phạm Hoàng Việt

**Nhóm 25:**
- B22DCCN482 - Trịnh Quang Lâm
- B22DCCN434 - Vũ Nhân Kiên  
- B22DCCN889 - Vũ Thế Văn

**Link sản phẩm:** [tvan16/Object_Detection_MNIST_SHAPE](https://github.com/tvan16/Object_Detection_MNIST_SHAPE)

**Hà Nội, 11/2025**

---

## 📋 MỤC LỤC

1. [Giới thiệu](#1-giới-thiệu)
2. [Bối cảnh & Tầm quan trọng](#2-bối-cảnh--tầm-quan-trọng)
3. [Động lực chọn MNIST mở rộng](#3-động-lực-chọn-mnist-mở-rộng)
4. [Mục tiêu nghiên cứu](#4-mục-tiêu-nghiên-cứu)
5. [Phạm vi thực hiện](#5-phạm-vi-thực-hiện)
6. [Tổng quan nghiên cứu & Công nghệ](#6-tổng-quan-nghiên-cứu--công-nghệ)
7. [Augmentation & Tiền xử lý](#7-augmentation--tiền-xử-lý)
8. [Kiến trúc mô hình & Công nghệ huấn luyện](#8-kiến-trúc-mô-hình--công-nghệ-huấn-luyện)
9. [Mô tả tập dữ liệu](#9-mô-tả-tập-dữ-liệu)
10. [Thực nghiệm](#10-thực-nghiệm)
11. [Ứng dụng & Triển khai](#11-ứng-dụng--triển-khai)
12. [Hướng cải thiện](#12-hướng-cải-thiện)
13. [Kết luận](#13-kết-luận)
14. [Tài liệu tham khảo](#14-tài-liệu-tham-khảo)

---

## 1. GIỚI THIỆU

Dự án **"Unified Digits & Shapes Recognition System"** là một hệ thống nhận diện đối tượng hoàn chỉnh, có khả năng phát hiện và phân loại đồng thời **chữ số viết tay** (0-9) và **hình học** (9 loại) trong cùng một ảnh. Hệ thống sử dụng kiến trúc **hai giai đoạn** (Two-Stage): **Detection** để tìm vị trí các đối tượng, sau đó **Classification** để nhận diện loại của từng đối tượng.

## 2. BỐI CẢNH & TẦM QUAN TRỌNG

Trong bối cảnh làn sóng ứng dụng thị giác máy tính đang lan rộng sang nhiều lĩnh vực như xe tự hành, sản xuất thông minh và công nghệ giáo dục, yêu cầu về những mô hình vừa nhẹ vừa chính xác trở nên cấp thiết hơn bao giờ hết. Các hệ thống triển khai trong môi trường thực, đặc biệt trên thiết bị nhúng hoặc biên, thường bị giới hạn tài nguyên tính toán nên không thể sử dụng các kiến trúc quá cồng kềnh, trong khi vẫn phải đảm bảo độ tin cậy đủ cao cho các tác vụ nhận dạng và quyết định tự động. Điều này đặt ra nhu cầu nghiên cứu các mô hình tối giản nhưng hiệu quả, có khả năng cân bằng giữa độ phức tạp, hiệu năng và khả năng triển khai.

Tập dữ liệu MNIST truyền thống từ lâu đã được xem như chuẩn mực cơ bản để đánh giá các thuật toán nhận dạng chữ số viết tay. Tuy nhiên, bài toán gốc chỉ dừng lại ở việc phân loại các chữ số đơn lẻ, trên nền ảnh đơn giản, nên chưa phản ánh đầy đủ những thách thức của các kịch bản thị giác máy tính ngoài đời thực, nơi mô hình cần xử lý nhiều đối tượng, bố cục phức tạp và các mối quan hệ không gian – hình học giữa các thành phần trong ảnh. Do đó, MNIST ở dạng nguyên bản không còn đủ để đánh giá năng lực của các kiến trúc hiện đại vốn hướng tới ứng dụng trong môi trường động, đa đối tượng.

Trong bối cảnh giáo dục và sáng tạo số, thị giác máy tính được sử dụng cho nhiều nhiệm vụ như theo dõi mức độ tương tác của người học, hỗ trợ học tập cá nhân hóa, xây dựng lớp học thông minh hay tạo nội dung học liệu trực quan. Những hệ thống như vậy thường phải xử lý các cảnh phức tạp với nhiều biểu tượng, vật thể học tập hoặc tương tác của người học trong không gian lớp học vật lý hoặc ảo. Vì vậy, việc mở rộng bài toán từ nhận dạng chữ số đơn lẻ sang phát hiện và định vị nhiều đối tượng trong một khung hình có ý nghĩa thiết thực, giúp mô hình tiến gần hơn với các bài toán thực tế của EdTech.

Một hướng nghiên cứu quan trọng là thiết kế các biến thể mở rộng của MNIST, trong đó các chữ số được kết hợp, sắp xếp theo cấu trúc hình học hoặc đặt trong những bố cục đa đối tượng, nhằm mô phỏng các tình huống chiến lược trong môi trường giáo dục và sáng tạo. Các tập dữ liệu như vậy cho phép đánh giá khả năng của mô hình trong việc phát hiện, phân tách và hiểu quan hệ giữa các đối tượng, đồng thời vẫn duy trì kích thước dữ liệu vừa phải để phù hợp cho việc thử nghiệm các kiến trúc nhẹ. Nhờ đó, người nghiên cứu có thể khảo sát sâu hơn cách tối ưu mạng nơ-ron cho những hệ thống thị giác máy tính áp dụng trong lớp học thông minh, trò chơi giáo dục hay công cụ hỗ trợ sáng tạo, nơi ràng buộc về tài nguyên và độ trễ là những yếu tố then chốt.

## 3. ĐỘNG LỰC CHỌN MNIST MỞ RỘNG

Việc lựa chọn MNIST làm nền tảng để mở rộng xuất phát từ chính tính biểu tượng của bộ dữ liệu này trong cộng đồng học máy và thị giác máy tính. MNIST đã được nghiên cứu rất kỹ, có tài liệu phong phú và nhiều ví dụ mã nguồn, nên việc tái lập thí nghiệm, so sánh mô hình và đánh giá cải tiến trở nên thuận lợi, đặc biệt cho mục đích giảng dạy và thử nghiệm nhanh các ý tưởng mới. Nhờ đó, mọi thay đổi trên MNIST mở rộng đều có thể đặt trong bối cảnh một chuẩn tham chiếu quen thuộc, giúp kết quả nghiên cứu dễ diễn giải và chia sẻ với cộng đồng.

Bên cạnh đó, cấu trúc ảnh đơn giản (thang xám 28×28) cho phép dễ dàng tùy biến để kết hợp chữ số với các hình dạng hình học như đường thẳng, hình tròn, hình đa giác, hoặc sắp xếp nhiều chữ số trong cùng một khung hình, tạo nên các "mini real-world" mô phỏng bảng điểm, ô bài tập hoặc giao diện trò chơi cho học sinh. Những bố cục này giúp chuyển bài toán từ phân loại đơn đối tượng sang phát hiện, định vị và hiểu quan hệ không gian giữa nhiều đối tượng, gần hơn với các kịch bản EdTech và game hóa học tập.

Một ưu điểm quan trọng khác là MNIST cho phép kiểm soát dữ liệu ở mức cao, từ đó có thể chủ động đưa vào các dạng nhiễu, chồng chéo đối tượng, biến đổi affine (quay, tịnh tiến, co giãn, biến dạng phối cảnh) hay thay đổi độ tương phản và độ sáng. Khả năng kiểm soát này giúp xây dựng các bộ dữ liệu "có chủ đích", trong đó từng yếu tố khó khăn được gia tăng có kế hoạch để đánh giá độ bền vững của mô hình, đo lường khả năng khái quát hóa trong điều kiện gần với thế giới thực nhưng vẫn an toàn, rẻ và dễ triển khai trong môi trường giáo dục.

## 4. MỤC TIÊU NGHIÊN CỨU

Nghiên cứu hướng tới xây dựng một pipeline thống nhất cho bài toán MNIST mở rộng, bao trùm toàn bộ các bước từ tiền xử lý dữ liệu, tạo mẫu đến huấn luyện và suy luận, với khả năng phát hiện đồng thời cả chữ số và các hình dạng hình học trong cùng một khung hình. Pipeline này được thiết kế sao cho có thể áp dụng lại dễ dàng cho các biến thể dữ liệu khác nhau, nhưng vẫn giữ cách tổ chức rõ ràng giữa các khối chức năng như tạo dữ liệu, huấn luyện mô hình và đánh giá kết quả. Một mục tiêu quan trọng là duy trì sự cân bằng hợp lý giữa độ chính xác và tốc độ, nhằm đảm bảo mô hình không chỉ đạt hiệu năng nhận dạng tốt trên bộ dữ liệu MNIST mở rộng mà còn có độ trễ thấp, phù hợp với yêu cầu triển khai trong các hệ thống thực tế như ứng dụng giáo dục tương tác hoặc trò chơi học tập. Trong bối cảnh tài nguyên tính toán bị giới hạn trên thiết bị biên, việc tối ưu mô hình và pipeline để đạt được sự đánh đổi hiệu quả giữa chi phí tính toán và chất lượng dự đoán là tiêu chí then chốt.

Bên cạnh đó, nghiên cứu đặt mục tiêu cung cấp bộ công cụ có khả năng tái lập cao dưới dạng script và notebook, cho phép người dùng dễ dàng tải dữ liệu, huấn luyện lại mô hình, điều chỉnh siêu tham số và đánh giá kết quả. Các tài liệu và mã nguồn đi kèm được tổ chức theo hướng thân thiện với cộng đồng, giúp sinh viên, nhà nghiên cứu hoặc nhà phát triển có thể nhanh chóng mở rộng, so sánh và tích hợp pipeline này vào những bài toán thị giác máy tính khác nhau trong môi trường giáo dục và sáng tạo.

## 5. PHẠM VI THỰC HIỆN

Trong khuôn khổ nghiên cứu này, dữ liệu đầu vào được nhóm tự sinh và tái cấu trúc, được quản lý tập trung trong thư mục dataset/, nhằm đảm bảo khả năng kiểm soát tốt quá trình tạo mẫu, gắn nhãn và tái lập thí nghiệm. Cách tiếp cận này giúp dễ dàng điều chỉnh các tham số sinh dữ liệu như phân bố vị trí, mức nhiễu hay mật độ đối tượng, đồng thời thuận lợi cho việc chia tách tập huấn luyện, kiểm thử và đánh giá.

Đề tài chỉ tập trung vào các hình dạng hình học cơ bản như hình tròn, tam giác, hình vuông (và một số biến thể đơn giản nếu có) cùng với các chữ số 0–9, qua đó giữ cho không gian lớp nhãn đủ đơn giản để phân tích nhưng vẫn đủ đa dạng để mô phỏng các kịch bản đa đối tượng. Việc giới hạn này giúp làm rõ tác động của thiết kế mô hình và pipeline lên bài toán phát hiện kết hợp shape + digit, tránh bị nhiễu bởi quá nhiều loại đối tượng khác nhau.

Về mặt triển khai, nghiên cứu giả định môi trường tính toán là các GPU phổ thông thường gặp trong phòng lab hoặc máy trạm, không đi sâu vào các tối ưu hóa phần cứng chuyên biệt cho IoT hoặc thiết bị edge. Những vấn đề như nén mô hình cực mạnh, triển khai trên vi điều khiển, hoặc tích hợp với hệ thống nhúng chỉ được đề cập ở mức định hướng tương lai, nhằm giữ phạm vi thực hiện phù hợp với nguồn lực và mục tiêu đánh giá mô hình trong bối cảnh học thuật.

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

---

## 📚 CHI TIẾT KỸ THUẬT VÀ GIẢI THÍCH

### 1. TẠI SAO DÙNG PRETRAINED MODEL? PRETRAINED MODEL CÓ GÌ ĐẶC BIỆT?

#### 1.1. Lý do sử dụng Pretrained Model

**Transfer Learning - Học chuyển giao:**
- **Định nghĩa**: Sử dụng kiến thức đã học từ một task lớn (ImageNet) để áp dụng vào task mới (nhận diện digits/shapes)
- **Lợi ích**:
  1. **Tiết kiệm thời gian training**: Thay vì train từ đầu (cần hàng triệu ảnh và hàng tuần), chỉ cần fine-tune vài giờ
  2. **Cần ít dữ liệu hơn**: Với pretrained model, chỉ cần ~100K ảnh thay vì hàng triệu ảnh
  3. **Đạt accuracy cao hơn**: Model đã học được các features cơ bản (edges, textures, shapes) từ ImageNet
  4. **Tránh overfitting**: Với dataset nhỏ, train từ đầu dễ bị overfitting

**So sánh:**
```
Train từ đầu:  100K ảnh → Accuracy ~85-90% (cần nhiều epochs)
Pretrained:    100K ảnh → Accuracy ~99% (chỉ cần 20 epochs)
```

#### 1.2. EfficientNet-B0 Pretrained trên ImageNet - Đặc điểm gì?

**ImageNet Dataset:**
- **Quy mô**: 1.2 triệu ảnh, 1000 classes
- **Đa dạng**: Động vật, đồ vật, thực phẩm, phương tiện, v.v.
- **Chất lượng**: Được label cẩn thận, đa dạng về góc chụp, ánh sáng, background

**EfficientNet-B0 Architecture:**
- **Compound Scaling**: Tối ưu đồng thời depth, width, và resolution
- **MobileNetV2 blocks**: Depthwise separable convolutions (hiệu quả hơn)
- **Squeeze-and-Excitation**: Attention mechanism để tập trung vào features quan trọng
- **Swish activation**: f(x) = x * sigmoid(x) - tốt hơn ReLU

**Features đã học được từ ImageNet:**
1. **Low-level features** (tầng đầu):
   - Edge detection (phát hiện cạnh)
   - Texture patterns (mẫu kết cấu)
   - Color blobs (vùng màu)
   
2. **Mid-level features** (tầng giữa):
   - Shapes và contours (hình dạng và đường viền)
   - Parts of objects (bộ phận đối tượng)
   - Spatial relationships (mối quan hệ không gian)

3. **High-level features** (tầng cuối):
   - Object recognition (nhận diện đối tượng)
   - Scene understanding (hiểu cảnh)

**Tại sao phù hợp với digits/shapes?**
- Digits và shapes cũng là **objects** với **edges, contours, shapes**
- Model đã biết cách nhận diện **geometric patterns** từ ImageNet
- Chỉ cần fine-tune classifier layer để phân biệt 19 classes cụ thể

#### 1.3. Fine-tuning Process

**Cách fine-tune:**
```python
# 1. Load pretrained weights
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

# 2. Thay đổi classifier layer (từ 1000 classes → 19 classes)
num_features = model.classifier[1].in_features  # 1280 features
model.classifier[1] = nn.Linear(num_features, 19)  # 19 classes

# 3. Train với learning rate nhỏ (1e-4) để không phá vỡ pretrained weights
optimizer = optim.Adam(model.parameters(), lr=1e-4)
```

**Tại sao learning rate nhỏ?**
- Pretrained weights đã tốt, chỉ cần điều chỉnh nhẹ
- Learning rate lớn sẽ "xóa" kiến thức đã học từ ImageNet
- Learning rate nhỏ giúp model học thêm features mới mà không quên cũ

---

### 2. CHI TIẾT VỀ CONTOUR DETECTION VÀ BOUNDING BOX

#### 2.1. Quy trình Contour Detection

**Bước 1: Preprocessing**
```python
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian Blur để giảm noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Kernel size (5,5): cửa sổ 5x5 pixels
# Sigma=0: tự động tính từ kernel size
```
**Tại sao blur?**
- Loại bỏ noise nhỏ (pixels lỗi, artifacts)
- Làm mịn ảnh để thresholding tốt hơn
- Giảm false positives từ noise

**Bước 2: Adaptive Thresholding**
```python
binary = cv2.adaptiveThreshold(
    blurred, 255,                          # Max value
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,        # Method
    cv2.THRESH_BINARY_INV,                 # Invert (objects = white)
    11,                                     # Block size (11x11)
    2                                       # C constant
)
```

**Adaptive Threshold vs Global Threshold:**
- **Global Threshold**: Dùng 1 giá trị cho toàn ảnh → không tốt với ánh sáng không đều
- **Adaptive Threshold**: Tính threshold riêng cho từng vùng 11x11 pixels

**Cách hoạt động:**
1. Chia ảnh thành các block 11x11 pixels
2. Tính mean của mỗi block
3. Threshold = mean - C (C=2)
4. Nếu pixel > threshold → white (255), ngược lại → black (0)

**Tại sao THRESH_BINARY_INV?**
- Objects (digits/shapes) thường tối trên nền sáng
- Invert để objects thành white (dễ tìm contours)

**Bước 3: Morphological Operations**
```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
```

**Morphological Closing:**
- **Dilation** (giãn nở) → **Erosion** (co lại)
- **Mục đích**: Đóng các lỗ hổng nhỏ trong objects, nối các phần bị đứt

**Ví dụ:**
```
Trước:  [1 0 1]  →  Sau:  [1 1 1]
        [0 0 0]           [0 0 0]
        [1 0 1]           [1 1 1]
```
- Đóng khoảng trống giữa các phần của chữ số "8"

**Bước 4: Find Contours**
```python
contours, _ = cv2.findContours(
    morph, 
    cv2.RETR_EXTERNAL,      # Chỉ lấy contours ngoài cùng
    cv2.CHAIN_APPROX_SIMPLE # Nén contours (chỉ giữ điểm góc)
)
```

**RETR_EXTERNAL vs RETR_TREE:**
- **RETR_EXTERNAL**: Chỉ lấy contours ngoài cùng (không lấy lỗ hổng bên trong)
- **RETR_TREE**: Lấy tất cả contours (bao gồm cả lỗ hổng)

**CHAIN_APPROX_SIMPLE vs CHAIN_APPROX_NONE:**
- **SIMPLE**: Nén contours, chỉ giữ điểm góc → tiết kiệm memory
- **NONE**: Giữ tất cả điểm → chính xác hơn nhưng tốn memory

**Bước 5: Extract Bounding Boxes**
```python
for contour in contours:
    area = cv2.contourArea(contour)
    
    # Filter by area
    if min_area < area < max_area:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by aspect ratio
        aspect_ratio = w / float(h)
        if min_ratio < aspect_ratio < max_ratio:
            bboxes.append((x, y, w, h))
```

**cv2.boundingRect(contour):**
- Tìm hình chữ nhật nhỏ nhất bao quanh contour
- Trả về: (x, y, w, h)
  - x, y: Tọa độ góc trên-trái
  - w, h: Chiều rộng và chiều cao

**Filtering:**
- **Area filter**: Loại bỏ noise nhỏ (< min_area) và objects quá lớn (> max_area)
- **Aspect ratio filter**: Loại bỏ objects quá dẹt hoặc quá cao (không phải digits/shapes)

**Ví dụ tham số:**
```python
min_area = 100      # Loại bỏ noise < 100 pixels²
max_area = 50000    # Loại bỏ objects > 50000 pixels²
aspect_ratio = (0.3, 3.0)  # Chấp nhận width/height từ 0.3 đến 3.0
```

#### 2.2. Bounding Box Format

**Format: (x, y, w, h)**
- **x, y**: Tọa độ góc trên-trái của bounding box
- **w, h**: Chiều rộng và chiều cao

**Ví dụ:**
```
Image: 800x600
Bounding box: (100, 50, 200, 150)
→ x=100, y=50, w=200, h=150
→ Góc trên-trái: (100, 50)
→ Góc dưới-phải: (300, 200)
```

**Tại sao dùng (x, y, w, h) thay vì (x1, y1, x2, y2)?**
- Dễ tính toán area: `area = w * h`
- Dễ resize: `new_w = w * scale`
- Chuẩn OpenCV

---

### 3. CHI TIẾT VỀ DATA AUGMENTATION

#### 3.1. Tại sao cần Data Augmentation?

**Vấn đề:**
- Dataset có hạn (100K ảnh)
- Model cần học được tính **invariant** (bất biến) với:
  - Rotation (xoay)
  - Translation (dịch chuyển)
  - Scale (thay đổi kích thước)
  - Lighting (ánh sáng)
  - Perspective (góc nhìn)

**Giải pháp: Data Augmentation**
- Tạo thêm dữ liệu từ dữ liệu có sẵn
- Tăng diversity mà không cần thu thập thêm ảnh
- Giảm overfitting

#### 3.2. Các phương pháp Augmentation được sử dụng

**1. RandomRotation (30°)**
```python
transforms.RandomRotation(30)
```
**Mục đích:**
- Model học được digits/shapes ở mọi góc xoay
- Thực tế: Ảnh có thể bị xoay khi scan/chụp

**Tại sao 30°?**
- Quá lớn (>45°): Digits/shapes khó nhận diện
- Quá nhỏ (<15°): Không đủ diversity
- 30°: Cân bằng tốt

**Ví dụ:**
```
Chữ số "6" xoay 30° → vẫn là "6"
Hình vuông xoay 30° → thành hình thoi (vẫn nhận diện được)
```

**2. RandomAffine (Translation)**
```python
transforms.RandomAffine(
    degrees=0,              # Không xoay (đã có RandomRotation)
    translate=(0.15, 0.15), # Dịch 15% theo x và y
    scale=(0.8, 1.2),       # Scale từ 80% đến 120%
    shear=10                # Shear 10°
)
```

**Translation (0.15, 0.15):**
- Dịch chuyển object 15% theo chiều ngang và dọc
- Mục đích: Model học được object ở mọi vị trí trong ảnh

**Scale (0.8, 1.2):**
- Thay đổi kích thước từ 80% đến 120%
- Mục đích: Model học được object ở mọi kích thước

**Shear (10°):**
- Biến dạng hình học (nghiêng)
- Mục đích: Mô phỏng góc chụp nghiêng

**3. RandomPerspective**
```python
transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
```

**Perspective Transformation:**
- Mô phỏng góc nhìn 3D (như nhìn từ góc nghiêng)
- distortion_scale=0.2: Độ biến dạng 20%
- p=0.5: Chỉ áp dụng 50% ảnh (không quá mạnh)

**Ví dụ:**
```
Hình vuông nhìn từ trên → Hình thang (perspective)
```

**4. ColorJitter**
```python
transforms.ColorJitter(brightness=0.3, contrast=0.3)
```

**Brightness (0.3):**
- Thay đổi độ sáng ±30%
- Mục đích: Model học được với mọi điều kiện ánh sáng

**Contrast (0.3):**
- Thay đổi độ tương phản ±30%
- Mục đích: Model học được với mọi độ tương phản

**Tại sao không dùng Saturation/Hue?**
- Digits/shapes là grayscale → không cần
- Chỉ cần brightness và contrast

**5. Resize (128x128)**
```python
transforms.Resize((128, 128))
```

**Tại sao 128x128?**
- **Tăng từ 64x64**: Để phân biệt tốt hơn các shapes có nhiều cạnh (Nonagon, Octagon)
- **Không quá lớn**: 128x128 đủ để nhận diện, không tốn quá nhiều memory
- **EfficientNet-B0**: Input size mặc định 224x224, nhưng 128x128 vẫn hoạt động tốt

**6. Grayscale → RGB**
```python
transforms.Grayscale(num_output_channels=3)
```

**Tại sao convert grayscale → RGB?**
- EfficientNet-B0 pretrained trên ImageNet (RGB 3 channels)
- Input phải có 3 channels để sử dụng pretrained weights
- Copy grayscale vào 3 channels: R=G=B

**7. Normalization**
```python
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # ImageNet mean
    std=[0.229, 0.224, 0.225]    # ImageNet std
)
```

**Tại sao normalize?**
- Chuẩn hóa pixel values về range [-1, 1]
- Model pretrained đã quen với distribution này
- Giúp training ổn định hơn

**Công thức:**
```
normalized = (pixel - mean) / std
```

#### 3.3. Augmentation Strategy

**Training:**
- Áp dụng TẤT CẢ augmentations
- Mỗi epoch, mỗi ảnh được augment khác nhau
- Tăng diversity tối đa

**Validation:**
- CHỈ resize và normalize
- Không augment để đánh giá chính xác

**Test Time Augmentation (TTA):**
- Áp dụng cho shapes (class_id >= 10)
- Rotations: ±5°, ±10°
- Average probabilities từ các augmentations
- Tăng accuracy inference

---

### 4. CHI TIẾT VỀ CRAFT DETECTOR

#### 4.1. CRAFT là gì?

**CRAFT (Character Region Awareness For Text detection):**
- Deep learning model để detect text/characters trong ảnh
- Pre-trained trên MLT dataset (25k ảnh đa ngôn ngữ)
- Tốt cho: Scene text, rotated text, complex backgrounds

#### 4.2. Cách hoạt động

**Architecture:**
- **Backbone**: VGG16 (feature extractor)
- **Output**: 2 heatmaps
  - **Text Region Map**: Vùng có text
  - **Character Link Map**: Kết nối giữa các characters

**Quy trình:**
```
1. Input image → Resize (giữ aspect ratio, max 1280px)
2. Forward pass qua CRAFT network
3. Output: 2 heatmaps (text regions + character links)
4. Post-processing: Tìm bounding boxes từ heatmaps
5. Adjust coordinates về kích thước gốc
```

**Thresholds:**
- **text_threshold=0.7**: Confidence để xác định vùng có text
- **link_threshold=0.4**: Confidence để kết nối characters
- **low_text=0.4**: Threshold thấp để detect text mờ

#### 4.3. Tại sao dùng CRAFT cho digits?

**Ưu điểm:**
- Tốt với **rotated text** (chữ số xoay)
- Tốt với **complex backgrounds** (nền phức tạp)
- Detect được **small characters** (chữ số nhỏ)

**Nhược điểm:**
- Chậm hơn Traditional CV (100-200ms vs 50-100ms)
- Cần GPU để chạy nhanh
- Model weights lớn (~85MB)

---

### 5. CHI TIẾT VỀ HYBRID DETECTOR

#### 5.1. Chiến lược Hybrid

**Vấn đề:**
- Traditional CV: Tốt cho shapes, nhưng kém với digits nhỏ/xoay
- CRAFT: Tốt cho digits, nhưng không detect shapes tốt

**Giải pháp: Hybrid**
- CRAFT detect digits/text
- Traditional CV detect shapes (sau khi mask out text regions)
- Merge và deduplicate

#### 5.2. Quy trình chi tiết

**Bước 1: CRAFT detect text/digits**
```python
text_bboxes = self.craft_detector.detect(image)
```

**Bước 2: Mask out text regions**
```python
masked_image = self._mask_regions(image, text_bboxes)
# Vẽ white rectangles lên các vùng text
```

**Tại sao mask?**
- Tránh Traditional CV detect lại digits (đã có từ CRAFT)
- Chỉ để lại vùng shapes cho Traditional CV

**Bước 3: Traditional CV detect shapes**
```python
shape_bboxes = self.cv_detector.detect(masked_image)
```

**Bước 4: Merge và NMS**
```python
all_bboxes = self._merge_bboxes(text_bboxes, shape_bboxes)
```

#### 5.3. Non-Maximum Suppression (NMS)

**Mục đích:**
- Loại bỏ overlapping boxes
- Giữ box tốt nhất (thường là box lớn hơn)

**IoU (Intersection over Union):**
```
IoU = (Intersection Area) / (Union Area)
```

**Ví dụ:**
```
Box 1: (100, 100, 200, 200)  # area = 40000
Box 2: (150, 150, 200, 200)  # area = 40000
Intersection: (150, 150, 200, 200)  # area = 2500
Union: (100, 100, 250, 250)  # area = 22500
IoU = 2500 / 22500 = 0.11
```

**NMS Algorithm:**
1. Sort boxes theo area (lớn → nhỏ)
2. Với mỗi box:
   - Tính IoU với các box còn lại
   - Nếu IoU > threshold → loại bỏ box nhỏ hơn
   - Nếu box bị contain hoàn toàn → loại bỏ

**IoU Threshold:**
- **0.5**: Loại bỏ boxes overlap >50%
- **0.2**: Loại bỏ nhiều hơn (cho CRAFT - nhiều overlapping boxes)

---

### 6. CHI TIẾT VỀ TRAINING PROCESS

#### 6.1. Loss Function: CrossEntropyLoss

**Công thức:**
```
Loss = -log(P(correct_class))
```

**Ví dụ:**
```
Predicted probabilities: [0.1, 0.8, 0.05, 0.05]  # 4 classes
True label: 1 (class thứ 2)
Loss = -log(0.8) = 0.223
```

**Tại sao dùng CrossEntropy?**
- Phù hợp với multi-class classification
- Penalize mạnh khi predict sai
- Stable và converge nhanh

#### 6.2. Optimizer: Adam

**Adam (Adaptive Moment Estimation):**
- Kết hợp **Momentum** (tốc độ) và **RMSprop** (adaptive learning rate)
- Tự động điều chỉnh learning rate cho từng parameter

**Ưu điểm:**
- Converge nhanh hơn SGD
- Không cần tune learning rate nhiều
- Phù hợp với sparse gradients

**Learning Rate: 1e-4**
- Nhỏ để fine-tune pretrained weights
- Không phá vỡ features đã học

#### 6.3. Scheduler: ReduceLROnPlateau

**Cách hoạt động:**
- Monitor validation accuracy
- Nếu accuracy không tăng trong 2 epochs (patience=2)
- Giảm learning rate xuống 50% (factor=0.5)

**Tại sao?**
- Khi accuracy plateau → có thể đang ở local minimum
- Giảm LR giúp tìm được minimum tốt hơn
- Fine-tuning tốt hơn

#### 6.4. Batch Size: 64

**Tại sao 64?**
- **Quá nhỏ (<32)**: Gradient không ổn định, training chậm
- **Quá lớn (>128)**: Tốn memory, có thể không fit vào GPU
- **64**: Cân bằng tốt giữa stability và speed

**Memory calculation:**
```
Batch size 64, Image 128x128x3
Memory per image: 128 * 128 * 3 * 4 bytes = 196KB
Memory per batch: 196KB * 64 = 12.5MB
+ Model weights: ~20MB
+ Gradients: ~20MB
Total: ~52.5MB (fit vào GPU 6GB+)
```

#### 6.5. Epochs: 20

**Tại sao 20?**
- Với pretrained model, chỉ cần vài epochs để fine-tune
- Sau epoch 5-10, accuracy đã đạt ~98%
- 20 epochs đảm bảo convergence

**Early Stopping:**
- Lưu model tốt nhất (best validation accuracy)
- Tránh overfitting

---

### 7. CHI TIẾT VỀ POST-PROCESSING

#### 7.1. Target Filtering

**Mục đích:**
- Cho phép user chọn chỉ detect digits, chỉ shapes, hoặc cả hai

**Implementation:**
```python
if target_classes == 'digits':
    return class_id in [0, 1, 2, ..., 9]
elif target_classes == 'shapes':
    return class_id in [10, 11, ..., 18]
else:  # 'all'
    return True
```

#### 7.2. Reading Order Sorting

**Mục đích:**
- Sắp xếp detections theo thứ tự đọc tự nhiên (top-to-bottom, left-to-right)

**Algorithm:**
1. Tính y_center của mỗi box
2. Group boxes vào rows (tolerance = 50% avg height)
3. Sort rows theo y (top → bottom)
4. Sort boxes trong mỗi row theo x (left → right)

**Ví dụ:**
```
Input boxes: [(100, 200), (50, 100), (300, 150), (200, 100)]
After sorting: [(50, 100), (200, 100), (300, 150), (100, 200)]
              Row 1      Row 1      Row 2      Row 3
```

#### 7.3. Test Time Augmentation (TTA)

**Mục đích:**
- Tăng accuracy inference bằng cách average predictions từ nhiều augmentations

**Chỉ áp dụng cho shapes:**
- Digits: Không cần (đã đủ chính xác)
- Shapes: Cần TTA để phân biệt tốt hơn (đặc biệt Nonagon/Octagon/Circle)

**Quy trình:**
```python
if predicted_class >= 10:  # Shape
    # Original prediction
    probs_original = model(crop)
    
    # Rotate +5°
    crop_rot5 = rotate(crop, 5)
    probs_rot5 = model(crop_rot5)
    
    # Rotate -5°
    crop_rot_neg5 = rotate(crop, -5)
    probs_rot_neg5 = model(crop_rot_neg5)
    
    # Average
    final_probs = (probs_original + probs_rot5 + probs_rot_neg5) / 3
```

**Kết quả:**
- Accuracy tăng ~0.5-1% cho shapes
- Trade-off: Inference chậm hơn 3-5x

---

### 8. CÁC CÂU HỎI THƯỜNG GẶP

#### Q1: Tại sao không dùng YOLO/SSD cho detection?

**Trả lời:**
- YOLO/SSD cần train riêng trên dataset có labels (bounding boxes)
- Dataset hiện tại chỉ có class labels, không có bounding box labels
- Traditional CV + CRAFT không cần training, hoạt động out-of-the-box
- Đủ tốt cho use case này (digits/shapes trên nền sáng)

#### Q2: Tại sao không dùng ResNet thay vì EfficientNet?

**Trả lời:**
- EfficientNet tối ưu hơn về accuracy/efficiency trade-off
- Với cùng accuracy, EfficientNet nhỏ hơn và nhanh hơn ResNet
- EfficientNet-B0: ~4M parameters vs ResNet-18: ~11M parameters

#### Q3: Tại sao input size 128x128 thay vì 224x224 (ImageNet standard)?

**Trả lời:**
- 128x128 đủ để nhận diện digits/shapes (objects đơn giản)
- Nhỏ hơn → nhanh hơn, ít memory hơn
- Trade-off: Có thể mất một chút accuracy, nhưng vẫn đạt 99%

#### Q4: Tại sao balanced sampling 67% shapes?

**Trả lời:**
- MNIST: 60K images
- Shapes: 90K images (nhưng chỉ sample 67% = ~60K)
- Balance dataset để model không bias về một class nào
- Nếu không balance: Model có thể học tốt digits nhưng kém shapes (hoặc ngược lại)

#### Q5: Tại sao dùng Grayscale → RGB thay vì train model mới cho grayscale?

**Trả lời:**
- Sử dụng pretrained weights (đã train trên RGB)
- Nếu train model mới cho grayscale → mất lợi ích của pretrained weights
- Grayscale → RGB đơn giản và hiệu quả hơn

---

### 9. TỐI ƯU HÓA VÀ CẢI TIẾN

#### 9.1. Tại sao tăng input size từ 64 → 128?

**Vấn đề với 64x64:**
- Khó phân biệt Nonagon (9 cạnh) và Circle
- Khó phân biệt Octagon (8 cạnh) và Circle
- Edges bị mờ khi resize nhỏ

**Giải pháp: 128x128**
- Giữ được nhiều chi tiết hơn
- Accuracy tăng: Circle 76% → 90%+, Nonagon 73% → 85%+

#### 9.2. Tại sao augmentation mạnh hơn?

**Rotation: 15° → 30°**
- Tăng diversity
- Model học được với góc xoay lớn hơn

**Thêm Perspective:**
- Mô phỏng góc chụp thực tế
- Tăng robustness

**Thêm ColorJitter:**
- Mô phỏng điều kiện ánh sáng khác nhau
- Tăng generalization

---

### 10. METRICS VÀ ĐÁNH GIÁ

#### 10.1. Accuracy Metrics

**Overall Accuracy:**
```
Accuracy = (Correct Predictions) / (Total Predictions)
```

**Per-Class Accuracy:**
```
Class Accuracy = (Correct for Class) / (Total for Class)
```

**Confusion Matrix:**
- Ma trận NxN (N=19 classes)
- Hàng i, cột j: Số lượng class i bị predict thành class j
- Đường chéo: Correct predictions
- Off-diagonal: Misclassifications

#### 10.2. Tại sao Nonagon khó nhất?

**Lý do:**
- Nonagon (9 cạnh) rất giống Circle khi nhìn từ xa hoặc khi resolution thấp
- Chỉ khác nhau ở số cạnh (9 vs vô số)
- Model dễ nhầm → accuracy thấp nhất (94.69%)

**Giải pháp:**
- Tăng input size (64→128)
- TTA với rotations
- Vẫn còn room for improvement

---

**Kết luận:**
README này đã giải thích chi tiết về:
- ✅ Pretrained models và Transfer Learning
- ✅ Contour detection và bounding boxes
- ✅ Data augmentation chi tiết
- ✅ CRAFT detector
- ✅ Hybrid detector
- ✅ Training process
- ✅ Post-processing
- ✅ Các câu hỏi thường gặp

Bạn có thể sử dụng các phần này để trả lời các câu hỏi sâu từ thầy giáo!

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

