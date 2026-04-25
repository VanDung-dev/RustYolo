# Nhận diện đối tượng với RustYolo

RustYolo cung cấp các bộ detector chuyên dụng cho các kiến trúc YOLO khác nhau. Việc lựa chọn class phù hợp phụ thuộc vào phiên bản mô hình bạn đang sử dụng.

## 🚀 Lựa chọn Detector

### 1. `YoloV8Detector`
Tương thích với **YOLOv8** và **YOLOv11**. Các mô hình này thường sử dụng cơ chế anchor-based và yêu cầu Non-Maximum Suppression (NMS) trong giai đoạn hậu xử lý.

```python
import rust_yolo

detector = rust_yolo.YoloV8Detector(
    model_path="models/yolov8n.onnx",
    conf_threshold=0.25,
    iou_threshold=0.45,
    execution_provider="coreml"  # "coreml", "webgpu", "cpu"
)
```

### 2. `YoloV26Detector`

Tương thích với **YOLOv10** và **YOLOv26**. Đây là các mô hình "NMS-Free", nghĩa là quá trình hậu xử lý nhanh hơn do không cần bước lọc IOU truyền thống.

```python
import rust_yolo

detector = rust_yolo.YoloV26Detector(
    model_path="models/yolov10n.onnx",
    conf_threshold=0.25,
    execution_provider="coreml"
)
```

---

## 📸 Chạy Inference

### Từ mảng NumPy (Phổ biến)

Đây là cách dễ nhất để nhận kết quả dưới dạng các đối tượng Python.

```python
import cv2

image = cv2.imread("image.jpg")
detections = detector.detect_from_numpy(image)

for det in detections:
    print(f"ID: {det.class_id}")
    print(f"Độ tin cậy: {det.confidence}")
    print(f"BBox: [{det.x}, {det.y}, {det.w}, {det.h}]")
```

### Vẽ trực tiếp lên ảnh

Nếu bạn chỉ muốn xem kết quả nhanh chóng, RustYolo có thể vẽ các bounding box trực tiếp lên ảnh đầu vào (sửa đổi buffer ngay trong bộ nhớ).

```python
# Frame được chỉnh sửa trực tiếp trong Rust để đạt tốc độ tối đa
detector.detect_and_draw(frame)
cv2.imshow("Detection", frame)
```

---

## 🛠 Các nhiệm vụ nâng cao

RustYolo tự động xác định loại nhiệm vụ (task) từ metadata của mô hình hoặc tên file.

### Ước lượng tư thế (Pose Estimation)

Nếu mô hình của bạn là loại `-pose`, đối tượng `YoloDetection` sẽ chứa các `keypoints`.

```python
detections = detector.detect_from_numpy(image)
for det in detections:
    if det.keypoints:
        for x, y, conf in det.keypoints:
            print(f"Điểm chốt: {x}, {y} (Độ tin cậy: {conf})")
```

### Phân đoạn thực thể (Instance Segmentation)

Nếu mô hình của bạn là loại `-seg`, đối tượng `YoloDetection` sẽ chứa `mask_coeffs`.

```python
for det in detections:
    if det.mask_coeffs:
        print(f"Hệ số Mask: {det.mask_coeffs}")
```

---

## ⚡ Execution Providers (Bộ thực thi)

| Provider | Mô tả | Phù hợp nhất cho |
| :--- | :--- | :--- |
| `coreml` | Tăng tốc phần cứng của Apple | macOS (ANE/GPU) |
| `webgpu` | Tăng tốc GPU đa nền tảng | Windows/Linux |
| `cpu` | Thực thi bằng CPU tiêu chuẩn | Dự phòng / Debug |

Bạn có thể kiểm tra kích thước đầu vào mà mô hình yêu cầu bằng cách:

```python
w, h = detector.get_input_size()
print(f"Mô hình yêu cầu: {w}x{h}")
```
