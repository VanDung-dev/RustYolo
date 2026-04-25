# Hướng dẫn tích hợp RustYolo với Python

RustYolo là một thư viện thị giác máy tính hiệu suất cao được viết bằng Rust, cung cấp các binding Python mượt mà thông qua PyO3 và Maturin. Thư viện hỗ trợ các mô hình YOLO tiên tiến nhất (v8, v10, v11, v26) và các công cụ sinh trắc học khuôn mặt chuyên dụng.

## 🛠 Điều kiện tiên quyết

Trước khi build hoặc sử dụng RustYolo, hãy đảm bảo bạn đã cài đặt các thành phần sau:

### 1. Yêu cầu hệ thống

* **macOS**: Khuyến khích sử dụng Apple Silicon để tận dụng tăng tốc phần cứng qua ANE (Apple Neural Engine).
* **Windows/Linux**: Được hỗ trợ thông qua các execution provider WebGPU hoặc CPU.

### 2. Thành phần phụ thuộc

* **Rust Toolchain**: [Cài đặt Rust](https://rustup.rs/) (phiên bản stable mới nhất).
* **Python**: Phiên bản 3.12 trở lên.
* **Maturin**: Dùng để build extension Rust.

    ```bash
    pip install maturin[patchelf]
    ```

* **ONNX Runtime**: Được quản lý tự động bởi crate `ort`, nhưng hãy đảm bảo hệ thống của bạn hỗ trợ các execution provider đã chọn (CoreML, WebGPU).

---

## 🏗 Build và Cài đặt

Để build thư viện từ mã nguồn và cài đặt vào môi trường Python hiện tại của bạn:

```bash
# Clone repository
git clone https://github.com/VanDung-dev/RustYolo.git
cd RustYolo

# Build và cài đặt cho mục đích phát triển
# Sử dụng --release để đạt hiệu suất tối đa
maturin develop --release
```

### Các tính năng (Features)

* **Mặc định**: Hỗ trợ CPU và CoreML (macOS).
* **WebGPU**: Thêm `--features webgpu` khi build để hỗ trợ tăng tốc GPU trên nhiều nền tảng.

---

## 📚 Các mục hướng dẫn

1. **[Nhận diện đối tượng (Detection)](detection.md)**: Cách sử dụng YOLOv8, v10, v11 và v26 để nhận diện vật thể chung.
2. **[Nhận diện khuôn mặt & Sinh trắc học](face-recognition.md)**: Phát hiện khuôn mặt hiệu suất cao (SCRFD) và trích xuất đặc trưng (ArcFace).
3. **[Theo dõi hiệu suất (Performance)](performance.md)**: Theo dõi FPS, độ trễ và mức độ sử dụng phần cứng (CPU/GPU/ANE) theo thời gian thực.
4. **[Dữ liệu tốc độ cao (Arrow)](advanced-arrow.md)**: Sử dụng Apache Arrow để truyền dữ liệu Zero-copy giữa Rust và Python.
5. **[Quy trình hoạt động (Workflow)](workflow.md)**: Giải thích chi tiết cách dữ liệu di chuyển giữa Rust và Python.

---

## 🚀 Ví dụ nhanh

```python
import rust_yolo
import cv2

# Khởi tạo detector YOLOv8
detector = rust_yolo.YoloV8Detector(
    model_path="models/yolov11n.onnx",
    execution_provider="coreml"
)

# Đọc ảnh
image = cv2.imread("bus.jpg")

# Chạy nhận diện
detections = detector.detect_from_numpy(image)

for det in detections:
    print(f"Phát hiện class {det.class_id} với độ tin cậy {det.confidence:.2f}")
```

---

## 📦 Các thư viện Python cần thiết

Để sử dụng đầy đủ các tính năng của RustYolo, khuyến nghị cài đặt:

```bash
pip install numpy opencv-python pyarrow
```
