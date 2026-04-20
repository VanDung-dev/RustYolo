# RustYolo - Hướng dẫn sử dụng Python bindings

Thư viện RustYolo cung cấp các PyO3 bindings mạnh mẽ để chạy inference YOLO với hiệu suất cao từ Python.

## 📦 Cài đặt

### Build từ source

```bash
# Clone repository
git clone https://github.com/your-repo/RustYolo.git
cd RustYolo

# Cài đặt Maturin
pip install maturin

# Build và cài đặt
maturin develop --release                     # Chỉ hỗ trợ mỗi macOS
maturin develop --release --features webgpu   # Hỗ trợ macOS/Windows/Linux
```

---

## 🚀 Bắt đầu nhanh

### 1. Khởi tạo Detector

```python
import rust_yolo

# YOLOv8/v11 (Anchor-based + NMS)
detector_v8 = rust_yolo.YoloV8Detector(
    model_path="yolov8n.onnx",
    conf_threshold=0.25,
    iou_threshold=0.45,
    execution_provider="coreml"  # "coreml", "webgpu", "cpu"
)

# YOLOv26/v10 (NMS-Free)
detector_v26 = rust_yolo.YoloV26Detector(
    model_path="yolo26n.onnx",
    conf_threshold=0.25,
    execution_provider="coreml"
)
```

### 2. Chạy Detection

```python
import numpy as np
import cv2

# Đọc ảnh bằng OpenCV (BGR format)
image = cv2.imread("test.jpg")
height, width = image.shape[:2]

# Cách 1: Detect và lấy kết quả Arrow (Zero-Copy)
arr_cap, sch_cap, proto_arr, proto_sch = detector_v8.detect_to_arrow(image)

# Cách 2: Detect và vẽ Bounding Box lên ảnh gốc
arr_cap, sch_cap, proto_arr, proto_sch = detector_v8.detect_and_draw(image)

# Cách 3: Lấy kết quả dạng Python List
detections = detector_v8.detect_from_numpy(image)
for det in detections:
    print(f"Class: {det.class_id}, Conf: {det.confidence:.3f}, Box: ({det.x:.0f}, {det.y:.0f}, {det.w:.0f}, {det.h:.0f})")
```

---

## 📋 API Chi tiết

### `YoloV8Detector`

```python
class YoloV8Detector:
    def __init__(
        self,
        model_path: str,              # Đường dẫn file .onnx
        conf_threshold: float = 0.25,    # Ngưỡng confidence
        iou_threshold: float = 0.45,   # Ngưỡng NMS IOU
        execution_provider: str = "coreml" # "coreml" | "webgpu" | "cpu"
    ):
        ...

    # --- Properties ---
    @property
    def is_cls_model(self) -> bool:  # Model phân loại?
    @property
    def is_obb_model(self) -> bool:  # Model OBB?
    @property
    def last_preprocess_ms(self) -> float:  # Thời gian preprocess
    @property
    def last_inference_ms(self) -> float:  # Thời gian inference
    @property
    def last_nms_ms(self) -> float:  # Thời gian NMS

    # --- Methods ---
    def detect_to_arrow(
        self,
        numpy_array: np.ndarray
    ) -> tuple[PyCapsule, PyCapsule, PyAny, PyAny]:
        """Trả về kết quả Zero-Copy qua Apache Arrow"""
        # Returns: (array_capsule, schema_capsule, proto_array, proto_schema)

    def detect_and_draw(
        self,
        numpy_array: np.ndarray
    ) -> tuple[PyCapsule, PyCapsule, PyAny, PyAny]:
        """Detection + vẽ Bounding Box lên ảnh gốc (Zero-Copy)"""

    def detect_from_numpy(self, numpy_array: np.ndarray) -> list[YoloDetection]:
        """Trả về kết quả dạng Python List"""

    def get_input_size(self) -> tuple[int, int]:
        """Lấy kích thước input của model (width, height)"""

    def set_conf_threshold(self, threshold: float):
        """Đặt lại ngưỡng confidence"""

    def set_iou_threshold(self, threshold: float):
        """Đặt lại ngưỡng IOU"""
```

### `YoloV26Detector`

```python
class YoloV26Detector:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        execution_provider: str = "coreml"
    ):
        ...

    @property
    def is_cls_model(self) -> bool
    @property
    def last_preprocess_ms(self) -> float
    @property
    def last_inference_ms(self) -> float
    @property
    def last_nms_ms(self) -> float  # decode time

    # Cùng interface với YoloV8Detector
    def detect_to_arrow(self, numpy_array) -> tuple
    def detect_and_draw(self, numpy_array) -> tuple
    def detect_from_numpy(self, numpy_array) -> list
```

### `YoloDetection`

```python
@dataclass
class YoloDetection:
    class_id: int           # ID class
    confidence: float       # Độ tin cậy [0, 1]
    x: float               # Tọa độ x góc trên trái
    y: float               # Tọa độ y góc trên trái
    width: float           # Chiều rộng bbox
    height: float          # Chiều cao bbox
    keypoints: list[tuple[float, float, float]]  # [(x, y, conf), ...]
    mask_coeffs: list[float]  # Coefficients cho segmentation
```

### `PerformanceMonitor`

```python
class PerformanceMonitor:
    def __init__(self):
        ...

    def set_backend(self, backend: ExecutionProviderType):
        """Đặt backend đang sử dụng"""

    def start_background_monitor(self):
        """Bắt đầu background thread monitoring"""

    def stop_background_monitor(self):
        """Dừng background thread"""

    def update_frame_time(self, latency_ms: float):
        """Cập nhật frame time và tính FPS"""

    def get_stats(self) -> dict:
        """Lấy tất cả stats"""
        # Returns:
        # {
        #     "fps": float,
        #     "engine_fps": float,
        #     "ai_latency": float,
        #     "rust_latency": float,
        #     "cpu_usage": float,
        #     "cpu_temp": float,
        #     "dt_dt": float,
        #     "memory_usage": {
        #         "used": str,    # "12.5 GB"
        #         "total": str,  # "24.0 GB"
        #         "percent": float
        #     },
        #     "gpu_info": {
        #         "available": bool,
        #         "name": str,
        #         "load": float,
        #         "power": float,
        #         "temperature": float
        #     },
        #     "ane_info": {
        #         "load": float,
        #         "status": str
        #     }
        # }
```

---

## 🎯 Zero-Copy với Apache Arrow

Để đạt hiệu suất cao nhất, sử dụng Arrow FFI thay vì copy dữ liệu:

```python
import pyarrow as pa
import rust_yolo

detector = rust_yolo.YoloV8Detector("model.onnx")
image = cv2.imread("test.jpg")

# Lấy Arrow capsules
arr_cap, sch_cap, _, _ = detector.detect_to_arrow(image)

# Import trực tiếp không copy
import numpy as np

# Cách đọc Arrow Array
array: pa.Array = pa.Capsule.get(arr_cap)
schema: pa.Schema = pa.Capsule.get(sch_cap)
table = pa.Table.from_arrays([array], schema=pa.schema(schema))

# Hoặc convert sang numpy không copy
arr = pa.array(arr_cap)
numpy_array = arr.to_numpy()  # Zero-copy view!
```

---

## 💻 Ví dụ đầy đủ

### Ví dụ 1: Detection cơ bản

```python
import rust_yolo
import cv2
import numpy as np

# Khởi tạo
detector = rust_yolo.YoloV8Detector(
    model_path="yolov8n.onnx",
    conf_threshold=0.3,
    execution_provider="coreml"
)

# Đọc và chạy
image = cv2.imread("input.jpg")
detections = detector.detect_from_numpy(image)

# In kết quả
for det in detections:
    print(f"Class {det.class_id}: {det.confidence:.2%} at ({det.x:.0f}, {det.y:.0f})")
```

### Ví dụ 2: Real-time với Camera

```python
import rust_yolo
import cv2

detector = rust_yolo.YoloV8Detector("yolov8n.onnx")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect + vẽ trực tiếp lên frame
    detector.detect_and_draw(frame)

    cv2.imshow("YOLO", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Ví dụ 3: Performance Monitoring

```python
import rust_yolo
import cv2
import time

detector = rust_yolo.YoloV8Detector(
    "yolov8n.onnx",
    execution_provider="coreml"
)

monitor = rust_yolo.PerformanceMonitor()
monitor.set_backend(rust_yolo.ExecutionProviderType.CoreML)

# Frame timing
for i in range(100):
    frame = cv2.imread(f"frame_{i}.jpg")
    start = time.perf_counter()

    detector.detect_to_arrow(frame)

    latency_ms = (time.perf_counter() - start) * 1000
    monitor.update_frame_time(latency_ms)

stats = monitor.get_stats()
print(f"FPS: {stats['fps']:.1f}")
print(f"AI Latency: {stats['ai_latency']:.1f}ms")
print(f"CPU: {stats['cpu_usage']:.1f}%")
```

### Ví dụ 4: Pose Estimation

```python
import rust_yolo

# Model pose estimation
detector = rust_yolo.YoloV26Detector(
    model_path="yolov8n-pose.onnx",
    conf_threshold=0.5,
    execution_provider="coreml"
)

image = cv2.imread("person.jpg")
detections = detector.detect_from_numpy(image)

for det in detections:
    if det.keypoints:
        print("Keypoints:")
        for i, (x, y, conf) in enumerate(det.keypoints):
            if conf > 0.5:
                cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
```

### Ví dụ 5: Segmentation

```python
import rust_yolo

detector = rust_yolo.YoloV26Detector(
    model_path="yolov8n-seg.onnx",
    conf_threshold=0.5,
    execution_provider="coreml"
)

image = cv2.imread("input.jpg")
detections = detector.detect_from_numpy(image)

for det in detections:
    if det.mask_coeffs:
        print(f"Mask coefficients: {det.mask_coeffs}")
        # Sử dụng mask coefficients để tạo mask segmentation
```

---

## ⚡ Tối ưu hiệu suất

### 1. Chọn đúng Execution Provider

| Platform | Provider | Ghi chú |
|----------|----------|---------|
| macOS (Apple Silicon) | `"coreml"` | Tăng tốc ANE |
| macOS (Intel) | `"coreml"` | Tăng tốc Metal |
| Windows/Linux | `"webgpu"` | Tăng tốc GPU |
| Fallback | `"cpu"` | Luôn khả dụng |

```python
# Tự động chọn provider tốt nhất
import platform
import rust_yolo

system = platform.system()
if system == "Darwin":
    provider = "coreml"
elif system == "Windows":
    provider = "webgpu"
else:
    provider = "cpu"

detector = rust_yolo.YoloV8Detector("model.onnx", execution_provider=provider)
```

### 2. Tái sử dụng Buffer

```python
# KHÔNG tạo detector mới cho mỗi frame
detector = rust_yolo.YoloV8Detector("model.onnx")  # Tạo một lần

# Sử dụng lại cho tất cả frames
for frame in video_frames:
    detector.detect_from_numpy(frame)  # Tái sử dụng buffer nội bộ
```

### 3. Zero-Copy Arrow khi xử lý batch

```python
# Sử dụng Arrow thay vì List để giảm overhead
arr_cap, sch_cap, _, _ = detector.detect_to_arrow(image)

# Convert sang PyArrow Table
table = pa.Table.from_pydict({
    "class_id": pa.array(arr_cap.field("class_id")),
    "confidence": pa.array(arr_cap.field("confidence")),
    # ...
})
```

### 4. Batch Processing

```python
import numpy as np

# Xử lý nhiều ảnh cùng lúc (nếu model hỗ trợ)
batch = np.stack([img1, img2, img3], axis=0)  # Shape: (3, H, W, 3)

# Một số model hỗ trợ batch
# Chú ý: Cần model ONNX có dynamic batch size
```

---

## 🔧 Xử lý lỗi

```python
import rust_yolo

try:
    detector = rust_yolo.YoloV8Detector("invalid.onnx")
except rust_yolo.exceptions.RuntimeError as e:
    print(f"Lỗi: {e}")

try:
    detector.detect_to_arrow(invalid_image)
except rust_yolo.exceptions.ValueError as e:
    print(f"Image không hợp lệ: {e}")

# Các exception types:
# - RuntimeError: Lỗi runtime (load model, inference, ...)
# - ValueError: Giá trị không hợp lệ (shape, path, ...)
```

---

## 📦 Model Requirements

### Định dạng Model

- **File**: `.onnx` (ONNX format)
- **Input**: `[1, 3, H, W]` (NCHW format, float32)
- **Output**: Tùy thuộc task

### Các loại Model

| Model Type | Output Format | Sử dụng |
|------------|---------------|---------|
| Detection | `[1, N, 6]` (x1,y1,x2,y2,score,class) | Object detection |
| Pose | `[1, N, 56]` (+ keypoints) | Pose estimation |
| Segmentation | `[1, N, 38]` (+ mask_coeffs) | Instance segmentation |
| OBB | `[1, N, 8]` (+ angle) | Oriented bounding box |
| Classification | `[1, N]` | Image classification |

### Auto-detection Model Type

```python
import rust_yolo

# Tự động xác định loại model từ filename
config = rust_yolo.YoloArchitecture  # Tự động detect
# Filename patterns:
# - *v8*.onnx -> YoloV8Detector
# - *v26*.onnx -> YoloV26Detector
# - *-pose*.onnx -> Pose task
# - *-seg*.onnx -> Segmentation task
# - *-cls*.onnx -> Classification task
```

---

## 📝 Logging

```python
import logging

# Bật logging chi tiết
logging.basicConfig(level=logging.DEBUG)

# Các log messages:
# [DEBUG] YoloV8Detector::new - Loading model
# [INFO] CoreML available! Activating hardware acceleration...
# [WARN] ⚠️ CoreML not available, falling back to CPU
# [ERROR] Inference failed: ...
```

---

## 🔄 Migration từ Ultralytics

```python
# TRƯỚC (Ultralytics)
from ultralytics import YOLO
model = YOLO("yolov8n.onnx")
results = model.predict("image.jpg")

# SAU (RustYolo)
import rust_yolo
detector = rust_yolo.YoloV8Detector("yolov8n.onnx")
image = cv2.imread("image.jpg")
if hasattr(cv2, 'cvtColor'):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
detections = detector.detect_from_numpy(image)
```
