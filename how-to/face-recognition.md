# Nhận diện khuôn mặt & Sinh trắc học

RustYolo bao gồm một công cụ chuyên dụng `FaceToolbox` để phát hiện khuôn mặt và xác thực danh tính hiệu suất cao. Nó kết hợp **SCRFD** để phát hiện khuôn mặt mạnh mẽ và **ArcFace** để trích xuất đặc trưng.

## 🛠 Khởi tạo

`FaceToolbox` quản lý cả việc phát hiện và tạo embedding. Bạn cần nạp các mô hình riêng biệt.

```python
import rust_yolo

toolbox = rust_yolo.FaceToolbox()

# Nạp bộ phát hiện khuôn mặt SCRFD
toolbox.load_detector(
    model_path="models/scrfd_2.5g_kps.onnx",
    input_size=(640, 640),
    execution_provider="coreml"
)

# Nạp bộ trích xuất embedding ArcFace
toolbox.load_embedder(
    model_path="models/arcface_w600k_r50.onnx",
    execution_provider="coreml"
)
```

---

## 🔍 Phát hiện khuôn mặt

Bạn có thể phát hiện khuôn mặt và lấy các điểm mốc (landmarks - 5 điểm căn chỉnh: mắt, mũi, khóe miệng).

### Phát hiện tiêu chuẩn

```python
import cv2
import numpy as np

image = cv2.imread("face.jpg")
# Lưu ý: FaceToolbox hiện tại trả về Arrow Capsules để đạt tốc độ cao
# Xem hướng dẫn 'Dữ liệu tốc độ cao' để biết cách đọc dữ liệu Arrow.
arr_cap, sch_cap = toolbox.detect_faces_to_arrow(image, score_threshold=0.5)
```

---

## 👤 Xác thực danh tính (Embeddings)

Để nhận diện một người, bạn cần trích xuất một "embedding" duy nhất (một vector gồm 512 số) từ khuôn mặt của họ.

### Quy trình:

1. **Phát hiện** các điểm mốc khuôn mặt.
2. **Căn chỉnh (Align)** và **Cắt (Crop)** khuôn mặt về kích thước 112x112.
3. **Trích xuất** embedding.

### Ví dụ: Căn chỉnh & Trích xuất embedding cho một khuôn mặt

```python
# Giả sử bạn đã có landmarks từ bộ detector
landmarks = [(120, 150), (180, 150), (150, 180), (130, 210), (170, 210)]

# Căn chỉnh và cắt về 112x112 (trả về dữ liệu RGB thô)
face_bytes = toolbox.align_face(image, landmarks)

# Trích xuất embedding 512 chiều
embedding = toolbox.get_embedding(face_bytes)
print(f"Độ dài embedding: {len(embedding)}") # 512
```

### Ví dụ: Trích xuất embedding hàng loạt (Hiệu suất cao)

RustYolo có thể xử lý nhiều khuôn mặt song song bằng cách sử dụng `rayon` của Rust và tính năng batching của `ort`.

```python
# landmarks_list: Danh sách landmarks đã làm phẳng [x1, y1, x2, y2, ...]
all_landmarks = [
    [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5], # Khuôn mặt 1
    [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]  # Khuôn mặt 2
]

# Trích xuất embedding cho tất cả khuôn mặt song song
# Trả về một mảng Arrow gồm các số thực
emb_arr, emb_sch = toolbox.get_embeddings_batch_to_arrow(image, all_landmarks)
```

---

## ⚡ Mẹo hiệu suất

- Sử dụng **CoreML** trên macOS để chạy SCRFD và ArcFace trên **Apple Neural Engine (ANE)**.
- Đối với xử lý hàng loạt, `get_embeddings_batch_to_arrow` nhanh hơn đáng kể so với việc lặp lại `get_embedding` trong Python.
- SCRFD chính xác hơn nhiều đối với khuôn mặt so với các mô hình YOLO thông thường, đặc biệt là với khuôn mặt nhỏ hoặc các góc nghiêng cực hạn.
