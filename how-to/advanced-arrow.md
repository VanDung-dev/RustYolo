# Dữ liệu tốc độ cao với Apache Arrow

RustYolo sử dụng **Apache Arrow** để truyền dữ liệu hiệu suất cao, không cần sao chép (zero-copy) giữa Rust và Python. Điều này rất quan trọng khi xử lý hàng ngàn kết quả nhận diện hoặc các embedding khuôn mặt đa chiều mà không gặp phải gánh nặng tạo đối tượng Python.

## 🏹 Tại sao dùng Arrow?

* **Zero-Copy**: Dữ liệu được chia sẻ trong bộ nhớ giữa Rust và Python. Không có dữ liệu nào bị sao chép dư thừa.
* **Tốc độ**: Việc đọc từ một mảng Arrow nhanh gần bằng việc đọc từ một mảng NumPy gốc.
* **Khả năng tương thích**: Tương thích tốt với `pyarrow`, `pandas`, `numpy`, và `polars`.

---

## 🛠 Các thư viện yêu cầu

```bash
pip install pyarrow numpy
```

---

## 📥 Nhận dữ liệu trong Python

Các hàm kết thúc bằng `_to_arrow` (ví dụ: `detect_to_arrow`, `detect_faces_to_arrow`) trả về một tuple chứa các **Arrow Capsules**.

### Ví dụ: Nhận diện đối tượng

```python
import rust_yolo
import pyarrow as pa
import numpy as np

detector = rust_yolo.YoloV8Detector("yolov8n.onnx")
image = cv2.imread("bus.jpg")

# Lấy Arrow Capsules (C-Data Interface)
arr_cap, sch_cap, _, _ = detector.detect_to_arrow(image)

# Chuyển đổi sang PyArrow
# Lưu ý: Trong các phiên bản PyArrow mới, bạn có thể dùng pa.array() hoặc pa.table()
array = pa.array(arr_cap) 

# Chuyển sang NumPy (Zero-copy view!)
# Thao tác này KHÔNG sao chép dữ liệu, nó chỉ tạo một con trỏ NumPy trỏ đến cùng vùng nhớ
numpy_detections = array.to_numpy()

# Nếu dữ liệu là một Struct (ví dụ: phát hiện khuôn mặt)
# faces_arr, faces_sch = toolbox.detect_faces_to_arrow(...)
table = pa.Table.from_arrays([pa.array(faces_arr)], schema=pa.schema(faces_sch))
print(table.to_pandas())
```

---

## 👤 Ví dụ: Embedding khuôn mặt

Embedding khuôn mặt thường có số chiều cao (512 số thực). Việc truyền 100 embedding qua danh sách Python sẽ tạo ra 51,200 đối tượng float trong Python, điều này rất chậm. Arrow xử lý việc này trong một khối dữ liệu duy nhất.

```python
# Trả về một mảng Arrow chứa các embedding đã làm phẳng
emb_arr, emb_sch = toolbox.get_embeddings_batch_to_arrow(image, landmarks)

# Chuyển đổi sang mảng NumPy 2D [N, 512]
embeddings = pa.array(emb_arr).to_numpy().reshape(-1, 512)

print(f"Kích thước: {embeddings.shape}") # (N, 512)
```

---

## ⚠️ Lưu ý quan trọng

1.  **Vòng đời bộ nhớ**: Bộ nhớ cho dữ liệu Arrow được sở hữu bởi extension Rust cho đến khi các đối tượng `arr_cap` và `sch_cap` được bộ thu gom rác (garbage collector) của Python dọn dẹp. Không cố gắng truy cập dữ liệu sau khi các đối tượng này đã bị xóa.
2.  **Tính bất biến (Immutability)**: Dữ liệu được truyền qua Arrow thường là chỉ đọc. Nếu bạn cần sửa đổi kết quả, hãy tạo một bản sao: `numpy_array.copy()`.
3.  **Phiên bản Capsule**: RustYolo sử dụng giao diện Arrow C-Data. Hãy đảm bảo phiên bản `pyarrow` của bạn >= 14.0 để có trải nghiệm tốt nhất.
