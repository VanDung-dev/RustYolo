# YOLO Engine Native Core (Rust)

Thư mục này chứa toàn bộ mã nguồn **Native Core** của hệ thống được viết bằng Rust. Core này chịu trách nhiệm cho các tác vụ nặng nhất để đảm bảo hiệu suất thời gian thực (Real-time).

## 🧩 Cấu trúc Modules

| File | Chức năng |
|---|---|
| `lib.rs` | Entry point của hệ thống, định nghĩa các PyO3 bindings để Python có thể gọi. |
| `yolo.rs` | Định nghĩa các kiểu dữ liệu chung (`YoloDetection`, `YoloTask`) và logic nhận diện kiến trúc model. |
| `v8.rs` | Engine thực thi cho các dòng YOLO truyền thống (v8, v11) - Anchor-based với NMS. |
| `v26.rs` | Engine thực thi cho dòng YOLO hiện đại (v26, v10) - NMS-Free (không cần hậu xử lý NMS). |
| `monitor.rs` | Hệ thống theo dõi hiệu năng native (CPU, GPU, Nhiệt độ, dT/dt). |
| `ffi.rs` | Cầu nối **Apache Arrow**, cho phép truyền dữ liệu lớn giữa Rust và Python mà không cần copy (Zero-copy). |
| `image_proc.rs` | Tối ưu hóa xử lý ảnh sử dụng Kornia và SIMD. |

## ⚙️ Luồng thực thi (Pipeline)

1. **Pre-processing**: Chuyển đổi ảnh từ Python sang mảng `ndarray`, resize và chuẩn hóa (Normalize) sử dụng Rayon để song song hóa.
2. **Inference**: Đưa dữ liệu vào ONNX Runtime, kích hoạt tăng tốc phần cứng qua CoreML (Mac) hoặc WebGPU (Windows/Linux).
3. **Post-processing**:

   * Đối với V8/V11: Giải mã output tensor và chạy thuật toán NMS cực nhanh.
   * Đối với V26/V10: Trích xuất trực tiếp tọa độ độ tin cậy.

4. **Data Bridge**: Đóng gói kết quả vào struct Arrow và gửi ngược lại cho Python qua FFI.

## 🚀 Công nghệ sử dụng

* **PyO3**: Binding Rust với Python mượt mà.
* **ONNX Runtime (ort)**: Backend thực thi mạnh mẽ từ Microsoft.
* **Rayon**: Tận dụng tối đa đa lõi (Multi-core) của CPU cho xử lý ảnh.
* **Apache Arrow**: Chìa khóa của hiệu năng "Zero-copy" dữ liệu.
* **sysinfo**: Đọc các thông số sensor hệ thống mức thấp.
