# Face ID Attendance System (Python)

Thư mục này chứa toàn bộ giải pháp **Điểm danh bằng Khuôn mặt (Face ID)** tích hợp các tính năng an toàn và tối ưu hóa hiệu năng cao trên Apple Silicon (CoreML).

Mọi cấu hình về đường dẫn mô hình, ngưỡng tin cậy và tham số hệ thống được quản lý tập trung tại `config.py`.

Hệ thống sử dụng cơ chế truyền dữ liệu **Zero-copy (Apache Arrow)** giữa Python và Engine Rust để đảm bảo tốc độ nhận diện thời gian thực (Real-time).

## 🧩 Cấu trúc Modules

| File | Chức năng |
|---|---|
| `config.py` | **Trung tâm cấu hình**: Quản lý toàn bộ đường dẫn mô hình, ngưỡng tin cậy (thresholds), và tham số camera cho hệ thống. |
| `core.py` | Engine điều phối chính: Quản lý SQLite, so khớp Vector Embedding bằng NumPy và gọi hàm xử lý từ Rust. |
| `check_in.py` | Ứng dụng điểm danh tự động: Tích hợp YOLO để phát hiện người/điện thoại (chống giả mạo) và tự động ghi log. |
| `register_user.py` | Quy trình đăng ký nhân viên: Thu thập 8 góc độ khuôn mặt (Portrait 3:4) để tạo bộ nhận diện chính xác nhất. |
| `face_models.py` | Công cụ quản lý mô hình: Tự động tải các mô hình khuôn mặt và chuyển đổi YOLO sang ONNX. |
| `ui_utils.py` | Tiện ích đồ họa: Hỗ trợ vẽ giao diện Tiếng Việt bằng Pillow và quản lý font chữ hệ thống. |
| `demo.py` | Phiên bản Demo đơn giản: Nhận diện khuôn mặt và hiển thị tên nhân viên thời gian thực. |

## 🛠 Luồng xử lý kỹ thuật

1. **Safety Filter (YOLOv8x)**: Trước khi quét mặt, hệ thống kiểm tra xem có **Người** trong khung hình không và đảm bảo không có **Điện thoại** (ngăn chặn việc dùng ảnh/video để gian lận).
2. **Face Detection (SCRFD)**: Sử dụng model SCRFD để phát hiện nhanh các khuôn mặt và 5 điểm mốc (landmarks).
3. **Batch Alignment**: Toàn bộ các mặt phát hiện được sẽ được Rust Engine căn chỉnh (Align) đồng thời để đưa về tư thế thẳng.
4. **ArcFace Embedding**: Trích xuất đặc trưng 512 chiều.
5. **Vector Search**: Sử dụng phép nhân ma trận (Matrix Multiplication) của NumPy để tìm người dùng khớp nhất trong Database chỉ trong vài mili giây.

## 🚀 Tính năng nổi bật

*   **Chống giả mạo (Anti-Spoofing)**: Sử dụng YOLOv8x để phát hiện các thiết bị điện tử khi quét mặt.
*   **Hỗ trợ Tiếng Việt**: Giao diện hiển thị tên nhân viên và hướng dẫn hoàn toàn bằng Tiếng Việt có dấu.
*   **Tối ưu Apple Silicon**: Tận dụng CoreML và Apple Neural Engine (ANE) cho mọi tác vụ inference.
*   **Chế độ Đăng ký 8 bước**: Đảm bảo AI "hiểu" khuôn mặt nhân viên ở mọi góc độ (ngước lên, cúi xuống, quay trái/phải).
*   **Hiệu năng cực cao**: Truyền dữ liệu qua Arrow giúp giảm 100% thời gian copy dữ liệu giữa các lớp xử lý.

## 📋 Yêu cầu hệ thống

*   Python 3.12+
*   Thư viện: `opencv-python`, `numpy`, `pyarrow`, `pillow`, `ultralytics`, `onnx`, `onnxsim`, `rust_yolo`.
*   Models: `yolov8m.onnx`, `scrfd_34g.onnx`, `arcface_w600k_r50.onnx`.
