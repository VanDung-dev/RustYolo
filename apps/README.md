# YOLO Application Wrapper (Python)

Thư mục này chứa toàn bộ mã nguồn **Python Application** chịu trách nhiệm cho các tác vụ cấp cao (High-level tasks):

* Quản lý luồng Camera (Multi-threading).
* Điều khiển vòng đời của Engine Rust.
* Vẽ giao diện hiển thị (UI/UX) và thông số hiệu năng chuyên nghiệp.
* Cấu hình và nạp các thư viện phần cứng (DLL/Dynamic Libs).

## 🧩 Cấu trúc Modules

| File | Chức năng |
|---|---|
| `config.py` | Cấu hình toàn bộ ứng dụng (kích thước Camera, màu sắc UI, ID Camera, ngưỡng Confidence). |
| `detector.py` | Wrapper chính cho Rust Engine, quản lý việc load model, nạp DLL cho Windows (Python 3.12+), và chạy inference. |
| `ui_panel.py` | Thiết kế giao diện Dashboard theo dõi FPS, Latency, Load GPU/CPU và Thermal Gradient. |
| `videostream.py` | Luồng đọc Camera tốc độ cao, sử dụng Buffer riêng để không làm nghẽn luồng xử lý AI. |
| `worker.py` | Luồng xử lý AI nền (AI Worker) với Adaptive Thermal Control cho Apple Silicon. |
| `camera_app.py` | Chứa logic chính cho camera detection và phát hiện độ phân giải màn hình theo platform. |
| `performance_monitor.py` | Cầu nối nhận dữ liệu đo đạc thực tế từ Rust để đẩy lên UI Dashboard. |

## 🛠 Cơ chế hoạt động

1. **Camera Thread**: Một luồng riêng biệt liên tục lấy ảnh từ phần cứng và đẩy vào hàng đợi (Queue).
2. **AI Engine**: Nạp Native Extension được biên dịch từ Rust, hỗ trợ cả CoreML (Mac) và WebGPU (Win/Linux).
3. **UI Compositing**: Kết hợp ảnh đã được AI vẽ khung hình (Annotated Frame) và Dashboard hiệu năng (Stats Panel) thành một khung hình HD duy nhất.
4. **Adaptive Scaling**: Thông minh tự động điều chỉnh kích thước hiển thị dựa trên độ phân giải thực tế của màn hình người dùng.

## 🚀 Tính năng nổi bật

* **Tối ưu Python 3.12+**: Tự động xử lý nạp thư viện động (DLL) cho WebGPU trên Windows 10/11.
* **Multiprocessing Friendly**: Thiết kế dạng hàng đợi giúp hệ thống luôn mượt ngay cả khi tải nặng.
* **Dễ cấu hình**: Mọi thông số từ kích thước hiển thị đến màu sắc Dashboard đều có thể chỉnh sửa trong `config.py` mà không cần biên dịch lại Code.
