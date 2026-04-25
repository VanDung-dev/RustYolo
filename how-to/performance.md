# Theo dõi hiệu suất & Tối ưu hóa

RustYolo tích hợp sẵn một `PerformanceMonitor` để theo dõi tình trạng ứng dụng AI của bạn, bao gồm FPS, độ trễ và mức độ sử dụng phần cứng.

## 📊 Sử dụng Performance Monitor

Bộ giám sát có thể chạy ngầm để lấy mẫu các chỉ số hệ thống (CPU, GPU, ANE) trong khi bạn cập nhật kết quả inference cho nó.

```python
import rust_yolo
import time

monitor = rust_yolo.PerformanceMonitor()

# Thiết lập backend để theo dõi chính xác
monitor.set_backend(rust_yolo.ExecutionProviderType.CoreML)

# Bắt đầu giám sát hệ thống trong background (CPU/Nhiệt độ/...)
monitor.start_background_monitor()

try:
    while True:
        start_time = time.perf_counter()
        
        # --- Mã xử lý Inference của bạn ---
        # detector.detect(...)
        # ---------------------------
        
        # Cập nhật độ trễ của frame hiện tại vào monitor
        latency_ms = (time.perf_counter() - start_time) * 1000
        monitor.update_frame_time(latency_ms)
        
        # Lấy số liệu thống kê
        stats = monitor.get_stats()
        print(f"FPS: {stats['fps']:.1f} | Độ trễ: {stats['ai_latency']:.1f}ms")
        print(f"CPU: {stats['cpu_usage']}% | Nhiệt độ: {stats['cpu_temp']}°C")
        
        if "ane_info" in stats:
            print(f"Tải trọng ANE: {stats['ane_info']['load']}%")
            
        time.sleep(0.01)
finally:
    monitor.stop_background_monitor()
```

---

## 🚀 Chiến lược tối ưu hóa

### 1. Tăng tốc phần cứng (Execution Providers)

| Engine | Phần cứng | Tốt nhất cho |
| :--- | :--- | :--- |
| **CoreML** | Apple Silicon (ANE/GPU) | Các ứng dụng thời gian thực trên macOS. |
| **WebGPU** | GPU đa nền tảng | Windows/Linux/Web. |
| **CPU** | Bộ vi xử lý hệ thống | Khả năng tương thích và thiết bị cấu hình thấp. |

### 2. Quản lý độ phân giải

Hầu hết các mô hình YOLO được huấn luyện trên kích thước **640x640**. Chạy ở độ phân giải cao hơn (ví dụ: 1280x1280) sẽ làm tăng đáng kể độ trễ.
- Kiểm tra kích thước đầu vào mô hình: `detector.get_input_size()`
- Thay đổi kích thước ảnh *trong Python* trước khi gửi vào Rust nếu detector không tự động làm hoặc nếu bạn muốn tiết kiệm bộ nhớ.

### 3. Lượng tử hóa mô hình (Quantization)

Sử dụng các mô hình **FP16** hoặc **INT8** có thể tăng gấp đôi hiệu suất trên phần cứng tương thích.
- **macOS**: CoreML hoạt động tốt nhất với mô hình FP16.
- **CPU**: Mô hình INT8 nhanh hơn đáng kể trên các CPU hiện đại.

---

## 📈 Hiểu các thông số

- **`fps`**: Số khung hình mỗi giây được xử lý bởi vòng lặp Python của bạn.
- **`engine_fps`**: FPS tối đa lý thuyết mà engine Rust có thể xử lý.
- **`ai_latency`**: Thời gian thực hiện bên trong phiên `ort` (ONNX Runtime).
- **`rust_latency`**: Tổng thời gian xử lý trong extension Rust (Tiền xử lý + Inference + Hậu xử lý).
- **`ane_info`**: Chỉ số riêng cho Apple Silicon, cho biết Apple Neural Engine có đang được sử dụng hay không.
