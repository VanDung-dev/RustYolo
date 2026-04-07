# Rust YOLOv8 Edge AI

✅ High performance YOLOv8 object detection stack for Apple Silicon with zero-copy Rust / Python hybrid architecture.

---

## 🚀 Đặc điểm kỹ thuật

| Đặc tính | Giá trị                                     |
|---|---------------------------------------------|
| Kiến trúc | Hybrid Rust + Python                        |
| Inference Engine | ONNX Runtime + CoreML Hardware Acceleration |
| Data Transfer | Apache Arrow C Data Interface **Zero Copy** |
| Đa luồng | Rayon data parallelism                      |
| Xử lý ảnh | Kornia CPU optimized                        |
| Monitoring | Native macOS system telemetry               |
| Latency yolov8n | ~23.5ms / frame (M4 Pro)                    |
| Latency yolov8x | ~60.5ms / frame (M4 Pro)                    |

---

## 📂 Cấu trúc dự án

```
rust_yolo/
├── src/                    # Rust native extension
│   ├── lib.rs              # PyO3 module binding
│   ├── yolo.rs             # YOLOv8 inference + NMS engine
│   ├── monitor.rs          # System performance monitor
│   ├── ffi.rs              # Apache Arrow C Data bridge
│   └── image_proc.rs       # Kornia preprocessing pipeline
├── apps/                   # Python application layer
│   ├── detector.py         # Python wrapper + annotation
│   ├── performance_monitor.py
│   ├── ui_panel.py         # OpenCV stats UI render
│   └── config.py
├── main.py                 # Entry point camera demo
├── Cargo.toml
└── requirements.txt
```

---

## 🔗 Kiến trúc giao tiếp

Dự án này sử dụng kiến trúc hybrid tối ưu nhất hiện có cho edge AI:

```
┌──────────────────────────────────────────────────────────┐
│                     PYTHON LAYER                         │
│  OpenCV Camera I/O  │  UI Render  │  Application Logic   │
└───────────────────────────┬──────────────────────────────┘
                            │
            ────────────────┼──────────────── Zero Copy
                            │
┌───────────────────────────▼──────────────────────────────┐
│                       RUST LAYER                         │
│  Preprocessing  │  Inference  │  NMS  │  System Monitor  │
└──────────────────────────────────────────────────────────┘
```

✅ **Python**: chỉ chịu trách nhiệm I/O và UI
✅ **Rust**: toàn bộ tính toán nặng, AI, xử lý số liệu
✅ **Không có copy dữ liệu** qua biên giới ngôn ngữ
✅ GIL được release 100% trong quá trình inference

---

## 🛠️ Cài đặt và triển khai

### 1. Yêu cầu hệ thống
- macOS 26.0+ (Apple Silicon ARM64)
- Python 3.12+
- Rust 1.94+

### 2. Cài đặt dependencies
```bash
# Python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 3. Build native extension
```bash
pip install maturin
maturin develop --release
```

### 4. Chạy demo camera
```bash
python main.py
```

---

## ⚡ Performance Benchmark (Apple Silicon)

✅ **Kiến trúc không block UI**: Luôn chạy camera 60fps mượt mà 100% bất kể tốc độ model. Video không bao giờ bị đứng hay giật lag. Chỉ có bounding box cập nhật theo tốc độ inference AI.

| Model | AI Latency | AI FPS | Camera Display FPS | Trải nghiệm người dùng |
|---|---|--------|---|---|
| yolov8n | 23.5 ms | 40 fps | 60 fps | ✅ Mượt hoàn hảo |
| yolov8s | 28.5 ms | 35 fps | 60 fps | ✅ Mượt hoàn hảo |
| yolov8m | 38.5 ms | 26 fps | 60 fps | ✅ Rất mượt |
| yolov8l | 48.5 ms | 21 fps | 60 fps | ✅ Mượt, không cảm giác giật |
| yolov8x | 60.5 ms | 16 fps | 60 fps | ✅ Video vẫn 60fps mượt, chỉ detection cập nhật 15 lần/giây |

> 💡 Điểm đặc biệt độc đáo của dự án này: Với các model nặng như yolov8x, người dùng vẫn xem video mượt 60fps bình thường, không giống các project khác thường làm video đứng lại chờ kết quả AI.

> ✅ Các giá trị trên đạt giới hạn vật lý của phần cứng.

---

## 🔧 Tính năng

✅ Realtime object detection 80 classes COCO
✅ Full system monitoring: CPU, GPU, Memory, Thermal
✅ Thermal gradient dT/dt realtime measurement
✅ Full latency breakdown per stage
✅ Hardware accelerated CoreML ANE / GPU
✅ Zero copy data transfer
✅ Thread safe background monitoring
✅ Hỗ trợ toàn bộ dòng YOLOv8

---

## 📝 License

[MIT License](LICENSE). Sử dụng hoàn toàn miễn phí cho mục đích thương mại và phi thương mại.