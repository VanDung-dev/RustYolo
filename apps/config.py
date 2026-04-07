"""
✅ Cấu hình toàn bộ ứng dụng

Tất cả giá trị trong file này có thể chỉnh sửa mà không cần biên dịch lại Rust.
Các thay đổi có hiệu lực ngay lập tức khi khởi động lại ứng dụng.
"""

# Cấu hình Camera
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

# Cấu hình UI
STATS_PANEL_WIDTH = 620
STATS_PANEL_HEIGHT = 1080
WINDOW_WIDTH = CAMERA_WIDTH + STATS_PANEL_WIDTH
WINDOW_HEIGHT = CAMERA_HEIGHT

# Cấu hình Performance Monitor
GPU_UPDATE_INTERVAL = 30  # cập nhật mỗi N frame
MONITOR_THREAD_INTERVAL = 1.0  # giây
STATS_CACHE_TTL = 0.5  # giây

# Cấu hình YOLO
DEFAULT_CONFIDENCE = 0.5
DEFAULT_MODEL = "yolov8n.onnx"
DEFAULT_CAMERA_ID = 0

# Màu sắc UI
COLORS = {
    "bg": (30, 30, 30),
    "white": (255, 255, 255),
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
    "cyan": (255, 255, 0),
    "gray": (150, 150, 150),
    "dark_gray": (60, 60, 60),
}
