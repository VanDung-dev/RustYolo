"""
Cấu hình cho ứng dụng YOLOv8 Object Detection
"""

# Cấu hình Camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Cấu hình UI
STATS_PANEL_WIDTH = 400
STATS_PANEL_HEIGHT = 480
WINDOW_WIDTH = CAMERA_WIDTH + STATS_PANEL_WIDTH
WINDOW_HEIGHT = CAMERA_HEIGHT

# Cấu hình Performance Monitor
GPU_UPDATE_INTERVAL = 30  # cập nhật mỗi N frame
MONITOR_THREAD_INTERVAL = 1.0  # giây
STATS_CACHE_TTL = 0.5  # giây

# Cấu hình YOLO
DEFAULT_CONFIDENCE = 0.5
DEFAULT_MODEL = "yolov8n.pt"
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
