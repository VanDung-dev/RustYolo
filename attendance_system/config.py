"""
Cấu hình tập trung cho toàn bộ hệ thống điểm danh RustYolo.
"""

import os

# --- Cấu hình Đường dẫn (Paths) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# Mô hình AI (ONNX)
# Lưu ý: Các file này nên nằm ở thư mục gốc của project
YOLO_MODEL_PATH = os.path.abspath(os.path.join(ROOT_DIR, "yolov8m.onnx"))
YOLO_PT_PATH = os.path.abspath(os.path.join(ROOT_DIR, "yolov8m.pt"))
FACE_DETECTOR_PATH = os.path.abspath(os.path.join(ROOT_DIR, "scrfd_34g.onnx"))
FACE_EMBEDDER_PATH = os.path.abspath(os.path.join(ROOT_DIR, "arcface_w600k_r50.onnx"))

# Cơ sở dữ liệu
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, "attendance.db"))

# Assets (Fonts)
FONT_PATH = os.path.abspath(os.path.join(ROOT_DIR, "assets", "JetBrainsMonoNF.ttf"))


# --- Tham số Hệ thống (System Parameters) ---
# Execution Provider: coreml (Mac), webgpu (Windows/Linux/Mac), cpu
DEFAULT_EP = "coreml"

# Ngưỡng tin cậy (Thresholds)
YOLO_CONF_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.4

FACE_DET_THRESHOLD = 0.5
FACE_REC_THRESHOLD = 0.45  # Ngưỡng chấp nhận danh tính (ArcFace Cosine Similarity)

# Cấu hình Camera & UI
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
RECOGNITION_COOLDOWN = 0.5  # Giây giữa các lần nhận diện (Tần suất quét Face ID)
STREAM_DELAY_MS = 50       # Độ trễ luồng video (ms)
LOG_ATTENDANCE_WINDOW = 0.05 # Khoảng thời gian xác nhận để ghi log điểm danh (giây)

# --- Cấu hình Màu sắc & Font (UI Aesthetics) ---
COLOR_DANGER = (0, 0, 255)      # Đỏ
COLOR_SUCCESS = (0, 255, 0)     # Xanh lá
COLOR_WARNING = (0, 165, 255)   # Cam
COLOR_INFO = (255, 255, 255)    # Trắng
COLOR_RESTING = (150, 150, 150) # Xám
COLOR_SCANNING = (0, 255, 255)  # Vàng

FONT_SIZE_LARGE = 24
FONT_SIZE_MEDIUM = 22
FONT_SIZE_SMALL = 20

# Cấu hình Đăng ký (Registration)
REG_CROP_W = 1080
REG_CROP_H = 1080


# --- Cấu hình Bảo mật & Tiết kiệm năng lượng (Security & Power Saving) ---
SECURITY_LOCK_DURATION = 5.0  # Thời gian khóa hệ thống khi thấy điện thoại (giây)
POWER_SAVING_THRESHOLD = 5.0  # Thời gian không thấy người để vào chế độ nghỉ (giây)
ACTIVE_MODE_FPS = 60          # FPS mục tiêu trong chế độ hoạt động bình thường
REST_MODE_FPS = 15            # FPS mục tiêu trong chế độ nghỉ


# --- Danh sách Class YOLO ---
CLASS_PERSON = 0
CLASS_PHONE = 67
