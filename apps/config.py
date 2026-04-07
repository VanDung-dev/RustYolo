"""
✅ Cấu hình toàn bộ ứng dụng

Tất cả giá trị trong file này có thể chỉnh sửa mà không cần biên dịch lại Rust.
Các thay đổi có hiệu lực ngay lập tức khi khởi động lại ứng dụng.
"""

import cv2
import numpy as np

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
# Model hỗ trợ: yolov8n.onnx | yolov8n-pose.onnx | yolov8n-seg.onnx
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

# COCO class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Segmentation: palette màu 80 class COCO (BGR)
# Tạo từ HSV cách đều nhau để tránh trùng màu giữa các class
def _gen_seg_palette(n: int) -> list:
    palette = []
    for i in range(n):
        hue = int(i * 180 / n)
        hsv = np.array([[[hue, 220, 220]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        palette.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return palette


SEG_PALETTE = _gen_seg_palette(len(COCO_CLASSES))

# Alpha blend opacity cho mask (0.0 = trong suốt, 1.0 = đục hoàn toàn)
SEG_ALPHA = 0.45

# Pose: skeleton edges theo nhóm cơ thể (BGR)
# Chuẩn COCO 17 keypoints:
# 0=mũi  1=mắt_T  2=mắt_P  3=tai_T  4=tai_P
# 5=vai_T  6=vai_P  7=khuỷu_T  8=khuỷu_P
# 9=cổ_tay_T  10=cổ_tay_P  11=hông_T  12=hông_P
# 13=đầu_gối_T  14=đầu_gối_P  15=mắt_cá_T  16=mắt_cá_P
SKELETON_EDGES = [
    # (kp_a, kp_b, color_BGR)
    # Đầu
    (0, 1, (0, 215, 255)),   # mũi - mắt trái     | vàng gold
    (0, 2, (0, 215, 255)),   # mũi - mắt phải     | vàng gold
    (1, 3, (147, 112, 219)),   # mắt trái - tai trái | tím medium
    (2, 4, (147, 112, 219)),   # mắt phải - tai phải | tím medium
    # Thân
    (5, 6, ( 50, 205, 50)),   # vai trái - vai phải  | xanh lá
    (5, 11, ( 50, 205, 50)),   # vai trái - hông trái | xanh lá
    (6, 12, ( 50, 205, 50)),   # vai phải - hông phải | xanh lá
    (11, 12, ( 50, 205, 50)),   # hông trái - hông phải| xanh lá
    # Tay trái
    (5, 7, (0, 165, 255)),   # vai trái - khuỷu trái    | cam
    (7, 9, (0, 165, 255)),   # khuỷu trái - cổ tay trái | cam
    # Tay phải
    (6, 8, (0, 69, 255)),   # vai phải - khuỷu phải    | cam đỏ
    (8, 10, (0, 69, 255)),   # khuỷu phải - cổ tay phải | cam đỏ
    # Chân trái
    (11, 13, (255, 144, 30)),   # hông trái - đầu gối trái   | xanh dương
    (13, 15, (255, 144, 30)),   # đầu gối trái - mắt cá trái | xanh dương
    # Chân phải
    (12, 14, (238, 130, 238)),   # hông phải - đầu gối phải   | tím hồng
    (14, 16, (238, 130, 238)),   # đầu gối phải - mắt cá phải | tím hồng
]

# Pose: màu từng keypoint theo nhóm (BGR)
KP_COLORS = [
    (0, 215, 255),  # 0  mũi            | vàng gold
    (147, 112, 219),  # 1  mắt trái        | tím medium
    (147, 112, 219),  # 2  mắt phải        | tím medium
    (147, 112, 219),  # 3  tai trái        | tím medium
    (147, 112, 219),  # 4  tai phải        | tím medium
    (0, 165, 255),  # 5  vai trái        | cam
    (0, 69, 255),  # 6  vai phải        | cam đỏ
    (0, 165, 255),  # 7  khuỷu trái      | cam
    (0, 69, 255),  # 8  khuỷu phải      | cam đỏ
    (0, 165, 255),  # 9  cổ tay trái     | cam
    (0, 69, 255),  # 10 cổ tay phải     | cam đỏ
    (255, 144, 30),  # 11 hông trái       | xanh dương sáng
    (238, 130, 238),  # 12 hông phải       | tím hồng
    (255, 144, 30),  # 13 đầu gối trái    | xanh dương sáng
    (238, 130, 238),  # 14 đầu gối phải    | tím hồng
    (255, 144, 30),  # 15 mắt cá trái     | xanh dương sáng
    (238, 130, 238),  # 16 mắt cá phải     | tím hồng
]

# Ngưỡng confidence để hiển thị keypoint (thấp hơn → skeleton đầy đủ hơn)
POSE_KP_CONF = 0.3
