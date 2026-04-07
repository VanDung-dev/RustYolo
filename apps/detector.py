"""
✅ Module YOLOv8 Object Detector - Giao tiếp với Rust Engine

Kiến trúc:
- File này là WRAPPER Python không chứa logic tính toán
- Toàn bộ inference, preprocess, NMS chạy 100% trên Rust
- Truyền dữ liệu Zero Copy qua Apache Arrow C Data Interface
- Không có copy dữ liệu frame giữa Python <> Rust
"""

import cv2
import numpy as np
import pyarrow as pa

# ✅ Import trực tiếp Native Rust Engine đã biên dịch
# Class này được export từ src/yolo.rs thông qua PyO3 binding
from rust_yolo import YoloV8Detector

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
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ─── Bảng màu skeleton theo nhóm cơ thể (BGR) ────────────────────────────────
# Chuẩn COCO 17 keypoints:
# 0=mũi  1=mắt_T  2=mắt_P  3=tai_T  4=tai_P
# 5=vai_T  6=vai_P  7=khuỷu_T  8=khuỷu_P
# 9=cổ_tay_T  10=cổ_tay_P  11=hông_T  12=hông_P
# 13=đầu_gối_T  14=đầu_gối_P  15=mắt_cá_T  16=mắt_cá_P
SKELETON_EDGES = [
    # (kp_a, kp_b, color_BGR)
    # Đầu
    (0,  1,  (  0, 215, 255)),   # mũi - mắt trái     | vàng gold
    (0,  2,  (  0, 215, 255)),   # mũi - mắt phải     | vàng gold
    (1,  3,  (147, 112, 219)),   # mắt trái - tai trái | tím medium
    (2,  4,  (147, 112, 219)),   # mắt phải - tai phải | tím medium
    # Thân
    (5,  6,  ( 50, 205,  50)),   # vai trái - vai phải  | xanh lá
    (5, 11,  ( 50, 205,  50)),   # vai trái - hông trái | xanh lá
    (6, 12,  ( 50, 205,  50)),   # vai phải - hông phải | xanh lá
    (11, 12, ( 50, 205,  50)),   # hông trái - hông phải| xanh lá
    # Tay trái
    (5,  7,  (  0, 165, 255)),   # vai trái - khuỷu trái   | cam
    (7,  9,  (  0, 165, 255)),   # khuỷu trái - cổ tay trái | cam
    # Tay phải
    (6,  8,  (  0,  69, 255)),   # vai phải - khuỷu phải   | cam đỏ
    (8, 10,  (  0,  69, 255)),   # khuỷu phải - cổ tay phải | cam đỏ
    # Chân trái
    (11, 13, (255, 144,  30)),   # hông trái - đầu gối trái  | xanh dương
    (13, 15, (255, 144,  30)),   # đầu gối trái - mắt cá trái | xanh dương
    # Chân phải
    (12, 14, (238, 130, 238)),   # hông phải - đầu gối phải  | tím
    (14, 16, (238, 130, 238)),   # đầu gối phải - mắt cá phải | tím
]

# ─── Màu từng keypoint theo nhóm (BGR) ───────────────────────────────────────
KP_COLORS = [
    (  0, 215, 255),  # 0  mũi           | vàng gold
    (147, 112, 219),  # 1  mắt trái       | tím medium
    (147, 112, 219),  # 2  mắt phải       | tím medium
    (147, 112, 219),  # 3  tai trái       | tím medium
    (147, 112, 219),  # 4  tai phải       | tím medium
    (  0, 165, 255),  # 5  vai trái       | cam
    (  0,  69, 255),  # 6  vai phải       | cam đỏ
    (  0, 165, 255),  # 7  khuỷu trái     | cam
    (  0,  69, 255),  # 8  khuỷu phải     | cam đỏ
    (  0, 165, 255),  # 9  cổ tay trái    | cam
    (  0,  69, 255),  # 10 cổ tay phải    | cam đỏ
    (255, 144,  30),  # 11 hông trái      | xanh dương sáng
    (238, 130, 238),  # 12 hông phải      | tím hồng
    (255, 144,  30),  # 13 đầu gối trái   | xanh dương sáng
    (238, 130, 238),  # 14 đầu gối phải   | tím hồng
    (255, 144,  30),  # 15 mắt cá trái    | xanh dương sáng
    (238, 130, 238),  # 16 mắt cá phải    | tím hồng
]

# Ngưỡng confidence để hiện keypoint (thấp hơn → skeleton đầy đủ hơn)
POSE_KP_CONF = 0.3


class YoloDetector:
    """YOLOv8 Object Detector tối ưu với Rayon và Apache Arrow."""

    def __init__(self, model_name: str = "yolov8n.onnx", confidence: float = 0.5):
        self.detector = YoloV8Detector(
            model_name,
            conf_threshold=confidence,
            iou_threshold=0.45
        )
        self.confidence = confidence
        self.input_w = 640
        self.input_h = 640
        self.is_pose_model = "-pose" in model_name.lower()

    def detect_frame(self, frame: np.ndarray) -> tuple:
        """
        ✅ Chạy detection AI trên 1 frame (Zero Copy qua Arrow).

        Returns:
            (results: list[dict], timing: dict)
        """
        try:
            array_capsule, schema_capsule = self.detector.detect_to_arrow(frame)
            results_arrow = pa.Array._import_from_c_capsule(
                schema_capsule, array_capsule
            )

            timing = {
                "preprocess_ms": self.detector.preprocess_ms,
                "inference_ms":  self.detector.inference_ms,
                "nms_ms":        self.detector.nms_ms,
            }

            if len(results_arrow) == 0:
                return [], timing

            return results_arrow.to_pylist(), timing
        except AttributeError as e:
            print(f"Lỗi Arrow: {e}")
            return [], {}

    def annotate_frame(self, frame: np.ndarray, results: list) -> np.ndarray:
        """
        Vẽ kết quả detection lên frame gốc bằng OpenCV.
        - Detection model : bounding box xanh + label
        - Pose model      : bounding box mỏng + skeleton màu chuẩn COCO
        """
        if not results:
            return frame

        annotated = frame.copy()
        h, w = frame.shape[:2]

        for det in results:
            conf = det["confidence"]
            x1 = max(0, int(det["x"]))
            y1 = max(0, int(det["y"]))
            x2 = min(w - 1, int(det["x"] + det["w"]))
            y2 = min(h - 1, int(det["y"] + det["h"]))

            # ── Bounding box ─────────────────────────────────────────────────
            box_thick = 1 if self.is_pose_model else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), box_thick)

            # ── Label ────────────────────────────────────────────────────────
            if self.is_pose_model:
                label = f"Person: {conf:.2f}"
            else:
                idx = det["class_id"]
                name = COCO_CLASSES[idx] if 0 <= idx < len(COCO_CLASSES) else "?"
                label = f"{name}: {conf:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), bl = cv2.getTextSize(label, font, 0.55, 1)
            lx1, ly1 = x1, max(0, y1 - th - bl - 4)
            cv2.rectangle(annotated, (lx1, ly1), (lx1 + tw, y1), (0, 0, 0), -1)
            cv2.putText(annotated, label, (lx1, max(0, y1 - 4)),
                        font, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

            # ── Skeleton pose ─────────────────────────────────────────────────
            if not (self.is_pose_model and "keypoints" in det):
                continue

            raw_kp = det["keypoints"]
            if not raw_kp or len(raw_kp) < 51:
                continue

            # Reshape flat [x0,y0,c0, x1,y1,c1, ...] → [(x,y,conf), ...]
            kp = [
                (raw_kp[i], raw_kp[i + 1], raw_kp[i + 2])
                for i in range(0, 51, 3)
            ]

            # Vẽ limb (đường nối) theo màu từng nhóm
            for (a, b, color) in SKELETON_EDGES:
                if a >= len(kp) or b >= len(kp):
                    continue
                xa, ya, ca = kp[a]
                xb, yb, cb = kp[b]
                if ca < POSE_KP_CONF or cb < POSE_KP_CONF:
                    continue
                px1, py1 = int(xa), int(ya)
                px2, py2 = int(xb), int(yb)
                if not (0 <= px1 < w and 0 <= py1 < h):
                    continue
                if not (0 <= px2 < w and 0 <= py2 < h):
                    continue
                cv2.line(annotated, (px1, py1), (px2, py2), color, 2, cv2.LINE_AA)

            # Vẽ keypoints (viền trắng 5px + chấm màu 3px ở giữa)
            for i, (kx, ky, kc) in enumerate(kp):
                if kc < POSE_KP_CONF:
                    continue
                if not (0 <= kx < w and 0 <= ky < h):
                    continue
                pt = (int(kx), int(ky))
                color = KP_COLORS[i] if i < len(KP_COLORS) else (0, 255, 0)
                cv2.circle(annotated, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(annotated, pt, 3, color, -1, cv2.LINE_AA)

        return annotated
