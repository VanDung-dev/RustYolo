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
from .config import COCO_CLASSES, SKELETON_EDGES, KP_COLORS, POSE_KP_CONF


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
