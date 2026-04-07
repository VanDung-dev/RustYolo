"""
✅ Module YOLOv8 Object Detector - Giao tiếp với Rust Engine

Kiến trúc:
- File này là WRAPPER Python không chứa hằng số hay cấu hình
- Toàn bộ constants/palette/màu sắc nằm trong apps/config.py
- Toàn bộ inference, preprocess, NMS chạy 100% trên Rust
- Truyền dữ liệu Zero Copy qua Apache Arrow C Data Interface
- Mask segmentation: proto tensor truyền zero-copy, tính matmul NumPy (~0.5ms)
"""

import cv2
import numpy as np
import pyarrow as pa

# ✅ Import trực tiếp Native Rust Engine đã biên dịch
from rust_yolo import YoloV8Detector

# ✅ Toàn bộ constants được quản lý tập trung tại apps/config.py
from apps.config import (
    COCO_CLASSES,
    SEG_PALETTE,
    SEG_ALPHA,
    SKELETON_EDGES,
    KP_COLORS,
    POSE_KP_CONF,
    IMAGENET_CLASSES,
)



class YoloDetector:
    """YOLOv8 Object / Pose / Segmentation Detector — Rust + Arrow zero-copy."""

    def __init__(self, model_name: str = "yolov8n.onnx", confidence: float = 0.5):
        self.detector = YoloV8Detector(
            model_name,
            conf_threshold=confidence,
            iou_threshold=0.45,
        )
        self.confidence = confidence
        self.input_w = 640
        self.input_h = 640
        self.is_pose_model = "-pose" in model_name.lower()
        self.is_seg_model = "-seg" in model_name.lower()
        self.is_cls_model = "-cls" in model_name.lower()


        # Proto tensor cache (cập nhật mỗi frame, dùng trong annotate_frame)
        self._proto: np.ndarray | None = None

    def detect_frame(self, frame: np.ndarray) -> tuple:
        """
        Chạy detection AI và trả về (results, timing).

        Zero-copy pipeline:
        1. Frame numpy → con trỏ → Rust (không copy pixel)
        2. Rust: preprocess + inference + NMS
        3. Kết quả → Arrow capsule → Python (không copy kết quả)
        4. [Seg] Proto tensor → Arrow capsule → Python (không copy proto)

        Returns:
            (results: list[dict], timing: dict)
        """
        try:
            arr_cap, sch_cap, proto_arr_cap, proto_sch_cap = \
                self.detector.detect_to_arrow(frame)

            results_arrow = pa.Array._import_from_c_capsule(sch_cap, arr_cap)

            timing = {
                "preprocess_ms": self.detector.preprocess_ms,
                "inference_ms": self.detector.inference_ms,
                "nms_ms": self.detector.nms_ms,
            }

            # Proto tensor cho seg model (zero-copy qua Arrow)
            if (self.is_seg_model
                    and proto_arr_cap is not None
                    and proto_sch_cap is not None):
                proto_arrow = pa.Array._import_from_c_capsule(
                    proto_sch_cap, proto_arr_cap
                )
                self._proto = np.array(proto_arrow, dtype=np.float32)
            else:
                self._proto = None

            if len(results_arrow) == 0:
                return [], timing

            return results_arrow.to_pylist(), timing

        except AttributeError as e:
            print(f"Lỗi Arrow: {e}")
            return [], {}

    def annotate_frame(self, frame: np.ndarray, results: list) -> np.ndarray:
        """
        Vẽ kết quả lên frame:
        - Detection : bounding box xanh + label
        - Pose      : bounding box mỏng + skeleton màu COCO
        - Seg       : instance mask màu per-class + bounding box + label
        """
        if not results:
            return frame

        annotated = frame.copy()
        h, w = frame.shape[:2]

        # ── Classification Model (Vẽ panel riêng, không có bbox) ───────────────
        if self.is_cls_model:
            return self._draw_classification_overlay(annotated, results)

        # ── Segmentation masks (vẽ trước để bbox/label ở trên) ───────────────
        if self.is_seg_model and self._proto is not None:
            annotated = self._draw_seg_masks(annotated, results, h, w)

        for det in results:
            conf = det["confidence"]
            x1 = max(0, int(det["x"]))
            y1 = max(0, int(det["y"]))
            x2 = min(w - 1, int(det["x"] + det["w"]))
            y2 = min(h - 1, int(det["y"] + det["h"]))

            box_thick = 1 if (self.is_pose_model or self.is_seg_model) else 2

            # Màu box: theo class (seg) hoặc xanh lá (detection/pose)
            if self.is_seg_model:
                box_color = SEG_PALETTE[det["class_id"] % len(SEG_PALETTE)]
            else:
                box_color = (0, 255, 0)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, box_thick)

            # Label text
            if self.is_pose_model:
                label = f"Person: {conf:.2f}"
            else:
                class_id = det["class_id"]
                label = f"{COCO_CLASSES[class_id]}" if class_id < len(COCO_CLASSES) else f"ID:{class_id}"
            label = f"{label} {conf:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), bl = cv2.getTextSize(label, font, 0.55, 1)
            ly1 = max(0, y1 - th - bl - 4)
            cv2.rectangle(annotated, (x1, ly1), (x1 + tw, y1), (0, 0, 0), -1)
            cv2.putText(annotated, label, (x1, max(0, y1 - 4)),
                        font, 0.55, box_color, 1, cv2.LINE_AA)

            # Skeleton (pose model only)
            if self.is_pose_model and "keypoints" in det:
                annotated = self._draw_skeleton(annotated, det, h, w)

        return annotated

    def _draw_seg_masks(
        self,
        annotated: np.ndarray,
        results: list,
        h: int,
        w: int,
    ) -> np.ndarray:
        """
        Tính và vẽ instance segmentation masks lên frame.

        Proto shape: flat [32 * 160 * 160]
        Mask = sigmoid(coefficients @ proto.reshape(32, 160*160))
               → reshape (160, 160) → upsample về frame → crop bbox → alpha blend
        """
        proto = self._proto
        if proto is None or len(proto) == 0:
            return annotated

        nm, mh, mw = 32, 160, 160

        if len(proto) != nm * mh * mw:
            return annotated

        proto_mat = proto.reshape(nm, mh * mw)          # (32, 25600)
        overlay = annotated.astype(np.float32)

        for det in results:
            class_id = det["class_id"]
            raw_mc = det.get("mask_coeffs")
            if not raw_mc or len(raw_mc) != nm:
                continue

            coeffs = np.array(raw_mc, dtype=np.float32) # (32,)

            # matmul + sigmoid: (32,) @ (32, 25600) → sigmoid → (160, 160)
            logits = coeffs @ proto_mat
            mask_160 = (1.0 / (1.0 + np.exp(-logits))).reshape(mh, mw)

            # Upsample về frame gốc
            mask_full = cv2.resize(mask_160, (w, h), interpolation=cv2.INTER_LINEAR)

            # Crop theo bounding box
            bx1 = max(0, int(det["x"]))
            by1 = max(0, int(det["y"]))
            bx2 = min(w, int(det["x"] + det["w"]))
            by2 = min(h, int(det["y"] + det["h"]))

            binary = (mask_full > 0.5).astype(np.uint8)
            binary[:by1, :] = 0
            binary[by2:, :] = 0
            binary[:, :bx1] = 0
            binary[:, bx2:] = 0

            # Alpha blend màu riêng theo class
            color = SEG_PALETTE[class_id % len(SEG_PALETTE)]
            color_layer = np.zeros_like(annotated, dtype=np.float32)
            color_layer[binary == 1] = color

            mask_3ch = binary[:, :, np.newaxis].astype(np.float32)
            overlay = (overlay * (1 - mask_3ch * SEG_ALPHA) + color_layer * mask_3ch * SEG_ALPHA)

        return np.clip(overlay, 0, 255).astype(np.uint8)

    def _draw_skeleton(
        self,
        annotated: np.ndarray,
        det: dict,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Vẽ skeleton pose chuẩn COCO với màu theo nhóm cơ thể."""
        raw_kp = det.get("keypoints", [])
        if not raw_kp or len(raw_kp) < 51:
            return annotated

        kp = [
            (raw_kp[i], raw_kp[i + 1], raw_kp[i + 2])
            for i in range(0, 51, 3)
        ]

        # Vẽ limb (đường nối), màu lấy từ config.SKELETON_EDGES
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

        # Vẽ keypoint (viền trắng 5px + fill màu 3px), màu từ config.KP_COLORS
        for i, (kx, ky, kc) in enumerate(kp):
            if kc < POSE_KP_CONF:
                continue
            if not (0 <= kx < w and 0 <= ky < h):
                continue
            pt = (int(kx), int(ky))
            color = KP_COLORS[i] if i < len(KP_COLORS) else (0, 255, 0)
            cv2.circle(annotated, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(annotated, pt, 3, color,           -1, cv2.LINE_AA)

        return annotated

    def _draw_classification_overlay(self, annotated: np.ndarray, results: list) -> np.ndarray:
        """Vẽ panel hiển thị Top-5 classification."""
        if not results:
            return annotated

        h, w = annotated.shape[:2]
        
        # Tạo semi-transparent overlay ở góc trái trên
        panel_w = 400
        panel_h = 30 + (len(results) * 35)
        
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (panel_w, panel_h), (0, 0, 0), -1)
        # Alpha blending
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        
        # Tiêu đề
        cv2.putText(annotated, "🏆 Top Classification:", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Hiển thị từng class
        for i, det in enumerate(results):
            class_id = det["class_id"]
            conf = det["confidence"]
            
            # Lấy tên từ IMAGENET_CLASSES
            if 0 <= class_id < len(IMAGENET_CLASSES):
                label = IMAGENET_CLASSES[class_id]
            else:
                label = f"Unknown ({class_id})"
                
            y_pos = 85 + (i * 35)
            
            # Vẽ số thứ tự
            cv2.putText(annotated, f"#{i+1}", (25, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            
            # Vẽ tên class
            cv2.putText(annotated, f"{label[:25]}", (70, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Vẽ confidence bar
            bar_start_x = 240
            bar_max_w = 120
            bar_w = int(conf * bar_max_w)
            
            # Bar background
            cv2.rectangle(annotated, (bar_start_x, y_pos - 12), 
                         (bar_start_x + bar_max_w, y_pos + 2), (50, 50, 50), -1)
            # Bar fill
            cv2.rectangle(annotated, (bar_start_x, y_pos - 12), 
                         (bar_start_x + bar_w, y_pos + 2), (0, 255, 0), -1)
            
            # % text
            cv2.putText(annotated, f"{conf*100:.1f}%", (bar_start_x + bar_max_w + 5, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
        return annotated
