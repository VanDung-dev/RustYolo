"""
Module YOLOv8 Object Detector - Giao tiếp với Rust Engine

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
import logging
import os
from PIL import ImageFont

# Cấu hình logger cho module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache font JetBrains Mono cho label (Size 32 cho nhãn vật thể)
FONT_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "JetBrainsMonoNF.ttf")
try:
    FONT_LABEL = ImageFont.truetype(FONT_PATH, 32)
except Exception:
    FONT_LABEL = None

# Tự động tìm DLL cho Windows (Dành cho CUDA/WebGPU)
import platform
import os
import sys
if platform.system() == "Windows":
    # Python 3.12+ tối ưu: Phải add_dll_directory trước khi import native module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Chỉ nạp DLL từ các đường dẫn tin cậy (Build output) để tránh DLL Hijacking.
    TRUSTED_DLL_PATHS = [
        os.path.join(project_root, "target", "release"),
    ]
    
    # Chỉ nạp debug DLL nếu explicitly yêu cầu (Dev mode)
    if os.environ.get("RUSTYOLO_DEBUG") == "1":
        TRUSTED_DLL_PATHS.append(os.path.join(project_root, "target", "debug"))

    for dll_path in TRUSTED_DLL_PATHS:
        if os.path.exists(dll_path):
            try:
                os.add_dll_directory(dll_path)
                logger.debug(f"🪟 Windows (Py 3.12+): Đã nạp DLL an toàn từ {dll_path}")
            except Exception as e:
                logger.warning(f"⚠️ Không thể nạp DLL từ {dll_path}: {e}")

    # Cập nhật PATH chỉ với các thư mục an toàn
    os.environ["PATH"] = ";".join(TRUSTED_DLL_PATHS) + ";" + os.environ.get("PATH", "")

# Import trực tiếp Native Rust Engine đã biên dịch
from rust_yolo import YoloV8Detector, YoloV26Detector, YoloArchitecture

# Toàn bộ constants được quản lý tập trung tại apps/config.py
from apps.config import (
    COCO_CLASSES,
    SEG_PALETTE,
    SEG_ALPHA,
    SKELETON_EDGES,
    KP_COLORS,
    POSE_KP_CONF,
    IMAGENET_CLASSES,
    DOTA_CLASSES,
)


class YoloDetector:
    """YOLOv8 Object / Pose / Segmentation Detector — Rust + Arrow zero-copy."""

    def __init__(self, model_name: str = "yolov8n.onnx", confidence: float = 0.5, ep: str = "coreml"):
        self.model_name = model_name
        self.confidence = confidence
        self.input_w = 640
        self.input_h = 640
        self.is_pose_model = "-pose" in model_name.lower()
        self.is_seg_model = "-seg" in model_name.lower()
        self.is_cls_model = "-cls" in model_name.lower()
        self.is_obb_model = "-obb" in model_name.lower()

        # Thử xác định architecture từ logic của Rust (nếu có thể gọi static)
        # Hoặc khởi tạo tạm và check config. Ở đây ta dùng logic string tương tự
        if "v26" in model_name.lower() or "26" in model_name.lower() or "v10" in model_name.lower():
            logger.info(f"🚀 Khởi tạo YOLOv26 (NMS-Free) Engine cho: {model_name} (EP: {ep})")
            self.detector = YoloV26Detector(model_name, conf_threshold=confidence, execution_provider=ep)
            self.arch = YoloArchitecture.V26
        else:
            logger.info(f"🎯 Khởi tạo YOLOv8 (Anchor-based) Engine cho: {model_name} (EP: {ep})")
            self.detector = YoloV8Detector(
                model_name,
                conf_threshold=confidence,
                iou_threshold=0.45,
                execution_provider=ep,
            )
            self.arch = YoloArchitecture.V8

        # Proto tensor cache (cập nhật mỗi frame, dùng trong annotate_frame)
        self._proto: np.ndarray | None = None

    @property
    def ep(self):
        """Trả về Execution Provider thực tế đang được sử dụng (CoreML, WebGPU, CPU)"""
        return self.detector.ep

    def detect_frame(self, frame: np.ndarray, benchmark_mode: bool = False) -> tuple:
        """
        Chạy detection AI và trả về (results, timing).

        Zero-copy pipeline:
        1. Frame numpy → con trỏ → Rust (không copy pixel)
        2. Rust: preprocess + inference + NMS
        3. Kết quả → Arrow capsule → Python (không copy kết quả)
        4. [Seg] Proto tensor → Arrow capsule → Python (không copy proto)

        Args:
            frame: Ảnh đầu vào (HWC uint8).
            benchmark_mode: Nếu True, bỏ qua to_pylist() để tránh O(N) copy
                           khi chỉ cần đo latency.

        Returns:
            (results: list[dict], timing: dict)
        """
        try:
            res = self.detector.detect_to_arrow(frame)
            if len(res) == 2:
                arr_cap, sch_cap = res
                proto_arr_cap, proto_sch_cap = None, None
            else:
                arr_cap, sch_cap, proto_arr_cap, proto_sch_cap = res

            results_arrow = pa.Array._import_from_c_capsule(sch_cap, arr_cap)

            timing = {
                "preprocess_ms": self.detector.preprocess_ms,
                "inference_ms": self.detector.inference_ms,
                "nms_ms": self.detector.nms_ms,
            }

            if self.is_seg_model and proto_arr_cap is not None and proto_sch_cap is not None:
                proto_arrow = pa.Array._import_from_c_capsule(proto_sch_cap, proto_arr_cap)
                self._proto = np.array(proto_arrow, dtype=np.float32)
            else:
                self._proto = None

            if len(results_arrow) == 0 or benchmark_mode:
                return [], timing

            return results_arrow.to_pylist(), timing

        except Exception as e:
            logger.error(f"Lỗi AI Pipeline: {e}")
            return [], {}

    def annotate_frame(self, frame: np.ndarray, results: list) -> np.ndarray:
        """Vẽ toàn bộ kết quả AI (Box, Seg, Pose, OBB) lên frame.
        
        Vẽ trực tiếp lên frame (không copy) vì caller không tái sử dụng frame gốc.
        Các hàm vẽ phức tạp (seg mask, classification overlay) tự quản lý copy riêng.
        """
        if not results:
            return frame

        if self.is_cls_model:
            return self._draw_classification_overlay(frame, results)

        annotated = frame
        h, w = frame.shape[:2]

        if self.is_seg_model and self._proto is not None:
            annotated = self._draw_seg_masks(annotated, results, h, w)

        for det in results:
            self._draw_detection_object(annotated, det, w, h)

        if self.is_pose_model:
            for det in results:
                if "keypoints" in det:
                    annotated = self._draw_skeleton(annotated, det, h, w)

        return annotated

    def _draw_detection_object(self, canvas: np.ndarray, det: dict, w: int, h: int):
        """Vẽ một vật thể bao gồm Box/OBB và Label."""
        conf = det["confidence"]
        class_id = det["class_id"]
        
        # Tọa độ cơ bản
        x1, y1 = max(0, int(det["x"])), max(0, int(det["y"]))
        x2, y2 = min(w - 1, int(det["x"] + det["w"])), min(h - 1, int(det["y"] + det["h"]))

        # Màu sắc & Độ dày
        color = SEG_PALETTE[class_id % len(SEG_PALETTE)] if self.is_seg_model else (0, 255, 0)
        thick = 1 if (self.is_pose_model or self.is_seg_model) else 2

        # 1. Vẽ Geometry (Rectangle hoặc OBB)
        label_y = y1 - 10
        if self.is_obb_model and len(det.get("keypoints", [])) >= 4:
            pts = np.array([(int(kp[0]), int(kp[1])) for kp in det["keypoints"][:4]], np.int32)
            cv2.polylines(canvas, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
            label_y = pts[0][1] - 10
        elif not self.is_obb_model:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick)

        # 2. Vẽ Label
        label_text = self._get_label_text(class_id, conf)
        self._draw_label(canvas, label_text, x1, label_y)

    def _get_label_text(self, class_id: int, conf: float) -> str:
        """Tạo chuỗi văn bản cho nhãn vật thể."""
        if self.is_pose_model:
            return f"Person {conf:.2f}"
        
        if self.is_obb_model:
            name = DOTA_CLASSES[class_id] if class_id < len(DOTA_CLASSES) else f"ID:{class_id}"
        else:
            name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"ID:{class_id}"
        
        return f"{name} {conf:.2f}"

    @staticmethod
    def _draw_label(canvas: np.ndarray, text: str, x: int, y: int):
        """Vẽ nhãn với nền đen và chữ trắng bằng OpenCV (Native - Nhanh)."""
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Vẽ nền đen
        cv2.rectangle(canvas, (x, y - text_h - baseline), (x + text_w, y + baseline), (0, 0, 0), -1)
        
        # Vẽ chữ trắng
        cv2.putText(
            canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2, cv2.LINE_AA
        )

    def _draw_seg_masks(
        self, annotated: np.ndarray, results: list, h: int, w: int,
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

    def _draw_skeleton(self, annotated: np.ndarray, det: dict, h: int, w: int) -> np.ndarray:
        """Vẽ skeleton pose chuẩn COCO."""
        raw_kp = det.get("keypoints", [])
        if not raw_kp or len(raw_kp) < 51:
            return annotated

        # 1. Parse keypoints
        kp = self._parse_keypoints(raw_kp)

        # 2. Vẽ Limbs (Đường nối)
        self._draw_pose_limbs(annotated, kp, w, h)

        # 3. Vẽ Keypoints (Các khớp)
        self._draw_pose_keypoints(annotated, kp, w, h)

        return annotated

    @staticmethod
    def _parse_keypoints(raw_kp: list) -> list:
        """Chuyển đổi list keypoints thô thành list các tuple (x, y, conf)."""
        return [(raw_kp[i], raw_kp[i + 1], raw_kp[i + 2]) for i in range(0, 51, 3)]

    @staticmethod
    def _draw_pose_limbs(canvas: np.ndarray, kp: list, w: int, h: int):
        """Vẽ các đường nối giữa các khớp dựa trên cấu trúc COCO."""
        for (a, b, color) in SKELETON_EDGES:
            if a >= len(kp) or b >= len(kp): continue
            
            xa, ya, ca = kp[a]
            xb, yb, cb = kp[b]
            
            if ca < POSE_KP_CONF or cb < POSE_KP_CONF: continue
            
            p1, p2 = (int(xa), int(ya)), (int(xb), int(yb))
            
            # Kiểm tra tọa độ hợp lệ
            if (0 <= p1[0] < w and 0 <= p1[1] < h and 
                0 <= p2[0] < w and 0 <= p2[1] < h):
                cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_pose_keypoints(canvas: np.ndarray, kp: list, w: int, h: int):
        """Vẽ các điểm khớp với viền trắng và tâm màu."""
        for i, (kx, ky, kc) in enumerate(kp):
            if kc < POSE_KP_CONF: continue
            
            pt = (int(kx), int(ky))
            if not (0 <= pt[0] < w and 0 <= pt[1] < h): continue
            
            color = KP_COLORS[i] if i < len(KP_COLORS) else (0, 255, 0)
            
            # Vẽ viền trắng ngoài cùng để nổi bật
            cv2.circle(canvas, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
            # Vẽ tâm màu theo cấu trúc cơ thể
            cv2.circle(canvas, pt, 3, color, -1, cv2.LINE_AA)

    @staticmethod
    def _draw_classification_overlay(annotated: np.ndarray, results: list) -> np.ndarray:
        """Vẽ panel hiển thị Top-5 classification tối ưu."""
        if not results:
            return annotated
        
        # 1. Tính toán kích thước panel linh hoạt
        num_res = len(results)
        row_h = 40
        header_h = 60
        panel_w = 480
        panel_h = header_h + (num_res * row_h) + 10
        
        # 2. Vẽ nền mờ (Glassmorphism effect)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (15, 15), (panel_w, panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        
        # Viền mỏng cho panel
        cv2.rectangle(annotated, (15, 15), (panel_w, panel_h), (60, 60, 60), 1, cv2.LINE_AA)
        
        # 3. Tiêu đề
        cv2.putText(
            annotated, "TOP CLASSIFICATION", (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2, cv2.LINE_AA
        )
        cv2.line(annotated, (35, 60), (panel_w - 20, 60), (80, 80, 80), 1, cv2.LINE_AA)
        
        # 4. Vẽ từng dòng kết quả
        for i, det in enumerate(results):
            class_id = det["class_id"]
            conf = det["confidence"]
            
            label = IMAGENET_CLASSES[class_id] if 0 <= class_id < len(IMAGENET_CLASSES) else f"ID:{class_id}"
            y_row = header_h + 35 + (i * row_h)
            
            # STT và Tên Class
            cv2.putText(
                annotated, f"#{i+1}", (35, y_row), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2, cv2.LINE_AA
            )
            
            # Cắt ngắn nhãn nếu quá dài
            display_label = label[:22] + ".." if len(label) > 22 else label
            cv2.putText(
                annotated, display_label, (85, y_row), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (240, 240, 240), 1, cv2.LINE_AA
            )
            
            # Thanh Progress Bar cho Confidence
            bar_x = 320
            bar_w_max = 100
            bar_w = int(conf * bar_w_max)
            # Background bar
            cv2.rectangle(annotated, (bar_x, y_row - 12), (bar_x + bar_w_max, y_row + 2), (40, 40, 40), -1)
            # Active bar (màu xanh lá)
            cv2.rectangle(annotated, (bar_x, y_row - 12), (bar_x + bar_w, y_row + 2), (0, 200, 0), -1)
            
            # Phần trăm %
            cv2.putText(
                annotated, f"{conf*100:.1f}%", (bar_x + bar_w_max + 8, y_row),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
            )
            
        return annotated
