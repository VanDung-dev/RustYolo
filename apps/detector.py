"""
Module YOLOv8 Object Detector (Rust Engine API)
"""

import cv2
import numpy as np
import pyarrow as pa
# Import engine phân tích từ CoreML/Rust siêu tối ưu!
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


class YoloDetector:
    """YOLOv8 Object Detector tối ưu với Rayon và Apache Arrow."""

    def __init__(self, model_name: str = "yolov8n.onnx", confidence: float = 0.5):
        # Mặc định sử dụng model ONNX và truyền cho Rust Engine
        self.detector = YoloV8Detector(
            model_name,
            conf_threshold=confidence,
            iou_threshold=0.45
        )
        self.confidence = confidence
        self.input_w = 640
        self.input_h = 640

    def detect_frame(self, frame: np.ndarray) -> tuple:
        """
        Chạy detection trên 1 frame với Apache Arrow (Zero-copy Results).
        Trả về (results, timing_dict) để hiển thị breakdown latency chi tiết.
        """
        try:
            array_capsule, schema_capsule = self.detector.detect_to_arrow(frame)
            results_arrow = pa.Array._import_from_c_capsule(
                schema_capsule, array_capsule
            )

            # Đọc timing breakdown từ Rust (đã được đo bằng Instant)
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

    def annotate_frame(self, frame: np.ndarray, results: list[dict]) -> np.ndarray:
        """
        Vẽ kết quả detection lên frame gốc bằng OpenCV tốc độ cao.
        """
        if not results:
            return frame

        annotated_frame = frame.copy()
        h, w = frame.shape[:2]

        for det in results:
            # det: {'class_id': 0, 'confidence': 0.9, 'x': 10, ...}
            class_id = det['class_id']
            conf = det['confidence']

            # Tọa độ từ Rust (Kornia) đã được trả về theo tỷ lệ frame gốc
            x1 = int(det['x'])
            y1 = int(det['y'])
            x2 = int(det['x'] + det['w'])
            y2 = int(det['y'] + det['h'])

            # Giới hạn tọa độ
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Vẽ Bounding Box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Vẽ Label Text
            idx = class_id
            name = COCO_CLASSES[idx] if 0 <= idx < len(COCO_CLASSES) else "Unk"
            label = f"{name}: {conf:.2f}"

            # Làm nền đen mờ
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), bl = cv2.getTextSize(label, font, 0.6, 2)
            cv2.rectangle(
                annotated_frame,
                (x1, max(0, y1 - th - bl - 5)),
                (x1 + tw, y1),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                annotated_frame,
                label,
                (x1, max(0, y1 - 5)),
                font,
                0.6,
                (0, 255, 0),
                2
            )

        return annotated_frame
