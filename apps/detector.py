"""
Module YOLOv8 Object Detector (Rust Engine API)
"""

import cv2
import numpy as np
from typing import Any, List
# Import engine phân tích từ CoreML/Rust siêu tối ưu!
from rust_yolo import YoloV8Detector

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

class YoloDetector:
    """YOLOv8 Object Detector đã được chuyển sang Rust."""

    def __init__(self, model_name: str = "yolov8n.onnx", confidence: float = 0.5):
        # Mặc định sử dụng model ONNX và truyền cho Rust Engine
        self.detector = YoloV8Detector(model_name, conf_threshold=confidence, iou_threshold=0.45)
        self.confidence = confidence
        self.input_w = 640
        self.input_h = 640

    def detect_frame(self, frame: np.ndarray) -> List[Any]:
        """
        Chạy detection trên 1 frame thông qua Rust Engine.

        Args:
            frame: Frame từ camera (numpy array BGR)

        Returns:
            Danh sách YoloDetection objects (từ rust_yolo)
        """
        # Resize và chuẩn hóa bytes
        frame_resized = cv2.resize(frame, (self.input_w, self.input_h))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_bytes = frame_rgb.tobytes()
        
        # Inference zero-copy bằng Rust
        detections = self.detector.detect_from_bytes(frame_bytes, self.input_w, self.input_h)
        return detections

    def annotate_frame(self, frame: np.ndarray, results: List[Any]) -> np.ndarray:
        """
        Vẽ kết quả detection (Bounding boxes, label) lên frame gốc bằng OpenCV tốc độ cao.

        Args:
            frame: Frame gốc
            results: Danh sách YoloDetection từ detect_frame()

        Returns:
            Frame đã được vẽ
        """
        if not results:
            return frame

        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        for det in results:
            # Map tọa độ về tỷ lệ frame gốc
            x1 = int(det.x * w / self.input_w)
            y1 = int(det.y * h / self.input_h)
            x2 = int((det.x + det.width) * w / self.input_w)
            y2 = int((det.y + det.height) * h / self.input_h)
            
            # Giới hạn tọa độ để không văng lỗi khi drawing
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            
            # Vẽ Bounding Box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Vẽ Label Text
            class_name = COCO_CLASSES[det.class_id] if 0 <= det.class_id < len(COCO_CLASSES) else "Unknown"
            label = f"{class_name}: {det.confidence:.2f}"
            
            # Làm nền đen mờ cho chữ dễ đọc
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, max(0, y1 - text_height - baseline - 5)), (x1 + text_width, y1), (0, 0, 0), -1)
            cv2.putText(annotated_frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        return annotated_frame
