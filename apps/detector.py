"""
Module YOLOv8 Object Detector
"""

import cv2
from ultralytics import YOLO
import numpy as np
from typing import Any


class YoloDetector:
    """YOLOv8 Object Detector wrapper với tối ưu tốc độ."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.5):
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.model.fuse()
        self.model.to("mps")
        self.model.half()

    def detect_frame(self, frame: np.ndarray) -> Any:
        """
        Chạy detection trên 1 frame.

        Args:
            frame: Frame từ camera (numpy array BGR)

        Returns:
            Ultralytics results object
        """
        return self.model(
            frame,
            conf=self.confidence,
            verbose=False,
            stream=False,
            iou=0.45,
            max_det=15,
            half=True,
            device="mps",
            agnostic_nms=True,
        )

    def annotate_frame(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Vẽ kết quả detection lên frame.

        Args:
            frame: Frame gốc
            results: Kết quả từ detect_frame()

        Returns:
            Frame đã được vẽ bounding box
        """
        return results[0].plot(line_width=1, font_size=0.8)
