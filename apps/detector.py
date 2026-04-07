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
        
        # ✅ Tự động phát hiện loại model
        self.is_pose_model = "-pose" in model_name.lower()

    def detect_frame(self, frame: np.ndarray) -> tuple:
        """
        ✅ Chạy detection AI trên 1 frame
        
        Kiến trúc Zero Copy:
        1. Frame numpy từ OpenCV được truyền CON TRỎ trực tiếp sang Rust
        2. Rust thực hiện toàn bộ preprocess, inference, NMS
        3. Kết quả trả về dưới dạng Apache Arrow C Capsule
        4. Python import lại trực tiếp không sao chép dữ liệu
        
        Args:
            frame: Ảnh từ camera OpenCV định dạng HWC uint8
            
        Returns:
            (results: list[dict], timing: dict)
            - results: Danh sách các object đã detect
            - timing: Breakdown latency từng giai đoạn từ Rust
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

            # ✅ Chỉ vẽ Bounding Box khi không phải model pose
            if not self.is_pose_model:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Vẽ Label Text
            if self.is_pose_model:
                label = f"Person: {conf:.2f}"
            else:
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

            # ✅ Vẽ skeleton pose nếu là model pose
            if self.is_pose_model and 'keypoints' in det:
                # Arrow trả về flat list [x0,y0,c0,x1,y1,c1,...]
                # Cần reshape thành [(x,y,conf), ...]
                raw_kp = det['keypoints']
                if raw_kp and len(raw_kp) >= 51:  # 17 * 3
                    kp = [(raw_kp[i], raw_kp[i+1], raw_kp[i+2]) for i in range(0, 51, 3)]
                else:
                    kp = []
                
                # Đường nối khung xương người chuẩn COCO
                skeleton = [
                    (0, 1), (0, 2), (1, 3), (2, 4),
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                    (5, 11), (6, 12), (11, 12),
                    (11, 13), (13, 15), (12, 14), (14, 16)
                ]
                
                # Vẽ các đường nối
                for (a, b) in skeleton:
                    try:
                        if kp[a][2] > 0.5 and kp[b][2] > 0.5:
                            sx1, sy1 = int(kp[a][0]), int(kp[a][1])
                            sx2, sy2 = int(kp[b][0]), int(kp[b][1])
                            cv2.line(annotated_frame, (sx1, sy1), (sx2, sy2), (0, 255, 255), 2)
                    except (IndexError, TypeError):
                        continue
                
                # Vẽ các điểm keypoint
                for (kx, ky, kconf) in kp:
                    if kconf > 0.5 and 0 <= kx < w and 0 <= ky < h:
                        cv2.circle(annotated_frame, (int(kx), int(ky)), 5, (0, 0, 255), -1)

        return annotated_frame
