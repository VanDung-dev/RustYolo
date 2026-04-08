"""
Kiểm tra modules YoloV8Detector
"""

import cv2
import numpy as np
from rust_yolo import YoloV8Detector
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

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

def test_yolov8n():
    logger.info("Đang khởi tạo YOLOv8n detector...")
    detector = YoloV8Detector("yolov8n.onnx", conf_threshold=0.25, iou_threshold=0.45)
    
    input_size = detector.get_input_size()
    logger.info(f"Input size: {input_size}")
    
    # Tạo ảnh test có nội dung thực tế để phát hiện object
    test_image = np.zeros((640, 640, 3), dtype=np.uint8)
    # Vẽ hình chữ nhật giả lập người
    cv2.rectangle(test_image, (100, 100), (300, 500), (128, 128, 128), -1)
    # Vẽ đầu người
    cv2.circle(test_image, (200, 70), 40, (180, 180, 180), -1)
    
    logger.info("Đang test với ảnh mẫu test...")
    detections = detector.detect_from_numpy(test_image)
    logger.info(f"Số object phát hiện: {len(detections)}")
    
    for det in detections:
        logger.info(f"  {det}")
    
    logger.info("\nTest thành công! YOLOv8n hoạt động bình thường.")

if __name__ == "__main__":
    test_yolov8n()
