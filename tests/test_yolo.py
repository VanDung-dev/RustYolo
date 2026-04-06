"""
Kiểm tra modules YoloV8Detector
"""

import cv2
import numpy as np
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

def test_yolov8x():
    print("Đang khởi tạo YOLOv8x detector...")
    detector = YoloV8Detector("yolov8x.onnx", conf_threshold=0.25, iou_threshold=0.45)
    
    input_size = detector.get_input_size()
    print(f"Input size: {input_size}")
    
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    print("Đang test với ảnh ngẫu nhiên...")
    detections = detector.detect_from_numpy(test_image)
    print(f"Số object phát hiện: {len(detections)}")
    
    for det in detections:
        print(f"  {det}")
    
    print("\nTest thành công! YOLOv8x hoạt động bình thường.")

if __name__ == "__main__":
    test_yolov8x()
