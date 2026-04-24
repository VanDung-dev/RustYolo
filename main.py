"""
YOLO Object Detection với Camera và hiển thị thông số hiệu năng
Main entry point - File chạy chính (Đã tối ưu High-Performance Multi-threading)
"""

import argparse
import logging

from apps import run_camera_detection

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Object Detection với Camera và Performance Monitor"
    )
    parser.add_argument(
        "--model", type=str, required=True, 
        help="Đường dẫn đến file model YOLOv8 (ví dụ: yolov8n.onnx)"
    )
    parser.add_argument(
        "--camera", type=str, default="0", 
        help="ID của camera (ví dụ: 0, 1) hoặc URL stream (ví dụ: rtsp://..., http://..., tcp://...)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, 
        help="Ngưỡng confidence (mặc định 0.5)"
    )
    parser.add_argument(
        "--ep", type=str, default="coreml",
        help="Execution Provider (coreml, webgpu, cpu)"
    )
    args = parser.parse_args()
    
    run_camera_detection(
        model_name=args.model, 
        camera_id=args.camera, 
        confidence_threshold=args.conf,
        execution_provider=args.ep
    )


if __name__ == "__main__":
    main()
