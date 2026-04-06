"""
YOLOv8 Object Detection với Camera và hiển thị thông số hiệu năng
Main entry point - File chạy chính
"""

import cv2
import numpy as np
import sys
import time
import os

# Import từ package apps
from apps.config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    STATS_PANEL_WIDTH,
    STATS_PANEL_HEIGHT,
    DEFAULT_CONFIDENCE,
    DEFAULT_MODEL,
    DEFAULT_CAMERA_ID,
)
from apps.detector import YoloDetector
from apps.performance_monitor import PerformanceMonitor
from apps.ui_panel import create_stats_panel


def run_camera_detection(
    model_name: str = DEFAULT_MODEL,
    camera_id: int = DEFAULT_CAMERA_ID,
    confidence_threshold: float = DEFAULT_CONFIDENCE,
):
    """
    Chạy object detection sử dụng camera và YOLOv8 với hiển thị thông số.

    Args:
        model_name: Tên model YOLOv8
        camera_id: ID của camera
        confidence_threshold: Ngưỡng confidence cho detection
    """
    # Load model YOLOv8
    print(f"Đang load model {model_name}...")
    detector = YoloDetector(model_name, confidence_threshold)
    print("Model đã được load thành công!")

    # Mở camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Không thể mở camera với ID {camera_id}")
        sys.exit(1)

    # Cấu hình camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Chỉ giữ lại 1 frame mới nhất

    # Tạo performance monitor và chạy background thread
    monitor = PerformanceMonitor()
    monitor.start_background_monitor()

    print(f"Đang mở camera... Nhấn 'q' để thoát.")
    print(f"Cửa sổ: {CAMERA_WIDTH + STATS_PANEL_WIDTH}x{CAMERA_HEIGHT}")

    try:
        while True:
            # Đọc frame từ camera
            ret, frame = cap.read()
            if not ret:
                print("Không thể đọc frame từ camera")
                break

            # Resize frame để phù hợp
            frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

            # Chạy detection và đo thời gian
            detect_start = time.time()
            results = detector.detect_frame(frame)
            detect_time = (time.time() - detect_start) * 1000  # ms

            # Vẽ kết quả lên frame
            annotated_frame = detector.annotate_frame(frame, results)

            # Cập nhật thông số FPS/Latency - non blocking
            monitor.update_frame_time(detect_time)

            # Lấy stats cached và tạo panel thống kê
            stats = monitor.get_stats()
            stats_panel = create_stats_panel(stats, STATS_PANEL_WIDTH, STATS_PANEL_HEIGHT)

            # Ghép frame và stats panel
            combined_frame = np.hstack((annotated_frame, stats_panel))

            # Hiển thị frame
            cv2.imshow("YOLOv8 Object Detection - Performance Monitor", combined_frame)

            # Thoát khi nhấn 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Giải phóng tài nguyên
        monitor.stop_background_monitor()
        cap.release()
        cv2.destroyAllWindows()
        print("Đã đóng camera.")


def main():
    """Hàm chính để chạy ứng dụng."""
    import argparse

    parser = argparse.ArgumentParser(
        description="YOLOv8 Object Detection với Camera và Performance Monitor"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Tên model YOLOv8",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=DEFAULT_CAMERA_ID,
        help="ID của camera",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Ngưỡng confidence cho detection",
    )

    args = parser.parse_args()
    run_camera_detection(
        model_name=args.model,
        camera_id=args.camera,
        confidence_threshold=args.conf,
    )


if __name__ == "__main__":
    main()
