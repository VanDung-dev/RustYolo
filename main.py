"""
YOLOv8 Object Detection với Camera và hiển thị thông số hiệu năng
Main entry point - File chạy chính (Đã tối ưu High-Performance Multi-threading)
"""

import os
import sys
import platform
import time
import threading
import queue
import logging
import argparse

import cv2
import numpy as np

# Import từ package apps
from apps.config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    STATS_PANEL_WIDTH,
    STATS_PANEL_HEIGHT,
    DEFAULT_CONFIDENCE,
    DEFAULT_CAMERA_ID,
)
from apps.detector import YoloDetector
from apps.performance_monitor import PerformanceMonitor
from apps.ui_panel import create_stats_panel

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoStream:
    """Threaded video capture stream to handle I/O without blocking"""
    def __init__(self, src=0, width=1920, height=1080):
        self.stream = cv2.VideoCapture(src)
        
        # Cấu hình camera độ phân giải cao và 60fps (tuỳ phần cứng)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, 60)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (grabbed, frame) = self.stream.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            return self.grabbed, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


def ai_worker_thread(detector, frame_queue, result_queue, monitor, stop_event):
    """Luồng worker nền để xử lý YOLO inference với Adaptive Thermal Control"""
    while not stop_event.is_set():
        try:
            # 1. Kiểm tra trạng thái nhiệt độ (Thermal Awareness)
            stats = monitor.get_stats()
            temp = stats.get("cpu_temp", 0.0)
            dt_dt = stats.get("dt_dt", 0.0)
            
            # Chiến lược điều tiết nhiệt (Adaptive Scheduling)
            thermal_delay = 0
            if temp > 82.0 or dt_dt > 0.4:
                thermal_delay = 0.01  # 10ms delay to cool down
            if temp > 88.0:
                thermal_delay = 0.03  # 30ms delay
            
            if thermal_delay > 0:
                time.sleep(thermal_delay)

            # 2. Lấy ảnh mới nhất ra xử lý
            frame = frame_queue.get(timeout=0.1)
            if frame is None:
                continue
                
            detect_start = time.perf_counter()
            results, timing = detector.detect_frame(frame)
            detect_time = (time.perf_counter() - detect_start) * 1000  # ms

            # Đẩy kết quả + thời gian xử l vào queue
            if result_queue.full():
                try: 
                    result_queue.get_nowait()
                except queue.Empty: 
                    pass

            result_queue.put((results, detect_time, timing))
            frame_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Lỗi AI Worker: {e}")


def run_camera_detection(
    model_name: str,
    camera_id: int = DEFAULT_CAMERA_ID,
    confidence_threshold: float = DEFAULT_CONFIDENCE,
    execution_provider: str = "coreml",
):
    # Kiểm tra sự tồn tại của file model
    if not os.path.exists(model_name):
        logger.error(f"Không tìm thấy file model: {model_name}")
        logger.error(f"Vui lòng kiểm tra lại đường dẫn hoặc chạy script export_onnx_for_rust.py trước.")
        sys.exit(1)

    logger.info(f"Đang load model {model_name}...")
    detector = YoloDetector(model_name, confidence_threshold, ep=execution_provider)
    logger.info("Model đã được load thành công!")

    # Khởi động luồng Camera siêu tốc
    stream = VideoStream(src=camera_id, width=CAMERA_WIDTH, height=CAMERA_HEIGHT).start()
    time.sleep(1.0) # Đợi camera warm up

    if stream.frame is None:
        logger.error(f"Không thể mở camera với ID {camera_id}")
        sys.exit(1)

    # 0. Xác định độ phân giải màn hình để scale UI hợp lý
    screen_w, screen_h = 1920, 1080 # Mặc định
    try:
        import tkinter as tk
        root = tk.Tk()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
        
        # Đặc biệt cho macOS: Logical pixels thường nhỏ hơn physical pixels (Retina)
        # Chúng ta sẽ ưu tiên giữ nguyên kích thước cho dòng Mac đời mới
        if platform.system() == "Darwin":
            logger.info(f"🍎 macOS Detected: {screen_w}x{screen_h} (Logical). Tự động tối ưu cho Retina...")
            # Virtualize screen_w for Mac to avoid unnecessary scaling
            screen_w = 2880 if screen_w >= 1440 else screen_w
        else:
            logger.info(f"🖥️ Phát hiện độ phân giải màn hình: {screen_w}x{screen_h}")
    except Exception:
        logger.warning("⚠️ Không thể lấy độ phân giải màn hình, dùng mặc định 1080p")

    # Queue giao tiếp Multi-threading
    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    # Tạo performance monitor và chạy background thread
    monitor = PerformanceMonitor()
    monitor.start_background_monitor()

    # Khởi động luồng AI Inference
    ai_worker = threading.Thread(
        target=ai_worker_thread, 
        args=(detector, frame_queue, result_queue, monitor, stop_event),
        daemon=True
    )
    ai_worker.start()

    logger.info(f"Đang quét... Nhấn 'q' để thoát.")
    
    current_results = []
    
    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                continue

            # Xử lý resize frame nếu cần
            h, w = frame.shape[:2]
            if h != CAMERA_HEIGHT or w != CAMERA_WIDTH:
                frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

            # Gắn frame vào hàng đợi cho AI
            if frame_queue.empty():
                frame_queue.put(frame)

            # Lấy kết quả AI mới nhất
            try:
                current_results, detect_time, rust_timing = result_queue.get_nowait()
                monitor.update_frame_time(detect_time)
                stats_extra = rust_timing
            except queue.Empty:
                stats_extra = {}

            # Vẽ bounding boxes cực nhanh
            annotated_frame = detector.annotate_frame(frame, current_results)

            # Gửi buffer frame sang Rust xử lý zero-copy
            frame_ptr = frame.__array_interface__['data'][0]
            frame_len = frame.size
            monitor.process_frame(frame_ptr, frame_len)

            # Lấy stats cached và hiển thị panel
            stats = monitor.get_stats()
            stats.update(stats_extra)
            
            # Đảm bảo chiều cao Stats Panel luôn khớp tuyệt đối với Frame Camera
            frame_h, frame_w = annotated_frame.shape[:2]
            stats_panel = create_stats_panel(stats, STATS_PANEL_WIDTH, frame_h)
            
            # Kiểm tra an toàn cuối cùng (Phòng trường hợp config bị ghi đè)
            if stats_panel.shape[0] != frame_h:
                stats_panel = cv2.resize(stats_panel, (STATS_PANEL_WIDTH, frame_h))

            # Ghép frame và stats panel
            try:
                combined_frame = np.hstack((annotated_frame, stats_panel))
            except Exception as e:
                logger.error(f"Lỗi ghép khung hình: Frame={annotated_frame.shape}, Stats={stats_panel.shape}")
                logger.error(f"Chi tiết lỗi: {e}")
                # Fallback: Chỉ hiển thị frame gốc nếu lỗi
                combined_frame = annotated_frame

            # Tự động scale lại nếu màn hình không đủ lớn (Tránh mất Stats Panel)
            # Nếu màn hình nhỏ hơn 2560x1440 (2.5K), giới hạn chiều rộng hiển thị 1600px
            if screen_w < 2560:
                screen_max_w = 1600 
                curr_h, curr_w = combined_frame.shape[:2]
                
                if curr_w > screen_max_w:
                    scale = screen_max_w / curr_w
                    new_w = int(curr_w * scale)
                    new_h = int(curr_h * scale)
                    combined_frame = cv2.resize(combined_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Hiển thị
            window_name = "YOLO Edge AI - Rust Engine Performance"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, combined_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        stop_event.set()
        monitor.stop_background_monitor()
        stream.stop()
        cv2.destroyAllWindows()
        logger.info("Đã đóng ứng dụng.")


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Object Detection với Camera và Performance Monitor"
    )
    parser.add_argument(
        "--model", type=str, required=True, 
        help="Đường dẫn đến file model YOLOv8 (ví dụ: yolov8n.onnx)"
    )
    parser.add_argument(
        "--camera", type=int, default=DEFAULT_CAMERA_ID, 
        help="ID của camera (mặc định 0)"
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONFIDENCE, 
        help="Ngưỡng confidence (mặc định 0.5)"
    )
    parser.add_argument(
        "--ep", type=str, default="coreml",
        help="Execution Provider (coreml, cuda, webgpu, cpu)"
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
