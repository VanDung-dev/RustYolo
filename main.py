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
    CAMERA_FPS,
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
    """Threaded video capture stream to handle I/O without blocking
    
    Args:
        src: Camera ID (int) hoặc URL stream (str) như 'rtsp://', 'http://', 'tcp://'
        width: Độ phân giải rộng (chỉ áp dụng cho camera local)
        height: Độ phân giải cao (chỉ áp dụng cho camera local)
        fps: FPS mục tiêu (chỉ áp dụng cho camera local)
    """
    def __init__(self, src=0, width=1920, height=1080, fps=60):
        # Xác định nếu src là URL string hay camera ID integer
        self.is_url = isinstance(src, str)
        
        # Nếu là URL, không áp dụng các cấu hình camera (không có ý nghĩa với stream)
        if self.is_url:
            self.stream = cv2.VideoCapture(src)
            logger.info(f"Đang kết nối đến stream URL: {src}")
        else:
            self.stream = cv2.VideoCapture(src)
            
            # Cấu hình camera độ phân giải cao và fps từ config
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.stream.set(cv2.CAP_PROP_FPS, fps)
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

        self.frame_queue = queue.Queue(maxsize=2)
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            (grabbed, frame) = self.stream.read()
            if grabbed:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)

    def read(self):
        try:
            # Thu thập frame mới nhất, timeout ngắn để không block main thread
            return True, self.frame_queue.get(timeout=0.1)
        except queue.Empty:
            return False, None

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
            
            # Chiến lược điều tiết nhiệt thực tế cho Apple Silicon
            thermal_delay = 0
            if temp > 85.0:
                thermal_delay = 0.002  # 2ms delay
            if temp > 92.0:
                thermal_delay = 0.005  # 5ms delay
            
            if thermal_delay > 0:
                if not getattr(ai_worker_thread, "_throttling", False):
                    logger.warning(f"⚠️ Thermal Control: {temp:.1f}°C. Đang điều tiết nhẹ để duy trì ổn định...")
                    ai_worker_thread._throttling = True
                time.sleep(thermal_delay)
            else:
                ai_worker_thread._throttling = False

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
    camera_id: str = str(DEFAULT_CAMERA_ID),  # Hỗ trợ cả integer ID và URL string
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

    # Xác định source: nếu camera_id là string số (ví dụ "0") thì chuyển thành int
    # Nếu là URL (chứa "://") thì giữ nguyên string
    if isinstance(camera_id, str) and "://" not in camera_id:
        try:
            src = int(camera_id)
        except ValueError:
            logger.error(f"Giá trị --camera không hợp lệ: {camera_id}")
            logger.error("Sử dụng: --camera 0 (camera local) hoặc --camera rtsp://... (stream URL)")
            sys.exit(1)
    else:
        src = camera_id  # URL string hoặc int
    
    # Khởi động luồng Camera siêu tốc
    stream = VideoStream(src=src, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS).start()
    time.sleep(1.0) # Đợi camera warm up

    ret, first_frame = stream.read()
    if not ret or first_frame is None:
        if stream.is_url:
            logger.error(f"Không thể kết nối đến stream URL: {camera_id}")
            logger.error("Vui lòng kiểm tra:")
            logger.error("  1. Địa chỉ IP và port đúng")
            logger.error("  2. Server stream đang chạy (ffplay, GStreamer, v.v.)")
            logger.error("  3. Firewall/network cho phép kết nối")
        else:
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
        if platform.system() == "Darwin":
            logger.info(f"🍎 macOS Detected: {screen_w}x{screen_h} (Logical). Tự động tối ưu cho Retina...")
            # Virtualize screen_w for Mac to avoid unnecessary scaling
            screen_w = 2880 if screen_w >= 1440 else screen_w
        else:
            logger.info(f"🖥️ Phát hiện độ phân giải màn hình: {screen_w}x{screen_h}")
    except Exception:
        logger.warning("⚠️ Không thể lấy độ phân giải màn hình, dùng mặc định 1080p")

    # Queue giao tiếp Multi-threading
    frame_queue = queue.Queue(maxsize=2)
    result_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    # Tạo performance monitor và chạy background thread
    monitor = PerformanceMonitor()
    monitor.set_backend(detector.ep)
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

            # Tối ưu: Cache trạng thái resize để tránh check shape mỗi frame
            if not hasattr(run_camera_detection, "_needs_resize"):
                curr_h, curr_w = frame.shape[:2]
                run_camera_detection._needs_resize = (curr_h != CAMERA_HEIGHT or curr_w != CAMERA_WIDTH)
                logger.info(f"Resize status: {run_camera_detection._needs_resize}")

            if run_camera_detection._needs_resize:
                frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_LINEAR)

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

            # ĐÃ GỠ BỎ: monitor.process_frame - Quá trình này quét pixel bằng Python gây tốn CPU vô ích

            # 3. Lấy stats cached và hiển thị panel (Tối ưu: Chỉ vẽ lại 10 lần/giây)
            stats = monitor.get_stats()
            stats.update(stats_extra)
            
            curr_time = time.perf_counter()
            if not hasattr(run_camera_detection, "_last_stats_time"):
                run_camera_detection._last_stats_time = 0
                run_camera_detection._cached_panel = None

            if curr_time - run_camera_detection._last_stats_time > 0.1: # 10 FPS UI Update
                run_camera_detection._cached_panel = create_stats_panel(stats, STATS_PANEL_WIDTH, CAMERA_HEIGHT)
                run_camera_detection._last_stats_time = curr_time
            
            stats_panel = run_camera_detection._cached_panel if run_camera_detection._cached_panel is not None else \
                          np.zeros((CAMERA_HEIGHT, STATS_PANEL_WIDTH, 3), dtype=np.uint8)

            # Ghép frame và stats panel
            try:
                combined_frame = np.hstack((annotated_frame, stats_panel))
            except Exception as e:
                logger.error(f"Lỗi ghép khung hình: Frame={annotated_frame.shape}, Stats={stats_panel.shape}")
                logger.error(f"Chi tiết lỗi: {e}")
                # Fallback: Chỉ hiển thị frame gốc nếu lỗi
                combined_frame = annotated_frame

            # 4. Hiển thị UI (Tối ưu: Chỉ scale nếu thực sự vượt quá màn hình)
            window_name = "YOLO Edge AI - M4 Pro ANE Optimization"
            
            # Tính toán scale một lần duy nhất để tiết kiệm CPU
            if not hasattr(run_camera_detection, "_ui_scale"):
                curr_h, curr_w = combined_frame.shape[:2]
                if screen_w < 2560 and curr_w > 1600:
                    run_camera_detection._ui_scale = 1600 / curr_w
                else:
                    run_camera_detection._ui_scale = 1.0
            
            if run_camera_detection._ui_scale < 1.0:
                new_w = int(combined_frame.shape[1] * run_camera_detection._ui_scale)
                new_h = int(combined_frame.shape[0] * run_camera_detection._ui_scale)
                final_display = cv2.resize(combined_frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            else:
                final_display = combined_frame

            cv2.imshow(window_name, final_display)

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
        "--camera", type=str, default=str(DEFAULT_CAMERA_ID), 
        help="ID của camera (ví dụ: 0, 1) hoặc URL stream (ví dụ: rtsp://..., http://..., tcp://...)"
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONFIDENCE, 
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
