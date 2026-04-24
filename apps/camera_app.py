"""
Module Camera Application - Chứa logic chính cho camera detection

Bao gồm:
- Hàm run_camera_detection: Chạy camera detection chính
- Các hàm hỗ trợ phát hiện độ phân giải màn hình theo platform
"""

import os
import sys
import platform
import time
import threading
import queue
import logging

import cv2
import numpy as np

from .config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    STATS_PANEL_WIDTH,
    DEFAULT_CONFIDENCE,
    DEFAULT_CAMERA_ID,
)
from .detector import YoloDetector
from .performance_monitor import PerformanceMonitor
from .ui_panel import create_stats_panel
from .videostream import VideoStream
from .worker import ai_worker_thread

logger = logging.getLogger(__name__)


def _get_screen_resolution():
    """
    Phát hiện độ phân giải màn hình theo từng platform.
    
    Returns:
        tuple: (width, height) của màn hình
    """
    screen_w, screen_h = 1920, 1080  # Mặc định
    
    try:
        system = platform.system()
        
        if system == "Darwin":
            # macOS: Sử dụng osascript (luôn khả dụng trên macOS)
            try:
                import subprocess
                result = subprocess.run(
                    ['osascript', '-e', 'tell application "Finder" to get bounds of window of desktop'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    bounds = result.stdout.strip().split(',')
                    if len(bounds) == 4:
                        screen_w = int(bounds[2]) - int(bounds[0])
                        screen_h = int(bounds[3]) - int(bounds[1])
                        logger.info(f"🍎 macOS: Phát hiện độ phân giải {screen_w}x{screen_h} (pixel vật lý)")
                    else:
                        raise ValueError("Định dạng bounds không hợp lệ")
                else:
                    # Fallback: system_profiler
                    result = subprocess.run(
                        ['system_profiler', 'SPDisplaysDataType'],
                        capture_output=True, text=True, timeout=5
                    )
                    import re
                    match = re.search(r'Resolution:\s+(\d+)\s*x\s*(\d+)', result.stdout)
                    if match:
                        screen_w = int(match.group(1))
                        screen_h = int(match.group(2))
                        logger.info(f"🍎 macOS: Phát hiện độ phân giải {screen_w}x{screen_h} (qua system_profiler)")
                    else:
                        raise ValueError("Không tìm thấy độ phân giải")
            except Exception:
                # Last resort: tkinter
                try:
                    import tkinter as tk
                    root = tk.Tk()
                    screen_w = root.winfo_screenwidth()
                    screen_h = root.winfo_screenheight()
                    root.destroy()
                    logger.info(f"🍎 macOS: Phát hiện độ phân giải {screen_w}x{screen_h} (Logical, qua tkinter)")
                except Exception:
                    raise
                    
        elif system == "Linux":
            # Linux: Sử dụng xrandr
            try:
                import subprocess
                result = subprocess.run(
                    ['xrandr'], capture_output=True, text=True, timeout=5
                )
                import re
                # Tìm dòng có "connected" và chứa độ phân giải
                for line in result.stdout.split('\n'):
                    if 'connected' in line and 'primary' in line:
                        match = re.search(r'(\d+)x(\d+)\+', line)
                        if match:
                            screen_w = int(match.group(1))
                            screen_h = int(match.group(2))
                            break
                else:
                    # Thử cách khác: primary display
                    for line in result.stdout.split('\n'):
                        if ' connected primary' in line:
                            match = re.search(r' (\d+)x(\d+)\+', line)
                            if match:
                                screen_w = int(match.group(1))
                                screen_h = int(match.group(2))
                                break
                if screen_w != 1920 or screen_h != 1080:
                    logger.info(f"🐧 Linux: Phát hiện độ phân giải {screen_w}x{screen_h}")
                else:
                    raise ValueError("Không tìm thấy độ phân giải")
            except Exception:
                # Fallback: tkinter
                try:
                    import tkinter as tk
                    root = tk.Tk()
                    screen_w = root.winfo_screenwidth()
                    screen_h = root.winfo_screenheight()
                    root.destroy()
                    logger.info(f"🐧 Linux: Phát hiện độ phân giải {screen_w}x{screen_h} (qua tkinter)")
                except Exception:
                    raise
                    
        elif system == "Windows":
            # Windows: Sử dụng ctypes
            try:
                import ctypes
                screen_w = ctypes.windll.user32.GetSystemMetrics(0)
                screen_h = ctypes.windll.user32.GetSystemMetrics(1)
                logger.info(f"🪟 Windows: Phát hiện độ phân giải {screen_w}x{screen_h}")
            except Exception:
                # Fallback: tkinter
                try:
                    import tkinter as tk
                    root = tk.Tk()
                    screen_w = root.winfo_screenwidth()
                    screen_h = root.winfo_screenheight()
                    root.destroy()
                    logger.info(f"🪟 Windows: Phát hiện độ phân giải {screen_w}x{screen_h} (qua tkinter)")
                except Exception:
                    raise
        else:
            # Unknown platform: thử tkinter
            try:
                import tkinter as tk
                root = tk.Tk()
                screen_w = root.winfo_screenwidth()
                screen_h = root.winfo_screenheight()
                root.destroy()
                logger.info(f"🖥️ Phát hiện độ phân giải màn hình: {screen_w}x{screen_h}")
            except Exception:
                raise
                
    except Exception:
        logger.warning("⚠️ Không thể lấy độ phân giải màn hình, dùng mặc định 1080p")
    
    return screen_w, screen_h


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
    time.sleep(1.0)  # Đợi camera warm up

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
    screen_w, screen_h = _get_screen_resolution()

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

            if curr_time - run_camera_detection._last_stats_time > 0.1:  # 10 FPS UI Update
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
