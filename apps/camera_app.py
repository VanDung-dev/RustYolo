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


def _get_macos_resolution():
    """Lấy độ phân giải màn hình trên macOS."""
    try:
        import subprocess
        # Cách 1: osascript
        result = subprocess.run(
            ['osascript', '-e', 'tell application "Finder" to get bounds of window of desktop'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            bounds = result.stdout.strip().split(',')
            if len(bounds) == 4:
                w = int(bounds[2]) - int(bounds[0])
                h = int(bounds[3]) - int(bounds[1])
                logger.info(f"🍎 macOS: Phát hiện độ phân giải {w}x{h} (pixel vật lý)")
                return w, h
        
        # Cách 2: system_profiler
        result = subprocess.run(
            ['system_profiler', 'SPDisplaysDataType'],
            capture_output=True, text=True, timeout=5
        )
        import re
        match = re.search(r'Resolution:\s+(\d+)\s*x\s*(\d+)', result.stdout)
        if match:
            w, h = int(match.group(1)), int(match.group(2))
            logger.info(f"🍎 macOS: Phát hiện độ phân giải {w}x{h} (qua system_profiler)")
            return w, h
    except Exception:
        pass
    return None

def _get_linux_resolution():
    """Lấy độ phân giải màn hình trên Linux."""
    try:
        import subprocess
        import re
        result = subprocess.run(['xrandr'], capture_output=True, text=True, timeout=5)
        for line in result.stdout.split('\n'):
            if 'connected' in line and ('primary' in line or ' connected primary' in line):
                match = re.search(r' (\d+)x(\d+)\+', line)
                if match:
                    w, h = int(match.group(1)), int(match.group(2))
                    logger.info(f"🐧 Linux: Phát hiện độ phân giải {w}x{h}")
                    return w, h
    except Exception:
        pass
    return None

def _get_windows_resolution():
    """Lấy độ phân giải màn hình trên Windows."""
    try:
        import ctypes
        w = ctypes.windll.user32.GetSystemMetrics(0)
        h = ctypes.windll.user32.GetSystemMetrics(1)
        logger.info(f"🪟 Windows: Phát hiện độ phân giải {w}x{h}")
        return w, h
    except Exception:
        pass
    return None

def _get_tk_resolution():
    """Dùng tkinter làm phương án cuối cùng để lấy độ phân giải."""
    try:
        import tkinter as tk
        root = tk.Tk()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        logger.info(f"🖥️ Phát hiện độ phân giải qua tkinter: {w}x{h}")
        return w, h
    except Exception:
        pass
    return None

def _get_screen_resolution():
    """Phát hiện độ phân giải màn hình theo từng platform."""
    system = platform.system()
    res = None
    
    if system == "Darwin":
        res = _get_macos_resolution()
    elif system == "Linux":
        res = _get_linux_resolution()
    elif system == "Windows":
        res = _get_windows_resolution()
        
    if res is None:
        res = _get_tk_resolution()
        
    return res if res else (1920, 1080)

def _initialize_detector(model_name, confidence_threshold, execution_provider):
    if not os.path.exists(model_name):
        logger.error(f"Không tìm thấy file model: {model_name}")
        sys.exit(1)
    logger.info(f"Đang load model {model_name}...")
    detector = YoloDetector(model_name, confidence_threshold, ep=execution_provider)
    logger.info("Model đã được load thành công!")
    return detector

def _get_camera_source(camera_id):
    """Xử lý ID camera hoặc URL stream."""
    if isinstance(camera_id, str) and "://" not in camera_id:
        try:
            return int(camera_id)
        except ValueError:
            logger.error(f"Giá trị --camera không hợp lệ: {camera_id}")
            sys.exit(1)
    return camera_id

def _handle_frame_processing(frame, monitor, frame_queue, result_queue):
    """Xử lý tiền xử lý frame, AI inference và lấy kết quả."""
    # VideoStream đã cấu hình đúng CAMERA_WIDTH x CAMERA_HEIGHT, không cần resize

    # Gửi sang luồng AI
    if frame_queue.empty():
        frame_queue.put(frame)

    # Lấy kết quả AI mới nhất
    try:
        current_results, detect_time, rust_timing = result_queue.get_nowait()
        monitor.update_frame_time(detect_time)
        return current_results, rust_timing
    except queue.Empty:
        return None, {}

def _setup_monitoring_and_worker(detector, frame_queue, result_queue):
    """Thiết lập performance monitor và AI worker thread."""
    stop_event = threading.Event()
    monitor = PerformanceMonitor()
    monitor.set_backend(detector.ep)
    monitor.start_background_monitor()

    threading.Thread(
        target=ai_worker_thread, 
        args=(detector, frame_queue, result_queue, monitor, stop_event),
        daemon=True
    ).start()
    
    return stop_event, monitor

def _get_stats_panel(monitor, stats_extra):
    """Tạo hoặc lấy stats panel từ cache (cập nhật 10 lần/giây)."""
    stats = monitor.get_stats()
    stats.update(stats_extra)
    
    curr_time = time.perf_counter()
    if not hasattr(_get_stats_panel, "_last_time"):
        _get_stats_panel._last_time = 0
        _get_stats_panel._cached_panel = None

    if curr_time - _get_stats_panel._last_time > 0.1:
        _get_stats_panel._cached_panel = create_stats_panel(stats, STATS_PANEL_WIDTH, CAMERA_HEIGHT)
        _get_stats_panel._last_time = curr_time
    
    return _get_stats_panel._cached_panel if _get_stats_panel._cached_panel is not None else \
           np.zeros((CAMERA_HEIGHT, STATS_PANEL_WIDTH, 3), dtype=np.uint8)

def _apply_ui_scaling(combined_frame, screen_w):
    """Scale khung hình hiển thị dựa trên độ phân giải màn hình."""
    if not hasattr(_apply_ui_scaling, "_scale"):
        curr_w = combined_frame.shape[1]
        _apply_ui_scaling._scale = 1600 / curr_w if (screen_w < 2560 and curr_w > 1600) else 1.0
    
    if _apply_ui_scaling._scale < 1.0:
        new_w = int(combined_frame.shape[1] * _apply_ui_scaling._scale)
        new_h = int(combined_frame.shape[0] * _apply_ui_scaling._scale)
        return cv2.resize(combined_frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return combined_frame

def run_camera_detection(
    model_name: str,
    camera_id: str = str(DEFAULT_CAMERA_ID),
    confidence_threshold: float = DEFAULT_CONFIDENCE,
    execution_provider: str = "coreml",
):
    # 1. Khởi tạo Engine & Stream
    detector = _initialize_detector(model_name, confidence_threshold, execution_provider)
    src = _get_camera_source(camera_id)
    stream = VideoStream(src=src, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS).start()
    time.sleep(1.0) 

    # 2. Thiết lập Concurrency & Monitoring
    screen_w, _ = _get_screen_resolution()
    frame_queue, result_queue = queue.Queue(maxsize=2), queue.Queue(maxsize=2)
    stop_event, monitor = _setup_monitoring_and_worker(detector, frame_queue, result_queue)

    logger.info(f"Đang quét... Nhấn 'q' để thoát.")
    current_results = []

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None: continue

            # 3. AI Inference & Results
            new_results, stats_extra = _handle_frame_processing(frame, monitor, frame_queue, result_queue)
            if new_results is not None:
                current_results = new_results

            # 4. UI Rendering
            annotated_frame = detector.annotate_frame(frame, current_results)
            stats_panel = _get_stats_panel(monitor, stats_extra)
            combined_frame = np.hstack((annotated_frame, stats_panel))
            final_display = _apply_ui_scaling(combined_frame, screen_w)

            cv2.imshow("YOLO Edge AI - M4 Pro ANE Optimization", final_display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop_event.set()
        monitor.stop_background_monitor()
        stream.stop()
        cv2.destroyAllWindows()
        logger.info("Đã đóng ứng dụng.")
