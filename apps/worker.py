"""
Module Worker - Luồng xử lý AI nền

Chứa các worker thread function để xử lý YOLO inference với Adaptive Thermal Control.
"""

import time
import threading
import queue
import logging

logger = logging.getLogger(__name__)


def ai_worker_thread(detector, frame_queue, result_queue, monitor, stop_event):
    """Luồng worker nền để xử lý YOLO inference với Adaptive Thermal Control."""
    while not stop_event.is_set():
        try:
            # 1. Kiểm soát nhiệt độ (Thermal Awareness)
            _check_thermal_throttling(monitor)

            # 2. Lấy frame và xử lý Inference
            frame = frame_queue.get(timeout=0.1)
            if frame is None: continue
            
            results, detect_time, timing = _process_ai_inference(detector, frame)

            # 3. Đẩy kết quả vào queue
            _push_ai_results(result_queue, results, detect_time, timing)
            frame_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Lỗi AI Worker: {e}")

def _check_thermal_throttling(monitor):
    """Kiểm tra nhiệt độ CPU và thực hiện delay nếu cần để hạ nhiệt."""
    stats = monitor.get_stats()
    temp = stats.get("cpu_temp", 0.0)
    
    thermal_delay = 0
    if temp > 85.0: thermal_delay = 0.002  # 2ms delay
    if temp > 92.0: thermal_delay = 0.005  # 5ms delay
    
    if thermal_delay > 0:
        if not getattr(ai_worker_thread, "_throttling", False):
            logger.warning(f"⚠️ Thermal Control: {temp:.1f}°C. Đang điều tiết nhẹ để bảo vệ chip...")
            ai_worker_thread._throttling = True
        time.sleep(thermal_delay)
    else:
        ai_worker_thread._throttling = False

def _process_ai_inference(detector, frame):
    """Thực hiện inference AI và tính toán thời gian thực thi."""
    start_time = time.perf_counter()
    results, timing = detector.detect_frame(frame)
    detect_time_ms = (time.perf_counter() - start_time) * 1000
    return results, detect_time_ms, timing

def _push_ai_results(result_queue, results, detect_time, timing):
    """Đẩy kết quả xử lý vào hàng đợi, đảm bảo không bị kẹt nếu queue đầy."""
    if result_queue.full():
        try:
            result_queue.get_nowait()
        except queue.Empty:
            pass
    result_queue.put((results, detect_time, timing))
