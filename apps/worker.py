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
