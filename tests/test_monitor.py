"""
Kiểm tra modules PerformanceMonitor
"""

import time
import numpy as np
from rust_yolo import PerformanceMonitor
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_performance_monitor():
    logger.info("Đang khởi tạo PerformanceMonitor...")
    monitor = PerformanceMonitor()
    
    logger.info("Bắt đầu background monitor...")
    monitor.start_background_monitor()
    
    # Đợi một chút để thu thập dữ liệu
    logger.info("Đang thu thập dữ liệu hệ thống...")
    time.sleep(2)
    
    # Test update_frame_time
    logger.info("Cập nhật frame time...")
    monitor.update_frame_time(15.5)  # 15.5 ms
    
    # Test process_frame với numpy array
    logger.info("Test process_frame...")
    dummy_frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    # Lấy memory address và length
    ptr = dummy_frame.ctypes.data
    length = dummy_frame.nbytes
    avg_value = monitor.process_frame(ptr, length)
    logger.info(f"Giá trị trung bình pixel: {avg_value:.2f}")
    
    # Lấy stats
    logger.info("\nSystem Stats:")
    stats = monitor.get_stats()
    
    logger.info(f"FPS: {stats['fps']}")
    logger.info(f"AI Latency: {stats['ai_latency']:.2f} ms")
    logger.info(f"Rust Latency: {stats['rust_latency']:.2f} ms")
    logger.info(f"CPU Usage: {stats['cpu_usage']:.1f}%")
    
    mem = stats['memory_usage']
    logger.info(f"Memory: {mem['used']} / {mem['total']} ({mem['percent']:.1f}%)")
    
    gpu = stats['gpu_info']
    logger.info(f"GPU: {gpu['name']} (Load: {gpu['load']}%)")
    
    logger.info("\nDừng background monitor...")
    monitor.stop_background_monitor()
    logger.info("Test thành công!")

if __name__ == "__main__":
    test_performance_monitor()
