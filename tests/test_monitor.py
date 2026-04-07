"""
Kiểm tra modules PerformanceMonitor
"""

import time
import numpy as np
from rust_yolo import PerformanceMonitor

def test_performance_monitor():
    print("Đang khởi tạo PerformanceMonitor...")
    monitor = PerformanceMonitor()
    
    print("Bắt đầu background monitor...")
    monitor.start_background_monitor()
    
    # Đợi một chút để thu thập dữ liệu
    print("Đang thu thập dữ liệu hệ thống...")
    time.sleep(2)
    
    # Test update_frame_time
    print("Cập nhật frame time...")
    monitor.update_frame_time(15.5)  # 15.5 ms
    
    # Test process_frame với numpy array
    print("Test process_frame...")
    dummy_frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    # Lấy memory address và length
    ptr = dummy_frame.ctypes.data
    length = dummy_frame.nbytes
    avg_value = monitor.process_frame(ptr, length)
    print(f"Giá trị trung bình pixel: {avg_value:.2f}")
    
    # Lấy stats
    print("\nSystem Stats:")
    stats = monitor.get_stats()
    
    print(f"FPS: {stats['fps']}")
    print(f"AI Latency: {stats['ai_latency']:.2f} ms")
    print(f"Rust Latency: {stats['rust_latency']:.2f} ms")
    print(f"CPU Usage: {stats['cpu_usage']:.1f}%")
    
    mem = stats['memory_usage']
    print(f"Memory: {mem['used']} / {mem['total']} ({mem['percent']:.1f}%)")
    
    gpu = stats['gpu_info']
    print(f"GPU: {gpu['name']} (Load: {gpu['load']}%)")
    
    print("\nDừng background monitor...")
    monitor.stop_background_monitor()
    print("Test thành công!")

if __name__ == "__main__":
    test_performance_monitor()
