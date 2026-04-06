"""
YOLOv8 Object Detection với Camera và hiển thị thông số hiệu năng
Main entry point - File chạy chính (Đã tối ưu High-Performance Multi-threading)
"""

import cv2
import numpy as np
import sys
import time
import threading
import queue

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

def ai_worker_thread(detector, frame_queue, result_queue, stop_event):
    """Luồng worker nền để xử lý YOLO inference không khóa camera"""
    while not stop_event.is_set():
        try:
            # Lấy ảnh mới nhất ra xử lý
            frame = frame_queue.get(timeout=0.1)
            if frame is None:
                continue
                
            detect_start = time.perf_counter()
            results = detector.detect_frame(frame)
            detect_time = (time.perf_counter() - detect_start) * 1000  # ms
            
            # Đẩy kết quả + thời gian xử lý xuống queue
            if result_queue.full():
                try: result_queue.get_nowait() # Vứt kết quả cũ chưa kịp vẽ
                except queue.Empty: pass
                
            result_queue.put((results, detect_time))
            frame_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Lỗi AI Worker: {e}")

def run_camera_detection(
    model_name: str = DEFAULT_MODEL,
    camera_id: int = DEFAULT_CAMERA_ID,
    confidence_threshold: float = DEFAULT_CONFIDENCE,
):
    print(f"Đang load model {model_name}...")
    detector = YoloDetector(model_name, confidence_threshold)
    print("Model đã được load thành công!")

    # Khởi động luồng Camera siêu tốc
    stream = VideoStream(src=camera_id, width=CAMERA_WIDTH, height=CAMERA_HEIGHT).start()
    time.sleep(1.0) # Đợi camera warm up

    if stream.frame is None:
        print(f"Không thể mở camera với ID {camera_id}")
        sys.exit(1)

    # Queue giao tiếp Multi-threading
    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    # Khởi động luồng AI Inference
    ai_worker = threading.Thread(
        target=ai_worker_thread, 
        args=(detector, frame_queue, result_queue, stop_event),
        daemon=True
    )
    ai_worker.start()

    # Tạo performance monitor và chạy background thread
    monitor = PerformanceMonitor()
    monitor.start_background_monitor()

    print(f"Đang quét (Gối đầu)... Nhấn 'q' để thoát.")
    print(f"Cửa sổ: {CAMERA_WIDTH + STATS_PANEL_WIDTH}x{CAMERA_HEIGHT}")

    current_results = []
    
    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                continue

            # Đo đạc FPS phần cứng hiển thị (Chỉ xử lý resize frame 1 lần)
            h, w = frame.shape[:2]
            if h != CAMERA_HEIGHT or w != CAMERA_WIDTH:
                frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

            # Gắn frame vào hàng đợi cho AI (Chỉ gửi nếu AI đang rảnh tay)
            if frame_queue.empty():
                frame_queue.put(frame)

            # Lấy kết quả AI mới nhất không block UI
            try:
                current_results, detect_time = result_queue.get_nowait()
                # Cập nhật thông số độ trễ (AI Engine Latency)
                monitor.update_frame_time(detect_time)
            except queue.Empty:
                pass

            # Vẽ bounding boxes cực nhanh
            annotated_frame = detector.annotate_frame(frame, current_results)

            # Gửi buffer frame sang Rust xử lý zero-copy
            frame_ptr = frame.__array_interface__['data'][0]
            frame_len = frame.size
            _avg_brightness = monitor.process_frame(frame_ptr, frame_len)

            # Lấy stats cached và tạo panel thống kê (Mượt 60hz)
            stats = monitor.get_stats()
            stats_panel = create_stats_panel(stats, STATS_PANEL_WIDTH, STATS_PANEL_HEIGHT)

            # Ghép frame và stats panel
            combined_frame = np.hstack((annotated_frame, stats_panel))

            # Hiển thị
            cv2.imshow("YOLOv8 Object Detection (Multi-Threaded) - Rust Engine", combined_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        stop_event.set()
        monitor.stop_background_monitor()
        stream.stop()
        cv2.destroyAllWindows()
        print("Đã đóng camera.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection với Camera và Performance Monitor")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Tên model YOLOv8")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_ID, help="ID của camera")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Ngưỡng confidence cho detection")
    args = parser.parse_args()
    
    run_camera_detection(model_name=args.model, camera_id=args.camera, confidence_threshold=args.conf)

if __name__ == "__main__":
    main()
