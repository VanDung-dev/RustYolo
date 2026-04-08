"""
Kiểm tra cơ chế xử lý dữ liệu giữa python và rust
"""

import cv2
import time
import threading
import queue
from rust_yolo import YoloV8Detector, PerformanceMonitor
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# COCO labels for YOLOv8 (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

class VideoStream:
    """Threaded video capture stream to handle I/O without blocking"""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        
        # Yêu cầu Camera chạy ở 1080p và 60 FPS (nếu phần cứng hỗ trợ)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.stream.set(cv2.CAP_PROP_FPS, 60)
        
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

def find_camera_src():
    """Tự động tìm camera khả dụng"""
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return i
    return None

# Hàng đợi để truyền dữ liệu giữa các luồng
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

def ai_worker(detector, input_w, input_h):
    """Luồng chuyên biệt để chạy suy luận AI"""
    logger.info("AI Worker Thread đã sẵn sàng.")
    while not stop_event.is_set():
        try:
            # Lấy ảnh mới nhất ra để xử lý
            frame_bytes = frame_queue.get(timeout=0.1)
            if frame_bytes is None: continue
            
            inf_start = time.time()
            detections = detector.detect_from_bytes(frame_bytes, input_w, input_h)
            inf_time = (time.time() - inf_start) * 1000
            
            # Gửi kết quả về kèm thời gian xử lý
            if result_queue.full():
                try: result_queue.get_nowait() # Xóa kết quả cũ nếu chưa ai lấy
                except queue.Empty: pass
            result_queue.put((detections, inf_time))
            
            frame_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Lỗi trong AI Worker: {e}")

def run_autonomous_scanner():
    logger.info("Khởi tạo hệ thống YOLOv8 Autonomous Scanner (Kiến trúc PIPELINE)...")
    
    # Kích thước input của mô hình
    input_w, input_h = 640, 640
    
    # Khởi tạo modules từ Rust (Release mode khuyên dùng)
    try:
        # Sử dụng bản Nano tối ưu
        detector = YoloV8Detector("yolov8n.onnx", conf_threshold=0.25, iou_threshold=0.45)
        monitor = PerformanceMonitor()
    except Exception as e:
        logger.error(f"Lỗi khởi tạo modules: {e}")
        return

    src = find_camera_src()
    if src is None:
        logger.error("Không tìm thấy camera khả dụng nào!")
        return

    # Khởi động AI Worker Thread
    worker = threading.Thread(target=ai_worker, args=(detector, input_w, input_h), daemon=True)
    worker.start()

    # Khởi động stream capture
    stream = VideoStream(src).start()
    time.sleep(1.0) # Đợi camera ổn định
    logger.info("Đang quét (Gối đầu)... Nhấn 'q' để thoát.")
    
    current_detections = []
    current_inf_time = 0.0

    try:
        while True:
            loop_start = time.time()
            
            ret, frame = stream.read()
            if not ret or frame is None:
                continue

            # 1. Pipeline Stage 1: Pre-processing (Khinh công)
            frame_resized = cv2.resize(frame, (input_w, input_h))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_bytes = frame_rgb.tobytes()

            # Gửi ảnh vào hàng đợi AI nếu hàng đợi đang trống
            if frame_queue.empty():
                frame_queue.put(frame_bytes)

            # 2. Pipeline Stage 2: Cập nhật kết quả AI (nếu đã xử lý xong)
            try:
                # Không block luồng hiển thị
                new_results = result_queue.get_nowait()
                current_detections, current_inf_time = new_results
            except queue.Empty:
                pass

            # 3. Pipeline Stage 3: Hiển thị mượt mà
            # Tính FPS của vòng lặp UI (Camera render)
            loop_time = time.time() - loop_start
            display_fps = 1.0 / loop_time if loop_time > 0 else 0
            
            # Tính FPS thực tế của Engine AI (Rust)
            ai_fps = 1000.0 / current_inf_time if current_inf_time > 0 else 0
            
            # Hiển thị monitor stats
            stats = monitor.get_stats()
            info_text = f"UI FPS: {display_fps:.1f} | AI FPS: {ai_fps:.1f} ({current_inf_time:.1f}ms)"
            cpu_text = f"CPU: {stats['cpu_usage']:.1f}%"
            
            cv2.putText(frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, cpu_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Vẽ overlay Bounding Boxes (kết quả gần nhất)
            h, w = frame.shape[:2]
            for det in current_detections:
                x1 = int(det.x * w / input_w)
                y1 = int(det.y * h / input_h)
                x2 = int((det.x + det.width) * w / input_w)
                y2 = int((det.y + det.height) * h / input_h)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{COCO_CLASSES[det.class_id]}: {det.confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("YOLOv8 HIGH-FPS Pipeline", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logger.error(f"Lỗi trong quá trình quét: {e}")
    finally:
        stop_event.set()
        stream.stop()
        cv2.destroyAllWindows()
        logger.info("Đã dừng quét.")

if __name__ == "__main__":
    run_autonomous_scanner()
