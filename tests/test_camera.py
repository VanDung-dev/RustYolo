import cv2
import time
import threading
import pyarrow as pa
from rust_yolo import YoloV8Detector, PerformanceMonitor

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

def run_autonomous_scanner():
    print("🚀 Khởi tạo hệ thống YOLOv8 Autonomous Scanner (Async Multi-Threaded)...")
    
    # Khởi tạo modules từ Rust (Release mode khuyên dùng)
    try:
        detector = YoloV8Detector("yolov8s.onnx", conf_threshold=0.25, iou_threshold=0.45)
        monitor = PerformanceMonitor()
    except Exception as e:
        print(f"❌ Lỗi khởi tạo modules: {e}")
        return

    src = find_camera_src()
    if src is None:
        print("❌ Không tìm thấy camera khả dụng nào!")
        return

    # Khởi động stream ngầm
    stream = VideoStream(src).start()
    print("📸 Đang bắt đầu quét... Nhấn 'q' để thoát.")
    
    # Kích thước input của mô hình
    input_w, input_h = 640, 640

    try:
        while True:
            loop_start = time.time()
            
            ret, frame = stream.read()
            if not ret or frame is None:
                continue

            # 1. Tối ưu khâu Resize bằng OpenCV
            frame_resized = cv2.resize(frame, (input_w, input_h))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # 2. Chuẩn bị dữ liệu CHW và Normalize (Float32)
            frame_norm = frame_rgb.transpose(2, 0, 1).astype('float32') / 255.0
            
            # 3. Tối ưu: Tạo PyArrow Array trực tiếp từ NumPy (Zero-copy)
            inf_start = time.time()
            arrow_array = pa.array(frame_norm.reshape(-1), type=pa.float32())

            # 4. Gọi Rust Inference qua Arrow
            detections = detector.detect_from_arrow(arrow_array)
            inf_time = (time.time() - inf_start) * 1000
            
            # Tính FPS
            loop_time = time.time() - loop_start
            current_fps = 1.0 / loop_time if loop_time > 0 else 0
            
            # Hiển thị monitor stats
            stats = monitor.get_stats()
            mem_percent = stats['memory_usage']['percent']
            info_text = f"FPS: {current_fps:.1f} | Inf: {inf_time:.1f}ms | CPU: {stats['cpu_usage']:.1f}%"
            cv2.putText(frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Vẽ overlay lên frame (chỉ để xem)
            for det in detections:
                h, w = frame.shape[:2]
                x1 = int(det.x * w / input_w)
                y1 = int(det.y * h / input_h)
                x2 = int((det.x + det.width) * w / input_w)
                y2 = int((det.y + det.height) * h / input_h)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{COCO_CLASSES[det.class_id]}: {det.confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("YOLOv8 Autonomous Scanner", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"❌ Lỗi trong quá trình quét: {e}")
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        print("👋 Đã dừng quét.")

if __name__ == "__main__":
    run_autonomous_scanner()
