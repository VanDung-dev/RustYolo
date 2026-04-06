import cv2
import time
import sys
from rust_yolo import YoloV8Detector, PerformanceMonitor

# COCO labels for YOLOv8 (80 classes)
# COCO_CLASSES = [
#     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
#     "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#     "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
#     "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
#     "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
#     "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
#     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
#     "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
#     "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
#     "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
# ]

COCO_CLASSES = [
    "person"
]

def find_camera():
    """Tự động tìm kiếm camera khả dụng."""
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Đã tìm thấy camera tại index: {i}")
            return cap
    return None

def run_autonomous_scanner():
    print("🚀 Khởi tạo hệ thống YOLOv8n Autonomous Scanner...")
    
    # Khởi tạo modules từ Rust
    try:
        detector = YoloV8Detector("yolov8n.onnx", conf_threshold=0.25, iou_threshold=0.45)
        monitor = PerformanceMonitor()
    except Exception as e:
        print(f"❌ Lỗi khởi tạo modules: {e}")
        return
        
    monitor.start_background_monitor()
    
    cap = find_camera()
    if not cap:
        print("❌ Không tìm thấy camera khả dụng nào!")
        return
    
    print("📋 Trạng thái: Đang quét tự động (Scanning...)")
    print("💡 Nhấn 'q' để dừng hệ thống.")
    
    # Dùng để track các vật thể đã log (tránh spam console)
    last_log_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Mất kết nối camera. Đang thử lại...")
                cap.release()
                time.sleep(1)
                cap = find_camera()
                if not cap: break
                continue
                
            # 1. Quét tự động
            detections = detector.detect_from_numpy(frame)
            
            # 2. Cập nhật performance metrics
            ptr = frame.ctypes.data
            length = frame.nbytes
            _ = monitor.process_frame(ptr, length)
            
            # 3. Log các vật thể phát hiện được lên console (mỗi 1 giây một lần để dễ nhìn)
            current_time = time.time()
            if current_time - last_log_time > 1.0 and len(detections) > 0:
                found_objects = [COCO_CLASSES[d.class_id] for d in detections if d.class_id < len(COCO_CLASSES)]
                print(f"🔍 [Scanner] Đã phát hiện: {', '.join(set(found_objects))}")
                last_log_time = current_time
            
            # 4. Hiển thị Overlay
            for det in detections:
                x, y, w, h = int(det.x), int(det.y), int(det.width), int(det.height)
                label = COCO_CLASSES[det.class_id] if det.class_id < len(COCO_CLASSES) else "Unknown"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 255, 50), 2)
                cv2.putText(frame, f"{label} {det.confidence:.2f}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 255, 50), 1)
            
            # Dashboard mini
            stats = monitor.get_stats()
            cv2.rectangle(frame, (5, 5), (220, 110), (0, 0, 0), -1) # Background cho stats
            cv2.putText(frame, f"FPS: {stats['fps']:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Latency: {stats['latency']:.1f}ms", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"CPU: {stats['cpu_usage']:.1f}%", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, "STATUS: SCANNING", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow('YOLO Autonomous Scanner', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        print("\n⏹️ Đang dừng hệ thống...")
        cap.release()
        cv2.destroyAllWindows()
        monitor.stop_background_monitor()
        print("✅ Hoàn tất.")

if __name__ == "__main__":
    run_autonomous_scanner()
