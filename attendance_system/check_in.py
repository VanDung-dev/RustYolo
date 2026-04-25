"""
Ứng dụng điểm danh tự động tích hợp phát hiện người và chống giả mạo.
"""

import cv2
import os
import time
import rust_yolo
from core import AttendanceCore
import argparse
import pyarrow as pa
import threading
import queue
import logging
from ui_utils import draw_text
import config

# Cấu hình logger cho module
logger = logging.getLogger(__name__)

class VideoStream:
    """
    VideoStream: Luồng đọc Camera đa luồng để đảm bảo AI xử lý 
    không làm chậm việc hiển thị hình ảnh.
    """
    def __init__(self, src=0, width=1280, height=720):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        self.frame_queue = queue.Queue(maxsize=2)
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                self.stopped = True
                break
            
            # Xóa frame cũ nếu queue đầy để luôn có ảnh mới nhất
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def read(self):
        try:
            return True, self.frame_queue.get(timeout=0.1)
        except queue.Empty:
            return False, None

    def stop(self):
        self.stopped = True
        self.stream.release()

def main():
    parser = argparse.ArgumentParser(description="Hệ thống điểm danh nâng cao với CoreML/WebGPU")
    parser.add_argument("--ep", type=str, default="coreml", help="Execution Provider (coreml, webgpu, cpu)")
    args = parser.parse_args()

    # Đường dẫn tuyệt đối tới các mô hình ONNX từ config
    PERSON_MODEL = config.YOLO_MODEL_PATH
    FACE_MODEL = config.FACE_DETECTOR_PATH
    ARCFACE_MODEL = config.FACE_EMBEDDER_PATH
    db_path = config.DB_PATH
    
    logger.info(f"\n--- HỆ THỐNG ĐIỂM DANH TỰ ĐỘNG (EP: {args.ep.upper()}) ---")
    
    # 1. Kiểm tra tài nguyên
    for m in [PERSON_MODEL, FACE_MODEL, ARCFACE_MODEL]:
        if not os.path.exists(m):
            logger.error(f"  [LỖI] Không tìm thấy model tại: {m}")
            return

    # 2. Khởi tạo Engine xử lý
    try:
        logger.info(f"-> Đang nạp Mô hình phát hiện Người ({args.ep})...")
        person_detector = rust_yolo.YoloV8Detector(PERSON_MODEL, 0.5, 0.4, args.ep)
        
        logger.info(f"-> Đang khởi tạo Logic Face ID ({args.ep})...")
        core = AttendanceCore(FACE_MODEL, ARCFACE_MODEL, db_path, execution_provider=args.ep)
        logger.info("-> Hệ thống đã sẵn sàng vận hành.")
    except Exception as e:
        logger.error(f"  [LỖI KHỞI TẠO] {e}")
        return

    # 3. Khởi chạy luồng Camera
    vstream = VideoStream(0, config.CAMERA_WIDTH, config.CAMERA_HEIGHT).start()
    
    prev_time = time.time()
    last_recognition_time = 0
    recognition_cooldown = config.RECOGNITION_COOLDOWN # Thời gian nghỉ giữa các lần nhận diện (giảm tải CPU)
    
    logger.info("\n[INFO] Hệ thống đang hoạt động. Nhấn 'q' để thoát.")

    while True:
        ret, frame = vstream.read()
        if not ret: break

        display_frame = frame.copy()
        
        # BƯỚC 1: Sử dụng YOLO để lọc Người & Điện thoại (Chống giả mạo cơ bản)
        # Tối ưu Zero-copy qua Arrow Capsule
        res_caps = person_detector.detect_to_arrow(frame)
        person_arr = pa.Array._import_from_c_capsule(res_caps[1], res_caps[0])
        
        has_person = False
        phone_detected = False
        
        if len(person_arr) > 0:
            class_ids = person_arr.field("class_id").to_numpy()
            boxes_x = person_arr.field("x").to_numpy()
            boxes_y = person_arr.field("y").to_numpy()
            boxes_w = person_arr.field("w").to_numpy()
            boxes_h = person_arr.field("h").to_numpy()

            for i in range(len(person_arr)):
                cid = class_ids[i]
                if cid == 0: # person
                    has_person = True
                    x1, y1 = int(boxes_x[i]), int(boxes_y[i])
                    x2, y2 = int(boxes_x[i] + boxes_w[i]), int(boxes_y[i] + boxes_h[i])
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                elif cid == 67: # cell phone
                    phone_detected = True
                    phx, phy = int(boxes_x[i]), int(boxes_y[i])
                    # Vẽ cảnh báo đỏ nếu phát hiện điện thoại (gian lận bằng ảnh/video)
                    cv2.rectangle(display_frame, (phx, phy), (int(boxes_x[i] + boxes_w[i]), int(boxes_y[i] + boxes_h[i])), (0, 0, 255), 2)
                    draw_text(display_frame, "PHÁT HIỆN ĐIỆN THOẠI", (phx, phy-25), 18, (0, 0, 255), 2)

        # BƯỚC 2: Chỉ nhận diện mặt khi CÓ NGƯỜI và KHÔNG thấy điện thoại
        if has_person and not phone_detected:
            current_time = time.time()
            if current_time - last_recognition_time > recognition_cooldown:
                face_results = core.process_frame(frame)
                for res in face_results:
                    identity = res['identity']
                    score = res['confidence']
                    x1, y1, x2, y2 = map(int, res['bbox'])
                    
                    # Màu sắc UI: Xanh cho nhân viên, Cam cho người lạ
                    color = (0, 255, 0) if identity != "Unknown" else (0, 165, 255)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    draw_text(display_frame, f"{identity} ({score:.2f})", (x1, y1-30),
                                20, color, 2)
                                
                    if identity != "Unknown":
                        # Tự động ghi nhận vào database điểm danh
                        core.log_attendance(res['user_id'])
                        logger.info(f"  [ĐIỂM DANH] {identity} - Độ tin cậy: {score:.2f}")
                
                last_recognition_time = current_time

        # Tính toán và hiển thị hiệu năng (FPS)
        curr_fps_time = time.time()
        fps = 1 / (curr_fps_time - prev_time)
        prev_time = curr_fps_time
        
        status_text = "DANG QUET..." if has_person else "CHO NGUOI..."
        draw_text(display_frame, f"FPS: {fps:.1f} | {status_text}", (20, 20), 
                    24, (0, 255, 255), 2)

        # Hiển thị kết quả cuối cùng
        cv2.imshow("He thong Diem danh Thong minh", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vstream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
