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

    logger.info(f"\n--- HỆ THỐNG ĐIỂM DANH TỰ ĐỘNG (EP: {args.ep.upper()}) ---")
    
    # 1. Kiểm tra tài nguyên
    for m in [config.YOLO_MODEL_PATH, config.FACE_DETECTOR_PATH, config.FACE_EMBEDDER_PATH]:
        if not os.path.exists(m):
            logger.error(f"  [LỖI] Không tìm thấy model tại: {m}")
            return

    # 2. Khởi tạo Engine xử lý
    try:
        logger.info(f"-> Đang nạp Mô hình phát hiện Người ({args.ep})...")
        person_detector = rust_yolo.YoloV8Detector(config.YOLO_MODEL_PATH, 0.5, 0.4, args.ep)
        
        logger.info(f"-> Đang khởi tạo Logic Face ID ({args.ep})...")
        core = AttendanceCore(config.FACE_DETECTOR_PATH, config.FACE_EMBEDDER_PATH, config.DB_PATH, execution_provider=args.ep)
        logger.info("-> Hệ thống đã sẵn sàng vận hành.")
    except Exception as e:
        logger.error(f"  [LỖI KHỞI TẠO] {e}")
        return

    # 3. Khởi chạy luồng Camera
    vstream = VideoStream(0, config.CAMERA_WIDTH, config.CAMERA_HEIGHT).start()
    
    prev_time = time.time()
    last_recognition_time = 0
    
    # Biến quản lý trạng thái
    lock_until_time = 0
    last_person_seen_time = time.time()
    is_resting = False
    
    logger.info("\n[INFO] Hệ thống đang hoạt động. Nhấn 'q' để thoát.")

    while True:
        ret, frame = vstream.read()
        if not ret: break

        # --- BẮT ĐẦU CHU TRÌNH XỬ LÝ ---
        now = time.time()
        display_frame = frame.copy()
        
        # BƯỚC 1: Sử dụng YOLO để lọc Người & Điện thoại (Zero-copy qua Arrow)
        res_caps = person_detector.detect_to_arrow(frame)
        person_arr = pa.Array._import_from_c_capsule(res_caps[1], res_caps[0])
        
        has_person = False
        phone_detected = False
        person_boxes = []

        if len(person_arr) > 0:
            class_ids = person_arr.field("class_id").to_numpy()
            boxes_x = person_arr.field("x").to_numpy()
            boxes_y = person_arr.field("y").to_numpy()
            boxes_w = person_arr.field("w").to_numpy()
            boxes_h = person_arr.field("h").to_numpy()

            # Tối ưu: Duyệt một lần để xác định cả Phone và Person
            for i in range(len(person_arr)):
                cid = class_ids[i]
                if cid == config.CLASS_PHONE:
                    phone_detected = True
                    # Tìm thấy điện thoại -> Ưu tiên cao nhất, thoát sớm
                    phx, phy, phw, phh = int(boxes_x[i]), int(boxes_y[i]), int(boxes_w[i]), int(boxes_h[i])
                    cv2.rectangle(display_frame, (phx, phy), (phx + phw, phy + phh), (0, 0, 255), -1)
                    cv2.rectangle(display_frame, (phx, phy), (phx + phw, phy + phh), (255, 255, 255), 1)
                    draw_text(display_frame, "DANGER: PHONE DETECTED", (phx, phy-35), 22, (0, 0, 255), 2)
                    has_person = False 
                    break
                elif cid == config.CLASS_PERSON:
                    has_person = True
                    person_boxes.append((int(boxes_x[i]), int(boxes_y[i]), int(boxes_w[i]), int(boxes_h[i])))

            # Vẽ box người nếu không bị khóa bởi điện thoại
            if not phone_detected:
                for box in person_boxes:
                    cv2.rectangle(display_frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255, 0, 0), 1)

        # BƯỚC 2: QUẢN LÝ TRẠNG THÁI KHÓA VÀ NGHỈ (POWER SAVING)
        is_system_ready = True
        countdown = 0
        
        if phone_detected:
            lock_until_time = now + config.SECURITY_LOCK_DURATION
            last_person_seen_time = now
            is_system_ready = False
            is_resting = False
        else:
            if now < lock_until_time:
                is_system_ready = False
                countdown = lock_until_time - now
                last_person_seen_time = now # Đang đếm ngược thì chưa tính là "không thấy người"
            
            if has_person:
                last_person_seen_time = now
                is_resting = False
            elif now - last_person_seen_time > config.POWER_SAVING_THRESHOLD:
                is_resting = True

        # BƯỚC 3: XỬ LÝ FACE ID - Chỉ khi hệ thống sẵn sàng
        if has_person and is_system_ready:
            if now - last_recognition_time > config.RECOGNITION_COOLDOWN:
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
                
                last_recognition_time = now

        # Tính toán và hiển thị hiệu năng (FPS)
        curr_fps_time = time.time()
        fps = 1 / (curr_fps_time - prev_time)
        prev_time = curr_fps_time
        
        # --- BƯỚC 4: HIỂN THỊ UI & FPS ---
        # Tối ưu hóa: Sử dụng Rule-based thay vì chuỗi if-elif dài
        status_rules = [
            (phone_detected, "HE THONG BI KHOA", (0, 0, 255)),
            (countdown > 0, f"KHOI DONG LAI TRONG {countdown:.1f}s...", (0, 165, 255)),
            (is_resting, "CHE DO NGHI", (150, 150, 150)),
            (has_person, "DANG QUET...", (0, 255, 255)),
            (True, "CHO NGUOI...", (0, 255, 255)) # Mặc định
        ]
        status_text, status_color = next((txt, clr) for cond, txt, clr in status_rules if cond)

        draw_text(display_frame, f"FPS: {fps:.1f} | {status_text}", (20, 20), 
                    24, status_color, 2)

        # Hiển thị kết quả
        cv2.imshow("He thong Diem danh Thong minh", display_frame)
        
        # Kiểm soát FPS: Tính toán thời gian xử lý thực tế để sleep chính xác
        elapsed = time.time() - now
        target_fps = config.REST_MODE_FPS if is_resting else config.ACTIVE_MODE_FPS
        sleep_time = max(0.001, (1.0 / target_fps) - elapsed)
        time.sleep(sleep_time)

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
