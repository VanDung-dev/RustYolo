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
from collections import deque
import config
import numpy as np

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

def initialize_engines(ep):
    """Kiểm tra tài nguyên và khởi tạo các engine AI."""
    # 1. Kiểm tra tài nguyên
    for m in [config.YOLO_MODEL_PATH, config.FACE_DETECTOR_PATH, config.FACE_EMBEDDER_PATH]:
        if not os.path.exists(m):
            raise FileNotFoundError(f"Không tìm thấy model tại: {m}")

    logger.info(f"-> Đang nạp Mô hình phát hiện Người ({ep})...")
    person_detector = rust_yolo.YoloV8Detector(
        config.YOLO_MODEL_PATH, 
        config.YOLO_CONF_THRESHOLD, 
        config.YOLO_IOU_THRESHOLD, 
        ep
    )
    
    logger.info(f"-> Đang khởi tạo Logic Face ID ({ep})...")
    core = AttendanceCore(config.FACE_DETECTOR_PATH, config.FACE_EMBEDDER_PATH, config.DB_PATH, execution_provider=ep)
    
    return person_detector, core

def run_attendance_system(vstream, person_detector, core):
    """Vòng lặp xử lý chính của hệ thống."""
    # Biến quản lý thời gian và hiệu năng
    prev_time = time.time()
    last_recognition_time = 0
    lock_until_time = 0
    last_person_seen_time = time.time()
    
    # Trạng thái hệ thống
    face_results = []
    
    # Khởi tạo bộ đệm Video (Stream Delay như OBS)
    frame_buffer = deque()
    buffer_size = int((config.STREAM_DELAY_MS / 1000) * config.ACTIVE_MODE_FPS)
    
    logger.info(f"\n[INFO] Hệ thống đang vận hành (Delay: {config.STREAM_DELAY_MS}ms). Nhấn 'q' để thoát.")

    while True:
        ret, frame = vstream.read()
        if not ret: break

        # 1. Quản lý bộ đệm (Stream Delay)
        frame_buffer.append(frame)
        if len(frame_buffer) < buffer_size:
            continue
            
        frame = frame_buffer.popleft()
        display_frame = frame.copy()
        now = time.time()
        
        # 2. Phát hiện đối tượng (YOLO)
        res_caps = person_detector.detect_to_arrow(frame)
        person_arr = pa.Array._import_from_c_capsule(res_caps[1], res_caps[0])
        class_ids = person_arr.field("class_id").to_numpy() if len(person_arr) > 0 else np.array([])
        
        # Xác định chỉ số Phone và Person ngay lập tức
        phone_idxs = np.where(class_ids == config.CLASS_PHONE)[0]
        phone_detected = len(phone_idxs) > 0
        person_idxs = np.where(class_ids == config.CLASS_PERSON)[0] if not phone_detected else []
        has_person = len(person_idxs) > 0

        # Vẽ kết quả YOLO (Nếu có phone thì ưu tiên vẽ phone và bỏ qua người)
        if phone_detected:
            idx = phone_idxs[0]
            bx, by, bw, bh = [int(person_arr.field(f).to_numpy()[idx]) for f in ["x", "y", "w", "h"]]
            cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), config.COLOR_DANGER, -1)
            cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), config.COLOR_INFO, 1)
            draw_text(
                display_frame, "DANGER: PHONE DETECTED", (bx, by-35),
                config.FONT_SIZE_MEDIUM, config.COLOR_DANGER, is_bold=True
            )
            
        if has_person:
            xs = person_arr.field("x").to_numpy()[person_idxs].astype(int)
            ys = person_arr.field("y").to_numpy()[person_idxs].astype(int)
            ws = person_arr.field("w").to_numpy()[person_idxs].astype(int)
            hs = person_arr.field("h").to_numpy()[person_idxs].astype(int)
            for bx, by, bw, bh in zip(xs, ys, ws, hs):
                cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (255, 0, 0), 1)

        # 3. Quản lý trạng thái Khóa/Nghỉ - Tối ưu logic làm phẳng
        if phone_detected:
            lock_until_time = now + config.SECURITY_LOCK_DURATION
        
        countdown = max(0, lock_until_time - now)
        is_system_ready = (countdown == 0) and not phone_detected
        
        # Cập nhật thời điểm nhìn thấy người (Hoặc trạng thái đang xử lý/khóa)
        if has_person or phone_detected or countdown > 0:
            last_person_seen_time = now
            is_resting = False
        else:
            is_resting = (now - last_person_seen_time > config.POWER_SAVING_THRESHOLD)

        # 4. Xử lý Face ID (Sampling Rate)
        can_recognize = has_person and is_system_ready and (now - last_recognition_time > config.RECOGNITION_COOLDOWN)
        if can_recognize:
            face_results = core.process_frame(frame)
            last_recognition_time = now
        
        # Xóa kết quả nếu mất dấu người hoặc bị khóa bảo mật
        if not (has_person and is_system_ready):
            face_results = []
            
        for res in face_results:
            identity, score = res['identity'], res['confidence']
            x1, y1, x2, y2 = map(int, res['bbox'])
            color = config.COLOR_SUCCESS if identity != "Unknown" else config.COLOR_WARNING
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ nền đen trong suốt cho label tên (giúp chữ nổi bật hơn)
            label_w = 160 # Chiều rộng ước tính cho text
            label_h = 35
            label_overlay = display_frame.copy()
            cv2.rectangle(label_overlay, (x1, y1 - label_h), (x1 + label_w, y1), (0, 0, 0), -1)
            cv2.addWeighted(label_overlay, 0.5, display_frame, 0.5, 0, display_frame)
            
            draw_text(
                display_frame, f"{identity} ({score:.2f})", (x1 + 5, y1 - 30),
                config.FONT_SIZE_SMALL, color, is_bold=True
            )
            if identity != "Unknown" and now - last_recognition_time < config.LOG_ATTENDANCE_WINDOW:
                core.log_attendance(res['user_id'])
                logger.info(f"  [ĐIỂM DANH] {identity} - Độ tin cậy: {score:.2f}")

        # 5. Hiển thị UI & FPS
        curr_fps_time = time.time()
        fps = 1 / (curr_fps_time - prev_time)
        prev_time = curr_fps_time
        
        # Vẽ nền đen trong suốt cho thanh trạng thái phía trên
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 65), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
        
        status_rules = [
            (phone_detected, "HỆ THỐNG BỊ KHÓA", config.COLOR_DANGER),
            (countdown > 0, f"KHỞI ĐỘNG LẠI TRONG {countdown:.1f}s...", config.COLOR_WARNING),
            (is_resting, "CHẾ ĐỘ NGHỈ", config.COLOR_RESTING),
            (has_person, "ĐANG QUÉT...", config.COLOR_SCANNING),
            (True, "CHỜ NGƯỜI...", config.COLOR_SCANNING)
        ]
        status_text, status_color = next((txt, clr) for cond, txt, clr in status_rules if cond)
        draw_text(
            display_frame, f"FPS: {fps:.1f} | {status_text}", (20, 20),
            config.FONT_SIZE_LARGE, status_color, is_bold=True
        )
        
        cv2.imshow("He thong Diem danh Thong minh", display_frame)
        
        # 6. Kiểm soát FPS (Power Saving)
        elapsed = time.time() - now
        target_fps = config.REST_MODE_FPS if is_resting else config.ACTIVE_MODE_FPS
        sleep_time = max(0.001, (1.0 / target_fps) - elapsed)
        time.sleep(sleep_time)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    vstream.stop()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Hệ thống điểm danh nâng cao với CoreML/WebGPU")
    parser.add_argument("--camera", type=str, default="0", help="Camera ID (0, 1) hoặc URL stream (rtsp://...)")
    parser.add_argument("--ep", type=str, default="coreml", help="Execution Provider (coreml, webgpu, cpu)")
    args = parser.parse_args()

    logger.info(f"\n--- HỆ THỐNG ĐIỂM DANH TỰ ĐỘNG (EP: {args.ep.upper()}) ---")
    
    try:
        # Khởi tạo Engine
        person_detector, core = initialize_engines(args.ep)
        
        # Khởi chạy luồng Camera
        cam_src = int(args.camera) if args.camera.isdigit() else args.camera
        vstream = VideoStream(cam_src, config.CAMERA_WIDTH, config.CAMERA_HEIGHT).start()
        
        # Chạy vòng lặp xử lý chính
        run_attendance_system(vstream, person_detector, core)
        
    except Exception as e:
        logger.error(f"  [LỖI HỆ THỐNG] {e}")
        return

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
