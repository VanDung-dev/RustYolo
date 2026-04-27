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
import logging
from ui_utils import draw_text
from collections import deque
import config
import numpy as np
from video_utils import VideoStream

# Cấu hình logger cho module
logger = logging.getLogger(__name__)


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

def _get_processed_frame(vstream, frame_buffer, buffer_size):
    """Lấy frame từ stream và quản lý bộ đệm delay."""
    ret, frame = vstream.read()
    if not ret:
        return False, None, None
        
    frame_buffer.append(frame)
    if len(frame_buffer) < buffer_size:
        return True, None, None
        
    frame = frame_buffer.popleft()
    return True, frame, frame.copy()

def _handle_detections(person_detector, frame, display_frame):
    """Thực hiện phát hiện người và thiết bị, vẽ kết quả lên display_frame."""
    res_caps = person_detector.detect_to_arrow(frame)
    person_arr = pa.Array._import_from_c_capsule(res_caps[1], res_caps[0])
    
    if len(person_arr) == 0:
        return False, False, person_arr

    class_ids = person_arr.field("class_id").to_numpy()
    
    # Phát hiện thiết bị cấm
    illegal_classes = [config.CLASS_PHONE, config.CLASS_TV, config.CLASS_LAPTOP]
    phone_idxs = np.where(np.isin(class_ids, illegal_classes))[0]
    phone_detected = len(phone_idxs) > 0
    
    # Phát hiện người (bỏ qua nếu có phone)
    person_idxs = np.where(class_ids == config.CLASS_PERSON)[0] if not phone_detected else []
    has_person = len(person_idxs) > 0

    if phone_detected:
        _draw_phone_warning(display_frame, person_arr, phone_idxs[0])
    
    if has_person:
        _draw_person_boxes(display_frame, person_arr, person_idxs)
        
    return phone_detected, has_person, person_arr

def _draw_phone_warning(display_frame, person_arr, idx):
    """Vẽ cảnh báo khi phát hiện điện thoại/thiết bị cấm."""
    bx, by, bw, bh = [int(person_arr.field(f).to_numpy()[idx]) for f in ["x", "y", "w", "h"]]
    
    # Shield đỏ
    shield_overlay = display_frame.copy()
    cv2.rectangle(shield_overlay, (bx, by), (bx + bw, by + bh), config.COLOR_DANGER, -1)
    cv2.addWeighted(
        shield_overlay, config.UI_ALPHA_SHIELD, display_frame, 1 - config.UI_ALPHA_SHIELD, 0, display_frame
    )
    
    cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), config.COLOR_INFO, config.UI_RECT_THICKNESS)
    draw_text(
        display_frame, "DANGER: DEVICE DETECTED", (bx, by-35),
        config.FONT_SIZE_MEDIUM, config.COLOR_DANGER, is_bold=1
    )

def _draw_person_boxes(display_frame, person_arr, person_idxs):
    """Vẽ khung bao quanh những người được phát hiện."""
    xs = person_arr.field("x").to_numpy()[person_idxs].astype(int)
    ys = person_arr.field("y").to_numpy()[person_idxs].astype(int)
    ws = person_arr.field("w").to_numpy()[person_idxs].astype(int)
    hs = person_arr.field("h").to_numpy()[person_idxs].astype(int)
    for bx, by, bw, bh in zip(xs, ys, ws, hs):
        cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (255, 0, 0), config.UI_RECT_THICKNESS)

def _update_system_state(now, phone_detected, has_person, lock_until_time, last_person_seen_time):
    """Cập nhật trạng thái khóa và chế độ nghỉ của hệ thống."""
    if phone_detected:
        lock_until_time = now + config.SECURITY_LOCK_DURATION
    
    countdown = max(0, lock_until_time - now)
    is_system_ready = (countdown == 0) and not phone_detected
    
    if has_person or phone_detected or countdown > 0:
        last_person_seen_time = now
        is_resting = False
    else:
        is_resting = (now - last_person_seen_time > config.POWER_SAVING_THRESHOLD)
        
    return lock_until_time, last_person_seen_time, is_resting, countdown, is_system_ready

def _handle_recognition(
    core, frame, display_frame, face_results, now, last_recognition_time, has_person, is_system_ready
):
    """Xử lý nhận diện khuôn mặt và log điểm danh."""
    can_recognize = has_person and is_system_ready and (now - last_recognition_time > config.RECOGNITION_COOLDOWN)
    
    if can_recognize:
        face_results = core.process_frame(frame)
        last_recognition_time = now
    
    if not (has_person and is_system_ready):
        face_results = []
        
    for res in face_results:
        _draw_recognition_result(display_frame, res, now, last_recognition_time, core)
        
    return face_results, last_recognition_time

def _draw_recognition_result(display_frame, res, now, last_recognition_time, core):
    """Vẽ kết quả nhận diện một khuôn mặt."""
    identity, score = res['identity'], res['confidence']
    x1, y1, x2, y2 = map(int, res['bbox'])
    color = config.COLOR_SUCCESS if identity != "Unknown" else config.COLOR_WARNING
    
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, config.UI_RECT_THICKNESS)
    
    # Label background
    label_overlay = display_frame.copy()
    cv2.rectangle(label_overlay, (x1, y1 - 35), (x1 + 200, y1), (0, 0, 0), -1)
    cv2.addWeighted(label_overlay, 0.5, display_frame, 0.5, 0, display_frame)
    
    draw_text(
        display_frame, f"{identity} ({score:.2f})", (x1 + 5, y1 - 30),
        config.FONT_SIZE_SMALL, color, is_bold=1
    )

    if identity != "Unknown" and now - last_recognition_time < config.LOG_ATTENDANCE_WINDOW:
        core.log_attendance(res['user_id'])
        logger.info(f"  [ĐIỂM DANH] {identity} - Độ tin cậy: {score:.2f}")

def _draw_ui_overlay(display_frame, fps, phone_detected, countdown, is_resting, has_person):
    """Vẽ overlay trạng thái hệ thống và FPS."""
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
        config.FONT_SIZE_LARGE, status_color, is_bold=1
    )

def run_attendance_system(vstream, person_detector, core):
    """Vòng lặp xử lý chính của hệ thống."""
    # Biến quản lý thời gian và hiệu năng
    prev_time = time.time()
    last_recognition_time = 0
    lock_until_time = 0
    last_person_seen_time = time.time()
    face_results = []
    
    # Khởi tạo bộ đệm Video (Stream Delay)
    frame_buffer = deque()
    buffer_size = int((config.STREAM_DELAY_MS / 1000) * config.ACTIVE_MODE_FPS)
    
    logger.info(f"\n[INFO] Hệ thống đang vận hành (Delay: {config.STREAM_DELAY_MS}ms). Nhấn 'q' để thoát.")

    while True:
        now = time.time()
        
        # 1. Lấy frame và quản lý bộ đệm
        ret, frame, display_frame = _get_processed_frame(vstream, frame_buffer, buffer_size)
        if not ret: break
        if frame is None: continue # Đang nạp buffer
        
        # 2. Phát hiện đối tượng (YOLO)
        phone_detected, has_person, _ = _handle_detections(person_detector, frame, display_frame)

        # 3. Quản lý trạng thái hệ thống
        lock_until_time, last_person_seen_time, is_resting, countdown, is_system_ready = \
            _update_system_state(now, phone_detected, has_person, lock_until_time, last_person_seen_time)

        # 4. Xử lý Face ID
        face_results, last_recognition_time = \
            _handle_recognition(core, frame, display_frame, face_results, now, last_recognition_time, has_person, is_system_ready)

        # 5. Hiển thị UI & FPS
        fps = 1 / (now - prev_time) if now > prev_time else 0
        prev_time = now
        _draw_ui_overlay(display_frame, fps, phone_detected, countdown, is_resting, has_person)
        
        cv2.imshow("He thong Diem danh Thong minh", display_frame)
        
        # 6. Kiểm soát FPS
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
