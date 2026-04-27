"""
Quy trình đăng ký nhân viên mới với 8 góc độ khuôn mặt.
"""

import cv2
import numpy as np
import time
import os
import pyarrow as pa
import rust_yolo
import logging
import argparse
from core import AttendanceCore
from ui_utils import draw_text
from collections import namedtuple
import config
from video_utils import VideoStream

# Cấu hình logger cho module
logger = logging.getLogger(__name__)

# Cấu trúc dữ liệu tạm cho phát hiện khuôn mặt
FaceDetection = namedtuple("FaceDetection", ["bbox", "score", "landmarks"])


def initialize_registration_engines(ep="coreml"):
    """Khởi tạo các engine AI cho quy trình đăng ký."""
    logger.info(f"Đang khởi tạo hệ thống đăng ký với EP: {ep.upper()}...")
    try:
        # Kiểm tra tài nguyên mô hình
        for m in [config.YOLO_MODEL_PATH, config.FACE_DETECTOR_PATH, config.FACE_EMBEDDER_PATH]:
            if not os.path.exists(m):
                raise FileNotFoundError(f"Không tìm thấy model tại: {m}")

        person_detector = rust_yolo.YoloV8Detector(
            config.YOLO_MODEL_PATH, 
            config.YOLO_CONF_THRESHOLD, 
            config.YOLO_IOU_THRESHOLD, 
            ep
        )
        core = AttendanceCore(
            config.FACE_DETECTOR_PATH, 
            config.FACE_EMBEDDER_PATH, 
            config.DB_PATH, 
            execution_provider=ep
        )
        logger.info("Hệ thống AI đã sẵn sàng.")
        return person_detector, core
    except Exception as e:
        raise RuntimeError(f"Không thể khởi tạo Engine: {e}")

def _prepare_frame(vstream):
    """Lấy frame, lật và cắt thành hình vuông."""
    ret, frame = vstream.read()
    if not ret:
        return False, None, None
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    target_size = min(h, w, config.REG_CROP_H)
    start_x, start_y = (w - target_size) // 2, (h - target_size) // 2
    frame_sq = frame[start_y:start_y+target_size, start_x:start_x+target_size]
    
    return True, frame, frame_sq

def _check_safety(person_detector, frame):
    """Kiểm tra xem có thiết bị cấm nào không."""
    res_caps = person_detector.detect_to_arrow(frame)
    person_arr = pa.Array._import_from_c_capsule(res_caps[1], res_caps[0])
    if len(person_arr) > 0:
        illegal_classes = [config.CLASS_PHONE, config.CLASS_TV, config.CLASS_LAPTOP]
        return any(cid in illegal_classes for cid in person_arr.field("class_id").to_numpy())
    return False

def _detect_registration_face(core, frame_sq, bx1, by1, bx2, by2):
    """Phát hiện khuôn mặt và kiểm tra xem có nằm trong Guide Box không."""
    rgb_sq = cv2.cvtColor(frame_sq, cv2.COLOR_BGR2RGB)
    target_face, face_in_box = None, False
    try:
        arr_cap, sch_cap = core.face_tools.detect_faces_to_arrow(rgb_sq, config.FACE_DET_THRESHOLD)
        detections = pa.Array._import_from_c_capsule(sch_cap, arr_cap)
        if len(detections) > 0:
            # Chọn khuôn mặt lớn nhất
            best_idx = np.argmax([(d['x2'].as_py()-d['x1'].as_py())*(d['y2'].as_py()-d['y1'].as_py()) for d in detections])
            det = detections[best_idx]
            fx1, fy1, fx2, fy2 = [det[f].as_py() for f in ['x1', 'y1', 'x2', 'y2']]
            target_face = FaceDetection(
                bbox=[fx1, fy1, fx2, fy2], score=det['score'].as_py(), landmarks=det['landmarks'].as_py()
            )
            margin = 20
            face_in_box = (fx1 > bx1-margin and fy1 > by1-margin and fx2 < bx2+margin and fy2 < by2+margin)
    except Exception as e:
        logger.error(f"Lỗi nhận diện khuôn mặt: {e}")
    return target_face, face_in_box, rgb_sq

def _draw_registration_ui(
    frame_sq, step_info, current_step, total_steps, bx1, by1, bx2, by2, phone_detected, target_face, face_in_box
):
    """Vẽ giao diện đăng ký."""
    target_size = frame_sq.shape[0]
    frame_display = frame_sq.copy()
    is_ready = face_in_box and not phone_detected
    
    # Nền UI
    ui_overlay = frame_display.copy()
    cv2.rectangle(ui_overlay, (0, 0), (target_size, 90), (0, 0, 0), -1)
    cv2.rectangle(ui_overlay, (0, target_size - 100), (target_size, target_size), (0, 0, 0), -1)
    cv2.addWeighted(ui_overlay, 0.5, frame_display, 0.5, 0, frame_display)

    # Guide Box
    box_color = config.COLOR_SUCCESS if is_ready else config.COLOR_DANGER
    cv2.rectangle(frame_display, (bx1, by1), (bx2, by2), box_color, config.UI_RECT_THICKNESS)
    
    # Text hướng dẫn
    draw_text(
        frame_display, f"BƯỚC {current_step+1}/{total_steps}: {step_info['label']}",
        (20, 20), config.FONT_SIZE_LARGE, config.COLOR_SUCCESS, is_bold=1
    )
    draw_text(frame_display, step_info["desc"], (20, 55), 18, config.COLOR_INFO)
    
    # Thông báo trạng thái
    status_rules = [
        (phone_detected, "CẢNH BÁO: PHÁT HIỆN THIẾT BỊ LẠ!", config.COLOR_DANGER),
        (not face_in_box and target_face, "VUI LÒNG ĐƯA MẶT VÀO Ô VUÔNG", config.COLOR_WARNING),
        (face_in_box, "NHẤN [SPACE] ĐỂ CHỤP ẢNH", config.COLOR_SUCCESS),
        (True, "ĐANG ĐỢI KHUÔN MẶT...", config.COLOR_INFO)
    ]
    status_msg, status_color = next((txt, clr) for cond, txt, clr in status_rules if cond)
    draw_text(
        frame_display, status_msg, (20, target_size - 80),
        config.FONT_SIZE_MEDIUM if is_ready else config.FONT_SIZE_SMALL, status_color, is_bold=1
    )

    return frame_display, is_ready

def _handle_capture(core, rgb_sq, target_face, frame_display):
    """Trích xuất embedding và thực hiện hiệu ứng flash."""
    try:
        lmarks = [(target_face.landmarks[i], target_face.landmarks[i+1]) for i in range(0, 10, 2)]
        face_bytes = core.face_tools.align_face(rgb_sq, lmarks)
        embedding = core.get_face_embedding(face_bytes)
        
        # Hiệu ứng Flash
        cv2.imshow("Dang ky Nhan vien - Square Mode", np.ones_like(frame_display)*255)
        cv2.waitKey(50)
        return embedding
    except Exception as e:
        logger.error(f"Lỗi lưu mẫu: {e}")
        return None

def collect_user_data(vstream, person_detector, core, name):
    """Vòng lặp thu thập 7 mẫu khuôn mặt từ người dùng."""
    steps = [
        {"label": "NHÌN THẲNG", "desc": "Giữ mặt thẳng vào ô vuông trung tâm", "close_up": False},
        {"label": "NHÌN THẲNG - GẦN", "desc": "Đưa mặt lại gần ô vuông lớn", "close_up": True},
        {"label": "QUAY TRÁI 45°", "desc": "Quay mặt sang trái một chút", "close_up": False},
        {"label": "QUAY TRÁI 90°", "desc": "Quay mặt sang trái hoàn toàn", "close_up": False},
        {"label": "QUAY PHẢI 45°", "desc": "Quay mặt sang phải một chút", "close_up": False},
        {"label": "QUAY PHẢI 90°", "desc": "Quay mặt sang phải hoàn toàn", "close_up": False},
        {"label": "NGƯỚC LÊN 45°", "desc": "Ngước mặt lên trên một chút", "close_up": False}
    ]
    
    collected_embeddings = []
    current_step = 0
    logger.info(f"\n[INFO] Bắt đầu quy trình thu thập đặc trưng cho: {name}")
    
    while current_step < len(steps):
        # 1. Chuẩn bị Frame
        ret, frame, frame_sq = _prepare_frame(vstream)
        if not ret: break
        
        # 2. Tính toán Guide Box và phát hiện an toàn
        step_info = steps[current_step]
        target_size = frame_sq.shape[0]
        side_ratio = config.REG_GUIDE_BOX_SIDE_RATIO_CLOSEUP if step_info["close_up"] else config.REG_GUIDE_BOX_SIDE_RATIO_NORMAL
        box_size = int(target_size * side_ratio)
        bx1 = by1 = (target_size - box_size) // 2
        bx2 = by2 = bx1 + box_size
        
        phone_detected = _check_safety(person_detector, frame)
        
        # 3. Phát hiện khuôn mặt
        target_face, face_in_box, rgb_sq = _detect_registration_face(core, frame_sq, bx1, by1, bx2, by2)
        
        # 4. Hiển thị UI
        frame_display, is_ready = _draw_registration_ui(
            frame_sq, step_info, current_step, len(steps), bx1, by1, bx2, by2, phone_detected, target_face, face_in_box
        )
        cv2.imshow("Dang ky Nhan vien - Square Mode", frame_display)
        
        # 5. Xử lý phím bấm
        key = cv2.waitKey(1) & 0xFF
        if key == 32 and is_ready:
            embedding = _handle_capture(core, rgb_sq, target_face, frame_display)
            if embedding is not None:
                collected_embeddings.append(embedding)
                logger.info(f"  [OK] Đã lưu mẫu {current_step+1}")
                current_step += 1
                time.sleep(0.3)
        elif key in [ord('q'), 27]:
            return None

    return collected_embeddings

def save_user_registration(name, embeddings, core):
    """Tính toán embedding trung bình và lưu vào database."""
    if not embeddings: return False
    
    logger.info("\nĐang tính toán đặc trưng trung bình...")
    avg_embedding = np.mean(embeddings, axis=0)
    final_embedding = avg_embedding / np.linalg.norm(avg_embedding)
    
    if core.add_user(name, final_embedding.tolist()):
        logger.info(f"\nCHÚC MỪNG: Nhân viên '{name}' đã đăng ký thành công.")
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Hệ thống đăng ký nhân viên mới")
    parser.add_argument("--camera", type=str, default="0", help="Camera ID (0, 1) hoặc URL stream (rtsp://...)")
    parser.add_argument("--ep", type=str, default="coreml", help="Execution Provider (coreml, webgpu, cpu)")
    args = parser.parse_args()

    try:
        person_detector, core = initialize_registration_engines(args.ep)
        
        print("\n--- HỆ THỐNG ĐĂNG KÝ NHÂN VIÊN MỚI ---")
        name = input("Nhập tên nhân viên: ").strip()
        if not name:
            logger.error("Tên nhân viên không được để trống.")
            return

        cam_src = int(args.camera) if args.camera.isdigit() else args.camera
        vstream = VideoStream(cam_src, 1280, config.CAMERA_HEIGHT).start()
        
        embeddings = collect_user_data(vstream, person_detector, core, name)
        
        if embeddings:
            save_user_registration(name, embeddings, core)
            
        vstream.stop()
        cv2.destroyAllWindows()
        
    except Exception as e:
        logger.error(f"Lỗi hệ thống: {e}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
