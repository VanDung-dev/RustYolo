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
import threading
import queue
from core import AttendanceCore
from ui_utils import draw_text
from collections import namedtuple
import config

# Cấu hình logger cho module
logger = logging.getLogger(__name__)

# Cấu trúc dữ liệu tạm cho phát hiện khuôn mặt
FaceDetection = namedtuple("FaceDetection", ["bbox", "score", "landmarks"])

class VideoStream:
    """
    VideoStream: Luồng đọc Camera đa luồng để đảm bảo hiển thị mượt mà.
    """
    def __init__(self, src=0, width=1280, height=720):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
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
            
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def read(self):
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True
        self.stream.release()

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
        frame = vstream.read()
        if frame is None: break
        
        frame = cv2.flip(frame, 1) 
        h, w, _ = frame.shape
        
        # 1. Cắt khung hình vuông 1:1 chính giữa
        target_size = min(h, w, config.REG_CROP_H)
        start_x, start_y = (w - target_size) // 2, (h - target_size) // 2
        frame_sq = frame[start_y:start_y+target_size, start_x:start_x+target_size]
        
        step_info = steps[current_step]
        # Guide Box: Tính toán trực tiếp dựa trên tỉ lệ cạnh so với chiều cao cửa sổ
        side_ratio = config.REG_GUIDE_BOX_SIDE_RATIO_CLOSEUP if step_info["close_up"] else config.REG_GUIDE_BOX_SIDE_RATIO_NORMAL
        box_size = int(target_size * side_ratio)
        bx1 = by1 = (target_size - box_size) // 2
        bx2 = by2 = bx1 + box_size
        
        # 2. Kiểm tra an toàn (YOLO)
        res_caps = person_detector.detect_to_arrow(frame)
        person_arr = pa.Array._import_from_c_capsule(res_caps[1], res_caps[0])
        phone_detected = False
        if len(person_arr) > 0:
            phone_detected = any(cid == config.CLASS_PHONE for cid in person_arr.field("class_id").to_numpy())

        # 3. Phát hiện khuôn mặt (SCRFD)
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
                target_face = FaceDetection(bbox=[fx1, fy1, fx2, fy2], score=det['score'].as_py(), landmarks=det['landmarks'].as_py())
                
                # Kiểm tra mặt trong Guide Box
                margin = 20
                face_in_box = (fx1 > bx1-margin and fy1 > by1-margin and fx2 < bx2+margin and fy2 < by2+margin)
        except Exception as e:
            logger.error(f"Lỗi nhận diện khuôn mặt: {e}")

        # 4. Hiển thị UI & Trạng thái
        frame_display = frame_sq.copy()
        is_ready_to_capture = face_in_box and not phone_detected
        box_color = config.COLOR_SUCCESS if is_ready_to_capture else config.COLOR_DANGER
        cv2.rectangle(frame_display, (bx1, by1), (bx2, by2), box_color, 2)
        
        # Tiêu đề bước thực hiện
        draw_text(frame_display, f"BƯỚC {current_step+1}/{len(steps)}: {step_info['label']}", (20, 20), config.FONT_SIZE_LARGE, config.COLOR_SUCCESS)
        draw_text(frame_display, step_info["desc"], (20, 55), 18, config.COLOR_INFO)
        
        # Xác định thông báo trạng thái dựa trên các quy tắc ưu tiên
        status_rules = [
            (phone_detected, "CẢNH BÁO: PHÁT HIỆN ĐIỆN THOẠI!", config.COLOR_DANGER),
            (not face_in_box and target_face, "VUI LÒNG ĐƯA MẶT VÀO Ô VUÔNG", config.COLOR_WARNING),
            (face_in_box, "NHẤN [SPACE] ĐỂ CHỤP ẢNH", config.COLOR_SUCCESS),
            (True, "ĐANG ĐỢI KHUÔN MẶT...", config.COLOR_INFO)
        ]
        status_msg, status_color = next((txt, clr) for cond, txt, clr in status_rules if cond)
        draw_text(frame_display, status_msg, (20, target_size - 80), config.FONT_SIZE_MEDIUM if is_ready_to_capture else config.FONT_SIZE_SMALL, status_color)

        cv2.imshow("Dang ky Nhan vien - Square Mode", frame_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32 and is_ready_to_capture:
            try:
                lmarks = [(target_face.landmarks[i], target_face.landmarks[i+1]) for i in range(0, 10, 2)]
                face_bytes = core.face_tools.align_face(rgb_sq, lmarks)
                collected_embeddings.append(core.get_face_embedding(face_bytes))
                
                # Hiệu ứng Flash khi chụp thành công
                cv2.imshow("Dang ky Nhan vien - Square Mode", np.ones_like(frame_display)*255)
                cv2.waitKey(50)
                
                logger.info(f"  [OK] Đã lưu mẫu {current_step+1}")
                current_step += 1
                time.sleep(0.3)
            except Exception as e:
                logger.error(f"Lỗi lưu mẫu: {e}")
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
