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
from core import AttendanceCore
from ui_utils import draw_text
from collections import namedtuple
import config

# Cấu hình logger cho module
logger = logging.getLogger(__name__)

# Cấu trúc dữ liệu tạm cho phát hiện khuôn mặt
FaceDetection = namedtuple("FaceDetection", ["bbox", "score", "landmarks"])

# --- KHÔNG SỬ DỤNG BIẾN ALIAS TẠI ĐÂY - SỬ DỤNG TRỰC TIẾP config.X ---

def main():
    """
    Quy trình đăng ký nhân viên mới:
    Thu thập 8 góc độ khuôn mặt khác nhau để tăng độ chính xác nhận diện.
    Sử dụng YOLO để đảm bảo tính an toàn (chống giả mạo bằng điện thoại).
    """
    db_path = config.DB_PATH
    
    # Kiểm tra tài nguyên mô hình
    for m in [config.YOLO_MODEL_PATH, config.FACE_DETECTOR_PATH, config.FACE_EMBEDDER_PATH]:
        if not os.path.exists(m):
            logger.error(f"  [LỖI] Không tìm thấy tệp tin model tại: {m}")
            return

    logger.info("Đang khởi tạo hệ thống đăng ký...")
    try:
        # Khởi tạo phát hiện người (YOLOv8x) để kiểm tra an toàn
        person_detector = rust_yolo.YoloV8Detector(config.YOLO_MODEL_PATH, config.YOLO_CONF_THRESHOLD, config.YOLO_IOU_THRESHOLD, config.DEFAULT_EP)
        
        # Khởi tạo Face System (SCRFD + ArcFace)
        core = AttendanceCore(config.FACE_DETECTOR_PATH, config.FACE_EMBEDDER_PATH, db_path, execution_provider=config.DEFAULT_EP)
        logger.info("Hệ thống đã sẵn sàng.")
    except Exception as e:
        logger.error(f"  [LỖI KHỞI TẠO] {e}")
        return

    logger.info("\n--- HỆ THỐNG ĐĂNG KÝ NHÂN VIÊN MỚI ---")
    name = input("Nhập tên nhân viên (Tiếng Việt có dấu): ").strip()
    if not name:
        logger.error("LỖI: Tên nhân viên không được để trống.")
        return

    cap = cv2.VideoCapture(0)
    # Thiết lập camera HD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    
    # Danh sách 7 tư thế cần thu thập (Loại bỏ cúi xuống)
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
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) # Chế độ gương
        h, w, _ = frame.shape
        
        # Cắt khung hình tỉ lệ 1:1 (Vuông) ở chính giữa
        # Đảm bảo khung hình vuông không vượt quá kích thước thực tế của camera
        target_size = min(h, w, config.REG_CROP_H)
        start_x = (w - target_size) // 2
        start_y = (h - target_size) // 2
        frame_sq = frame[start_y:start_y+target_size, start_x:start_x+target_size]
        
        step_info = steps[current_step]
        is_close_up = step_info["close_up"]
        
        # Thiết lập Guide Box (Ô vuông hướng dẫn) dựa trên kích thước vuông thực tế
        box_size = (2 * target_size) // 3 if is_close_up else target_size // 2
        bx1 = (target_size - box_size) // 2
        by1 = (target_size - box_size) // 2
        bx2, by2 = bx1 + box_size, by1 + box_size
        
        # BƯỚC 1: Kiểm tra an toàn bằng YOLO (Dùng frame gốc 16:9 để quét rộng hơn)
        res_caps = person_detector.detect_to_arrow(frame)
        person_arr = pa.Array._import_from_c_capsule(res_caps[1], res_caps[0])
        
        has_person_yolo = False
        phone_detected = False
        
        if len(person_arr) > 0:
            class_ids = person_arr.field("class_id").to_numpy()
            for cid in class_ids:
                if cid == 0: has_person_yolo = True
                if cid == 67: phone_detected = True

        # BƯỚC 2: Phát hiện khuôn mặt trên khung hình Vuông
        rgb_sq = cv2.cvtColor(frame_sq, cv2.COLOR_BGR2RGB)
        target_face = None
        face_in_box = False
        
        try:
            # Gửi ảnh vuông xuống Rust để phát hiện mặt
            arr_cap, sch_cap = core.face_tools.detect_faces_to_arrow(rgb_sq, config.FACE_DET_THRESHOLD)
            detections_arr = pa.Array._import_from_c_capsule(sch_cap, arr_cap)
            
            if len(detections_arr) > 0:
                # Chọn khuôn mặt lớn nhất trong khung hình (tránh nhiễu người phía sau)
                best_idx = -1
                max_area = 0
                for i in range(len(detections_arr)):
                    det = detections_arr[i]
                    x1, y1, x2, y2 = det['x1'].as_py(), det['y1'].as_py(), det['x2'].as_py(), det['y2'].as_py()
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        best_idx = i
                
                if best_idx != -1:
                    det = detections_arr[best_idx]
                    fx1, fy1, fx2, fy2 = det['x1'].as_py(), det['y1'].as_py(), det['x2'].as_py(), det['y2'].as_py()
                    target_face = FaceDetection(
                        bbox=[fx1, fy1, fx2, fy2],
                        score=det['score'].as_py(),
                        landmarks=det['landmarks'].as_py()
                    )
                    
                    # Kiểm tra xem mặt có nằm gọn trong ô vuông chỉ dẫn không
                    margin = 20
                    if (fx1 > bx1 - margin and fy1 > by1 - margin and 
                        fx2 < bx2 + margin and fy2 < by2 + margin):
                        face_in_box = True
        except Exception as e:
            logger.error(f"  [LỖI] Phát hiện khuôn mặt: {e}")

        # Hiển thị UI và hướng dẫn
        frame_display = frame_sq.copy()
        
        # Đổi màu ô vuông: Xanh khi hợp lệ, Đỏ khi có vấn đề
        box_color = (0, 255, 0) if (face_in_box and not phone_detected) else (0, 0, 255)
        cv2.rectangle(frame_display, (bx1, by1), (bx2, by2), box_color, 2)
        
        # Văn bản hướng dẫn bằng Tiếng Việt
        draw_text(frame_display, f"BƯỚC {current_step+1}/8: {step_info['label']}", (20, 20), 24, (0, 255, 0))
        draw_text(frame_display, step_info["desc"], (20, 55), 18, (255, 255, 255))
        
        if phone_detected:
            draw_text(frame_display, "CẢNH BÁO: PHÁT HIỆN ĐIỆN THOẠI!", (20, config.REG_CROP_H - 80), 20, (0, 0, 255))
        elif not face_in_box and target_face:
            draw_text(frame_display, "VUI LÒNG ĐƯA MẶT VÀO Ô VUÔNG", (20, config.REG_CROP_H - 80), 20, (0, 165, 255))
        elif face_in_box:
            draw_text(frame_display, "NHẤN [CÁCH] ĐỂ CHỤP ẢNH", (20, config.REG_CROP_H - 80), 22, (0, 255, 0))

        cv2.imshow("Dang ky Nhan vien - Square Mode", frame_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32: # Phím Space (Cách)
            # Kiểm tra các điều kiện an toàn trước khi lưu
            if not has_person_yolo and not target_face:
                continue
            if phone_detected:
                logger.warning("  [WAIT] Vui lòng bỏ điện thoại ra khỏi khung hình.")
                continue
            if not face_in_box:
                logger.warning("  [WAIT] Vui lòng đưa khuôn mặt vào đúng ô vuông hướng dẫn.")
                continue
                
            try:
                # Trích xuất Embedding bằng Rust Engine
                # Chuyển landmarks phẳng sang danh sách các cặp tọa độ (x, y)
                lmarks = [(target_face.landmarks[i], target_face.landmarks[i+1]) for i in range(0, 10, 2)]
                
                # Cân chỉnh (Align) khuôn mặt dựa trên landmarks
                face_bytes = core.face_tools.align_face(rgb_sq, lmarks)
                embedding = core.get_face_embedding(face_bytes)
                
                collected_embeddings.append(embedding)
                
                # Hiệu ứng nháy màn hình (Flash) khi chụp thành công
                flash = np.ones_like(frame_display) * 255
                cv2.imshow("Dang ky Nhan vien - Square Mode", flash)
                cv2.waitKey(50)
                
                logger.info(f"  [OK] Đã lưu mẫu {current_step+1}: {step_info['label']}")
                current_step += 1
                time.sleep(0.4)
            except Exception as e:
                logger.error(f"  [LỖI] Không thể lưu mẫu: {e}")
            
        elif key == ord('q') or key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    # Lưu kết quả sau khi thu thập đủ 8 mẫu
    if len(collected_embeddings) == len(steps):
        logger.info("\nĐang tính toán đặc trưng trung bình...")
        # Lấy trung bình cộng các embeddings để có vector đại diện ổn định nhất
        avg_embedding = np.mean(collected_embeddings, axis=0)
        # Chuẩn hóa L2
        final_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        if core.add_user(name, final_embedding.tolist()):
            logger.info(f"\nCHÚC MỪNG: Nhân viên '{name}' đã được đăng ký thành công với 8 góc độ mặt.")
        else:
            logger.error("\nLỖI: Không thể lưu thông tin vào cơ sở dữ liệu SQLite.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
