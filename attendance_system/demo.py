"""
Ứng dụng Demo điểm danh Face ID đơn giản sử dụng VideoStream.
"""

import sys
import os
import cv2
import time
import logging

# Cấu hình logger cho module
logger = logging.getLogger(__name__)

# Thêm thư mục gốc vào path để import được các module từ apps/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apps.videostream import VideoStream
from core import AttendanceCore
from ui_utils import draw_text
import config

def main():
    """
    Ứng dụng Demo điểm danh Face ID cơ bản.
    Sử dụng VideoStream đa luồng và UI Tiếng Việt.
    """
    # Cấu hình model từ config
    DETECTOR_MODEL = config.FACE_DETECTOR_PATH
    ARCFACE_MODEL = config.FACE_EMBEDDER_PATH
    
    # Kiểm tra sự tồn tại của models
    if not os.path.exists(DETECTOR_MODEL) or not os.path.exists(ARCFACE_MODEL):
        logger.error(f"LỖI: Không tìm thấy tệp tin models trong thư mục hiện tại.")
        logger.error(f"  Thiếu: {DETECTOR_MODEL} hoặc {ARCFACE_MODEL}")
        return

    logger.info("Đang khởi tạo hệ thống nhận diện khuôn mặt...")
    try:
        core = AttendanceCore(DETECTOR_MODEL, ARCFACE_MODEL)
        logger.info("Hệ thống đã sẵn sàng.")
    except Exception as e:
        logger.error(f"LỖI KHỞI TẠO: {e}")
        return

    # Khởi tạo camera qua thư viện apps (Tối ưu cho đa nền tảng)
    logger.info("Đang mở camera...")
    vs = VideoStream(src=0, width=config.CAMERA_WIDTH, height=config.CAMERA_HEIGHT, fps=30).start()
    
    logger.info("\n--- CHƯƠNG TRÌNH ĐIỂM DANH FACE ID ---")
    logger.info("Điều khiển:")
    logger.info("  - Nhấn 'q' để thoát chương trình.")
    logger.info("  - Nhấn 'r' để đăng ký nhân viên mới (Chạy file register_user.py).")

    last_time = time.time()
    
    while True:
        # Đọc khung hình từ Camera luồng riêng
        grabbed, frame = vs.read()
        if not grabbed:
            continue
            
        # Xử lý nhận diện khuôn mặt thông qua module Core
        # Trả về danh sách: [{bbox, identity, confidence, user_id}, ...]
        detections = core.process_frame(frame)
        
        # Vẽ kết quả nhận diện lên khung hình hiển thị
        for det in detections:
            bbox = det['bbox']
            name = det.get('identity', "Unknown")
            score = det.get('confidence', 0.0)
            
            # Màu sắc: Xanh lá nếu khớp, Cam nếu không xác định
            color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
            
            # Vẽ khung chữ nhật quanh mặt (OpenCV)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            
            # Vẽ tên và độ tin cậy bằng font Tiếng Việt (Pillow)
            text = f"{name} ({score:.2f})"
            draw_text(frame, text, (int(bbox[0]), int(bbox[1] - 30)), 
                        20, color, 2)

        # Tính toán và hiển thị chỉ số FPS thực tế
        current_time = time.time()
        fps = 1 / (current_time - last_time)
        last_time = current_time
        draw_text(frame, f"FPS: {fps:.1f}", (10, 10), 22, (255, 255, 0), 2)

        # Hiển thị cửa sổ
        cv2.imshow("Hệ thống Điểm danh Face ID", frame)
        
        # Xử lý phím bấm từ người dùng
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    logger.info("Đang dừng hệ thống và giải phóng tài nguyên...")
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
