"""
Ứng dụng Demo điểm danh Face ID đơn giản sử dụng VideoStream.
"""

import sys
import os
import cv2
import time
import logging

# Thêm thư mục gốc vào path để import được các module từ apps/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apps.videostream import VideoStream
from core import AttendanceCore
from ui_utils import draw_text
import config

# Cấu hình logger cho module
logger = logging.getLogger(__name__)

def main():
    """
    Ứng dụng Demo điểm danh Face ID cơ bản.
    Sử dụng VideoStream đa luồng và UI Tiếng Việt.
    """
    
    # Kiểm tra sự tồn tại của models từ config
    for m in [config.FACE_DETECTOR_PATH, config.FACE_EMBEDDER_PATH]:
        if not os.path.exists(m):
            logger.error(f"LỖI: Không tìm thấy tệp tin model tại: {m}")
            return

    logger.info("Đang khởi tạo hệ thống nhận diện khuôn mặt...")
    try:
        # Khởi tạo Core với cấu hình tập trung
        core = AttendanceCore(
            config.FACE_DETECTOR_PATH, 
            config.FACE_EMBEDDER_PATH,
            config.DB_PATH,
            execution_provider=config.DEFAULT_EP
        )
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
        detections = core.process_frame(frame)
        
        # Vẽ kết quả nhận diện lên khung hình hiển thị
        for det in detections:
            bbox = det['bbox']
            name = det.get('identity', "Unknown")
            score = det.get('confidence', 0.0)
            
            # Màu sắc từ config: Xanh lá nếu khớp, Cam nếu không xác định
            color = config.COLOR_SUCCESS if name != "Unknown" else config.COLOR_WARNING
            
            # Vẽ khung chữ nhật quanh mặt (OpenCV)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            
            # Vẽ tên và độ tin cậy bằng font Tiếng Việt
            text = f"{name} ({score:.2f})"
            draw_text(frame, text, (int(bbox[0]), int(bbox[1] - 30)), 
                        config.FONT_SIZE_SMALL, color)

        # Tính toán và hiển thị chỉ số FPS thực tế
        current_time = time.time()
        fps_val = 1 / (current_time - last_time) if (current_time - last_time) > 0 else 0
        last_time = current_time
        draw_text(frame, f"FPS: {fps_val:.1f}", (10, 10), config.FONT_SIZE_MEDIUM, config.COLOR_SCANNING)

        # Hiển thị cửa sổ
        cv2.imshow("He thong Diem danh Face ID", frame)
        
        # Xử lý phím bấm từ người dùng
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            logger.info("Đang chuyển hướng sang quy trình đăng ký...")

    logger.info("Đang dừng hệ thống và giải phóng tài nguyên...")
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
