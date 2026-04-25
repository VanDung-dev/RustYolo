"""
Công cụ chuẩn bị mô hình cho Hệ thống Điểm danh RustYolo.
Tải các mô hình khuôn mặt chính thức và chuyển đổi YOLOv8m từ .pt sang .onnx.
"""

import urllib.request
import os
import sys
import logging
import config

# Cấu hình logging chuyên nghiệp
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def download_file(url, output_path):
    """Tải tệp tin từ URL với thanh hiển thị tiến trình"""
    if os.path.exists(output_path):
        logger.info(f"  [OK] Đã có: {output_path}")
        return True

    logger.info(f"  [DOWN] Đang tải: {url}")
    logger.info(f"         Đến: {output_path}")
    
    try:
        def progress(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                percent = min(percent, 100)
                sys.stdout.write(f"\r  Tiến độ: {percent}% [{count * block_size}/{total_size} bytes]")
                sys.stdout.flush()

        # Thiết lập User-Agent để tránh bị chặn bởi một số server (như HuggingFace)
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, output_path, reporthook=progress)
        print("\n  Xong!")
        return True
    except Exception as e:
        print(f"\n  [LỖI] Tải xuống thất bại: {e}")
        return False

def export_yolo_to_onnx():
    """
    Chuyển đổi mô hình YOLO từ định dạng .pt sang .onnx tối ưu cho Rust.
    Giữ nguyên file .pt gốc và tạo ra file .onnx cùng tên dựa trên config.
    """
    pt_file = config.YOLO_PT_PATH
    onnx_file = config.YOLO_MODEL_PATH

    logger.info(f"\n--- Quy trình chuẩn bị mô hình YOLO ---")

    # Kiểm tra file .onnx đã tồn tại chưa
    if os.path.exists(onnx_file):
        logger.info(f"  [OK] File ONNX đã tồn tại: {onnx_file}")
        return True

    try:
        from ultralytics import YOLO
        import onnx
        import onnxsim

        # 1. Tải/Nạp weights .pt (Ultralytics tự động tải nếu chưa có file .pt)
        logger.info(f"  Đang nạp mô hình {pt_file}...")
        model = YOLO(pt_file)
        
        # 2. Thực hiện Export sang ONNX với cấu hình tối ưu cho CoreML/WebGPU
        # Các tham số này đảm bảo tính ổn định khi chạy trên Rust ONNX Runtime
        logger.info(f"  Đang thực hiện Export sang ONNX (FP32)...")
        exported_path = model.export(
            format="onnx",
            opset=12,         # Opset 12 tương thích tốt với CoreML trên Mac
            simplify=False,
            dynamic=False,
            imgsz=640,
            half=False,       # Sử dụng FP32 để đạt độ chính xác và ổn định cao
            nms=False,        # Tắt NMS tích hợp để Rust xử lý thủ công (linh hoạt hơn)
            agnostic_nms=False,
            optimize=True,
            verbose=False
        )

        # 3. Quản lý tệp tin sau khi export
        # Ultralytics đôi khi tạo thư mục con, ta cần đưa file .onnx ra thư mục gốc
        if os.path.exists(exported_path):
            if exported_path != onnx_file:
                if os.path.exists(onnx_file):
                    os.remove(onnx_file)
                os.rename(exported_path, onnx_file)
            logger.info(f"  [XONG] Đã tạo thành công: {onnx_file}")
            # Lưu ý: File .pt vẫn được giữ nguyên trong thư mục hiện tại.
        else:
            logger.error("  [LỖI] Không tìm thấy file sau khi export.")
            return False

        # 4. Tối ưu hóa đồ thị (Graph Optimization) bằng onnxsim
        logger.info("  Đang tối ưu hóa đồ thị ONNX bằng onnxsim...")
        try:
            model_onnx = onnx.load(onnx_file)
            model_simp, check = onnxsim.simplify(model_onnx)
            if check:
                onnx.save(model_simp, onnx_file)
                logger.info("  Tối ưu hóa thành công.")
        except Exception as sim_err:
            logger.warning(f"  [!] Tối ưu hóa (onnxsim) gặp lỗi: {sim_err}. Sẽ sử dụng file ONNX gốc.")

        return True

    except ImportError:
        logger.error("  [LỖI] Thiếu thư viện cần thiết. Vui lòng cài đặt:")
        logger.error("  pip install ultralytics onnx onnxsim")
        return False
    except Exception as e:
        logger.error(f"  [LỖI] Quá trình chuyển đổi thất bại: {e}")
        return False

if __name__ == "__main__":
    logger.info("\n================================================")
    logger.info("   CHƯƠNG TRÌNH CHUẨN BỊ MÔ HÌNH RUSTYOLO")
    logger.info("================================================\n")
    
    # GIAI ĐOẠN 1: Tải các mô hình khuôn mặt từ InsightFace (Nguồn công khai tin cậy)
    # scrfd_34g: Mô hình phát hiện khuôn mặt mạnh mẽ
    # arcface: Mô hình nhận diện (trích xuất đặc trưng)
    face_models = {
        config.FACE_DETECTOR_PATH: "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/det_10g.onnx",
        config.FACE_EMBEDDER_PATH: "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx"
    }

    logger.info("[PHẦN 1] KIỂM TRA MÔ HÌNH KHUÔN MẶT:")
    for path, url in face_models.items():
        download_file(url, path)

    # Hệ thống sẽ tự động tải yolov8m.pt và chuyển sang yolov8m.onnx
    logger.info("\n[PHẦN 2] KIỂM TRA VÀ CHUYỂN ĐỔI YOLOv8m:")
    export_yolo_to_onnx()

    logger.info("\n================================================")
    logger.info("   TẤT CẢ MÔ HÌNH ĐÃ SẴN SÀNG!")
    logger.info("================================================\n")
