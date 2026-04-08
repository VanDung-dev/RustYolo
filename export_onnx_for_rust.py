"""
Export YOLO model sang ONNX tối ưu dành riêng cho Rust ONNX Runtime
Người dùng nhập tên model (ví dụ: yolo11n, yolov8s-pose.pt, ...)
Script tự động xử lý phần đuôi file và export sang ONNX cùng tên.
"""
from ultralytics import YOLO
import onnxsim
import onnx
import os
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def export_yolo_by_name():
    logger.info("\n--- YOLO ONNX Exporter for Rust ---")
    
    # Bước 1: Nhập tên model
    model_input = input("\nNhập tên model muốn export (vd: yolo11n, yolov8x-seg.pt): ").strip()
    if not model_input:
        logger.error("Tên model không được để trống!")
        return

    # Xử lý tên file .pt để load
    if model_input.lower().endswith(".pt"):
        model_pt = model_input
        model_name_base = model_input[:-3] # Bỏ .pt
    else:
        model_pt = f"{model_input}.pt"
        model_name_base = model_input

    # Tên file ONNX xuất ra (giữ nguyên tên gốc, đổi đuôi)
    onnx_file = f"{model_name_base}.onnx"
    
    logger.info(f"\nĐang chuẩn bị export: {model_pt} -> {onnx_file}")

    try:
        # Load model weights (tự động tải nếu chưa có)
        model = YOLO(model_pt)
        logger.info(f"Load model {model_pt} thành công")
        
        # Export sang ONNX với cấu hình tối ưu cho Rust
        logger.info("\nĐang export ONNX...")
        exported_path = model.export(
            format="onnx",
            opset=12,         # Opset 12 thường ổn định hơn cho nhiều runtime
            simplify=False,
            dynamic=False,
            imgsz=640,
            nms=False,
            agnostic_nms=False,
            optimize=True,
            verbose=False
        )

        # Đảm bảo đúng tên file mong muốn (keeping original name base)
        if os.path.exists(exported_path) and exported_path != onnx_file:
            if os.path.exists(onnx_file):
                os.remove(onnx_file)
            os.rename(exported_path, onnx_file)

        logger.info(f"\nFile ONNX gốc đã tạo: {onnx_file}")

        # Chạy onnxsim tối ưu graph
        logger.info("\nĐang tối ưu ONNX graph bằng onnxsim...")
        try:
            model_onnx = onnx.load(onnx_file)
            model_simp, check = onnxsim.simplify(model_onnx)
            if check:
                onnx.save(model_simp, onnx_file)
                logger.info("ONNX đã được tối ưu thành công")
        except Exception as e:
            logger.warning(f"Lỗi tối ưu: {e}. Sử dụng file gốc.")

        logger.info("\nHoàn thành!")
        logger.info(f"File cuối cùng: {onnx_file}")
        
    except Exception as e:
        logger.error(f"Lỗi: {e}")
        logger.info(f"Hãy đảm bảo '{model_pt}' là tên mô hình hợp lệ của Ultralytics.")


if __name__ == "__main__":
    export_yolo_by_name()
