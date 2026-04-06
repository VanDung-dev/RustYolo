"""
Export YOLOv8x model sang ONNX tối ưu dành riêng cho Rust ONNX Runtime
Tương thích 100% và tối ưu tốc độ
"""
from ultralytics import YOLO
import onnxsim
import onnx
import os


def export_yolov8_for_rust():
    print("🚀 Bắt đầu export YOLOv8x sang ONNX cho Rust")

    # Load model weights (tự động tải nếu chưa có)
    model_name = "yolov8x.pt"
    model = YOLO(model_name)

    print(f"✅ Load model {model_name} thành công")
    print(f"📊 Số lớp: {len(model.names)} lớp")

    # Export sang ONNX với cấu hình tối ưu cho Rust
    print("\n⚙️  Đang export ONNX...")
    exported_path = model.export(
        format="onnx",
        opset=17,
        simplify=False,  # Chúng ta sẽ chạy onnxsim thủ công sau
        dynamic=False,
        imgsz=640,
        nms=False,
        agnostic_nms=False,
        optimize=True,
        verbose=False
    )

    # Đảm bảo file output là yolov8x.onnx (hoặc theo ý người dùng)
    onnx_file = "yolov8x.onnx"
    if exported_path != onnx_file and os.path.exists(exported_path):
        if os.path.exists(onnx_file):
            os.remove(onnx_file)
        os.rename(exported_path, onnx_file)

    print(f"\n✅ File ONNX đã tạo: {onnx_file}")

    # ✅ Chạy onnxsim tối ưu đồ thị: BƯỚC BẮT BUỘC cho Rust
    print("\n🔧 Đang tối ưu ONNX graph...")
    model_onnx = onnx.load(onnx_file)
    model_simp, check = onnxsim.simplify(model_onnx)

    if check:
        onnx.save(model_simp, onnx_file)
        print("✅ ONNX đã được tối ưu thành công")
    else:
        print("⚠️  Không thể tối ưu ONNX, dùng file gốc")

    print("\n🎉 Hoàn thành!")
    print(f"📂 File cuối cùng: {onnx_file}")
    print("\n✅ Bạn có thể dùng file này trực tiếp trong Rust ONNX Runtime")


if __name__ == "__main__":
    export_yolov8_for_rust()
