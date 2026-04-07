"""
Export YOLOv8 model sang ONNX tối ưu dành riêng cho Rust ONNX Runtime
Hỗ trợ chọn phiên bản từ Nano đến Extra Large
"""
from ultralytics import YOLO
import onnxsim
import onnx
import os


def export_yolov8_interactive():
    print("\n--- YOLOv8 ONNX Exporter for Rust ---")
    print("Chọn phiên bản mô hình bạn muốn export:")
    print("1. YOLOv8n (Nano)  - [Nhanh nhất, Nhẹ nhất]")
    print("2. YOLOv8s (Small) - [Cân bằng Tốc độ/Độ chính xác]")
    print("3. YOLOv8m (Medium)- [Độ chính xác tốt]")
    print("4. YOLOv8l (Large) - [Chậm, Độ chính xác cao]")
    print("5. YOLOv8x (X-Large)- [Chậm nhất, Đỉnh nhất]")
    print("6. YOLOv8n-pose     - [Pose Estimation Nano]")
    print("7. YOLOv8s-pose     - [Pose Estimation Small]")
    print("8. YOLOv8m-pose     - [Pose Estimation Medium]")
    print("9. YOLOv8l-pose     - [Pose Estimation Large]")
    print("10. YOLOv8x-pose    - [Pose Estimation X-Large]")
    
    choice = input("\nNhập số (1-5) hoặc tên model (vd: yolov8n): ").strip().lower()
    
    model_map = {
        "1": "yolov8n",
        "2": "yolov8s",
        "3": "yolov8m",
        "4": "yolov8l",
        "5": "yolov8x",
        "6": "yolov8n-pose",
        "7": "yolov8s-pose",
        "8": "yolov8m-pose",
        "9": "yolov8l-pose",
        "10": "yolov8x-pose",
        "yolov8n": "yolov8n",
        "yolov8s": "yolov8s",
        "yolov8m": "yolov8m",
        "yolov8l": "yolov8l",
        "yolov8x": "yolov8x",
        "yolov8n-pose": "yolov8n-pose",
        "yolov8s-pose": "yolov8s-pose",
        "yolov8m-pose": "yolov8m-pose",
        "yolov8l-pose": "yolov8l-pose",
        "yolov8x-pose": "yolov8x-pose"
    }
    
    model_base = model_map.get(choice)
    if not model_base:
        print("❌ Lựa chọn không hợp lệ. Sử dụng mặc định: yolov8n")
        model_base = "yolov8n"
    
    model_pt = f"{model_base}.pt"
    onnx_file = f"{model_base}.onnx"
    
    print(f"\n🚀 Bắt đầu xử lý: {model_pt} -> {onnx_file}")

    # Load model weights (tự động tải)
    model = YOLO(model_pt)

    print(f"✅ Load model {model_pt} thành công")
    
    # Export sang ONNX với cấu hình tối ưu cho Rust
    print("\n⚙️  Đang export ONNX...")
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

    # Đảm bảo đúng tên file mong muốn
    if exported_path != onnx_file and os.path.exists(exported_path):
        if os.path.exists(onnx_file):
            os.remove(onnx_file)
        os.rename(exported_path, onnx_file)

    print(f"\n✅ File ONNX gốc đã tạo: {onnx_file}")

    # ✅ Chạy onnxsim tối ưu đồ thị: BƯỚC BẮT BUỘC cho Rust
    print("\n🔧 Đang tối ưu ONNX graph bằng onnxsim...")
    try:
        model_onnx = onnx.load(onnx_file)
        model_simp, check = onnxsim.simplify(model_onnx)
        if check:
            onnx.save(model_simp, onnx_file)
            print("✅ ONNX đã được tối ưu thành công")
    except Exception as e:
        print(f"⚠️  Lỗi tối ưu: {e}. Sử dụng file gốc.")

    print("\n🎉 Hoàn thành!")
    print(f"📂 File cuối cùng: {onnx_file}")
    
    # Gợi ý cho người dùng nếu tên file khác yolov8x.onnx
    if onnx_file != "yolov8x.onnx":
        print(f"\n💡 LƯU Ý: Để chạy với các script test hiện tại, bạn có thể cần:")
        print(f"   1. Đổi tên {onnx_file} thành yolov8x.onnx")
        print(f"   2. Hoặc cập nhật model_path trong test_camera.py thành '{onnx_file}'")


if __name__ == "__main__":
    export_yolov8_interactive()
