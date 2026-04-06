#!/usr/bin/env python3
"""
Export YOLOv8x model sang ONNX tối ưu dành riêng cho Rust ONNX Runtime
Tương thích 100% và tối ưu tốc độ
"""
from ultralytics import YOLO
import onnxsim
import onnx

def export_yolov8_for_rust():
    print("🚀 Bắt đầu export YOLOv8x sang ONNX cho Rust")
    
    # Load model weights đã train sẵn trong project
    model = YOLO("./yolov8x.pt")
    
    print("✅ Load model thành công")
    print(f"📊 Số lớp: {len(model.names)} lớp")
    
    # Export sang ONNX với cấu hình tối ưu cho Rust
    print("\n⚙️  Đang export ONNX...")
    onnx_file = model.export(
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
    
    print(f"\n✅ File ONNX gốc đã tạo: {onnx_file}")
    
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
    print("\n💡 Thông tin cho Rust:")
    print("   - Input shape:  [1, 3, 640, 640]")
    print("   - Output shape: [1, 84, 8400]")
    print("   - Opset: 17")
    print("   - Dynamic shape: OFF")
    print("\n✅ Bạn có thể dùng file này trực tiếp trong Rust ONNX Runtime")

if __name__ == "__main__":
    export_yolov8_for_rust()
