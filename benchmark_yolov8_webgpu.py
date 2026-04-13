"""
Benchmark khả năng xử lý của WebGPU.
"""

import os
import time
import numpy as np
import logging
import sys
from apps.detector import YoloDetector

class TeeLogger:
    """Ghi log ra cả màn hình terminal và file cùng lúc"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def run_benchmark():
    os.makedirs("output", exist_ok=True)
    log_path = "output/log_webgpu.txt"
    sys.stdout = TeeLogger(log_path)
    
    # Cấu hình logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s',
        stream=sys.stdout
    )

    models = [
        'yolov8n.onnx', 
        'yolov8s.onnx', 
        'yolov8m.onnx', 
        'yolov8l.onnx', 
        'yolov8x.onnx'
    ]
    ep = "webgpu"
    
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    print(f"\n🌐 BẮT ĐẦU BENCHMARK YOLOv8 TRÊN GPU ĐA NỀN TẢNG (ENGINE: {ep.upper()})")
    print(f"Log sẽ được lưu tự động tại: {log_path}")
    print("="*60)
    
    for model_name in models:
        if not os.path.exists(model_name):
            print(f"⚠️ Bỏ qua {model_name}: Không tìm thấy file model.")
            continue
            
        print(f"\n[MODEL]: {model_name}")
        print(f"Command simulation: python main.py --model {model_name} --ep {ep}")
        
        try:
            detector = YoloDetector(model_name, confidence=0.25, ep=ep)
            
            print("Đang khởi động phần cứng (Warmup)...")
            for _ in range(10):
                detector.detect_frame(frame)
            
            print("Đang đo lường hiệu suất...")
            start_bench = time.perf_counter()
            num_iter=100
            for i in range(num_iter):
                detector.detect_frame(frame)
            end_bench = time.perf_counter()
            
            avg_time = (end_bench - start_bench) / num_iter * 1000
            print(f"✅ Hoàn thành: Average Latency = {avg_time:.2f}ms | Estimated FPS = {1000/avg_time:.1f}")
            print("-" * 30)
            
        except Exception as e:
            print(f"❌ Lỗi khi benchmark {model_name}: {e}")
            
    print("\n" + "="*60)
    print(f"🏁 Benchmark WebGPU kết thúc. Log đã được lưu tại {log_path}")

if __name__ == "__main__":
    run_benchmark()
