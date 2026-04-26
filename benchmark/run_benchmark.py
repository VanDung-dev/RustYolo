"""
Công cụ Benchmark YOLOv8 đa nền tảng (Unified Benchmark Tool)
Hỗ trợ: CPU, CoreML, WebGPU
"""

import os
import time
import sys
import argparse

# Khởi tạo path để có thể chạy từ bất kỳ đâu
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from apps.detector import YoloDetector
from utils import prepare_benchmark_images, TeeLogger

def run_benchmark():
    parser = argparse.ArgumentParser(description="YOLOv8 Unified Benchmark Tool")
    parser.add_argument("--models", nargs="+", default=['yolov8n.onnx', 'yolov8s.onnx', 'yolov8m.onnx', 'yolov8l.onnx', 'yolov8x.onnx'],
                        help="Danh sách file model (.onnx)")
    parser.add_argument("--loop", action="store_true", help="Chế độ benchmark liên tục (xoay vòng tất cả engine)")
    parser.add_argument("--iter", type=int, default=100, help="Số lần lặp cho mỗi model trong một chu kỳ")
    
    args = parser.parse_args()
    
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Chuẩn bị tập hợp 100 ảnh benchmark từ COCO
    benchmark_images = prepare_benchmark_images(target_num=100)
    num_images = len(benchmark_images)
    
    # Danh sách các engine cần benchmark
    engines = ["cpu", "coreml", "webgpu"]
    titles = {
        "cpu": "CPU FALLBACK (ENGINE: CPU)",
        "coreml": "APPLE SILICON (ENGINE: COREML)",
        "webgpu": "GPU ĐA NỀN TẢNG (ENGINE: WEBGPU)"
    }
    
    cycle = 0
    try:
        while True:
            cycle += 1
            if args.loop:
                print(f"\n{'='*20} CHU KỲ BENCHMARK TỔNG THỂ #{cycle} {'='*20}")
            
            for ep in engines:
                log_path = os.path.join(output_dir, f"log_{ep}.txt")
                # Reset stdout cho mỗi engine để ghi vào file log tương ứng
                sys.stdout = TeeLogger(log_path)
                
                print(f"\n🚀 ĐANG CHẠY BENCHMARK TRÊN: {titles.get(ep)}")
                print(f"Log: {log_path}")
                print("-" * 60)
                
                for model_name in args.models:
                    model_path = os.path.join(PROJECT_ROOT, model_name)
                    if not os.path.exists(model_path):
                        # Thử tìm trong thư mục benchmark nếu không thấy ở root
                        model_path = os.path.join(CURRENT_DIR, model_name)
                    
                    if not os.path.exists(model_path):
                        print(f"⚠️ Bỏ qua {model_name}: Không tìm thấy file model tại {PROJECT_ROOT} hoặc {CURRENT_DIR}.")
                        continue
                        
                    print(f"\n[MODEL]: {model_name}")
                    
                    try:
                        # Khởi tạo detector
                        detector = YoloDetector(model_path, confidence=0.25, ep=ep)
                        
                        # Warmup
                        warmup_count = 5 if ep == "cpu" else 10
                        print(f"Đang khởi động phần cứng (Warmup {warmup_count} lần)...")
                        for i in range(warmup_count):
                            detector.detect_frame(benchmark_images[i % num_images])
                        
                        # Đo lường
                        print(f"Đang đo lường trên {args.iter} lượt chạy...")
                        start_bench = time.perf_counter()
                        for i in range(args.iter):
                            current_frame = benchmark_images[i % num_images]
                            detector.detect_frame(current_frame)
                        end_bench = time.perf_counter()
                        
                        avg_time = (end_bench - start_bench) / args.iter * 1000
                        print(f"✅ Kết quả: Latency TB = {avg_time:.2f}ms | FPS TB = {1000/avg_time:.1f}")
                        
                    except Exception as e:
                        print(f"❌ Lỗi khi benchmark {model_name} trên {ep}: {e}")
                    
                    print("-" * 30)
                
                # Khôi phục stdout về terminal sau mỗi engine để chuẩn bị cho engine tiếp theo (hoặc kết thúc)
                sys.stdout.flush()
                sys.stdout = sys.stdout.terminal
                
            if not args.loop:
                break
                
    except KeyboardInterrupt:
        print("\n👋 Đã dừng benchmark theo yêu cầu người dùng.")
        
    print("\n" + "="*60)
    print(f"🏁 TẤT CẢ BENCHMARK ĐÃ HOÀN TẤT. Kết quả lưu tại thư mục output/")

if __name__ == "__main__":
    run_benchmark()
