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

def _parse_args():
    """Thiết lập và phân tích các tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="YOLOv8 Unified Benchmark Tool")
    parser.add_argument(
        "--models", nargs="+",
        default=['yolov8n.onnx', 'yolov8s.onnx', 'yolov8m.onnx', 'yolov8l.onnx', 'yolov8x.onnx'],
        help="Danh sách file model (.onnx)"
    )
    parser.add_argument("--loop", action="store_true", help="Chế độ benchmark liên tục (xoay vòng tất cả engine)")
    parser.add_argument("--iter", type=int, default=100, help="Số lần lặp cho mỗi model trong một chu kỳ")
    return parser.parse_args()

def _benchmark_model(model_name, ep, iterations, benchmark_images):
    """Thực hiện benchmark cho một model cụ thể trên một engine cụ thể."""
    num_images = len(benchmark_images)
    model_path = os.path.join(PROJECT_ROOT, model_name)
    if not os.path.exists(model_path):
        model_path = os.path.join(CURRENT_DIR, model_name)
    
    if not os.path.exists(model_path):
        print(f"⚠️ Bỏ qua {model_name}: Không tìm thấy file model.")
        return

    print(f"\n[MODEL]: {model_name}")
    try:
        detector = YoloDetector(model_path, confidence=0.25, ep=ep)
        
        # Warmup
        warmup_count = 10 if ep == "cpu" else 15
        print(f"Đang warmup {warmup_count} lần...")
        for i in range(warmup_count):
            detector.detect_frame(benchmark_images[i % num_images], benchmark_mode=True)
        
        # Benchmarking
        print(f"Đang đo lường {iterations} lượt chạy...")
        start_bench = time.perf_counter()
        for i in range(iterations):
            detector.detect_frame(benchmark_images[i % num_images], benchmark_mode=True)
        end_bench = time.perf_counter()
        
        avg_time = (end_bench - start_bench) / iterations * 1000
        print(f"✅ Kết quả: Latency TB = {avg_time:.2f}ms | FPS TB = {1000/avg_time:.1f}")
        
    except Exception as e:
        print(f"❌ Lỗi khi benchmark {model_name} trên {ep}: {e}")

def _run_engine_cycle(ep, title, args, benchmark_images, output_dir):
    """Chạy toàn bộ danh sách model cho một engine cụ thể."""
    log_path = os.path.join(output_dir, f"log_{ep}.txt")
    logger = TeeLogger(log_path)
    old_stdout = sys.stdout
    sys.stdout = logger
    
    try:
        print(f"\n🚀 ĐANG CHẠY BENCHMARK TRÊN: {title}")
        print(f"Log: {log_path}")
        print("-" * 60)
        
        for model_name in args.models:
            _benchmark_model(model_name, ep, args.iter, benchmark_images)
            print("-" * 30)
    finally:
        sys.stdout.flush()
        sys.stdout = old_stdout
        logger.close()

def run_benchmark():
    """Hàm chính điều phối toàn bộ quy trình benchmark."""
    args = _parse_args()
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    benchmark_images = prepare_benchmark_images(target_num=100)
    engines = {
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
            
            for ep, title in engines.items():
                _run_engine_cycle(ep, title, args, benchmark_images, output_dir)
                
            if not args.loop: break
                
    except KeyboardInterrupt:
        print("\n👋 Đã dừng benchmark theo yêu cầu người dùng.")
        
    print("\n" + "="*60)
    print(f"🏁 TẤT CẢ BENCHMARK ĐÃ HOÀN TẤT. Kết quả lưu tại thư mục output/")

if __name__ == "__main__":
    run_benchmark()
