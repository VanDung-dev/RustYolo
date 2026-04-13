"""
Vẽ sơ đồ so sánh CoreML, WebGPU và CPU
"""

import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def parse_log(log_path):
    """
    Phân tích file log để lấy độ trễ trung bình cho từng model.
    Hỗ trợ cả format benchmark tự động và log thủ công.
    """
    if not os.path.exists(log_path):
        print(f"⚠️ Cảnh báo: Không tìm thấy {log_path}, bỏ qua...")
        return {}
        
    data = {}
    current_model = None
    latencies = []
    
    # Map model names to standard format used in charts
    model_map = {
        'yolov8n': 'YOLOv8n',
        'yolov8s': 'YOLOv8s',
        'yolov8m': 'YOLOv8m',
        'yolov8l': 'YOLOv8l',
        'yolov8x': 'YOLOv8x'
    }
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 1. Tìm tên model (Từ benchmark script hoặc lệnh command)
            model_match = re.search(r'(yolov8[nsmlx])\.onnx', line.lower())
            if model_match:
                found_model = model_map.get(model_match.group(1))
                if found_model and found_model != current_model:
                    # Lưu kết quả model cũ trước khi chuyển sang model mới
                    if current_model and latencies:
                        data[current_model] = sum(latencies) / len(latencies)
                    
                    current_model = found_model
                    latencies = []
                    continue
            
            # 2. Tìm giá trị Total latency từ Rust Engine hoặc tóm tắt benchmark
            # Format: Total=19.17ms HOẶC Average Latency = 12.89ms
            perf_match = re.search(r'(?:Total=|Average Latency\s*=\s*)([\d\.]+)ms', line)
            if perf_match:
                latencies.append(float(perf_match.group(1)))
                
    # Lưu model cuối cùng trong file
    if current_model and latencies:
        data[current_model] = sum(latencies) / len(latencies)
        
    return data

def main():
    # 1. Đọc dữ liệu từ các file log
    providers = {
        'CoreML': 'output/log_coreml.txt',
        'WebGPU': 'output/log_webgpu.txt',
        'CPU': 'output/log_cpu.txt'
    }
    
    all_results = []
    models_order = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x']
    
    for provider_name, log_file in providers.items():
        results = parse_log(log_file)
        for model in models_order:
            if model in results:
                all_results.append({
                    'Model': model,
                    'Latency (ms)': results[model],
                    'Execution Provider': provider_name,
                    'FPS': 1000.0 / results[model]
                })
    
    if not all_results:
        print("❌ Không tìm thấy dữ liệu benchmark nào trong các file log. Vui lòng chạy các script benchmark trước.")
        return

    df = pd.DataFrame(all_results)
    
    # 2. Cấu hình giao diện biểu đồ
    sns.set_theme(style="whitegrid")
    colors = ["#007AFF", "#34C759", "#8E8E93"] # Apple Style: Blue, Green, Gray
    
    # --- Biểu đồ Latency ---
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x="Model", y="Latency (ms)", hue="Execution Provider", data=df, palette=colors)
    
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.1f}ms', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 9), 
                        textcoords='offset points', fontsize=10, fontweight='bold')

    plt.title('Độ trễ YOLOv8 trên M4 Pro (Càng thấp càng tốt)', fontsize=18, fontweight='bold', pad=25)
    plt.ylabel('Độ trễ xử lý (ms)', fontsize=13)
    plt.xlabel('Phiên bản Model', fontsize=13)
    plt.tight_layout()
    
    # Lưu dưới dạng WebP
    out_latency = 'output/performance_chart.webp'
    plt.savefig(out_latency, format='webp', dpi=300)
    print(f"✅ Đã lưu biểu đồ Độ trễ: {out_latency}")
    
    # --- Biểu đồ FPS ---
    plt.figure(figsize=(14, 8))
    ax_fps = sns.barplot(x="Model", y="FPS", hue="Execution Provider", data=df, palette=colors)
    
    for p in ax_fps.patches:
        if p.get_height() > 0:
            ax_fps.annotate(f'{p.get_height():.1f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 9), 
                            textcoords='offset points', fontsize=10, fontweight='bold')

    plt.title('Tốc độ xử lý YOLOv8 trên M4 Pro (Càng cao càng tốt)', fontsize=18, fontweight='bold', pad=25)
    plt.ylabel('Số khung hình / giây (FPS)', fontsize=13)
    plt.xlabel('Phiên bản Model', fontsize=13)
    plt.tight_layout()
    
    # Lưu dưới dạng WebP
    out_fps = 'output/fps_chart.webp'
    plt.savefig(out_fps, format='webp', dpi=300)
    print(f"✅ Đã lưu biểu đồ FPS: {out_fps}")

if __name__ == "__main__":
    main()
