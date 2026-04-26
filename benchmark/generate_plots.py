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
            
            # 2. Tìm giá trị độ trễ (Latency)
            # Hỗ trợ các format: Total=19.17ms, Average Latency = 12.89ms, Latency TB = 26.98ms
            perf_match = re.search(r'(?:Total=|Average Latency\s*=\s*|Latency TB\s*=\s*)([\d\.]+)ms', line)
            if perf_match:
                latencies.append(float(perf_match.group(1)))
                
    # Lưu model cuối cùng trong file
    if current_model and latencies:
        data[current_model] = sum(latencies) / len(latencies)
        
    return data

def main():
    # 1. Đọc dữ liệu từ các file log
    providers = {
        'CoreML': '../output/log_coreml.txt',
        'WebGPU': '../output/log_webgpu.txt',
        'CPU': '../output/log_cpu.txt'
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
    
    # 2. Cấu hình giao diện biểu đồ (Premium Dark Mode)
    plt.rcParams.update({
        "figure.facecolor": "#0F172A", # Slate 900
        "axes.facecolor": "#1E293B",   # Slate 800
        "text.color": "#F8FAFC",       # Slate 50
        "axes.labelcolor": "#CBD5E1",  # Slate 300
        "xtick.color": "#94A3B8",      # Slate 400
        "ytick.color": "#94A3B8",      # Slate 400
        "grid.color": "#334155",       # Slate 700
        "font.family": "sans-serif"
    })
    
    colors = ["#38BDF8", "#4ADE80", "#94A3B8"] # Cyan, Green, Slate
    
    # --- Biểu đồ Latency ---
    plt.figure(figsize=(14, 8), dpi=120)
    ax = sns.barplot(x="Model", y="Latency (ms)", hue="Execution Provider", data=df, palette=colors, edgecolor="#0F172A", linewidth=2)
    
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.2f}ms',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 12), 
                        textcoords='offset points', fontsize=10, fontweight='bold', color="#F8FAFC")

    plt.title('YOLOv8 Latency Comparison - M4 Pro', fontsize=20, fontweight='bold', pad=30, color="#F1F5F9")
    plt.ylabel('Độ trễ xử lý (ms)', fontsize=14, labelpad=15)
    plt.xlabel('Phiên bản Model', fontsize=14, labelpad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(title="Engine", frameon=True, facecolor="#1E293B", edgecolor="#334155")
    plt.tight_layout()
    
    # Lưu dưới dạng WebP
    out_latency = '../output/performance_chart.webp'
    plt.savefig(out_latency, format='webp', dpi=300, facecolor="#0F172A")
    print(f"✅ Đã lưu biểu đồ Độ trễ: {out_latency}")
    
    # --- Biểu đồ FPS ---
    plt.figure(figsize=(14, 8), dpi=120)
    ax_fps = sns.barplot(x="Model", y="FPS", hue="Execution Provider", data=df, palette=colors, edgecolor="#0F172A", linewidth=2)
    
    for p in ax_fps.patches:
        if p.get_height() > 0:
            ax_fps.annotate(f'{p.get_height():.1f}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 12), 
                            textcoords='offset points', fontsize=11, fontweight='bold', color="#F8FAFC")

    plt.title('YOLOv8 Throughput (FPS) - M4 Pro', fontsize=20, fontweight='bold', pad=30, color="#F1F5F9")
    plt.ylabel('Số khung hình / giây (FPS)', fontsize=14, labelpad=15)
    plt.xlabel('Phiên bản Model', fontsize=14, labelpad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(title="Engine", frameon=True, facecolor="#1E293B", edgecolor="#334155")
    plt.tight_layout()
    
    # Lưu dưới dạng WebP
    out_fps = '../output/fps_chart.webp'
    plt.savefig(out_fps, format='webp', dpi=300, facecolor="#0F172A")
    print(f"✅ Đã lưu biểu đồ FPS: {out_fps}")

if __name__ == "__main__":
    main()
