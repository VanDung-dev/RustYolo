"""
Vẽ sơ đồ so sánh CoreML, WebGPU và CPU
"""

import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def _extract_model_name(line):
    """Trích xuất tên model chuẩn hóa từ một dòng log."""
    model_map = {
        'yolov8n': 'YOLOv8n',
        'yolov8s': 'YOLOv8s',
        'yolov8m': 'YOLOv8m',
        'yolov8l': 'YOLOv8l',
        'yolov8x': 'YOLOv8x'
    }
    match = re.search(r'(yolov8[nsmlx])\.onnx', line.lower())
    if match:
        return model_map.get(match.group(1))
    return None

def _extract_latency(line):
    """Trích xuất giá trị độ trễ (ms) từ một dòng log."""
    # Hỗ trợ các format: Total=19.17ms, Average Latency = 12.89ms, Latency TB = 26.98ms
    match = re.search(r'(?:Total=|Average Latency\s*=\s*|Latency TB\s*=\s*)([\d\.]+)ms', line)
    if match:
        return float(match.group(1))
    return None

def parse_log(log_path):
    """Phân tích file log để lấy độ trễ trung bình cho từng model."""
    if not os.path.exists(log_path):
        print(f"⚠️ Cảnh báo: Không tìm thấy {log_path}, bỏ qua...")
        return {}
        
    data = {}
    current_model = None
    latencies = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 1. Kiểm tra tên model
            found_model = _extract_model_name(line)
            if found_model:
                if current_model and latencies:
                    data[current_model] = sum(latencies) / len(latencies)
                
                current_model = found_model
                latencies = []
                continue
            
            # 2. Kiểm tra giá trị độ trễ
            val = _extract_latency(line)
            if val is not None:
                latencies.append(val)
                
    # Lưu model cuối cùng
    if current_model and latencies:
        data[current_model] = sum(latencies) / len(latencies)
        
    return data

def _load_benchmark_data():
    """Đọc dữ liệu từ các file log và trả về DataFrame."""
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
    
    return pd.DataFrame(all_results) if all_results else None

def _setup_plot_style():
    """Cấu hình giao diện biểu đồ Premium Dark Mode."""
    plt.rcParams.update({
        "figure.facecolor": "#0F172A", "axes.facecolor": "#1E293B",
        "text.color": "#F8FAFC", "axes.labelcolor": "#CBD5E1",
        "xtick.color": "#94A3B8", "ytick.color": "#94A3B8",
        "grid.color": "#334155", "font.family": "sans-serif"
    })
    return ["#38BDF8", "#4ADE80", "#94A3B8"] # Cyan, Green, Slate

def _create_latency_chart(df, colors):
    """Vẽ và lưu biểu đồ so sánh Latency."""
    plt.figure(figsize=(14, 8), dpi=120)
    ax = sns.barplot(x="Model", y="Latency (ms)", hue="Execution Provider", data=df, palette=colors, edgecolor="#0F172A", linewidth=2)
    
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.2f}ms', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 12), textcoords='offset points', 
                        fontsize=10, fontweight='bold', color="#F8FAFC")

    plt.title('YOLOv8 Latency Comparison - Apple Silicon M4 Pro', fontsize=20, fontweight='bold', pad=30, color="#F1F5F9")
    plt.ylabel('Độ trễ xử lý (ms)', fontsize=14, labelpad=15)
    plt.xlabel('Phiên bản Model', fontsize=14, labelpad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(title="Engine", frameon=True, facecolor="#1E293B", edgecolor="#334155")
    plt.tight_layout()
    
    out_path = '../output/performance_chart.webp'
    plt.savefig(out_path, format='webp', dpi=300, facecolor="#0F172A")
    print(f"✅ Đã lưu biểu đồ Độ trễ: {out_path}")

def _create_fps_chart(df, colors):
    """Vẽ và lưu biểu đồ so sánh FPS."""
    plt.figure(figsize=(14, 8), dpi=120)
    ax_fps = sns.barplot(x="Model", y="FPS", hue="Execution Provider", data=df, palette=colors, edgecolor="#0F172A", linewidth=2)
    
    for p in ax_fps.patches:
        if p.get_height() > 0:
            ax_fps.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 12), textcoords='offset points', 
                            fontsize=11, fontweight='bold', color="#F8FAFC")

    plt.title('YOLOv8 Throughput (FPS) - Apple Silicon M4 Pro', fontsize=20, fontweight='bold', pad=30, color="#F1F5F9")
    plt.ylabel('Số khung hình / giây (FPS)', fontsize=14, labelpad=15)
    plt.xlabel('Phiên bản Model', fontsize=14, labelpad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(title="Engine", frameon=True, facecolor="#1E293B", edgecolor="#334155")
    plt.tight_layout()
    
    out_path = '../output/fps_chart.webp'
    plt.savefig(out_path, format='webp', dpi=300, facecolor="#0F172A")
    print(f"✅ Đã lưu biểu đồ FPS: {out_path}")

def main():
    """Hàm chính điều phối quy trình vẽ biểu đồ."""
    # 1. Tải dữ liệu
    df = _load_benchmark_data()
    if df is None:
        print("❌ Không tìm thấy dữ liệu benchmark. Vui lòng chạy benchmark trước.")
        return

    # 2. Thiết lập giao diện
    colors = _setup_plot_style()

    # 3. Vẽ biểu đồ
    _create_latency_chart(df, colors)
    _create_fps_chart(df, colors)

if __name__ == "__main__":
    main()
