"""
Module vẽ UI Panel Thống kê Hiệu năng

Module này chỉ chịu trách nhiệm VẼ hình ảnh, không tính toán gì cả:
- Nhận dictionary stats đã được tính toán hoàn toàn từ Rust
- Vẽ text, thanh progress, màu sắc lên mảng numpy
- Render bằng OpenCV tốc độ cao
- Tất cả logic tính toán đã được hoàn thành trước đó
"""

import cv2
import numpy as np
import ctypes
from typing import Any

from .config import COLORS, STATS_PANEL_WIDTH, STATS_PANEL_HEIGHT


def get_display_scale() -> float:
    """
    Lấy độ phân giải màn hình hiện tại và tính tỉ lệ scale tự động
    hỗ trợ Windows, macOS, Linux
    """
    try:
        # Windows API
        if hasattr(ctypes, 'windll'):
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            screen_width = user32.GetSystemMetrics(0)
            screen_height = user32.GetSystemMetrics(1)
            
            # Tính tỉ lệ DPI scale
            dpi = user32.GetDpiForSystem()
            scale = dpi / 96.0
            
        # macOS / Linux
        else:
            import tkinter as tk
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            scale = root.winfo_fpixels('1i') / 72.0
            root.destroy()

        # Tính tỉ lệ scale phù hợp dựa trên chiều cao màn hình
        base_height = 1080
        height_scale = screen_height / base_height
        
        # Giới hạn scale trong khoảng 0.6 -> 1.6
        final_scale = max(0.6, min(1.6, height_scale * scale))
        
        return final_scale

    except Exception:
        # Fallback về scale 1.0 nếu không lấy được thông tin màn hình
        return 1.0

# Tính scale 1 lần duy nhất khi import module
GLOBAL_SCALE = get_display_scale()


def create_stats_panel(
    stats: dict[str, Any],
    width: int = STATS_PANEL_WIDTH,
    height: int = STATS_PANEL_HEIGHT,
    target_height: int | None = None,
    scale: float = GLOBAL_SCALE,
) -> np.ndarray:
    
    # ✅ Scale toàn bộ kích thước theo tỉ lệ màn hình
    width = int(width * scale)
    height = int(height * scale)
    
    # ✅ Nếu có target_height thì điều chỉnh scale để khớp chính xác chiều cao frame camera
    if target_height is not None:
        # Tính lại scale để stats panel khớp chính xác chiều cao
        height_scale = target_height / height
        width = int(width * height_scale)
        height = target_height
        scale = scale * height_scale
    """
    Tạo bảng thống kê hiệu năng.

    Args:
        stats: Dictionary chứa các thông số từ PerformanceMonitor
        width: Chiều rộng của panel
        height: Chiều cao của panel

    Returns:
        Panel thống kê dưới dạng numpy array
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = COLORS["bg"]  # Màu nền tối

    y_offset = int(35 * scale)
    line_height = int(44 * scale)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.76 * scale

    # Tiêu đề
    cv2.putText(
        panel,
        "PERFORMANCE MONITOR",
        (15, y_offset),
        font,
        1.4,
        COLORS["green"],
        3,
    )
    y_offset += 70

    # Đường phân cách
    cv2.line(panel, (15, y_offset), (width - 15, y_offset), COLORS["green"], 3)
    y_offset += 45

    # FPS
    fps = stats.get("fps", 0.0)
    fps_color = (
        COLORS["green"] if fps > 20
        else COLORS["yellow"] if fps > 10
        else COLORS["red"]
    )
    cv2.putText(
        panel,
        "AI FPS:",
        (15, y_offset),
        font,
        font_scale,
        COLORS["white"],
        2,
    )
    cv2.putText(
        panel,
        f"{fps:.2f}",
        (200, y_offset),
        font,
        font_scale + 0.2,
        fps_color,
        3,
    )
    y_offset += line_height

    # AI Latency
    ai_latency = stats.get("ai_latency", 0.0)
    ai_latency_color = (
        COLORS["green"] if ai_latency < 50
        else COLORS["yellow"] if ai_latency < 100
        else COLORS["red"]
    )
    cv2.putText(
        panel,
        "AI Latency:",
        (15, y_offset),
        font,
        font_scale,
        COLORS["white"],
        2,
    )
    cv2.putText(
        panel,
        f"{ai_latency:.2f} ms",
        (201, y_offset),
        font,
        font_scale,
        ai_latency_color,
        2,
    )
    y_offset += line_height

    # Rust Latency Breakdown
    pre_ms  = stats.get("preprocess_ms", 0.0)
    inf_ms  = stats.get("inference_ms",  0.0)
    nms_ms  = stats.get("nms_ms",        0.0)

    cv2.putText(panel, "Preprocess:", (15, y_offset), font,
                font_scale - 0.1, COLORS["gray"], 1)
    cv2.putText(panel, f"{pre_ms:.1f} ms", (200, y_offset), font,
                font_scale - 0.1, COLORS["cyan"], 2)
    y_offset += int(line_height * 0.8)

    cv2.putText(panel, "Inference:", (15, y_offset), font,
                font_scale - 0.1, COLORS["gray"], 1)
    cv2.putText(panel, f"{inf_ms:.1f} ms", (200, y_offset), font,
                font_scale - 0.1, COLORS["yellow"], 2)
    y_offset += int(line_height * 0.8)

    cv2.putText(panel, "NMS:", (15, y_offset), font,
                font_scale - 0.1, COLORS["gray"], 1)
    cv2.putText(panel, f"{nms_ms:.1f} ms", (200, y_offset), font,
                font_scale - 0.1, COLORS["cyan"], 2)
    y_offset += line_height

    # Đường phân cách
    cv2.line(panel, (15, y_offset), (width - 15, y_offset), (80, 80, 80), 2)
    y_offset += 35

    # GPU Section
    cv2.putText(
        panel,
        "GPU",
        (15, y_offset),
        font,
        1.3,
        COLORS["cyan"],
        3,
    )
    y_offset += 55

    gpu_info = stats.get("gpu_info", {})

    # GPU Name
    cv2.putText(
        panel,
        f"Name: {gpu_info.get('name', 'N/A')[:35]}",
        (15, y_offset),
        font,
        font_scale - 0.1,
        COLORS["white"],
        2,
    )
    y_offset += line_height

    # GPU Load với progress bar
    gpu_load = gpu_info.get("load", 0.0)
    gpu_load_color = (
        COLORS["green"] if gpu_load < 50 
        else COLORS["yellow"] if gpu_load < 80 
        else COLORS["red"]
    )
    cv2.putText(
        panel,
        f"Load: {gpu_load:.2f}%",
        (15, y_offset),
        font,
        font_scale,
        COLORS["white"],
        2,
    )
    # Thanh progress
    progress_x = 220
    progress_width = 350
    progress_y = y_offset - 14
    cv2.rectangle(
        panel,
        (progress_x, progress_y),
        (progress_x + progress_width, progress_y + 22),
        COLORS["dark_gray"],
        2,
    )
    fill_width = int(progress_width * min(gpu_load, 100) / 100)
    if fill_width > 0:
        cv2.rectangle(
            panel,
            (progress_x, progress_y),
            (progress_x + fill_width, progress_y + 22),
            gpu_load_color,
            -1,
        )
    y_offset += line_height

    # GPU Power
    gpu_power = gpu_info.get("power", 0.0)
    gpu_power_color = (
        COLORS["green"] if gpu_power < 8
        else COLORS["yellow"] if gpu_power < 15
        else COLORS["red"]
    )
    cv2.putText(
        panel,
        f"Power:",
        (15, y_offset),
        font,
        font_scale,
        COLORS["white"],
        2,
    )
    cv2.putText(
        panel,
        f"{gpu_power:.1f} W",
        (200, y_offset),
        font,
        font_scale,
        gpu_power_color,
        2,
    )
    y_offset += line_height

    # GPU Temperature
    gpu_temp = gpu_info.get("temperature", 0.0)
    gpu_temp_color = (
        COLORS["green"] if gpu_temp < 65
        else COLORS["yellow"] if gpu_temp < 85
        else COLORS["red"]
    )
    cv2.putText(
        panel,
        f"Temp:",
        (15, y_offset),
        font,
        font_scale,
        COLORS["white"],
        2,
    )
    cv2.putText(
        panel,
        f"{gpu_temp:.1f} °C",
        (200, y_offset),
        font,
        font_scale,
        gpu_temp_color,
        2,
    )
    y_offset += line_height

    y_offset += line_height

    # Đường phân cách
    cv2.line(panel, (15, y_offset), (width - 15, y_offset), (80, 80, 80), 2)
    y_offset += 35

    # CPU Section
    cv2.putText(
        panel,
        "CPU",
        (15, y_offset),
        font,
        1.3,
        COLORS["cyan"],
        3,
    )
    y_offset += 55

    # CPU Usage
    cpu_usage = stats.get("cpu_usage", 0.0)
    cpu_usage_color = (
        COLORS["green"] if cpu_usage < 50 
        else COLORS["yellow"] if cpu_usage < 80 
        else COLORS["red"]
    )
    cv2.putText(
        panel,
        f"Usage: {cpu_usage:.2f}%",
        (15, y_offset),
        font,
        font_scale,
        COLORS["white"],
        2,
    )
    # Progress bar cho CPU
    progress_y = y_offset - 14
    cv2.rectangle(
        panel,
        (progress_x, progress_y),
        (progress_x + progress_width, progress_y + 22),
        COLORS["dark_gray"],
        2,
    )
    fill_width = int(progress_width * min(cpu_usage, 100) / 100)
    if fill_width > 0:
        cv2.rectangle(
            panel,
            (progress_x, progress_y),
            (progress_x + fill_width, progress_y + 22),
            cpu_usage_color,
            -1,
        )
    y_offset += line_height

    # CPU Temperature
    cpu_temp = stats.get("cpu_temp", 0.0)
    dt_dt = stats.get("dt_dt", 0.0)
    
    if cpu_temp > 0:
        cpu_temp_color = (
            COLORS["green"] if cpu_temp < 60
            else COLORS["yellow"] if cpu_temp < 80
            else COLORS["red"]
        )
        cv2.putText(
            panel,
            f"Temp: {cpu_temp:.2f}°C",
            (15, y_offset),
            font,
            font_scale,
            cpu_temp_color,
            2,
        )
        
        # Thêm Thermal Gradient (dT/dt)
        dt_color = (
            COLORS["white"] if abs(dt_dt) < 0.1 
            else COLORS["yellow"] if dt_dt < 0.5 
            else COLORS["red"]
        )
        cv2.putText(
            panel,
            f"dT/dt: {dt_dt:+.2f} °C/s",
            (250, y_offset),
            font,
            font_scale - 0.1,
            dt_color,
            2,
        )
    else:
        cv2.putText(
            panel,
            "Temp: N/A",
            (15, y_offset),
            font,
            font_scale,
            COLORS["gray"],
            2,
        )
    y_offset += line_height

    # Đường phân cách
    cv2.line(panel, (15, y_offset), (width - 15, y_offset), (80, 80, 80), 2)
    y_offset += 35

    # Memory Section
    mem_info = stats.get("memory_usage", {})
    cv2.putText(
        panel,
        "MEMORY",
        (15, y_offset),
        font,
        1.3,
        COLORS["cyan"],
        3,
    )
    y_offset += 55

    cv2.putText(
        panel,
        f"Used: {mem_info.get('used', 'N/A')} / {mem_info.get('total', 'N/A')}",
        (15, y_offset),
        font,
        font_scale,
        COLORS["white"],
        2,
    )
    y_offset += line_height

    mem_percent = mem_info.get("percent", 0.0)
    mem_color = COLORS["green"] if mem_percent < 50 else COLORS["yellow"] if mem_percent < 80 else COLORS["red"]
    cv2.putText(
        panel,
        f"Usage: {mem_percent:.2f}%",
        (15, y_offset),
        font,
        font_scale,
        COLORS["white"],
        2,
    )
    # Progress bar cho Memory
    progress_y = y_offset - 14
    cv2.rectangle(
        panel,
        (progress_x, progress_y),
        (progress_x + progress_width, progress_y + 22),
        COLORS["dark_gray"],
        2,
    )
    fill_width = int(progress_width * min(mem_percent, 100) / 100)
    if fill_width > 0:
        cv2.rectangle(
            panel,
            (progress_x, progress_y),
            (progress_x + fill_width, progress_y + 22),
            mem_color,
            -1,
        )

    return panel
