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
from typing import Any

from .config import COLORS, STATS_PANEL_WIDTH, STATS_PANEL_HEIGHT


def create_stats_panel(
    stats: dict[str, Any],
    width: int = STATS_PANEL_WIDTH,
    height: int = STATS_PANEL_HEIGHT,
) -> np.ndarray:
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

    y_offset = 35
    line_height = 44
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.76

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

    # FPS Section
    actual_fps = stats.get("fps", 0.0)
    engine_fps = stats.get("engine_fps", 0.0)
    
    fps_color = COLORS["green"] if actual_fps > 55 else COLORS["yellow"] if actual_fps > 30 else COLORS["red"]
    
    cv2.putText(panel, "ACTUAL FPS:", (15, y_offset), font, font_scale, COLORS["white"], 2)
    cv2.putText(panel, f"{actual_fps:.2f}", (240, y_offset), font, font_scale + 0.1, fps_color, 3)
    y_offset += line_height
    
    engine_color = COLORS["cyan"] if engine_fps > 60 else COLORS["yellow"]
    cv2.putText(panel, "ENGINE FPS:", (15, y_offset), font, font_scale, (200, 200, 200), 2)
    cv2.putText(panel, f"{engine_fps:.1f} (POTENTIAL)", (240, y_offset), font, font_scale, engine_color, 2)
    y_offset += line_height + 10

    # AI Latency Breakdown
    pre_ms  = stats.get("preprocess_ms", 0.0)
    inf_ms  = stats.get("inference_ms",  0.0)
    nms_ms  = stats.get("nms_ms",        0.0)
    total_ms = pre_ms + inf_ms + nms_ms

    ai_latency_color = COLORS["green"] if total_ms < 16.6 else COLORS["yellow"] if total_ms < 33.3 else COLORS["red"]
    
    cv2.putText(panel, "TOTAL LATENCY:", (15, y_offset), font, font_scale, COLORS["white"], 2)
    cv2.putText(panel, f"{total_ms:.2f} ms", (240, y_offset), font, font_scale, ai_latency_color, 3)
    y_offset += line_height

    cv2.putText(panel, " - Preprocess:", (15, y_offset), font, font_scale - 0.2, COLORS["gray"], 1)
    cv2.putText(panel, f"{pre_ms:.2f} ms", (240, y_offset), font, font_scale - 0.2, COLORS["cyan"], 1)
    y_offset += int(line_height * 0.7)

    cv2.putText(panel, " - Inference:", (15, y_offset), font, font_scale - 0.2, COLORS["gray"], 1)
    cv2.putText(panel, f"{inf_ms:.2f} ms", (240, y_offset), font, font_scale - 0.2, COLORS["yellow"], 1)
    y_offset += int(line_height * 0.7)

    cv2.putText(panel, " - NMS/Post:", (15, y_offset), font, font_scale - 0.2, COLORS["gray"], 1)
    cv2.putText(panel, f"{nms_ms:.2f} ms", (240, y_offset), font, font_scale - 0.2, COLORS["cyan"], 1)
    y_offset += line_height

    # Đường phân cách
    cv2.line(panel, (15, y_offset), (width - 15, y_offset), (80, 80, 80), 2)
    y_offset += 35

    # --- GPU Section ---
    cv2.line(panel, (15, y_offset), (width - 15, y_offset), (80, 80, 80), 2)
    y_offset += 30
    cv2.putText(panel, "CHIP GRAPHICS (GPU)", (15, y_offset), font, 1.0, COLORS["cyan"], 2)
    y_offset += 40

    gpu_info = stats.get("gpu_info", {})
    gpu_load = gpu_info.get("load", 0.0)
    
    cv2.putText(panel, f"Load: {gpu_load:.1f}%", (15, y_offset), font, font_scale - 0.1, COLORS["white"], 1)
    
    progress_x = 180
    progress_width = 280
    progress_y = y_offset - 12
    cv2.rectangle(panel, (progress_x, progress_y), (progress_x + progress_width, progress_y + 16), COLORS["dark_gray"], -1)
    fill_w = int(progress_width * (gpu_load / 100.0))
    if fill_w > 0:
        cv2.rectangle(panel, (progress_x, progress_y), (progress_x + fill_w, progress_y + 16), COLORS["cyan"], -1)
    y_offset += 30

    cv2.putText(panel, f"Temp: {gpu_info.get('temperature', 0.0):.1f} C", (15, y_offset), font, font_scale - 0.2, COLORS["gray"], 1)
    cv2.putText(panel, f"Power: {gpu_info.get('power', 0.0):.1f} W", (230, y_offset), font, font_scale - 0.2, COLORS["gray"], 1)
    y_offset += 35

    # --- ANE Section (Apple Neural Engine) ---
    cv2.line(panel, (15, y_offset), (width - 15, y_offset), (80, 80, 80), 2)
    y_offset += 30
    cv2.putText(panel, "NEURAL ENGINE (ANE)", (15, y_offset), font, 1.0, COLORS["green"], 2)
    y_offset += 40

    ane_info = stats.get("ane_info", {})
    ane_load = ane_info.get("load", 0.0)
    ane_status = ane_info.get("status", "Idle")
    
    status_color = COLORS["green"] if ane_status == "Active" else COLORS["gray"]
    cv2.putText(panel, f"Status: {ane_status}", (15, y_offset), font, font_scale - 0.1, status_color, 2)
    y_offset += 30

    cv2.putText(panel, f"AI Load: {ane_load:.1f}%", (15, y_offset), font, font_scale - 0.1, COLORS["white"], 1)
    
    ane_progress_y = y_offset - 12
    cv2.rectangle(panel, (progress_x, ane_progress_y), (progress_x + progress_width, ane_progress_y + 16), COLORS["dark_gray"], -1)
    ane_fill_w = int(progress_width * (ane_load / 100.0))
    if ane_fill_w > 0:
        cv2.rectangle(panel, (progress_x, ane_progress_y), (progress_x + ane_fill_w, ane_progress_y + 16), COLORS["green"], -1)
    y_offset += 45

    # --- System Memory Section ---
    cv2.line(panel, (15, y_offset), (width - 15, y_offset), (80, 80, 80), 2)
    y_offset += 30
    mem_info = stats.get("memory_usage", {})
    cv2.putText(panel, "SYSTEM MEMORY", (15, y_offset), font, 1.0, COLORS["yellow"], 2)
    y_offset += 40
    
    mem_percent = mem_info.get("percent", 0.0)
    cv2.putText(panel, f"Usage: {mem_percent:.1f}%", (15, y_offset), font, font_scale - 0.1, COLORS["white"], 1)
    
    mem_progress_y = y_offset - 12
    cv2.rectangle(panel, (progress_x, mem_progress_y), (progress_x + progress_width, mem_progress_y + 16), COLORS["dark_gray"], -1)
    mem_fill_w = int(progress_width * (mem_percent / 100.0))
    if mem_fill_w > 0:
        cv2.rectangle(panel, (progress_x, mem_progress_y), (progress_x + mem_fill_w, mem_progress_y + 16), COLORS["yellow"], -1)
    
    y_offset += 30
    cv2.putText(panel, f"RAM: {mem_info.get('used', 'N/A')} / {mem_info.get('total', 'N/A')}", (15, y_offset), font, font_scale - 0.2, COLORS["gray"], 1)

    return panel

    return panel
