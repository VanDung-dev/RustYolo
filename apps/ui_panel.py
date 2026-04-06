"""
Module tạo giao diện panel thống kê hiệu năng
"""

import cv2
import numpy as np
from typing import Dict, Any

from .config import COLORS, STATS_PANEL_WIDTH, STATS_PANEL_HEIGHT


def create_stats_panel(
    stats: Dict[str, Any],
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

    # FPS
    fps = stats.get("fps", 0.0)
    fps_color = COLORS["green"] if fps > 20 else COLORS["yellow"] if fps > 10 else COLORS["red"]
    cv2.putText(
        panel,
        f"FPS:",
        (15, y_offset),
        font,
        font_scale,
        COLORS["white"],
        2,
    )
    cv2.putText(
        panel,
        f"{fps:.1f}",
        (200, y_offset),
        font,
        font_scale + 0.2,
        fps_color,
        3,
    )
    y_offset += line_height

    # Latency
    latency = stats.get("latency", 0.0)
    latency_color = COLORS["green"] if latency < 50 else COLORS["yellow"] if latency < 100 else COLORS["red"]
    cv2.putText(
        panel,
        f"Latency:",
        (15, y_offset),
        font,
        font_scale,
        COLORS["white"],
        2,
    )
    cv2.putText(
        panel,
        f"{latency:.1f} ms",
        (200, y_offset),
        font,
        font_scale,
        latency_color,
        2,
    )
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
    gpu_load_color = COLORS["green"] if gpu_load < 50 else COLORS["yellow"] if gpu_load < 80 else COLORS["red"]
    cv2.putText(
        panel,
        f"Load: {gpu_load:.1f}%",
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
    cv2.putText(
        panel,
        f"Power: {gpu_info.get('power', 'N/A')}",
        (15, y_offset),
        font,
        font_scale,
        COLORS["white"],
        2,
    )
    y_offset += line_height

    # GPU Temperature
    cv2.putText(
        panel,
        f"Temp: {gpu_info.get('temperature', 'N/A')}",
        (15, y_offset),
        font,
        font_scale,
        COLORS["white"],
        2,
    )
    y_offset += line_height

    # GPU Memory
    cv2.putText(
        panel,
        f"Memory: {gpu_info.get('memory_used', 'N/A')} / {gpu_info.get('memory_total', 'N/A')}",
        (15, y_offset),
        font,
        font_scale,
        COLORS["white"],
        2,
    )
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
    cpu_usage_color = COLORS["green"] if cpu_usage < 50 else COLORS["yellow"] if cpu_usage < 80 else COLORS["red"]
    cv2.putText(
        panel,
        f"Usage: {cpu_usage:.1f}%",
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
    if cpu_temp > 0:
        cpu_temp_color = COLORS["green"] if cpu_temp < 60 else COLORS["yellow"] if cpu_temp < 80 else COLORS["red"]
        cv2.putText(
            panel,
            f"Temp: {cpu_temp:.1f}°C",
            (15, y_offset),
            font,
            font_scale,
            COLORS["white"],
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
        f"Usage: {mem_percent:.1f}%",
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
