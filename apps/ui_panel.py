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
import os
from PIL import Image, ImageDraw, ImageFont
from typing import Any

from .config import COLORS, STATS_PANEL_WIDTH, STATS_PANEL_HEIGHT, CAMERA_FPS

# Cache font JetBrains Mono
FONT_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "JetBrainsMonoNF.ttf")
try:
    # Đưa size tối thiểu lên 24-26 cho mọi thành phần
    FONT_HEADER = ImageFont.truetype(FONT_PATH, 36)
    FONT_MAIN = ImageFont.truetype(FONT_PATH, 26)
    FONT_SUB = ImageFont.truetype(FONT_PATH, 24)
    FONT_SMALL = ImageFont.truetype(FONT_PATH, 22)
except Exception:
    print(f"⚠️ Warning: Could not load font at {FONT_PATH}. Falling back to default.")
    FONT_HEADER = FONT_MAIN = FONT_SUB = FONT_SMALL = None


def create_stats_panel(
    stats: dict[str, Any],
    width: int = STATS_PANEL_WIDTH,
    height: int = STATS_PANEL_HEIGHT,
) -> np.ndarray:
    """Tạo bảng thống kê hiệu năng với font JetBrains Mono NF."""
    # 1. Tạo nền canvas
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = COLORS["bg"]

    # 2. Chuyển sang PIL để vẽ chữ đẹp
    pil_img = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    def draw_text(text, pos, font, color_bgr):
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        draw.text(pos, str(text), font=font, fill=color_rgb)

    y_offset = 35
    
    # Tiêu đề
    draw_text("PERFORMANCE MONITOR", (20, y_offset), FONT_HEADER, COLORS["green"])
    y_offset += 60

    # Vẽ đường phân cách
    draw.line([(15, y_offset), (width - 15, y_offset)], fill=(0, 255, 0), width=2)
    y_offset += 35

    # FPS Section
    # CAMERA FPS: Tốc độ hardware của camera (cố định 60fps theo cấu hình)
    # ACTUAL FPS: Số frame thực tế xử lý/hiển thị (phụ thuộc pipeline)
    # ENGINE FPS: Tốc độ tối đa của AI model (1000ms / latency)
    
    # Camera hardware FPS (từ config)
    camera_color = COLORS["green"] if CAMERA_FPS == 60 else COLORS["yellow"]
    draw_text("CAMERA FPS: ", (20, y_offset), FONT_SUB, (180, 180, 180))
    draw_text(f"{CAMERA_FPS}", (260, y_offset), FONT_MAIN, camera_color)
    y_offset += 35
    
    actual_fps = stats.get("fps", 0.0)
    engine_fps = stats.get("engine_fps", 0.0)
    
    # Actual FPS <= Engine FPS là bình thường (do pipeline delays, frame drops)
    fps_color = COLORS["green"] if actual_fps > 55 else COLORS["yellow"] if actual_fps > 30 else COLORS["red"]
    
    draw_text("ACTUAL FPS:", (20, y_offset), FONT_MAIN, COLORS["white"])
    draw_text(f"{actual_fps:.2f}", (260, y_offset - 2), FONT_HEADER, fps_color)
    y_offset += 40
    
    # Engine FPS = 1000ms / latency_ms (tốc độ tối đa AI có thể đạt được)
    engine_color = COLORS["cyan"] if engine_fps > 60 else COLORS["yellow"]
    draw_text("ENGINE FPS:", (20, y_offset), FONT_MAIN, (200, 200, 200))
    draw_text(f"{engine_fps:.1f} (MAX)", (260, y_offset), FONT_MAIN, engine_color)
    y_offset += 50

    # Latency Breakdown
    pre_ms  = stats.get("preprocess_ms", 0.0)
    inf_ms  = stats.get("inference_ms",  0.0)
    nms_ms  = stats.get("nms_ms",        0.0)
    total_ms = pre_ms + inf_ms + nms_ms
    ai_color = COLORS["green"] if total_ms < 16.6 else COLORS["yellow"]
    
    draw_text("TOTAL LATENCY:", (20, y_offset), FONT_MAIN, COLORS["white"])
    draw_text(f"{total_ms:.2f} ms", (260, y_offset), FONT_MAIN, ai_color)
    y_offset += 35

    draw_text(f" - Pre: {pre_ms:.2f} ms", (30, y_offset), FONT_SUB, COLORS["gray"])
    y_offset += 25
    draw_text(f" - Inf: {inf_ms:.2f} ms", (30, y_offset), FONT_SUB, COLORS["gray"])
    y_offset += 25
    draw_text(f" - NMS: {nms_ms:.2f} ms", (30, y_offset), FONT_SUB, COLORS["gray"])
    y_offset += 40

    # --- Hardware Sections (GPU / ANE) ---
    def draw_progress(title, value, color_bgr, current_y):
        draw_text(title, (20, current_y), FONT_SUB, COLORS["white"])
        # Thanh progress
        px, py, pw, ph = 150, current_y + 4, 300, 24
        draw.rectangle([px, py, px + pw, py + ph], fill=(40, 40, 40))
        fill_w = int(pw * (value / 100.0))
        if fill_w > 0:
            rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            draw.rectangle([px, py, px + fill_w, py + ph], fill=rgb)
        draw_text(f"{value:.1f}%", (px + pw + 10, current_y), FONT_SMALL, COLORS["gray"])
        return current_y + 35

    draw.line([(15, y_offset), (width - 15, y_offset)], fill=(80, 80, 80), width=1)
    y_offset += 25
    draw_text("CHIP GRAPHICS (GPU)", (20, y_offset), FONT_MAIN, COLORS["cyan"])
    y_offset += 35
    
    gpu_info = stats.get("gpu_info", {})
    y_offset = draw_progress("Load:", gpu_info.get("load", 0.0), COLORS["cyan"], y_offset)
    draw_text(f"Temp: {gpu_info.get('temperature', 0.0):.1f} C", (30, y_offset), FONT_SMALL, COLORS["gray"])
    draw_text(f"Power: {gpu_info.get('power', 0.0):.1f} W", (230, y_offset), FONT_SMALL, COLORS["gray"])
    y_offset += 40

    draw.line([(15, y_offset), (width - 15, y_offset)], fill=(80, 80, 80), width=1)
    y_offset += 25
    draw_text("NEURAL ENGINE (ANE)", (20, y_offset), FONT_MAIN, COLORS["green"])
    y_offset += 35
    
    ane_info = stats.get("ane_info", {})
    ane_status = ane_info.get("status", "Idle")
    status_color = COLORS["green"] if ane_status == "Active" else COLORS["gray"]
    draw_text(f"Status: {ane_status}", (30, y_offset), FONT_SMALL, status_color)
    y_offset += 25
    y_offset = draw_progress("AI Load:", ane_info.get("load", 0.0), COLORS["green"], y_offset)
    y_offset += 25

    # Memory
    draw.line([(15, y_offset), (width - 15, y_offset)], fill=(80, 80, 80), width=1)
    y_offset += 25
    mem_info = stats.get("memory_usage", {})
    draw_text("SYSTEM MEMORY", (20, y_offset), FONT_MAIN, COLORS["yellow"])
    y_offset += 35
    y_offset = draw_progress("Usage:", mem_info.get("percent", 0.0), COLORS["yellow"], y_offset)
    draw_text(f"RAM: {mem_info.get('used', 'N/A')} / {mem_info.get('total', 'N/A')}", (30, y_offset), FONT_SMALL, COLORS["gray"])
    y_offset += 45
    
    # Hướng dẫn thoát
    draw.line([(15, y_offset), (width - 15, y_offset)], fill=(80, 80, 80), width=1)
    y_offset += 25
    draw_text("CONTROLS", (20, y_offset), FONT_MAIN, COLORS["white"])
    y_offset += 35
    draw_text("Press 'q' to quit", (30, y_offset), FONT_SUB, COLORS["gray"])

    # 3. Chuyển ngược lại numpy BGR
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
