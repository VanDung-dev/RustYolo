# YOLOv8 Object Detection Package
"""
Package chứa các module cho ứng dụng YOLOv8 object detection với camera.
"""

from .config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    STATS_PANEL_WIDTH,
    STATS_PANEL_HEIGHT,
    DEFAULT_CONFIDENCE,
    DEFAULT_CAMERA_ID,
    COLORS,
)
from .detector import YoloDetector
from .performance_monitor import PerformanceMonitor
from .ui_panel import create_stats_panel

__all__ = [
    "CAMERA_WIDTH",
    "CAMERA_HEIGHT",
    "STATS_PANEL_WIDTH",
    "STATS_PANEL_HEIGHT",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_CAMERA_ID",
    "COLORS",
    "YoloDetector",
    "PerformanceMonitor",
    "create_stats_panel",
]
