"""
✅ Module Performance Monitor

⚠️ QUAN TRỌNG: Đây chỉ là wrapper RỖNG 100%
Toàn bộ logic monitoring chạy hoàn toàn trên Rust:
- Background thread riêng refresh metrics mỗi giây
- Đọc trực tiếp system sensor macOS không qua trung gian
- Tính toán FPS, thermal gradient, GPU/CPU metrics
- Không có overhead nào ở phía Python
"""

# ✅ Import trực tiếp class PerformanceMonitor từ Rust native
# Được export từ src/monitor.rs thông qua PyO3
from rust_yolo import PerformanceMonitor

__all__ = ["PerformanceMonitor"]
