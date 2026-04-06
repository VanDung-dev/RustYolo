"""
Module theo dõi hiệu năng hệ thống
Chạy trong thread riêng biệt không block main loop
"""

import time
import platform
import psutil
import threading
from typing import Dict, Any
import subprocess


class PerformanceMonitor:
    """Theo dõi và hiển thị các thông số hiệu năng."""

    def __init__(self):
        self._lock = threading.Lock()
        self.fps = 0.0
        self.latency = 0.0
        self.frame_times = []
        self.gpu_info: Dict[str, Any] = {}
        self.cpu_temp = 0.0
        self.cpu_usage = 0.0
        self.memory_usage: Dict[str, Any] = {}
        self.running = False
        self.thread = None
        self.last_update = 0.0

        self._init_gpu_info()
        self._init_cpu_info()

    def _init_gpu_info(self):
        """Khởi tạo thông tin GPU."""
        with self._lock:
            self.gpu_info = {
                "available": False,
                "name": "N/A",
                "load": 0.0,
                "power": "N/A",
                "temperature": "N/A",
                "memory_used": "N/A",
                "memory_total": "N/A",
            }

            # Kiểm tra CUDA (NVIDIA GPU)
            try:
                import torch

                if torch.cuda.is_available():
                    self.gpu_info["available"] = True
                    self.gpu_info["name"] = torch.cuda.get_device_name(0)
            except ImportError:
                pass

            # Kiểm tra Apple Silicon (macOS)
            if platform.system() == "Darwin":
                try:
                    result = subprocess.run(
                        ["system_profiler", "SPDisplaysDataType"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        for line in result.stdout.split("\n"):
                            if "Chip" in line and any(
                                keyword in line for keyword in ["Apple", "M1", "M2", "M3", "M4"]
                            ):
                                self.gpu_info["available"] = True
                                self.gpu_info["name"] = line.strip().split(":")[-1].strip()
                                break
                except Exception:
                    pass

    def _init_cpu_info(self):
        """Khởi tạo thông tin CPU và Memory."""
        with self._lock:
            self.memory_usage = {
                "total": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
                "used": f"{psutil.virtual_memory().used / (1024**3):.1f} GB",
                "percent": psutil.virtual_memory().percent,
            }

    def start_background_monitor(self):
        """Chạy monitor trong thread riêng biệt."""
        self.running = True
        self.thread = threading.Thread(target=self._background_loop, daemon=True)
        self.thread.start()

    def stop_background_monitor(self):
        """Dừng background monitor thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def _background_loop(self):
        """Vòng lặp chạy trong background thread."""
        while self.running:
            # Cập nhật các thông số không cần realtime
            self.get_cpu_usage()
            self.get_memory_usage()
            self.update_gpu_info()

            time.sleep(1.0)  # Cập nhật mỗi 1 giây

    def update_frame_time(self, latency_ms: float):
        """Cập nhật thời gian xử lý frame - gọi từ main thread."""
        with self._lock:
            self.latency = latency_ms
            self.frame_times.append(time.time())
            # Giữ lại frame times trong 1 giây
            self.frame_times = [
                t for t in self.frame_times if time.time() - t < 1.0
            ]
            self.fps = len(self.frame_times)

    def update_gpu_info(self):
        """Cập nhật thông tin GPU - chạy trong background."""
        with self._lock:
            if platform.system() == "Darwin":
                self._update_macos_gpu_info()
            else:
                self._update_nvidia_gpu_info()

    def _update_macos_gpu_info(self):
        """Cập nhật GPU info trên macOS - không block main thread."""
        try:
            self.gpu_info["load"] = psutil.cpu_percent(interval=0.1)
            self.gpu_info["power"] = "~3-5W"
        except Exception:
            pass

    def _update_nvidia_gpu_info(self):
        """Cập nhật GPU info trên NVIDIA GPU."""
        if not self.gpu_info["available"]:
            return

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 5:
                    self.gpu_info["load"] = float(parts[0].strip())
                    self.gpu_info["temperature"] = f"{parts[1].strip()}°C"
                    self.gpu_info["power"] = f"{parts[2].strip()}W"
                    self.gpu_info["memory_used"] = f"{parts[3].strip()} MB"
                    self.gpu_info["memory_total"] = f"{parts[4].strip()} MB"
        except Exception:
            pass

    def get_cpu_temperature(self):
        """Lấy nhiệt độ CPU (macOS/Linux)."""
        # Tạm bỏ đọc nhiệt độ để không làm chậm main loop
        return 0.0

    def get_cpu_usage(self):
        """Lấy CPU usage."""
        with self._lock:
            self.cpu_usage = psutil.cpu_percent(interval=None)
            return self.cpu_usage

    def get_memory_usage(self):
        """Lấy memory usage."""
        with self._lock:
            self.memory_usage = {
                "total": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
                "used": f"{psutil.virtual_memory().used / (1024**3):.1f} GB",
                "percent": psutil.virtual_memory().percent,
            }
            return self.memory_usage

    def get_stats(self) -> Dict[str, Any]:
        """Lấy tất cả stats thread-safe."""
        with self._lock:
            return {
                "fps": self.fps,
                "latency": self.latency,
                "gpu_info": self.gpu_info.copy(),
                "cpu_usage": self.cpu_usage,
                "cpu_temp": self.cpu_temp,
                "memory_usage": self.memory_usage.copy(),
            }
