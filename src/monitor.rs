//! Hệ thống Monitoring hiệu năng Native macOS
//!
//! Chạy hoàn toàn độc lập trên background thread riêng:
//! - Đọc trực tiếp sensor hệ thống macOS không qua trung gian
//! - Theo dõi CPU, Memory, nhiệt độ
//! - Tính toán Thermal Gradient dT/dt realtime
//! - Đo FPS và breakdown latency
//! - Tính toán ước lượng hiệu năng GPU Apple Silicon
//!
//! Không gây block hay overhead cho luồng chính AI

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use sysinfo::{Components, CpuRefreshKind, RefreshKind, System};

/// Struct quản lý monitoring hệ thống
#[pyclass]
pub struct PerformanceMonitor {
    system: Arc<Mutex<System>>,
    components: Arc<Mutex<Components>>,
    running: Arc<Mutex<bool>>,
    thread_handle: Option<thread::JoinHandle<()>>,

    fps: f64,
    ai_latency: f64,
    rust_latency: f64,
    frame_times: Vec<f64>,

    // Thermal data
    current_temp: f32,
    prev_temp: f32,
    last_temp_update: Instant,
    dt_dt: f32, // degC/sec
}

#[pymethods]
impl PerformanceMonitor {
    #[new]
    pub fn new() -> Self {
        let mut system = System::new_with_specifics(
            RefreshKind::everything().with_cpu(CpuRefreshKind::everything()),
        );
        system.refresh_all();

        let mut components = Components::new();
        components.refresh_list();

        PerformanceMonitor {
            system: Arc::new(Mutex::new(system)),
            components: Arc::new(Mutex::new(components)),
            running: Arc::new(Mutex::new(false)),
            thread_handle: None,
            fps: 0.0,
            ai_latency: 0.0,
            rust_latency: 0.0,
            frame_times: Vec::with_capacity(60),
            current_temp: 0.0,
            prev_temp: 0.0,
            last_temp_update: Instant::now(),
            dt_dt: 0.0,
        }
    }

    /// Bắt đầu background monitor thread
    pub fn start_background_monitor(&mut self) {
        *self.running.lock().unwrap() = true;

        let system_clone = self.system.clone();
        let components_clone = self.components.clone();
        let running_clone = self.running.clone();

        self.thread_handle = Some(thread::spawn(move || {
            while *running_clone.lock().unwrap() {
                if let Ok(mut sys) = system_clone.lock() {
                    sys.refresh_cpu_all();
                    sys.refresh_memory();
                }

                if let Ok(mut comp) = components_clone.lock() {
                    comp.refresh_list();
                }

                thread::sleep(Duration::from_millis(1000));
            }
        }));
    }

    /// Dừng background thread
    pub fn stop_background_monitor(&mut self) {
        *self.running.lock().unwrap() = false;

        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }

    /// Cập nhật thời gian xử lý AI và tính toán thermal gradient
    pub fn update_frame_time(&mut self, latency_ms: f64) {
        self.ai_latency = latency_ms;

        let now_sys = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        self.frame_times.push(now_sys);
        self.frame_times.retain(|&t| now_sys - t < 1.0);
        self.fps = self.frame_times.len() as f64;

        // Cập nhật chỉ số nhiệt độ
        if let Ok(comp) = self.components.lock() {
            // Tìm sensor nhiệt độ CPU / GPU từ danh sách sensor hệ thống
            let mut max_temp = 0.0;
            for component in comp.iter() {
                let name = component.label().to_lowercase();
                if name.contains("cpu") || name.contains("gpu") || name.contains("die") {
                    if component.temperature() > max_temp {
                        max_temp = component.temperature();
                    }
                }
            }

            if max_temp > 0.0 {
                let now = Instant::now();
                let duration = now.duration_since(self.last_temp_update).as_secs_f32();
                if duration >= 1.0 {
                    self.prev_temp = self.current_temp;
                    self.current_temp = max_temp;
                    if self.prev_temp > 0.0 {
                        self.dt_dt = (self.current_temp - self.prev_temp) / duration;
                    }
                    self.last_temp_update = now;
                }
            }
        }
    }

    /// Nhận buffer pointer trực tiếp từ Python numpy (zero copy)
    fn process_frame(&mut self, _py: Python, ptr: usize, length: usize) -> PyResult<f64> {
        let start = std::time::Instant::now();

        unsafe {
            let data_ptr = ptr as *const u8;

            let mut sum: u64 = 0;
            // Tối ưu hóa: Chỉ tính toán trung bình cho 1 phần nhỏ frame để tiết kiệm CPU
            // do đây chỉ là ví dụ logic. Thực tế sẽ làm thermal logic cao cấp hơn.
            let step = (length / 1000).max(1);
            for i in (0..length).step_by(step) {
                sum += *data_ptr.add(i) as u64;
            }
            let avg = sum as f64 / (length / step) as f64;

            let latency = start.elapsed().as_secs_f64() * 1000.0;
            self.rust_latency = latency;

            Ok(avg)
        }
    }

    /// Lấy tất cả stats dưới dạng Python dict
    fn get_stats(&self, py: Python) -> Py<PyDict> {
        let stats = PyDict::new(py);

        let sys = self.system.lock().unwrap();

        // Frame stats
        stats.set_item(pyo3::intern!(py, "fps"), self.fps).unwrap();
        stats
            .set_item(pyo3::intern!(py, "ai_latency"), self.ai_latency)
            .unwrap();
        stats
            .set_item(pyo3::intern!(py, "rust_latency"), self.rust_latency)
            .unwrap();

        // CPU info
        let cpu_usage = sys.global_cpu_usage();
        stats
            .set_item(pyo3::intern!(py, "cpu_usage"), cpu_usage)
            .unwrap();
        stats
            .set_item(pyo3::intern!(py, "cpu_temp"), self.current_temp)
            .unwrap();
        stats
            .set_item(pyo3::intern!(py, "dt_dt"), self.dt_dt)
            .unwrap();

        // Memory info
        let mem_used = sys.used_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        let mem_total = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        let mem_percent = (mem_used / mem_total) * 100.0;

        let memory_usage = PyDict::new(py);
        memory_usage
            .set_item(pyo3::intern!(py, "used"), format!("{:.1} GB", mem_used))
            .unwrap();
        memory_usage
            .set_item(pyo3::intern!(py, "total"), format!("{:.1} GB", mem_total))
            .unwrap();
        memory_usage
            .set_item(pyo3::intern!(py, "percent"), mem_percent)
            .unwrap();

        stats
            .set_item(pyo3::intern!(py, "memory_usage"), memory_usage)
            .unwrap();

        // Thông tin GPU Apple Silicon
        // macOS không cung cấp API đọc trực tiếp GPU metrics nên ước lượng từ nhiệt độ CPU
        let gpu_info = PyDict::new(py);

        // Ước tính hiệu năng GPU dựa trên nhiệt độ và tải CPU
        let gpu_load = (cpu_usage * 0.85).clamp(0.0, 100.0);
        let gpu_temp = self.current_temp + 2.5;
        let gpu_power = 2.2 + (gpu_load / 100.0) * 18.0; // ✅ Phạm vi 2.2W idle -> 20W max

        gpu_info
            .set_item(pyo3::intern!(py, "available"), true)
            .unwrap();
        gpu_info
            .set_item(pyo3::intern!(py, "name"), "Apple Silicon GPU")
            .unwrap();
        gpu_info
            .set_item(pyo3::intern!(py, "load"), gpu_load)
            .unwrap();
        gpu_info
            .set_item(pyo3::intern!(py, "power"), gpu_power)
            .unwrap();
        gpu_info
            .set_item(pyo3::intern!(py, "temperature"), gpu_temp)
            .unwrap();

        stats
            .set_item(pyo3::intern!(py, "gpu_info"), gpu_info)
            .unwrap();

        stats.into()
    }
}
