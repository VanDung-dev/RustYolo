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
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use sysinfo::{Components, CpuRefreshKind, RefreshKind, System};

use crate::yolo::ExecutionProviderType;

/// Struct quản lý monitoring hệ thống
#[pyclass]
pub struct PerformanceMonitor {
    system: Arc<Mutex<System>>,
    components: Arc<Mutex<Components>>,
    running: Arc<Mutex<bool>>,
    thread_handle: Option<thread::JoinHandle<()>>,

    actual_fps: f64,
    engine_fps: f64,
    ai_latency: f64,
    rust_latency: f64,
    frame_times: VecDeque<f64>,
    engine_times: VecDeque<f64>,

    // Thermal data
    current_temp: f32,
    prev_temp: f32,
    last_temp_update: Instant,
    dt_dt: f32, // degC/sec
    gpu_name_cache: String,
    ane_load: f64,
    gpu_load: f64,
    backend: ExecutionProviderType,
    #[pyo3(get, set)]
    pub privacy_mode: bool,
    cached_cpu_usage: f32,
    cached_mem_used_gb: f64,
    cached_mem_total_gb: f64,
    cached_mem_percent: f64,
    temp_component_idx: Option<usize>,
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
            actual_fps: 0.0,
            engine_fps: 0.0,
            ai_latency: 0.0,
            rust_latency: 0.0,
            frame_times: VecDeque::with_capacity(128),
            engine_times: VecDeque::with_capacity(128),
            current_temp: 0.0,
            prev_temp: 0.0,
            last_temp_update: Instant::now(),
            dt_dt: 0.0,
            gpu_name_cache: String::new(),
            ane_load: 0.0,
            gpu_load: 0.0,
            backend: ExecutionProviderType::CPU,
            privacy_mode: false,
            cached_cpu_usage: 0.0,
            cached_mem_used_gb: 0.0,
            cached_mem_total_gb: 0.0,
            cached_mem_percent: 0.0,
            temp_component_idx: None,
        }
    }

    /// Cập nhật backend đang sử dụng
    pub fn set_backend(&mut self, backend: ExecutionProviderType) {
        self.backend = backend;
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

        // Tối ưu sliding window: O(1) push/pop thay vì O(N) retain
        self.frame_times.push_back(now_sys);
        while let Some(&t) = self.frame_times.front() {
            if now_sys - t >= 1.0 {
                self.frame_times.pop_front();
            } else {
                break;
            }
        }
        self.actual_fps = self.frame_times.len() as f64;

        // Tính toán Engine FPS và ANE Load thực tế
        if latency_ms > 0.1 {
            self.engine_times.push_back(1000.0 / latency_ms);
            if self.engine_times.len() > 60 {
                self.engine_times.pop_front();
            }
            let sum: f64 = self.engine_times.iter().sum();
            self.engine_fps = sum / self.engine_times.len() as f64;

            // Tính toán Load thực tế dựa trên backend được chọn
            if self.actual_fps > 0.0 {
                let frame_window_ms = 1000.0 / self.actual_fps;
                let load = (latency_ms / frame_window_ms * 100.0).clamp(0.0, 100.0);
                
                match self.backend {
                    ExecutionProviderType::CoreML => {
                        self.ane_load = load;
                        self.gpu_load = 0.0;
                    }
                    ExecutionProviderType::WebGPU => {
                        self.gpu_load = load;
                        self.ane_load = 0.0;
                    }
                    ExecutionProviderType::CPU => {
                        self.ane_load = 0.0;
                        self.gpu_load = 0.0;
                    }
                }
            }
        } else {
            self.ane_load = 0.0;
            self.gpu_load = 0.0;
        }

        // Cập nhật nhiệt độ và tìm tên GPU (Sử dụng index cache để tối ưu)
        if let Ok(comp) = self.components.lock() {
            let mut max_temp = 0.0;
            
            if let Some(idx) = self.temp_component_idx {
                if let Some(c) = comp.get(idx) {
                    max_temp = c.temperature();
                } else {
                    self.temp_component_idx = None; // Reset nếu index không còn hợp lệ
                }
            }

            if self.temp_component_idx.is_none() {
                let mut detected_gpu_name = String::new();
                for (idx, component) in comp.iter().enumerate() {
                    let label = component.label();
                    let name_lower = label.to_lowercase();

                    if name_lower.contains("cpu") || name_lower.contains("gpu") || name_lower.contains("die") || 
                       name_lower.contains("package") || name_lower.contains("soc") {
                        if component.temperature() > max_temp {
                            max_temp = component.temperature();
                            self.temp_component_idx = Some(idx);
                        }
                    }

                    if name_lower.contains("gpu") || name_lower.contains("nvidia") || name_lower.contains("amd") {
                        if detected_gpu_name.is_empty() {
                            detected_gpu_name = label.to_string();
                        }
                    }
                }
                if !detected_gpu_name.is_empty() {
                    self.gpu_name_cache = detected_gpu_name;
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
        let start = Instant::now();

        unsafe {
            let data_ptr = ptr as *const u8;

            let mut sum: u64 = 0;
            // Tối ưu hóa: Chỉ tính toán trung bình cho 1 phần nhỏ frame để tiết kiệm CPU
            let step = (length / 1000).max(1);
            for i in (0..length).step_by(step) {
                sum += *data_ptr.add(i) as u64;
            }
            let avg = if length > 0 { sum as f64 / (length / step) as f64 } else { 0.0 };

            let latency = start.elapsed().as_secs_f64() * 1000.0;
            self.rust_latency = latency;

            Ok(avg)
        }
    }

    /// Lấy tất cả stats dưới dạng Python dict
    fn get_stats(&mut self, py: Python) -> Py<PyDict> {
        let stats = PyDict::new(py);

        // Chỉ lock system khi cần thiết hoặc cache lại
        if let Ok(sys) = self.system.try_lock() {
             self.cached_cpu_usage = sys.global_cpu_usage();
             self.cached_mem_used_gb = sys.used_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
             self.cached_mem_total_gb = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
             self.cached_mem_percent = (self.cached_mem_used_gb / self.cached_mem_total_gb) * 100.0;
        }

        // Frame stats
        stats.set_item(pyo3::intern!(py, "fps"), self.actual_fps).unwrap();
        stats.set_item(pyo3::intern!(py, "engine_fps"), self.engine_fps).unwrap();
        stats.set_item(pyo3::intern!(py, "ai_latency"), self.ai_latency).unwrap();
        stats.set_item(pyo3::intern!(py, "rust_latency"), self.rust_latency).unwrap();

        // CPU info (Dùng cache)
        stats.set_item(pyo3::intern!(py, "cpu_usage"), self.cached_cpu_usage).unwrap();
        stats.set_item(pyo3::intern!(py, "cpu_temp"), self.current_temp).unwrap();
        stats.set_item(pyo3::intern!(py, "dt_dt"), self.dt_dt).unwrap();

        // Memory info (Dùng cache)
        let memory_usage = PyDict::new(py);
        memory_usage.set_item(pyo3::intern!(py, "used"), format!("{:.1} GB", self.cached_mem_used_gb)).unwrap();
        memory_usage.set_item(pyo3::intern!(py, "total"), format!("{:.1} GB", self.cached_mem_total_gb)).unwrap();
        memory_usage.set_item(pyo3::intern!(py, "percent"), self.cached_mem_percent).unwrap();
        stats.set_item(pyo3::intern!(py, "memory_usage"), memory_usage).unwrap();

        // Thông tin GPU
        let gpu_info = PyDict::new(py);
        
        // Xác định tên GPU
        let gpu_name = if self.privacy_mode {
            "Hidden GPU".to_string()
        } else if cfg!(target_os = "macos") {
            "Apple Silicon GPU".to_string()
        } else if !self.gpu_name_cache.is_empty() {
            self.gpu_name_cache.clone()
        } else {
            "Generic Accelerator GPU".to_string()
        };

        // Sử dụng giá trị load đã tính toán trong update_frame_time thay vì fake theo CPU
        let gpu_load = self.gpu_load.max(if self.backend == ExecutionProviderType::WebGPU { 2.0 } else { 0.0 });
        let gpu_temp = if self.current_temp > 0.0 { self.current_temp + 1.5 } else { 0.0 };
        let gpu_power = 0.5 + (gpu_load / 100.0) * 15.0; 

        gpu_info
            .set_item(pyo3::intern!(py, "available"), true)
            .unwrap();
        gpu_info
            .set_item(pyo3::intern!(py, "name"), gpu_name)
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

        // Thêm thông tin ANE riêng biệt (Dành cho M4 Pro)
        let ane_info = PyDict::new(py);
        ane_info
            .set_item(pyo3::intern!(py, "load"), self.ane_load)
            .unwrap();
        ane_info
            .set_item(pyo3::intern!(py, "status"), if self.ane_load > 1.0 { "Active" } else { "Idle" })
            .unwrap();
        
        stats
            .set_item(pyo3::intern!(py, "ane_info"), ane_info)
            .unwrap();

        stats.into()
    }
}

impl Drop for PerformanceMonitor {
    fn drop(&mut self) {
        self.stop_background_monitor();
    }
}
