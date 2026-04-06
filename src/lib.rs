//! Rust extension module cho macOS system monitoring
//! Sử dụng PyO3 0.28 và sysinfo 0.32

use pyo3::prelude::*;
use pyo3::types::PyDict;
use sysinfo::{System, CpuRefreshKind, RefreshKind};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use std::ffi::c_void;
use std::os::raw::c_char;


/// Struct quản lý monitoring hệ thống
#[pyclass]
struct PerformanceMonitor {
    system: Arc<Mutex<System>>,
    running: Arc<Mutex<bool>>,
    thread_handle: Option<thread::JoinHandle<()>>,

    fps: f64,
    latency: f64,
    frame_times: Vec<f64>,
}


#[pymethods]
impl PerformanceMonitor {
    #[new]
    fn new() -> Self {
        let mut system = System::new_with_specifics(
            RefreshKind::everything()
                .with_cpu(CpuRefreshKind::everything())
        );
        system.refresh_all();

        PerformanceMonitor {
            system: Arc::new(Mutex::new(system)),
            running: Arc::new(Mutex::new(false)),
            thread_handle: None,
            fps: 0.0,
            latency: 0.0,
            frame_times: Vec::with_capacity(60),
        }
    }

    /// Bắt đầu background monitor thread
    fn start_background_monitor(&mut self) {
        *self.running.lock().unwrap() = true;

        let system_clone = self.system.clone();
        let running_clone = self.running.clone();

        self.thread_handle = Some(thread::spawn(move || {
            while *running_clone.lock().unwrap() {
                if let Ok(mut sys) = system_clone.lock() {
                    sys.refresh_cpu_all();
                    sys.refresh_memory();
                }

                thread::sleep(Duration::from_millis(1000));
            }
        }));
    }

    /// Dừng background thread
    fn stop_background_monitor(&mut self) {
        *self.running.lock().unwrap() = false;

        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }

    /// Cập nhật thời gian xử lý frame
    fn update_frame_time(&mut self, latency_ms: f64) {
        self.latency = latency_ms;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        self.frame_times.push(now);

        // Xóa các frame cũ hơn 1 giây
        self.frame_times.retain(|&t| now - t < 1.0);
        self.fps = self.frame_times.len() as f64;
    }

    /// Nhận buffer pointer trực tiếp từ Python numpy (zero copy)
    fn process_frame(&mut self, py: Python, ptr: usize, length: usize) -> PyResult<f64> {
        let start = std::time::Instant::now();
        
        unsafe {
            // Truy cập trực tiếp memory buffer từ OpenCV numpy array
            let data_ptr = ptr as *const c_char;
            
            let mut sum: u64 = 0;
            for i in 0..length {
                sum += *data_ptr.add(i) as u64;
            }
            let avg = sum as f64 / length as f64;
            
            let latency = start.elapsed().as_secs_f64() * 1000.0;
            self.update_frame_time(latency);
            
            Ok(avg)
        }
    }

    /// Lấy tất cả stats dưới dạng Python dict
    fn get_stats(&self, py: Python) -> Py<PyDict> {
        let stats = PyDict::new(py);

        let sys = self.system.lock().unwrap();

        // Frame stats
        stats.set_item(pyo3::intern!(py, "fps"), self.fps).unwrap();
        stats.set_item(pyo3::intern!(py, "latency"), self.latency).unwrap();

        // CPU info
        let cpu_usage = sys.global_cpu_usage();
        stats.set_item(pyo3::intern!(py, "cpu_usage"), cpu_usage).unwrap();
        stats.set_item(pyo3::intern!(py, "cpu_temp"), 0.0).unwrap();

        // Memory info
        let mem_used = sys.used_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        let mem_total = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        let mem_percent = (mem_used / mem_total) * 100.0;

        let memory_usage = PyDict::new(py);
        memory_usage.set_item(pyo3::intern!(py, "used"), format!("{:.1} GB", mem_used)).unwrap();
        memory_usage.set_item(pyo3::intern!(py, "total"), format!("{:.1} GB", mem_total)).unwrap();
        memory_usage.set_item(pyo3::intern!(py, "percent"), mem_percent).unwrap();

        stats.set_item(pyo3::intern!(py, "memory_usage"), memory_usage).unwrap();

        // GPU info
        let gpu_info = PyDict::new(py);
        gpu_info.set_item(pyo3::intern!(py, "available"), true).unwrap();
        gpu_info.set_item(pyo3::intern!(py, "name"), "Apple Silicon GPU").unwrap();
        gpu_info.set_item(pyo3::intern!(py, "load"), cpu_usage).unwrap();
        gpu_info.set_item(pyo3::intern!(py, "power"), "~3-5W").unwrap();
        gpu_info.set_item(pyo3::intern!(py, "temperature"), "N/A").unwrap();
        gpu_info.set_item(pyo3::intern!(py, "memory_used"), "N/A").unwrap();
        gpu_info.set_item(pyo3::intern!(py, "memory_total"), "N/A").unwrap();

        stats.set_item(pyo3::intern!(py, "gpu_info"), gpu_info).unwrap();

        stats.into()
    }
}


/// Module initialization
#[pymodule]
#[allow(unused_variables)]
fn rust_yolo(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PerformanceMonitor>()?;

    Ok(())
}
