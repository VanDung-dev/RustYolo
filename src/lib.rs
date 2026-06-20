//! Native Extension Rust cho Python
//!
//! Đây là entry point chính của toàn bộ engine.
//! File này chịu trách nhiệm:
//! - Export các class Rust sang Python thông qua PyO3
//! - Khởi tạo module native khi Python import
//! - Đăng ký binding cho tất cả các chức năng
//!
//! Toàn bộ logic tính toán nằm trong các file con, không ở đây.

use pyo3::prelude::*;

mod face;
mod ffi;
pub mod image_proc;
mod monitor;
mod security;
mod v8;
mod v26;
mod yolo;

pub use monitor::PerformanceMonitor;
pub use v8::YoloV8Detector;
pub use v26::YoloV26Detector;
pub use yolo::YoloDetection;

/// Khởi tạo module
#[pymodule]
#[allow(unused_variables)]
fn rust_yolo(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    // Cấu hình Rayon global thread pool: tối ưu cho mixed CPU/GPU workloads
    // Dùng available_parallelism, tránh oversubscription khi GPU đang chạy inference
    if let Ok(n) = std::thread::available_parallelism() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n.get().max(2))
            .build_global()
            .ok();
    }

    m.add_class::<PerformanceMonitor>()?;
    m.add_class::<YoloV8Detector>()?;
    m.add_class::<YoloV26Detector>()?;
    m.add_class::<YoloDetection>()?;
    m.add_class::<yolo::YoloArchitecture>()?;
    m.add_class::<yolo::YoloTask>()?;
    m.add_class::<yolo::ExecutionProviderType>()?;
    m.add_class::<face::ffi::FaceToolbox>()?;

    Ok(())
}
