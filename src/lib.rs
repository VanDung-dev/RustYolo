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

    m.add_class::<PerformanceMonitor>()?;
    m.add_class::<YoloV8Detector>()?;
    m.add_class::<YoloV26Detector>()?;
    m.add_class::<YoloDetection>()?;
    m.add_class::<yolo::YoloArchitecture>()?;
    m.add_class::<yolo::YoloTask>()?;
    m.add_class::<yolo::ExecutionProviderType>()?;
    m.add_class::<face::ffi::FaceToolbox>()?;
    m.add_class::<face::ffi::FaceToolbox>()?;

    Ok(())
}
