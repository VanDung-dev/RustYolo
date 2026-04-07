//! ✅ Native Extension Rust cho Python
//!
//! Đây là entry point chính của toàn bộ engine.
//! File này chịu trách nhiệm:
//! - Export các class Rust sang Python thông qua PyO3
//! - Khởi tạo module native khi Python import
//! - Đăng ký binding cho tất cả các chức năng
//!
//! ⚠️ Toàn bộ logic tính toán nằm trong các file con, không ở đây.

use pyo3::prelude::*;

mod monitor;
mod yolo;
mod ffi;
pub mod image_proc;

pub use monitor::PerformanceMonitor;
pub use yolo::{YoloDetection, YoloV8Detector};

/// Khởi tạo module
#[pymodule]
#[allow(unused_variables)]
fn rust_yolo(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PerformanceMonitor>()?;
    m.add_class::<YoloV8Detector>()?;
    m.add_class::<YoloDetection>()?;

    Ok(())
}
