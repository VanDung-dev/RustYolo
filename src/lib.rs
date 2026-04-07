//! Rust extension module cho macOS system monitoring và YOLOv8x inference
//! Sử dụng PyO3 0.28, sysinfo 0.32 và ONNX Runtime

use pyo3::prelude::*;

mod monitor;
mod yolo;
mod ffi;

pub use monitor::PerformanceMonitor;
pub use yolo::{YoloDetection, YoloV8Detector};

/// Module initialization
#[pymodule]
#[allow(unused_variables)]
fn rust_yolo(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PerformanceMonitor>()?;
    m.add_class::<YoloV8Detector>()?;
    m.add_class::<YoloDetection>()?;

    Ok(())
}
