//! Tiện ích Bảo mật cho RustYolo
//!
//! Module này cung cấp các hàm validate để ngăn chặn các lỗ hổng bảo mật phổ biến
//! như Path Traversal, Buffer Overflow và Unrestricted Input.

use std::path::Path;
use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};

/// Kiểm tra đường dẫn model có an toàn và trỏ đến định dạng file được cho phép không.
/// Ngăn chặn tấn công Path Traversal.
pub fn validate_model_path(path: &str) -> PyResult<()> {
    let p = Path::new(path);
    
    // Kiểm tra các thành phần path traversal
    if p.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
        return Err(PyValueError::new_err("Đường dẫn model không hợp lệ: phát hiện path traversal"));
    }
    
    // Chỉ cho phép model .onnx
    if let Some(ext) = p.extension() {
        if ext != "onnx" {
            return Err(PyValueError::new_err("Chỉ cho phép sử dụng model .onnx"));
        }
    } else {
        return Err(PyValueError::new_err("File model phải có phần mở rộng .onnx"));
    }
    
    Ok(())
}

/// Kiểm tra kích thước đầu vào để ngăn chặn DoS (cạn kiệt bộ nhớ).
pub fn validate_input_shape(width: usize, height: usize, channels: usize) -> PyResult<()> {
    const MAX_DIMENSION: usize = 4096;
    const EXPECTED_CHANNELS: usize = 3;

    if width == 0 || height == 0 {
        return Err(PyValueError::new_err("Kích thước không hợp lệ: chiều rộng và chiều cao phải > 0"));
    }

    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(PyValueError::new_err(format!(
            "Kích thước đầu vào quá lớn: {}x{} (tối đa {}x{})",
            width, height, MAX_DIMENSION, MAX_DIMENSION
        )));
    }

    if channels != EXPECTED_CHANNELS {
        return Err(PyValueError::new_err(format!(
            "Yêu cầu {} kênh màu (BGR/RGB), nhưng nhận được {}",
            EXPECTED_CHANNELS, channels
        )));
    }

    Ok(())
}

/// Kiểm tra kích thước buffer có khớp với kích thước dự kiến dựa trên các chiều không.
/// Ngăn chặn Buffer Overflow/Over-read.
pub fn validate_buffer_size(actual_len: usize, width: usize, height: usize, channels: usize) -> PyResult<()> {
    let expected_len = width * height * channels;
    if actual_len < expected_len {
        return Err(PyRuntimeError::new_err(format!(
            "Kích thước buffer không khớp: dự kiến ít nhất {} bytes, nhận được {}",
            expected_len, actual_len
        )));
    }
    Ok(())
}
