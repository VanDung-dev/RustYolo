//! Common types for YOLO Inference Engine
//!
//! File này chứa các định nghĩa dùng chung giữa các phiên bản YOLO (v8, v26, ...):
//! - Struct YoloDetection: Kết quả trả về cho mỗi đối tượng phát hiện được
//! - Các hằng số hoặc tiện ích chung khác

use pyo3::prelude::*;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct YoloDetection {
    pub class_id: i32,
    pub confidence: f32,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub keypoints: Vec<(f32, f32, f32)>, // (x, y, conf)
    pub mask_coeffs: Vec<f32>,           // Segmentation coefficients
}

#[pymethods]
impl YoloDetection {
    #[getter]
    fn class_id(&self) -> i32 {
        self.class_id
    }
    #[getter]
    fn confidence(&self) -> f32 {
        self.confidence
    }
    #[getter]
    fn x(&self) -> f32 {
        self.x
    }
    #[getter]
    fn y(&self) -> f32 {
        self.y
    }
    #[getter]
    fn width(&self) -> f32 {
        self.width
    }
    #[getter]
    fn height(&self) -> f32 {
        self.height
    }
    #[getter]
    fn keypoints(&self) -> Vec<(f32, f32, f32)> {
        self.keypoints.clone()
    }
    #[getter]
    fn mask_coeffs(&self) -> Vec<f32> {
        self.mask_coeffs.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "YoloDetection(class_id={}, confidence={:.3}, x={:.1}, y={:.1}, w={:.1}, h={:.1})",
            self.class_id, self.confidence, self.x, self.y, self.width, self.height
        )
    }
}
