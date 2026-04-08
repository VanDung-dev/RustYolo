//! Common types for YOLO Inference Engine
//!
//! File này chứa các định nghĩa dùng chung giữa các phiên bản YOLO (v8, v26, ...):
//! - Struct YoloDetection: Kết quả trả về cho mỗi đối tượng phát hiện được
//! - Các hằng số hoặc tiện ích chung khác

use pyo3::prelude::*;
use ort::session::Session;
use log::{info};

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

#[pyclass(from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YoloArchitecture {
    V8, // YOLOv8, v11 (Anchor-based + NMS)
    V26, // YOLOv26, v10 (NMS-Free)
}

#[pyclass(from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YoloTask {
    Detection,
    Pose,
    Segmentation,
    OBB,
    Classification,
}

pub struct ModelConfig {
    pub arch: YoloArchitecture,
    pub task: YoloTask,
    pub input_size: (usize, usize),
    pub num_classes: usize,
    pub num_keypoints: usize,
    pub num_mask_coeffs: usize,
}

impl ModelConfig {
    /// Tự động xác định cấu trúc model dựa trên filename và Session outputs
    pub fn identify(path: &str, _session: &Session) -> Self {
        let name = path.to_lowercase();
        
        // 1. Xác định Kiến trúc (Arch)
        let mut arch = YoloArchitecture::V8;
        if name.contains("v26") || name.contains("26") || name.contains("v10") || name.contains("nms-free") {
            arch = YoloArchitecture::V26;
        }

        // 2. Xác định Task
        let task = if name.contains("-pose") {
            YoloTask::Pose
        } else if name.contains("-seg") {
            YoloTask::Segmentation
        } else if name.contains("-obb") {
            YoloTask::OBB
        } else if name.contains("-cls") {
            YoloTask::Classification
        } else {
            YoloTask::Detection
        };

        // 3. Cấu hình mặc định theo task
        let mut num_classes = 80;
        let mut num_keypoints = 0;
        let mut num_mask_coeffs = 0;
        let input_size = (640, 640);

        match task {
            YoloTask::Pose => {
                num_classes = 1;
                num_keypoints = 17;
            }
            YoloTask::Segmentation => {
                num_mask_coeffs = 32;
            }
            YoloTask::OBB => {
                num_classes = 15;
            }
            _ => {}
        }

        info!(
            "Identified Model: Arch={:?}, Task={:?}, Input={}x{}, path={}",
            arch, task, input_size.0, input_size.1, path
        );

        Self {
            arch,
            task,
            input_size,
            num_classes,
            num_keypoints,
            num_mask_coeffs,
        }
    }
}
