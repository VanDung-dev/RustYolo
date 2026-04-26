//! Common types for YOLO Inference Engine
//!
//! File này chứa các định nghĩa dùng chung giữa các phiên bản YOLO (v8, v26, ...):
//! - Struct YoloDetection: Kết quả trả về cho mỗi đối tượng phát hiện được
//! - Các hằng số hoặc tiện ích chung khác

use pyo3::prelude::*;
use pyo3::types::PyAny;
use ort::session::Session;
use ort::ep::{CoreML, ExecutionProvider};
use log::{info, warn};
use ndarray::Array4;
use std::time::Instant;

/// Different execution providers available for ONNX Runtime
#[pyclass(from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProviderType {
    /// Apple CoreML (macOS only)
    CoreML,
    /// WebGPU (cross-platform graphics API, includes Vulkan/Metal/DirectX12)
    WebGPU,
    /// CPU fallback
    CPU,
}

impl ExecutionProviderType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "coreml" => {
                if CoreML::default().is_available().unwrap_or(false) {
                    Self::CoreML
                } else {
                    warn!("⚠️ CoreML không khả dụng. Đang chuyển sang sử dụng CPU.");
                    Self::CPU
                }
            }
            "webgpu" => {
                #[cfg(feature = "webgpu")]
                {
                    if ort::ep::WebGPU::default().is_available().unwrap_or(false) {
                        Self::WebGPU
                    } else {
                        warn!("⚠️ WebGPU không khả dụng. Đang chuyển sang sử dụng CPU.");
                        Self::CPU
                    }
                }
                #[cfg(not(feature = "webgpu"))]
                {
                    warn!("⚠️ Tính năng WebGPU không được bật. Đang chuyển sang sử dụng CPU.");
                    Self::CPU
                }
            }
            "cpu" => Self::CPU,
            _ => {
                warn!("Không rõ bộ thực thi '{}', đang chuyển sang sử dụng CPU.", s);
                Self::CPU
            }
        }
    }

    pub fn get_dispatch(&self) -> Vec<ort::ep::ExecutionProviderDispatch> {
        match self {
            Self::CoreML => {
                vec![CoreML::default()
                    .with_subgraphs(true)
                    .with_low_precision_accumulation_on_gpu(true)
                    .with_compute_units(ort::ep::coreml::ComputeUnits::All)
                    .build()]
            }
            Self::WebGPU => {
                #[cfg(feature = "webgpu")]
                { vec![ort::ep::WebGPU::default().build()] }
                #[cfg(not(feature = "webgpu"))]
                { vec![] }
            }
            Self::CPU => vec![],
        }
    }

    pub fn configure_session(&self, builder: ort::session::builder::SessionBuilder) -> PyResult<ort::session::builder::SessionBuilder> {
        builder.with_execution_providers(self.get_dispatch())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}

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
    fn class_id(&self) -> i32 { self.class_id }
    #[getter]
    fn confidence(&self) -> f32 { self.confidence }
    #[getter]
    fn x(&self) -> f32 { self.x }
    #[getter]
    fn y(&self) -> f32 { self.y }
    #[getter]
    fn width(&self) -> f32 { self.width }
    #[getter]
    fn height(&self) -> f32 { self.height }
    #[getter]
    fn keypoints(&self) -> Vec<(f32, f32, f32)> { self.keypoints.clone() }
    #[getter]
    fn mask_coeffs(&self) -> Vec<f32> { self.mask_coeffs.clone() }

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
    #[allow(dead_code)]
    pub arch: YoloArchitecture,
    pub task: YoloTask,
    pub input_size: (usize, usize),
    pub num_classes: usize,
    pub num_keypoints: usize,
    pub num_mask_coeffs: usize,
}

impl ModelConfig {
    pub fn identify(path: &str, _session: &Session) -> Self {
        let name = path.to_lowercase();
        let arch = if name.contains("v26") || name.contains("26") || name.contains("v10") || name.contains("nms-free") {
            YoloArchitecture::V26
        } else {
            YoloArchitecture::V8
        };

        let task = if name.contains("-pose") { YoloTask::Pose }
            else if name.contains("-seg") { YoloTask::Segmentation }
            else if name.contains("-obb") { YoloTask::OBB }
            else if name.contains("-cls") { YoloTask::Classification }
            else { YoloTask::Detection };

        let mut num_classes = 80;
        let mut num_keypoints = 0;
        let mut num_mask_coeffs = 0;

        match task {
            YoloTask::Pose => { num_classes = 1; num_keypoints = 17; }
            YoloTask::Segmentation => { num_mask_coeffs = 32; }
            YoloTask::OBB => { num_classes = 15; }
            _ => {}
        }

        info!("Xác định Model: Kiến trúc={:?}, Nhiệm vụ={:?}, đầu vào=640x640, đường dẫn={}", arch, task, path);

        Self { arch, task, input_size: (640, 640), num_classes, num_keypoints, num_mask_coeffs }
    }
}

/// Shared utilities for YOLO detectors to reduce redundancy
pub(crate) struct YoloCommon {
    pub(crate) session: Session,
    pub(crate) input_width: usize,
    pub(crate) input_height: usize,
    pub(crate) num_classes: usize,
    pub(crate) num_keypoints: usize,
    pub(crate) num_mask_coeffs: usize,
    pub(crate) input_tensor_buffer: Array4<f32>,
    pub(crate) ep: ExecutionProviderType,
}

impl YoloCommon {
    pub fn prepare_input(&mut self, numpy_array: &Bound<'_, PyAny>) -> PyResult<(usize, usize, f64)> {
        let shape: (usize, usize, usize) = numpy_array.getattr("shape")?.extract()?;
        let (height, width, _channels) = shape;
        
        crate::security::validate_input_shape(width, height, _channels)?;
        let data_ptr = numpy_array.getattr("ctypes")?.getattr("data")?.extract::<usize>()?;
        let expected_size = width * height * 3;
        let actual_size = numpy_array.getattr("size")?.extract::<usize>()?;
        crate::security::validate_buffer_size(actual_size, width, height, 3)?;

        let raw_data = unsafe { std::slice::from_raw_parts(data_ptr as *const u8, expected_size) };
        let t_pre = Instant::now();
        
        crate::image_proc::preprocess_image_kornia(
            raw_data, width, height, self.input_width, self.input_height, 
            &mut self.input_tensor_buffer, true
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        
        let elapsed = t_pre.elapsed().as_secs_f64() * 1000.0;
        Ok((width, height, elapsed))
    }

    #[inline]
    pub fn compute_iou(
        box1: (f32, f32, f32, f32),
        box2: (f32, f32, f32, f32),
    ) -> f32 {
        let x1 = box1.0.max(box2.0);
        let y1 = box1.1.max(box2.1);
        let x2 = (box1.0 + box1.2).min(box2.0 + box2.2);
        let y2 = (box1.1 + box1.3).min(box2.1 + box2.3);

        let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        let area1 = box1.2 * box1.3;
        let area2 = box2.2 * box2.3;
        let union = area1 + area2 - intersection;

        if union == 0.0 { 0.0 } else { intersection / union }
    }

    pub fn perform_nms(
        boxes: &[(f32, f32, f32, f32)],
        class_ids: &[usize],
        iou_threshold: f32,
    ) -> Vec<usize> {
        let mut keep = vec![true; boxes.len()];
        let mut result = Vec::new();

        for i in 0..boxes.len() {
            if !keep[i] { continue; }
            result.push(i);

            for j in (i + 1)..boxes.len() {
                if keep[j] && class_ids[j] == class_ids[i] {
                    if Self::compute_iou(boxes[i], boxes[j]) > iou_threshold {
                        keep[j] = false;
                    }
                }
            }
        }
        result
    }

    pub fn decode_classification(out_data: &[f32]) -> Vec<YoloDetection> {
        let max_val = out_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = out_data.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|x| x / sum).collect();

        let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        indexed.into_iter().take(5).map(|(idx, prob)| YoloDetection {
            class_id: idx as i32,
            confidence: prob,
            x: 0.0, y: 0.0, width: 0.0, height: 0.0,
            keypoints: vec![],
            mask_coeffs: vec![],
        }).collect()
    }
}
