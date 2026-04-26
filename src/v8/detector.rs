//! YOLOv8 Main Detector Implementation
//!
//! Chịu trách nhiệm khởi tạo Session và điều phối luồng inference.

use arrow::array::{Array, Float32Array};

use log::{debug, info, warn};
use ndarray::{Array4};
use ort::ep::{CoreML, ExecutionProvider};
use ort::session::Session;
use ort::value::Value;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList};
use std::time::Instant;

use crate::yolo::{YoloDetection, ModelConfig, YoloTask, ExecutionProviderType};
use crate::image_proc::draw_rect_native;

#[pyclass]
pub struct YoloV8Detector {
    pub(crate) session: Session,
    pub(crate) input_width: usize,
    pub(crate) input_height: usize,
    pub(crate) conf_threshold: f32,
    pub(crate) iou_threshold: f32,
    pub(crate) num_classes: usize,
    pub(crate) num_keypoints: usize,
    pub(crate) num_mask_coeffs: usize,
    #[pyo3(get)]
    pub is_cls_model: bool,
    #[pyo3(get)]
    pub is_obb_model: bool,
    pub(crate) last_preprocess_ms: f64,
    #[pyo3(get)]
    pub last_inference_ms: f64,
    #[pyo3(get)]
    pub last_nms_ms: f64,
    // Buffer tái sử dụng để tránh cấp phát bộ nhớ liên tục (640*640*3*4 bytes)
    pub(crate) input_tensor_buffer: Array4<f32>,
    #[pyo3(get)]
    pub ep: ExecutionProviderType,
}

#[pymethods]
impl YoloV8Detector {
    #[new]
    #[pyo3(signature = (model_path, conf_threshold=0.25, iou_threshold=0.45, execution_provider="coreml"))]
    fn new(model_path: &str, conf_threshold: f32, iou_threshold: f32, execution_provider: &str) -> PyResult<Self> {
        debug!("YoloV8Detector::new called with model: {}, execution_provider: {}", model_path, execution_provider);
        
        crate::security::validate_model_path(model_path)?;
        
        let ep = match execution_provider.to_lowercase().as_str() {
            "coreml" => {
                if !CoreML::default().is_available().unwrap_or(false) {
                    warn!("⚠️ CoreML không khả dụng. Đang chuyển sang sử dụng CPU.");
                    ExecutionProviderType::CPU
                } else {
                    info!("🍎 CoreML khả dụng! Đang kích hoạt tăng tốc phần cứng...");
                    ExecutionProviderType::CoreML
                }
            }
            "webgpu" => {
                #[cfg(feature = "webgpu")]
                {
                    if !ort::ep::WebGPU::default().is_available().unwrap_or(false) {
                        warn!("⚠️ WebGPU không khả dụng. Đang chuyển sang sử dụng CPU.");
                        ExecutionProviderType::CPU
                    } else {
                        info!("🌐 WebGPU khả dụng! Đang sử dụng tăng tốc GPU đa nền tảng...");
                        ExecutionProviderType::WebGPU
                    }
                }
                #[cfg(not(feature = "webgpu"))]
                {
                    warn!("⚠️ Tính năng WebGPU không được bật trong bản build này. Đang chuyển sang sử dụng CPU.");
                    ExecutionProviderType::CPU
                }
            }
            "cpu" => ExecutionProviderType::CPU,
            _ => {
                warn!("Không rõ bộ thực thi '{}', đang chuyển sang sử dụng CPU.", execution_provider);
                ExecutionProviderType::CPU
            }
        };
        
        let session_builder = Session::builder()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Không thể tạo session builder: {}",
                    e
                ))
            })?;
        
        let session_builder = match ep {
            ExecutionProviderType::CoreML => {
                session_builder.with_execution_providers([CoreML::default()
                    .with_subgraphs(true)
                    .with_low_precision_accumulation_on_gpu(true)
                    .with_compute_units(ort::ep::coreml::ComputeUnits::All)
                    .build()])
            }
            ExecutionProviderType::WebGPU => {
                #[cfg(feature = "webgpu")]
                { session_builder.with_execution_providers([ort::ep::WebGPU::default().build()]) }
                #[cfg(not(feature = "webgpu"))]
                {
                    warn!("Tính năng WebGPU không được bật. Đang chuyển sang sử dụng CPU.");
                    Ok(session_builder)
                }
            }
            ExecutionProviderType::CPU => {
                Ok(session_builder)
            }
        };
        
        // Cấu hình Session tối ưu cho Apple Silicon
        let session = session_builder
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Không thể kích hoạt bộ thực thi: {}",
                    e
                ))
            })?
            .with_intra_threads(1) // M4 Pro có nhân hiệu năng cao, giảm tranh chấp thread
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Không thể tải model từ {}: {}",
                    model_path, e
                ))
            })?;
        
        let config = ModelConfig::identify(model_path, &session);
        
        info!(
            "Cấu hình Model: kiến trúc={:?}, nhiệm vụ={:?}, số lớp={}, đầu vào={}x{}",
            config.arch, config.task, config.num_classes, config.input_size.0, config.input_size.1
        );
        
        Ok(Self {
            session,
            input_width: config.input_size.0,
            input_height: config.input_size.1,
            conf_threshold,
            iou_threshold,
            num_classes: config.num_classes,
            num_keypoints: config.num_keypoints,
            num_mask_coeffs: config.num_mask_coeffs,
            is_cls_model: config.task == YoloTask::Classification,
            is_obb_model: config.task == YoloTask::OBB,
            last_preprocess_ms: 0.0,
            last_inference_ms: 0.0,
            last_nms_ms: 0.0,
            input_tensor_buffer: Array4::zeros((1, 3, config.input_size.1, config.input_size.0)),
            ep,
        })
    }

    #[getter]
    fn preprocess_ms(&self) -> f64 {
        self.last_preprocess_ms
    }
    #[getter]
    fn inference_ms(&self) -> f64 {
        self.last_inference_ms
    }
    #[getter]
    fn nms_ms(&self) -> f64 {
        self.last_nms_ms
    }

    /// Chạy inference và trả về kết quả Zero Copy qua Arrow Capsule
    fn detect_to_arrow<'py>(
        &mut self,
        py: Python<'py>,
        numpy_array: &Bound<'py,PyAny>,
    ) -> PyResult<(
        Bound<'py, PyCapsule>,
        Bound<'py, PyCapsule>,
        Py<PyAny>,
        Py<PyAny>,
    )> {
        let (detections, proto_flat, _width, _height) = self.run_detection_pipeline(py, numpy_array)?;

        let (arr_cap, sch_cap) = crate::ffi::export_detections_to_arrow(
            py,
            &detections,
            self.num_keypoints,
            self.num_mask_coeffs,
        )?;

        let (proto_arr, proto_sch) = if let Some(proto) = proto_flat {
            let proto_array = Float32Array::from(proto);
            let (pa, ps) = crate::ffi::export_to_python(py, proto_array.to_data())?;
            (pa.into_any().unbind(), ps.into_any().unbind())
        } else {
            (py.None(), py.None())
        };

        Ok((arr_cap, sch_cap, proto_arr, proto_sch))
    }

    /// Nhận diện AI và vẽ trực tiếp Bounding Box lên ảnh (Vẽ Native trong Rust)
    /// Trả về kết quả Arrow như cũ để không làm gãy logic Python
    fn detect_and_draw<'py>(
        &mut self,
        py: Python<'py>,
        numpy_array: &Bound<'py, PyAny>,
    ) -> PyResult<(
        Bound<'py, PyCapsule>,
        Bound<'py, PyCapsule>,
        Py<PyAny>,
        Py<PyAny>,
    )> {
        // 1. Chạy AI Pipeline
        let (detections, proto_flat, width, height) = self.run_detection_pipeline(py, numpy_array)?;

        // 2. Lấy con trỏ bộ nhớ của ảnh để vẽ Native
        let data_ptr = numpy_array
            .getattr("ctypes")?
            .getattr("data")?
            .extract::<usize>()?;
        
        let expected_size = width * height * 3;
        let actual_size = numpy_array.getattr("size")?.extract::<usize>()?;
        crate::security::validate_buffer_size(actual_size, width, height, 3)?;

        // Tạo mutable slice từ con trỏ (BGR/RGB)
        let data = unsafe { std::slice::from_raw_parts_mut(data_ptr as *mut u8, expected_size) };

        // 3. Vẽ Native Bounding Box
        let colors: [[u8; 3]; 6] = [
            [0, 255, 0],   // Green
            [0, 255, 255], // Yellow
            [255, 255, 0], // Cyan
            [255, 0, 0],   // Red
            [255, 0, 255], // Magenta
            [0, 165, 255], // Orange
        ];

        for det in &detections {
            let color = colors[det.class_id as usize % colors.len()];
            draw_rect_native(
                data, 
                width, 
                height, 
                det.x, 
                det.y, 
                det.x + det.width, 
                det.y + det.height, 
                color, 
                2
            );
        }

        // 4. Đóng gói kết quả Arrow
        let (arr_cap, sch_cap) = crate::ffi::export_detections_to_arrow(
            py,
            &detections,
            self.num_keypoints,
            self.num_mask_coeffs,
        )?;

        let (proto_arr, proto_sch) = if let Some(proto) = proto_flat {
            let proto_array = Float32Array::from(proto);
            let (pa, ps) = crate::ffi::export_to_python(py, proto_array.to_data())?;
            (pa.into_any().unbind(), ps.into_any().unbind())
        } else {
            (py.None(), py.None())
        };

        Ok((arr_cap, sch_cap, proto_arr, proto_sch))
    }

    fn detect_from_numpy(
        &mut self,
        py: Python,
        numpy_array: &Bound<PyAny>,
    ) -> PyResult<Py<PyList>> {
        let (detections, _, _width, _height) = self.run_detection_pipeline(py, numpy_array)?;

        let py_list = PyList::empty(py);
        for det in detections {
            let py_det = Py::new(py, det)?;
            py_list.append(py_det)?;
        }
        Ok(py_list.into())
    }

    fn get_input_size(&self) -> (usize, usize) {
        (self.input_width, self.input_height)
    }

    fn set_conf_threshold(&mut self, threshold: f32) {
        self.conf_threshold = threshold;
    }

    fn set_iou_threshold(&mut self, threshold: f32) {
        self.iou_threshold = threshold;
    }
}

impl YoloV8Detector {
    #[inline]
    pub(crate) fn create_empty_detection(class_id: i32, confidence: f32) -> YoloDetection {
        YoloDetection {
            class_id,
            confidence,
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
            keypoints: vec![],
            mask_coeffs: vec![],
        }
    }

    #[inline]
    pub(crate) fn run_detection_pipeline<'py>(&mut self, py: Python<'py>, numpy_array: &Bound<'py, PyAny>) -> PyResult<(Vec<YoloDetection>, Option<Vec<f32>>, usize, usize)> {
        let (width, height) = self.prepare_input(numpy_array)?;
        let (detections, proto_flat) = self.run_inference_internal(py, (width, height))?;
        Ok((detections, proto_flat, width, height))
    }

    #[inline]
    pub(crate) fn prepare_input(&mut self, numpy_array: &Bound<'_, PyAny>) -> PyResult<(usize, usize)> {
        let shape_obj = numpy_array.getattr("shape")?;
        let shape: (usize, usize, usize) = shape_obj.extract()?;
        let (height, width, _channels) = shape;

        crate::security::validate_input_shape(width, height, _channels)?;
        
        let data_ptr = numpy_array
            .getattr("ctypes")?
            .getattr("data")?
            .extract::<usize>()?;

        let expected_size = width * height * 3;
        let actual_size = numpy_array.getattr("size")?.extract::<usize>()?;
        crate::security::validate_buffer_size(actual_size, width, height, 3)?;

        let raw_data =
            unsafe { std::slice::from_raw_parts(data_ptr as *const u8, expected_size) };

        let t_pre = Instant::now();
        // Zero-allocation: Pre-allocated buffer reuse
        crate::image_proc::preprocess_image_kornia(
            raw_data,
            width,
            height,
            self.input_width,
            self.input_height,
            &mut self.input_tensor_buffer,
            true, // is_bgr
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Preprocessing failed: {}",
                e
            ))
        })?;
        self.last_preprocess_ms = t_pre.elapsed().as_secs_f64() * 1000.0;

        Ok((width, height))
    }

    pub(crate) fn run_inference_internal(
        &mut self,
        py: Python,
        orig_dim: (usize, usize),
    ) -> PyResult<(Vec<YoloDetection>, Option<Vec<f32>>)> {
        // Create tensor from pre-allocated buffer (Cloned for ORT ownership)
        let input_tensor = Value::from_array(self.input_tensor_buffer.clone()).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create input tensor: {}",
                e
            ))
        })?;

        let t_infer = Instant::now();
        let outputs = py
            .detach(|| self.session.run(ort::inputs![input_tensor]))
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Inference failed: {}",
                    e
                ))
            })?;
        self.last_inference_ms = t_infer.elapsed().as_secs_f64() * 1000.0;


        let out_value = &outputs["output0"];
        let out_extract = out_value.try_extract_tensor::<f32>().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Extract error: {}", e))
        })?;

        let shape_usize: Vec<usize> = out_extract.0.iter().map(|&d| d as usize).collect();
        let out_data = ndarray::ArrayViewD::from_shape(shape_usize, out_extract.1).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("ArrayView error: {}", e))
        })?;

        let proto_flat: Option<Vec<f32>> = if self.num_mask_coeffs > 0 {
            match outputs.get("output1") {
                Some(v) => {
                    let t = v.try_extract_tensor::<f32>().map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Proto error: {}", e))
                    })?;
                    Some(t.1.to_vec())
                }
                None => {
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "Thiếu đầu ra mô hình output1 (proto)"
                    ));
                }
            }
        } else {
            None
        };

        if self.is_cls_model {
            let cls_results = Self::decode_cls_v8(out_extract.1, self.num_classes);
            return Ok((cls_results, proto_flat));
        }

        let t_nms = Instant::now();
        
        let detections = if self.is_obb_model {
            Self::decode_obb_v8(
                self.conf_threshold,
                self.iou_threshold,
                self.num_classes,
                self.input_width,
                self.input_height,
                &out_data,
                orig_dim,
            )
        } else if self.num_keypoints > 0 {
            Self::decode_pose_v8(
                self.conf_threshold,
                self.iou_threshold,
                self.num_classes,
                self.num_keypoints,
                self.input_width,
                self.input_height,
                &out_data,
                orig_dim,
            )
        } else if self.num_mask_coeffs > 0 {
            Self::decode_seg_v8(
                self.conf_threshold,
                self.iou_threshold,
                self.num_classes,
                self.num_mask_coeffs,
                self.input_width,
                self.input_height,
                &out_data,
                orig_dim,
            )
        } else {
            Self::decode_base_v8(
                self.conf_threshold,
                self.iou_threshold,
                self.num_classes,
                self.input_width,
                self.input_height,
                &out_data,
                orig_dim,
            )
        }?;

        self.last_nms_ms = t_nms.elapsed().as_secs_f64() * 1000.0;

        Ok((detections, proto_flat))
    }

    pub(crate) fn compute_iou_internal(
        box1: &(f32, f32, f32, f32, f32, usize, Vec<(f32, f32, f32)>, Vec<f32>),
        box2: &(f32, f32, f32, f32, f32, usize, Vec<(f32, f32, f32)>, Vec<f32>),
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
}
