//! YOLOv26 Main Detector Implementation
//!
//! Chịu trách nhiệm khởi tạo Session NMS-Free (v26/v10).

use arrow::array::{Array, Float32Array};

use log::{debug, info, warn};
use ndarray::Array4;
use ort::ep::{CoreML, ExecutionProvider};
use ort::session::Session;
use ort::value::Value;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList};
use std::time::Instant;

use crate::yolo::{YoloDetection, ModelConfig, YoloTask, ExecutionProviderType};
use crate::image_proc::draw_rect_native;

#[pyclass]
pub struct YoloV26Detector {
    pub(crate) session: Session,
    pub(crate) input_width: usize,
    pub(crate) input_height: usize,
    pub(crate) conf_threshold: f32,
    pub(crate) task: YoloTask,
    pub(crate) num_classes: usize,
    pub(crate) num_keypoints: usize,
    pub(crate) num_mask_coeffs: usize,
    #[pyo3(get)]
    pub is_cls_model: bool,
    pub(crate) last_preprocess_ms: f64,
    #[pyo3(get)]
    pub last_inference_ms: f64,
    #[pyo3(get)]
    pub last_nms_ms: f64, // decode time
    // Buffer tái sử dụng để tránh cấp phát bộ nhớ liên tục
    pub(crate) input_tensor_buffer: Array4<f32>,
    #[pyo3(get)]
    pub ep: ExecutionProviderType,
}

pub(crate) struct YoloResultsV26 {
    pub(crate) detections: Vec<YoloDetection>,
    pub(crate) proto: Option<ndarray::ArrayD<f32>>,
}

#[pymethods]
impl YoloV26Detector {
    #[new]
    #[pyo3(signature = (model_path, conf_threshold=0.25, execution_provider="coreml"))]
    fn new(model_path: &str, conf_threshold: f32, execution_provider: &str) -> PyResult<Self> {
        debug!("YoloV26Detector::new called with model: {}, execution_provider: {}", model_path, execution_provider);

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lỗi khởi tạo Session: {}", e)))?;

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

        let session = session_builder
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lỗi cấu hình bộ thực thi: {}", e)))?
            .with_intra_threads(1)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lỗi tải model: {}", e)))?;

        let config = ModelConfig::identify(model_path, &session);

        info!(
            "Cấu hình Model V26: nhiệm vụ={:?}, đầu vào={}x{}",
            config.task, config.input_size.0, config.input_size.1
        );

        Ok(Self {
            session,
            input_width: config.input_size.0,
            input_height: config.input_size.1,
            conf_threshold,
            task: config.task,
            num_classes: config.num_classes,
            num_keypoints: config.num_keypoints,
            num_mask_coeffs: config.num_mask_coeffs,
            is_cls_model: config.task == YoloTask::Classification,
            last_preprocess_ms: 0.0,
            last_inference_ms: 0.0,
            last_nms_ms: 0.0,
            input_tensor_buffer: Array4::zeros((1, 3, config.input_size.1, config.input_size.0)),
            ep,
        })
    }

    #[getter]
    fn preprocess_ms(&self) -> f64 { self.last_preprocess_ms }
    #[getter]
    fn inference_ms(&self) -> f64 { self.last_inference_ms }
    #[getter]
    fn nms_ms(&self) -> f64 { self.last_nms_ms }

    fn detect_to_arrow<'py>(
        &mut self,
        py: Python<'py>,
        numpy_array: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>, Py<PyAny>, Py<PyAny>)> {
        let (width, height) = self.prepare_input(numpy_array)?;
        let results = self.run_inference_internal(py, (width, height))?;

        let (arr_cap, sch_cap) = crate::ffi::export_detections_to_arrow(
            py,
            &results.detections,
            self.num_keypoints,
            self.num_mask_coeffs,
        )?;
        
        let (p_arr, p_sch) = if let Some(p) = results.proto {
            let proto_array = Float32Array::from(p.into_raw_vec_and_offset().0);
            let (pa, ps) = crate::ffi::export_to_python(py, proto_array.to_data())?;
            (pa.into_any().unbind(), ps.into_any().unbind())
        } else {
            (py.None(), py.None())
        };

        Ok((arr_cap, sch_cap, p_arr, p_sch))
    }

    /// Nhận diện YOLOv26 và vẽ native trực tiếp lên buffer (Vòng lặp đóng)
    fn detect_and_draw<'py>(
        &mut self,
        py: Python<'py>,
        numpy_array: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>, Py<PyAny>, Py<PyAny>)> {
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
        
        let (p_arr, p_sch) = if let Some(p) = proto_flat {
            let data_vec: Vec<f32> = p.into_raw_vec_and_offset().0;
            let proto_array = Float32Array::from(data_vec);
            let (pa, ps) = crate::ffi::export_to_python(py, proto_array.to_data())?;
            (pa.into_any().unbind(), ps.into_any().unbind())
        } else {
            (py.None(), py.None())
        };

        Ok((arr_cap, sch_cap, p_arr, p_sch))
    }

    fn detect_from_numpy(&mut self, py: Python, numpy_array: &Bound<PyAny>) -> PyResult<Py<PyList>> {
        let (width, height) = self.prepare_input(numpy_array)?;
        let results = self.run_inference_internal(py, (width, height))?;
        let py_list = PyList::empty(py);
        for det in results.detections { py_list.append(Py::new(py, det)?)?; }
        Ok(py_list.into())
    }
}

impl YoloV26Detector {
    #[inline]
    pub(crate) fn run_detection_pipeline<'py>(&mut self, py: Python<'py>, numpy_array: &Bound<'py, PyAny>) -> PyResult<(Vec<YoloDetection>, Option<ndarray::ArrayD<f32>>, usize, usize)> {
        let (width, height) = self.prepare_input(numpy_array)?;
        let results = self.run_inference_internal(py, (width, height))?;
        Ok((results.detections, results.proto, width, height))
    }

    pub(crate) fn prepare_input(&mut self, numpy_array: &Bound<'_, PyAny>) -> PyResult<(usize, usize)> {
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
            raw_data, 
            width, 
            height, 
            self.input_width, 
            self.input_height, 
            &mut self.input_tensor_buffer,
            true
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        
        self.last_preprocess_ms = t_pre.elapsed().as_secs_f64() * 1000.0;
        Ok((width, height))
    }

    pub(crate) fn run_inference_internal(
        &mut self,
        py: Python,
        orig_dim: (usize, usize),
    ) -> PyResult<YoloResultsV26> {
        let input_tensor = Value::from_array(self.input_tensor_buffer.clone()).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let t_infer = Instant::now();
        let outputs = py.detach(|| self.session.run(ort::inputs![input_tensor])).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        self.last_inference_ms = t_infer.elapsed().as_secs_f64() * 1000.0;


        let t_decode = Instant::now();
        
        let results = match self.task {
            YoloTask::Classification => {
                let out_value = outputs.values().next().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No output found"))?;
                let out_extract = out_value.try_extract_tensor::<f32>().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let detections = Self::decode_cls_v26(out_extract.1, self.num_classes);
                YoloResultsV26 { detections, proto: None }
            },
            YoloTask::Pose => Self::decode_pose_v26(
                self.conf_threshold,
                self.input_width,
                self.input_height,
                self.num_keypoints,
                &outputs,
                orig_dim,
            )?,
            YoloTask::Segmentation => Self::decode_seg_v26(
                self.conf_threshold,
                self.input_width,
                self.input_height,
                self.num_mask_coeffs,
                &outputs,
                orig_dim,
            )?,
            _ => Self::decode_base_v26(
                self.conf_threshold,
                self.input_width,
                self.input_height,
                &outputs,
                orig_dim,
            )?,
        };

        self.last_nms_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        Ok(results)
    }
}
