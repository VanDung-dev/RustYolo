//! YOLOv8 Main Detector Implementation
//!
//! Chịu trách nhiệm khởi tạo Session và điều phối luồng inference.

use arrow::array::{Array, Float32Array};

use log::{debug};
use ndarray::{Array4};
use ort::session::Session;
use ort::value::Value;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList};
use std::time::Instant;

use crate::yolo::{YoloDetection, ModelConfig, YoloTask, ExecutionProviderType, YoloCommon};
use crate::image_proc::draw_rect_native;

#[pyclass]
pub struct YoloV8Detector {
    pub(crate) common: YoloCommon,
    pub(crate) conf_threshold: f32,
    pub(crate) iou_threshold: f32,
    #[pyo3(get)]
    pub is_cls_model: bool,
    #[pyo3(get)]
    pub is_obb_model: bool,
    pub(crate) last_preprocess_ms: f64,
    #[pyo3(get)]
    pub last_inference_ms: f64,
    #[pyo3(get)]
    pub last_nms_ms: f64,
}

#[pymethods]
impl YoloV8Detector {
    #[new]
    #[pyo3(signature = (model_path, conf_threshold=0.25, iou_threshold=0.45, execution_provider="coreml"))]
    fn new(model_path: &str, conf_threshold: f32, iou_threshold: f32, execution_provider: &str) -> PyResult<Self> {
        debug!("YoloV8Detector::new: model={}, ep={}", model_path, execution_provider);
        crate::security::validate_model_path(model_path)?;
        
        let ep = ExecutionProviderType::from_str(execution_provider);
        let session_builder = Session::builder()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Session builder error: {}", e)))?;
        
        let num_threads = if ep == ExecutionProviderType::CPU { 
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
        } else { 1 };

        let session = ep.configure_session(session_builder)?
            .with_intra_threads(num_threads)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Load model error: {}", e)))?;
        
        let config = ModelConfig::identify(model_path, &session);
        
        Ok(Self {
            common: YoloCommon {
                session,
                input_width: config.input_size.0,
                input_height: config.input_size.1,
                num_classes: config.num_classes,
                num_keypoints: config.num_keypoints,
                num_mask_coeffs: config.num_mask_coeffs,
                input_tensor_buffer: Array4::zeros((1, 3, config.input_size.1, config.input_size.0)),
                ep,
            },
            conf_threshold,
            iou_threshold,
            is_cls_model: config.task == YoloTask::Classification,
            is_obb_model: config.task == YoloTask::OBB,
            last_preprocess_ms: 0.0,
            last_inference_ms: 0.0,
            last_nms_ms: 0.0,
        })
    }

    #[getter]
    fn ep(&self) -> ExecutionProviderType { self.common.ep }
    #[getter]
    fn preprocess_ms(&self) -> f64 { self.last_preprocess_ms }
    #[getter]
    fn inference_ms(&self) -> f64 { self.last_inference_ms }
    #[getter]
    fn nms_ms(&self) -> f64 { self.last_nms_ms }

    fn detect_to_arrow<'py>(
        &mut self,
        py: Python<'py>,
        numpy_array: &Bound<'py,PyAny>,
    ) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>, Py<PyAny>, Py<PyAny>)> {
        let (detections, proto_flat, _, _) = self.run_detection_pipeline(py, numpy_array)?;
        let (arr_cap, sch_cap) = crate::ffi::export_detections_to_arrow(
            py, &detections, self.common.num_keypoints, self.common.num_mask_coeffs,
        )?;

        let (proto_arr, proto_sch) = if let Some(proto) = proto_flat {
            let proto_array = Float32Array::from(proto);
            let (pa, ps) = crate::ffi::export_to_python(py, proto_array.to_data())?;
            (pa.into_any().unbind(), ps.into_any().unbind())
        } else { (py.None(), py.None()) };

        Ok((arr_cap, sch_cap, proto_arr, proto_sch))
    }

    fn detect_and_draw<'py>(
        &mut self,
        py: Python<'py>,
        numpy_array: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>, Py<PyAny>, Py<PyAny>)> {
        let (detections, proto_flat, width, height) = self.run_detection_pipeline(py, numpy_array)?;

        let data_ptr = numpy_array.getattr("ctypes")?.getattr("data")?.extract::<usize>()?;
        let expected_size = width * height * 3;
        let actual_size = numpy_array.getattr("size")?.extract::<usize>()?;
        crate::security::validate_buffer_size(actual_size, width, height, 3)?;

        let data = unsafe { std::slice::from_raw_parts_mut(data_ptr as *mut u8, expected_size) };
        let colors: [[u8; 3]; 6] = [[0, 255, 0], [0, 255, 255], [255, 255, 0], [255, 0, 0], [255, 0, 255], [0, 165, 255]];

        for det in &detections {
            let color = colors[det.class_id as usize % colors.len()];
            draw_rect_native(data, width, height, det.x, det.y, det.x + det.width, det.y + det.height, color, 2);
        }

        let (arr_cap, sch_cap) = crate::ffi::export_detections_to_arrow(
            py, &detections, self.common.num_keypoints, self.common.num_mask_coeffs,
        )?;

        let (proto_arr, proto_sch) = if let Some(proto) = proto_flat {
            let proto_array = Float32Array::from(proto);
            let (pa, ps) = crate::ffi::export_to_python(py, proto_array.to_data())?;
            (pa.into_any().unbind(), ps.into_any().unbind())
        } else { (py.None(), py.None()) };

        Ok((arr_cap, sch_cap, proto_arr, proto_sch))
    }

    fn detect_from_numpy(&mut self, py: Python, numpy_array: &Bound<PyAny>) -> PyResult<Py<PyList>> {
        let (detections, _, _, _) = self.run_detection_pipeline(py, numpy_array)?;
        let py_list = PyList::empty(py);
        for det in detections {
            py_list.append(Py::new(py, det)?)?;
        }
        Ok(py_list.into())
    }

    fn get_input_size(&self) -> (usize, usize) { (self.common.input_width, self.common.input_height) }
    fn set_conf_threshold(&mut self, threshold: f32) { self.conf_threshold = threshold; }
    fn set_iou_threshold(&mut self, threshold: f32) { self.iou_threshold = threshold; }
}

impl YoloV8Detector {
    pub(crate) fn run_detection_pipeline<'py>(&mut self, _py: Python<'py>, numpy_array: &Bound<'py, PyAny>) -> PyResult<(Vec<YoloDetection>, Option<Vec<f32>>, usize, usize)> {
        let (width, height, prep_ms) = self.common.prepare_input(numpy_array)?;
        self.last_preprocess_ms = prep_ms;
        let (detections, proto_flat) = self.run_inference_internal(_py, (width, height))?;
        Ok((detections, proto_flat, width, height))
    }

    pub(crate) fn run_inference_internal(&mut self, py: Python, orig_dim: (usize, usize)) -> PyResult<(Vec<YoloDetection>, Option<Vec<f32>>)> {
        let input_buffer = std::mem::replace(
            &mut self.common.input_tensor_buffer,
            Array4::zeros((1, 3, self.common.input_height, self.common.input_width)),
        );
        let input_tensor = Value::from_array(input_buffer)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Input tensor error: {}", e)))?;

        let t_infer = Instant::now();
        let outputs = py.detach(|| self.common.session.run(ort::inputs![input_tensor]))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Inference error: {}", e)))?;
        self.last_inference_ms = t_infer.elapsed().as_secs_f64() * 1000.0;

        let out_value = &outputs["output0"];
        let out_extract = out_value.try_extract_tensor::<f32>().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let shape_usize: Vec<usize> = out_extract.0.iter().map(|&d| d as usize).collect();
        let out_data = ndarray::ArrayViewD::from_shape(shape_usize, out_extract.1).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let proto_flat = if self.common.num_mask_coeffs > 0 {
            outputs.get("output1").map(|v| v.try_extract_tensor::<f32>().map(|t| t.1.to_vec())).transpose()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
        } else { None };

        if self.is_cls_model {
            return Ok((YoloCommon::decode_classification(out_extract.1), proto_flat));
        }

        let t_nms = Instant::now();
        let detections = if self.is_obb_model {
            Self::decode_obb_v8(self.conf_threshold, self.iou_threshold, self.common.num_classes, self.common.input_width, self.common.input_height, &out_data, orig_dim)
        } else if self.common.num_keypoints > 0 {
            Self::decode_pose_v8(self.conf_threshold, self.iou_threshold, self.common.num_classes, self.common.num_keypoints, self.common.input_width, self.common.input_height, &out_data, orig_dim)
        } else if self.common.num_mask_coeffs > 0 {
            Self::decode_seg_v8(self.conf_threshold, self.iou_threshold, self.common.num_classes, self.common.num_mask_coeffs, self.common.input_width, self.common.input_height, &out_data, orig_dim)
        } else {
            Self::decode_base_v8(self.conf_threshold, self.iou_threshold, self.common.num_classes, self.common.input_width, self.common.input_height, &out_data, orig_dim)
        }?;
        self.last_nms_ms = t_nms.elapsed().as_secs_f64() * 1000.0;

        Ok((detections, proto_flat))
    }
}
