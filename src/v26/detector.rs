//! YOLOv26 Main Detector Implementation
//!
//! Chịu trách nhiệm khởi tạo Session NMS-Free (v26/v10).

use arrow::array::{Array, Float32Array};
use log::{debug, info};
use ndarray::Array4;
use ort::session::Session;
use ort::value::Value;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList};
use std::time::Instant;

use crate::yolo::{YoloDetection, ModelConfig, YoloTask, ExecutionProviderType, YoloCommon};
use crate::image_proc::draw_rect_native;

#[pyclass]
pub struct YoloV26Detector {
    pub(crate) common: YoloCommon,
    pub(crate) conf_threshold: f32,
    pub(crate) task: YoloTask,
    #[pyo3(get)]
    pub is_cls_model: bool,
    pub(crate) last_preprocess_ms: f64,
    #[pyo3(get)]
    pub last_inference_ms: f64,
    #[pyo3(get)]
    pub last_nms_ms: f64,
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
        debug!("YoloV26Detector::new: model={}, ep={}", model_path, execution_provider);
        crate::security::validate_model_path(model_path)?;
        
        let ep = ExecutionProviderType::from_str(execution_provider);
        
        // Kiểm tra an toàn: YOLOv26 (NMS-Free) có các Op không tương thích ổn định với CoreML trên macOS
        if ep == ExecutionProviderType::CoreML {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "⚠️ YOLOv26 (NMS-Free) KHÔNG hỗ trợ CoreML do lỗi tương thích Op (GatherElements). \n\
                 Vui lòng khởi chạy lại với: CPU hoặc WebGPU"
            ));
        }

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
        
        info!(
            "Cấu hình Model V26: nhiệm vụ={:?}, đầu vào={}x{}",
            config.task, config.input_size.0, config.input_size.1
        );

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
            task: config.task,
            is_cls_model: config.task == YoloTask::Classification,
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
        numpy_array: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>, Py<PyAny>, Py<PyAny>)> {
        let (detections, proto_flat, _, _) = self.run_detection_pipeline(py, numpy_array)?;
        let (arr_cap, sch_cap) = crate::ffi::export_detections_to_arrow(
            py, &detections, self.common.num_keypoints, self.common.num_mask_coeffs,
        )?;

        let (proto_arr, proto_sch) = if let Some(proto) = proto_flat {
            let proto_array = Float32Array::from(proto.into_raw_vec_and_offset().0);
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
            let proto_array = Float32Array::from(proto.into_raw_vec_and_offset().0);
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
}

impl YoloV26Detector {
    pub(crate) fn run_detection_pipeline<'py>(&mut self, py: Python<'py>, numpy_array: &Bound<'py, PyAny>) -> PyResult<(Vec<YoloDetection>, Option<ndarray::ArrayD<f32>>, usize, usize)> {
        let (width, height, prep_ms) = self.common.prepare_input(numpy_array)?;
        self.last_preprocess_ms = prep_ms;
        let results = self.run_inference_internal(py, (width, height))?;
        Ok((results.detections, results.proto, width, height))
    }

    pub(crate) fn run_inference_internal(&mut self, py: Python, _orig_dim: (usize, usize)) -> PyResult<YoloResultsV26> {
        let input_tensor = Value::from_array(self.common.input_tensor_buffer.clone())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let t_infer = Instant::now();
        let outputs = py.detach(|| self.common.session.run(ort::inputs![input_tensor]))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        self.last_inference_ms = t_infer.elapsed().as_secs_f64() * 1000.0;

        let t_decode = Instant::now();
        let results = match self.task {
            YoloTask::Classification => {
                let out_value = outputs.values().next().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No output found"))?;
                let out_extract = out_value.try_extract_tensor::<f32>().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                YoloResultsV26 {
                    detections: YoloCommon::decode_classification(out_extract.1),
                    proto: None,
                }
            },
            YoloTask::Pose => Self::decode_pose_v26(self.conf_threshold, self.common.input_width, self.common.input_height, self.common.num_keypoints, &outputs, _orig_dim)?,
            YoloTask::Segmentation => Self::decode_seg_v26(self.conf_threshold, self.common.input_width, self.common.input_height, self.common.num_mask_coeffs, &outputs, _orig_dim)?,
            _ => Self::decode_base_v26(self.conf_threshold, self.common.input_width, self.common.input_height, &outputs, _orig_dim)?,
        };
        self.last_nms_ms = t_decode.elapsed().as_secs_f64() * 1000.0;

        Ok(results)
    }
}
