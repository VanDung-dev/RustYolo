//! Engine YOLOv26 (NMS-Free) Inference Implementation
//!
//! File này chứa logic thực thi riêng cho YOLOv26 và YOLOv10:
//! - Load model ONNX NMS-Free
//! - Preprocessing (Kornia)
//! - Postprocessing đơn giản (không cần NMS)
//! - Export kết quả ra định dạng Arrow

use arrow::array::{Array, Float32Array, Int32Array, StructArray};
use arrow::datatypes::{DataType, Field, Fields};
use log::{debug, info, warn};
use ndarray::Array4;
use ort::execution_providers::{CoreML, ExecutionProvider};
use ort::session::Session;
use ort::value::Value;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList};
use std::sync::Arc;
use std::time::Instant;

use crate::yolo::{YoloDetection, ModelConfig, YoloTask};

    #[pyclass]
pub struct YoloV26Detector {
    session: Session,
    input_width: usize,
    input_height: usize,
    conf_threshold: f32,
    #[pyo3(get)]
    pub is_cls_model: bool,
    pub last_preprocess_ms: f64,
    #[pyo3(get)]
    pub last_inference_ms: f64,
    #[pyo3(get)]
    pub last_nms_ms: f64, //decode time
}

#[pymethods]
impl YoloV26Detector {
    #[new]
    #[pyo3(signature = (model_path, conf_threshold=0.25))]
    fn new(model_path: &str, conf_threshold: f32) -> PyResult<Self> {
        debug!("YoloV26Detector::new called with model: {}", model_path);

        // Ưu tiên CPU cho YOLOv26 vì CoreML thường lỗi với GatherElements của NMS-Free
        let session = Session::builder()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Session builder error: {}", e)))?
            .commit_from_file(model_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Load model error: {}", e)))?;

        let config = ModelConfig::identify(model_path, &session);

        info!(
            "V26 Model Config: task={:?}, input={}x{}",
            config.task, config.input_size.0, config.input_size.1
        );

        Ok(Self {
            session,
            input_width: config.input_size.0,
            input_height: config.input_size.1,
            conf_threshold,
            is_cls_model: config.task == YoloTask::Classification,
            last_preprocess_ms: 0.0,
            last_inference_ms: 0.0,
            last_nms_ms: 0.0,
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
        let (width, height, input_array) = self.prepare_input(numpy_array)?;
        let detections = self.run_inference_internal(py, input_array, (width, height))?;

        let class_ids = Int32Array::from(detections.iter().map(|d| d.class_id).collect::<Vec<_>>());
        let confidences = Float32Array::from(detections.iter().map(|d| d.confidence).collect::<Vec<_>>());
        let boxes_x = Float32Array::from(detections.iter().map(|d| d.x).collect::<Vec<_>>());
        let boxes_y = Float32Array::from(detections.iter().map(|d| d.y).collect::<Vec<_>>());
        let boxes_w = Float32Array::from(detections.iter().map(|d| d.width).collect::<Vec<_>>());
        let boxes_h = Float32Array::from(detections.iter().map(|d| d.height).collect::<Vec<_>>());

        let fields = vec![
            Field::new("class_id", DataType::Int32, false),
            Field::new("confidence", DataType::Float32, false),
            Field::new("x", DataType::Float32, false),
            Field::new("y", DataType::Float32, false),
            Field::new("w", DataType::Float32, false),
            Field::new("h", DataType::Float32, false),
        ];
        let arrays: Vec<Arc<dyn Array>> = vec![
            Arc::new(class_ids), Arc::new(confidences),
            Arc::new(boxes_x), Arc::new(boxes_y), Arc::new(boxes_w), Arc::new(boxes_h),
        ];

        let struct_array = StructArray::try_new(Fields::from(fields), arrays, None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Arrow error: {}", e)))?;

        let (arr_cap, sch_cap) = crate::ffi::export_to_python(py, struct_array.to_data())?;
        Ok((arr_cap, sch_cap, py.None(), py.None()))
    }

    fn detect_from_numpy(&mut self, py: Python, numpy_array: &Bound<PyAny>) -> PyResult<Py<PyList>> {
        let (width, height, input_array) = self.prepare_input(numpy_array)?;
        let detections = self.run_inference_internal(py, input_array, (width, height))?;

        let py_list = PyList::empty(py);
        for det in detections {
            py_list.append(Py::new(py, det)?)?;
        }
        Ok(py_list.into())
    }
}

impl YoloV26Detector {
    fn prepare_input(&mut self, numpy_array: &Bound<'_, PyAny>) -> PyResult<(usize, usize, Array4<f32>)> {
        let shape: (usize, usize, usize) = numpy_array.getattr("shape")?.extract()?;
        let (height, width, _) = shape;
        let data_ptr = numpy_array.getattr("ctypes")?.getattr("data")?.extract::<usize>()?;
        let raw_data = unsafe { std::slice::from_raw_parts(data_ptr as *const u8, width * height * 3) };

        let t_pre = Instant::now();
        let input_array = crate::image_proc::preprocess_image_kornia(raw_data, width, height, self.input_width, self.input_height)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        self.last_preprocess_ms = t_pre.elapsed().as_secs_f64() * 1000.0;
        Ok((width, height, input_array))
    }

    fn run_inference_internal(
        &mut self,
        py: Python,
        input_array: Array4<f32>,
        orig_dim: (usize, usize),
    ) -> PyResult<Vec<YoloDetection>> {
        let conf_threshold = self.conf_threshold;
        let scale_x = orig_dim.0 as f32 / self.input_width as f32;
        let scale_y = orig_dim.1 as f32 / self.input_height as f32;

        let input_tensor = Value::from_array(input_array)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Tensor error: {}", e)))?;

        let t_infer = Instant::now();
        let outputs = py.detach(|| self.session.run(ort::inputs![input_tensor]))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Inference failed: {}", e)))?;
        self.last_inference_ms = t_infer.elapsed().as_secs_f64() * 1000.0;

        let t_decode = Instant::now();
        let out_value = &outputs["output0"];
        let out_extract = out_value.try_extract_tensor::<f32>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Extract error: {}", e)))?;

        // YOLOv26 Output shape: [1, 300, 6] -> [x1, y1, x2, y2, score, class]
        let shape = out_extract.0.iter().map(|&d| d as usize).collect::<Vec<_>>();
        let out_data = ndarray::ArrayViewD::from_shape(shape, out_extract.1)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("ArrayView error: {}", e)))?;

        let num_detections = out_data.shape()[1];
        let mut detections = Vec::with_capacity(32);

        for i in 0..num_detections {
            let score = out_data[[0, i, 4]];
            if score < conf_threshold { continue; }

            let x1 = out_data[[0, i, 0]];
            let y1 = out_data[[0, i, 1]];
            let x2 = out_data[[0, i, 2]];
            let y2 = out_data[[0, i, 3]];
            let class_id = out_data[[0, i, 5]] as i32;

            detections.push(YoloDetection {
                class_id,
                confidence: score,
                x: x1 * scale_x,
                y: y1 * scale_y,
                width: (x2 - x1) * scale_x,
                height: (y2 - y1) * scale_y,
                keypoints: vec![],
                mask_coeffs: vec![],
            });
        }

        self.last_nms_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        Ok(detections)
    }
}
