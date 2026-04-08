//! Engine YOLOv26 (NMS-Free) Inference Implementation
//!
//! File này chứa logic thực thi riêng cho YOLOv26 và YOLOv10:
//! - Load model ONNX NMS-Free
//! - Preprocessing (Kornia)
//! - Postprocessing đơn giản (không cần NMS)
//! - Hỗ trợ Detection, Pose, Segmentation, Classification
//! - Export kết quả ra định dạng Arrow

use arrow::array::{Array, Float32Array, Int32Array, StructArray};
use arrow::datatypes::{DataType, Field, Fields};
use log::{debug, info};
use ndarray::Array4;
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
    task: YoloTask,
    num_classes: usize,
    num_keypoints: usize,
    num_mask_coeffs: usize,
    #[pyo3(get)]
    pub is_cls_model: bool,
    pub last_preprocess_ms: f64,
    #[pyo3(get)]
    pub last_inference_ms: f64,
    #[pyo3(get)]
    pub last_nms_ms: f64, // decode time
}

struct YoloResultsV26 {
    detections: Vec<YoloDetection>,
    proto: Option<ndarray::ArrayD<f32>>,
}

#[pymethods]
impl YoloV26Detector {
    #[new]
    #[pyo3(signature = (model_path, conf_threshold=0.25))]
    fn new(model_path: &str, conf_threshold: f32) -> PyResult<Self> {
        debug!("YoloV26Detector::new called with model: {}", model_path);

        let session = Session::builder()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Session error: {}", e)))?
            .commit_from_file(model_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Load error: {}", e)))?;

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
            task: config.task,
            num_classes: config.num_classes,
            num_keypoints: config.num_keypoints,
            num_mask_coeffs: config.num_mask_coeffs,
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
        let results = self.run_inference_internal(py, input_array, (width, height))?;

        let class_ids = Int32Array::from(results.detections.iter().map(|d| d.class_id).collect::<Vec<_>>());
        let confidences = Float32Array::from(results.detections.iter().map(|d| d.confidence).collect::<Vec<_>>());
        let boxes_x = Float32Array::from(results.detections.iter().map(|d| d.x).collect::<Vec<_>>());
        let boxes_y = Float32Array::from(results.detections.iter().map(|d| d.y).collect::<Vec<_>>());
        let boxes_w = Float32Array::from(results.detections.iter().map(|d| d.width).collect::<Vec<_>>());
        let boxes_h = Float32Array::from(results.detections.iter().map(|d| d.height).collect::<Vec<_>>());

        let mut fields = vec![
            Field::new("class_id", DataType::Int32, false),
            Field::new("confidence", DataType::Float32, false),
            Field::new("x", DataType::Float32, false),
            Field::new("y", DataType::Float32, false),
            Field::new("w", DataType::Float32, false),
            Field::new("h", DataType::Float32, false),
        ];
        let mut arrays: Vec<Arc<dyn Array>> = vec![
            Arc::new(class_ids), Arc::new(confidences),
            Arc::new(boxes_x), Arc::new(boxes_y), Arc::new(boxes_w), Arc::new(boxes_h),
        ];

        if self.num_keypoints > 0 {
            fields.push(Field::new(
                "keypoints",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                true,
            ));
            let mut kp_builder = arrow::array::ListBuilder::new(arrow::array::Float32Builder::new());
            for det in &results.detections {
                for (x, y, conf) in &det.keypoints {
                    kp_builder.values().append_value(*x);
                    kp_builder.values().append_value(*y);
                    kp_builder.values().append_value(*conf);
                }
                kp_builder.append(true);
            }
            arrays.push(Arc::new(kp_builder.finish()));
        }

        if self.num_mask_coeffs > 0 {
            fields.push(Field::new(
                "mask_coeffs",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                true,
            ));
            let mut mc_builder = arrow::array::ListBuilder::new(arrow::array::Float32Builder::new());
            for det in &results.detections {
                for c in &det.mask_coeffs { mc_builder.values().append_value(*c); }
                mc_builder.append(true);
            }
            arrays.push(Arc::new(mc_builder.finish()));
        }

        let struct_array = StructArray::try_new(Fields::from(fields), arrays, None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Arrow error: {}", e)))?;

        let (arr_cap, sch_cap) = crate::ffi::export_to_python(py, struct_array.to_data())?;
        
        let (p_arr, p_sch) = if let Some(p) = results.proto {
            let (raw_vec, _) = p.into_raw_vec_and_offset();
            let proto_array = Float32Array::from(raw_vec);
            let (pa, ps) = crate::ffi::export_to_python(py, proto_array.to_data())?;
            (pa.into_any().unbind(), ps.into_any().unbind())
        } else {
            (py.None(), py.None())
        };

        Ok((arr_cap, sch_cap, p_arr, p_sch))
    }

    fn detect_from_numpy(&mut self, py: Python, numpy_array: &Bound<PyAny>) -> PyResult<Py<PyList>> {
        let (width, height, input_array) = self.prepare_input(numpy_array)?;
        let results = self.run_inference_internal(py, input_array, (width, height))?;
        let py_list = PyList::empty(py);
        for det in results.detections { py_list.append(Py::new(py, det)?)?; }
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
    ) -> PyResult<YoloResultsV26> {
        let conf_threshold = self.conf_threshold;
        let scale_x = orig_dim.0 as f32 / self.input_width as f32;
        let scale_y = orig_dim.1 as f32 / self.input_height as f32;

        let input_tensor = Value::from_array(input_array).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let t_infer = Instant::now();
        let outputs = py.detach(|| self.session.run(ort::inputs![input_tensor])).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        self.last_inference_ms = t_infer.elapsed().as_secs_f64() * 1000.0;

        let t_decode = Instant::now();
        let mut detections = Vec::with_capacity(32);

        if self.task == YoloTask::Classification {
            // CLS: [1, num_classes]
            let out_value = outputs.values().next().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No output found"))?;
            let out_extract = out_value.try_extract_tensor::<f32>().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            detections = Self::decode_cls_v26(out_extract.1, self.num_classes);
        } else {
            // YOLOv26 NMS-Free: [1, 300, 6 + Extra]
            let out_value = outputs.values().next().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No output found"))?;
            let out_extract = out_value.try_extract_tensor::<f32>().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let shape = out_extract.0.iter().map(|&d| d as usize).collect::<Vec<_>>();
            let out_data = ndarray::ArrayViewD::from_shape(shape, out_extract.1).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let num_detections = out_data.shape()[1];
            for i in 0..num_detections {
                let score = out_data[[0, i, 4]];
                if score < conf_threshold { continue; }

                let x1 = out_data[[0, i, 0]];
                let y1 = out_data[[0, i, 1]];
                let x2 = out_data[[0, i, 2]];
                let y2 = out_data[[0, i, 3]];
                let class_id = out_data[[0, i, 5]] as i32;

                let mut keypoints = Vec::new();
                if self.task == YoloTask::Pose {
                    for k in 0..self.num_keypoints {
                        let kx = out_data[[0, i, 6 + k * 3]] * scale_x;
                        let ky = out_data[[0, i, 7 + k * 3]] * scale_y;
                        let kconf = out_data[[0, i, 8 + k * 3]];
                        keypoints.push((kx, ky, kconf));
                    }
                }

                let mut mask_coeffs = Vec::new();
                if self.task == YoloTask::Segmentation {
                    for m in 0..self.num_mask_coeffs {
                        mask_coeffs.push(out_data[[0, i, 6 + m]]);
                    }
                }

                detections.push(YoloDetection {
                    class_id, confidence: score,
                    x: x1 * scale_x, y: y1 * scale_y,
                    width: (x2 - x1) * scale_x, height: (y2 - y1) * scale_y,
                    keypoints, mask_coeffs,
                });
            }
        }

        let proto = if self.task == YoloTask::Segmentation {
            outputs.values().nth(1).and_then(|v| {
                v.try_extract_tensor::<f32>().ok().map(|ev| {
                    let p_shape = ev.0.iter().map(|&d| d as usize).collect::<Vec<_>>();
                    ndarray::ArrayViewD::from_shape(p_shape, ev.1).unwrap().to_owned()
                })
            })
        } else { None };

        self.last_nms_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        Ok(YoloResultsV26 { detections, proto })
    }

    fn decode_cls_v26(out_data: &[f32], _num_classes: usize) -> Vec<YoloDetection> {
        let max_val = out_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = out_data.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|x| x / sum).collect();
        let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.into_iter().take(5).map(|(idx, prob)| YoloDetection {
            class_id: idx as i32, confidence: prob,
            x: 0.0, y: 0.0, width: 0.0, height: 0.0,
            keypoints: vec![], mask_coeffs: vec![],
        }).collect()
    }
}
