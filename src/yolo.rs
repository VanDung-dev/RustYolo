//! ✅ Engine YOLOv8 Inference Native
//!
//! Toàn bộ logic AI chạy ở file này 100%:
//! - Load model ONNX
//! - Tăng tốc CoreML phần cứng Apple Silicon
//! - Preprocessing ảnh
//! - Inference ONNX Runtime
//! - Decode output tensor
//! - Non Maximum Suppression (NMS) song song Rayon
//! - Export kết quả Zero Copy qua Apache Arrow
//!
//! ✅ Không có bất kỳ logic nào ở phía Python

use ndarray::Array4;
use ort::execution_providers::{CoreML, ExecutionProvider};
use ort::session::Session;
use ort::value::Value;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyCapsule};
use rayon::prelude::*;
use arrow::array::{Array, Float32Array, Int32Array, StructArray};
use arrow::datatypes::{DataType, Field, Fields};
use std::sync::Arc;
use std::time::Instant;

#[pyclass]
pub struct YoloDetection {
    pub class_id: i32,
    pub confidence: f32,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
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

    fn __repr__(&self) -> String {
        format!(
            "YoloDetection(class_id={}, confidence={:.3}, x={:.1}, y={:.1}, w={:.1}, h={:.1})",
            self.class_id, self.confidence, self.x, self.y, self.width, self.height
        )
    }
}

#[pyclass]
pub struct YoloV8Detector {
    session: Session,
    input_width: usize,
    input_height: usize,
    conf_threshold: f32,
    iou_threshold: f32,
    num_classes: usize,
    pub last_preprocess_ms: f64,
    pub last_inference_ms: f64,
    pub last_nms_ms: f64,
}

#[pymethods]
impl YoloV8Detector {
    #[new]
    #[pyo3(signature = (model_path, conf_threshold=0.25, iou_threshold=0.45))]
    fn new(model_path: &str, conf_threshold: f32, iou_threshold: f32) -> PyResult<Self> {
        println!("DEB: YoloV8Detector::new called with model: {}", model_path);
        if !CoreML::default().is_available().unwrap_or(false) {
            println!("⚠️ CẢNH BÁO: CoreML không khả dụng trên thiết bị này. Đang lùi về CPU.");
        } else {
            println!("🚀 CoreML khả dụng! Đang kích hoạt tăng tốc phần cứng...");
        }

        let session = Session::builder()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to create session builder: {}",
                    e
                ))
            })?
            .with_execution_providers([
                CoreML::default()
                    .with_subgraphs(true)
                    .with_compute_units(ort::execution_providers::coreml::ComputeUnits::All)
                    .build()
            ])
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to enable CoreML: {}",
                    e
                ))
            })?
            .commit_from_file(model_path)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to load model from {}: {}",
                    model_path, e
                ))
            })?;

        let input_width = 640;
        let input_height = 640;
        let num_classes = 80;

        Ok(YoloV8Detector {
            session,
            input_width,
            input_height,
            conf_threshold,
            iou_threshold,
            num_classes,
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

    /// ✅ Chạy inference YOLO và trả về kết quả Zero Copy qua Arrow Capsule
    fn detect_to_arrow<'py>(
        &mut self,
        py: Python<'py>,
        numpy_array: &Bound<'py, pyo3::PyAny>,
    ) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
        let shape_obj = numpy_array.getattr("shape")?;
        let shape: (usize, usize, usize) = shape_obj.extract()?;
        let (height, width, _channels) = shape;

        let data_ptr = numpy_array
            .getattr("ctypes")?
            .getattr("data")?
            .extract::<usize>()?;
        
        let raw_data = unsafe { 
            std::slice::from_raw_parts(data_ptr as *const u8, width * height * 3) 
        };

        // ✅ Đo thời gian tiền xử lý ảnh
        let t_pre = Instant::now();
        let input_array = crate::image_proc::preprocess_image_kornia(
            raw_data,
            width,
            height,
            self.input_width,
            self.input_height,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Preprocessing failed: {}", e)))?;
        self.last_preprocess_ms = t_pre.elapsed().as_secs_f64() * 1000.0;

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

        let struct_array = StructArray::try_new(
            Fields::from(fields),
            vec![
                Arc::new(class_ids),
                Arc::new(confidences),
                Arc::new(boxes_x),
                Arc::new(boxes_y),
                Arc::new(boxes_w),
                Arc::new(boxes_h),
            ],
            None,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Arrow error: {}", e)))?;

        crate::ffi::export_to_python(py, struct_array.to_data())
    }

    fn detect_from_numpy(
        &mut self,
        py: Python,
        numpy_array: &Bound<pyo3::PyAny>,
    ) -> PyResult<Py<PyList>> {
        let shape_obj = numpy_array.getattr("shape")?;
        let shape: (usize, usize, usize) = shape_obj.extract()?;
        let (height, width, _channels) = shape;

        let data_ptr = numpy_array
            .getattr("ctypes")?
            .getattr("data")?
            .extract::<usize>()?;
        
        let raw_data = unsafe { 
            std::slice::from_raw_parts(data_ptr as *const u8, width * height * 3) 
        };
        
        let input_array = crate::image_proc::preprocess_image_kornia(
            raw_data,
            width,
            height,
            self.input_width,
            self.input_height,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Preprocessing failed: {}", e)))?;

        let detections = self.run_inference_internal(py, input_array, (width, height))?;

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

/// Internal methods for YoloV8Detector
impl YoloV8Detector {
    fn run_inference_internal(
        &mut self,
        py: Python,
        input_array: Array4<f32>,
        orig_dim: (usize, usize)
    ) -> PyResult<Vec<YoloDetection>> {
        let input_tensor = Value::from_array(input_array).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create input tensor: {}",
                e
            ))
        })?;

        // ✅ Đo thời gian inference ONNX Runtime
        let t_infer = Instant::now();
        let outputs = py.detach(|| {
            self.session.run(ort::inputs![input_tensor])
        }).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Inference failed: {}", e))
        })?;
        self.last_inference_ms = t_infer.elapsed().as_secs_f64() * 1000.0;

        let out_value = &outputs["output0"];
        let out_shape = out_value.try_extract_tensor::<f32>().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Extract error: {}", e)))?.0;
        let out_data = out_value.try_extract_tensor::<f32>().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Extract error: {}", e)))?.1;
        
        let num_classes = (out_shape[1] - 4) as usize;
        let num_anchors = out_shape[2] as usize;
        self.num_classes = num_classes;
        
        let scale_x = orig_dim.0 as f32 / self.input_width as f32;
        let scale_y = orig_dim.1 as f32 / self.input_height as f32;
        let conf_threshold = self.conf_threshold;

        // ✅ Đo thời gian decode output và Non Maximum Suppression
        let t_nms = Instant::now();
        
        // ✅ Optimized decode: single pass filter only valid boxes first
        let mut all_boxes = Vec::with_capacity(128);
        
        // Single iteration over all anchors
        for i in 0..num_anchors {
            // Find max confidence class for this anchor
            let mut max_conf = 0.0f32;
            let mut max_class = 0usize;
            
            for c in 0..num_classes {
                let conf = out_data[(4 + c) * num_anchors + i];
                if conf > max_conf {
                    max_conf = conf;
                    max_class = c;
                }
            }
            
            if max_conf > conf_threshold {
                let cx = out_data[i];
                let cy = out_data[num_anchors + i];
                let w = out_data[2 * num_anchors + i];
                let h = out_data[3 * num_anchors + i];

                let x = (cx - w / 2.0) * scale_x;
                let y = (cy - h / 2.0) * scale_y;
                let width = w * scale_x;
                let height = h * scale_y;
                
                all_boxes.push((x, y, width, height, max_conf, max_class));
            }
        }
        
        let iou_threshold = self.iou_threshold;
        
        // Sort all boxes descending by confidence
        all_boxes.sort_unstable_by(|a, b| b.4.partial_cmp(&a.4).unwrap());
        
        let mut keep = vec![true; all_boxes.len()];
        let mut detections = Vec::with_capacity(all_boxes.len());

        // Fast single pass NMS
        for i in 0..all_boxes.len() {
            if !keep[i] { continue; }
            
            let (x, y, w, h, conf, class_id) = all_boxes[i];
            detections.push(YoloDetection {
                class_id: class_id as i32,
                confidence: conf,
                x, y, width: w, height: h,
            });

            // Suppress overlapping boxes
            for j in (i + 1)..all_boxes.len() {
                if !keep[j] || all_boxes[j].5 != class_id { continue; }
                if Self::compute_iou_internal(&all_boxes[i], &all_boxes[j]) > iou_threshold {
                    keep[j] = false;
                }
            }
        }
        
        self.last_nms_ms = t_nms.elapsed().as_secs_f64() * 1000.0;

        Ok(detections)
    }

    fn compute_iou_internal(
        box1: &(f32, f32, f32, f32, f32, usize),
        box2: &(f32, f32, f32, f32, f32, usize),
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
