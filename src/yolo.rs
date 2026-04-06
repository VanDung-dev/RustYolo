//! Rust extension module cho macOS system monitoring và YOLOv8x inference
//!

use ndarray::Array4;
use ort::execution_providers::{CoreML, ExecutionProvider};
use ort::session::Session;
use ort::value::Value;
use pyo3::prelude::*;
use pyo3::types::PyList;
use arrow::pyarrow::PyArrowType;
use arrow::array::ArrayData;
use arrow::array::Float32Array;

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
}

#[pymethods]
impl YoloV8Detector {
    #[new]
    #[pyo3(signature = (model_path, conf_threshold=0.25, iou_threshold=0.45))]
    fn new(model_path: &str, conf_threshold: f32, iou_threshold: f32) -> PyResult<Self> {
        if !CoreML::default().is_available().unwrap_or(false) {
            println!("⚠️ CẢNH BÁO: CoreML không khả dụng trên thiết bị này. Đang lùi về CPU.");
        }

        let session = Session::builder()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to create session builder: {}",
                    e
                ))
            })?
            .with_execution_providers([CoreML::default().build()])
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

        let _input_info = &session.inputs()[0];
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
        })
    }

    fn detect_from_arrow(
        &mut self,
        py: Python,
        arrow_array: PyArrowType<ArrayData>,
    ) -> PyResult<Py<PyList>> {
        // Zero-copy access to the Arrow ArrayData
        let array_data = arrow_array.0;
        
        let float_array = Float32Array::from(array_data);
        let data = float_array.values();
        
        // Tạo ndarray view từ Arrow buffer (Zero-copy)
        let input_array = ndarray::ArrayView4::from_shape(
            (1, 3, self.input_height, self.input_width),
            data
        ).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid array shape: {}", e))
        })?;

        let input_owned = input_array.to_owned();
        self.run_inference(py, input_owned, (self.input_width, self.input_height))
    }

    fn detect_from_numpy(
        &mut self,
        py: Python,
        numpy_array: &Bound<pyo3::PyAny>,
    ) -> PyResult<Py<PyList>> {
        // ... giữ lại để tương thích nhưng khuyên dùng Arrow ...
        let shape_obj = numpy_array.getattr("shape")?;
        let shape: (usize, usize, usize) = shape_obj.extract()?;
        let (height, width, _channels) = shape;

        let data_ptr = numpy_array
            .getattr("ctypes")?
            .getattr("data")?
            .extract::<usize>()?;

        // Zero-copy mapping numpy -> ndarray
        let raw_data = unsafe { 
            std::slice::from_raw_parts(data_ptr as *const u8, width * height * 3) 
        };
        
        let input_array = Array4::from_shape_fn((1, 3, self.input_height, self.input_width), |(_, c, y, x)| {
            let offset = (y as usize * width + x as usize) * 3 + c;
            raw_data[offset] as f32 / 255.0
        });

        self.run_inference(py, input_array, (width, height))
    }

    fn detect_image_bytes(
        &mut self,
        py: Python,
        data: Vec<u8>,
        width: usize,
        height: usize,
    ) -> PyResult<Py<PyList>> {
        if data.len() != width * height * 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid data size: expected {} bytes, got {}",
                width * height * 3,
                data.len()
            )));
        }

        let img = image::DynamicImage::ImageRgb8(
            image::ImageBuffer::from_raw(width as u32, height as u32, data).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to create image buffer")
            })?,
        );

        self.detect_image(py, img)
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
    fn detect_image(&mut self, py: Python, img: image::DynamicImage) -> PyResult<Py<PyList>> {
        let resized = img.resize_exact(
            self.input_width as u32,
            self.input_height as u32,
            image::imageops::FilterType::Triangle,
        );

        let rgb = resized.to_rgb8();
        let (img_width, img_height) = (img.width(), img.height());

        // Optimize v2: Sử dụng raw samples để tránh overhead của get_pixel()
        let rgb_raw = rgb.as_raw();
        let input_array = Array4::from_shape_fn((1, 3, self.input_height, self.input_width), |(_, c, y, x)| {
            let offset = (y as usize * self.input_width + x as usize) * 3 + c;
            rgb_raw[offset] as f32 / 255.0
        });

        let input_tensor = Value::from_array(input_array).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create input tensor: {}",
                e
            ))
        })?;

        let outputs = self.session.run(ort::inputs![input_tensor]).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Inference failed: {}", e))
        })?;

        let (_out_shape, out_data) =
            outputs["output0"]
                .try_extract_tensor::<f32>()
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to extract output: {}",
                        e
                    ))
                })?;

        let out_data: Vec<f32> = out_data.to_vec();
        drop(outputs);

        let num_anchors = 8400; // Standard for 640x640 YOLOv8
        let scale_x = img_width as f32 / self.input_width as f32;
        let scale_y = img_height as f32 / self.input_height as f32;

        let mut detections = Vec::new();
        let mut boxes_by_class: Vec<Vec<(f32, f32, f32, f32, f32)>> =
            vec![Vec::new(); self.num_classes];

        for i in 0..num_anchors {
            let mut max_conf = 0.0f32;
            let mut max_class = 0usize;

            // Tìm class có confidence cao nhất
            for c in 0..self.num_classes {
                // Layout is [84, 8400], so each class score is at (4 + c) * 8400 + i
                let raw_score = out_data[(4 + c) * num_anchors + i];
                
                // YOLOv8 output scores are usually already sigmoid-activated if using standard export,
                // but if they are raw logits, we'd need: 1.0 / (1.0 + (-raw_score).exp())
                // In most Ultralytics exports, they are ALREADY scores (0.0 to 1.0).
                let conf = raw_score; 
                
                if conf > max_conf {
                    max_conf = conf;
                    max_class = c;
                }
            }

            if max_conf > self.conf_threshold {
                let cx = out_data[i];
                let cy = out_data[num_anchors + i];
                let w = out_data[2 * num_anchors + i];
                let h = out_data[3 * num_anchors + i];

                let x = (cx - w / 2.0) * scale_x;
                let y = (cy - h / 2.0) * scale_y;
                let width = w * scale_x;
                let height = h * scale_y;

                boxes_by_class[max_class].push((x, y, width, height, max_conf));
            }
        }

        for (class_id, boxes) in boxes_by_class.iter().enumerate() {
            if boxes.is_empty() { continue; }
            
            let mut nms_boxes: Vec<(f32, f32, f32, f32, f32)> = boxes.clone();
            nms_boxes.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());

            let mut keep = vec![true; nms_boxes.len()];

            for i in 0..nms_boxes.len() {
                if !keep[i] {
                    continue;
                }

                for j in (i + 1)..nms_boxes.len() {
                    if !keep[j] {
                        continue;
                    }

                    let iou = self.compute_iou(&nms_boxes[i], &nms_boxes[j]);

                    if iou > self.iou_threshold {
                        keep[j] = false;
                    }
                }
            }

            for (idx, &(x, y, w, h, conf)) in nms_boxes.iter().enumerate() {
                if keep[idx] {
                    detections.push(YoloDetection {
                        class_id: class_id as i32,
                        confidence: conf,
                        x,
                        y,
                        width: w,
                        height: h,
                    });
                }
            }
        }

        let py_list = PyList::empty(py);
        for det in detections {
            let py_det = Py::new(py, det)?;
            py_list.append(py_det)?;
        }

        Ok(py_list.into())
    }

    fn run_inference(
        &mut self,
        py: Python,
        input_array: Array4<f32>,
        orig_dim: (usize, usize)
    ) -> PyResult<Py<PyList>> {
        let input_tensor = Value::from_array(input_array).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create input tensor: {}",
                e
            ))
        })?;

        let outputs = self.session.run(ort::inputs![input_tensor]).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Inference failed: {}", e))
        })?;

        let (_out_shape, out_data) =
            outputs["output0"]
                .try_extract_tensor::<f32>()
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to extract output: {}",
                        e
                    ))
                })?;

        let out_data: Vec<f32> = out_data.to_vec();
        drop(outputs);

        let num_anchors = 8400; // Standard for 640x640 YOLOv8
        let scale_x = orig_dim.0 as f32 / self.input_width as f32;
        let scale_y = orig_dim.1 as f32 / self.input_height as f32;

        let mut detections = Vec::new();
        let mut boxes_by_class: Vec<Vec<(f32, f32, f32, f32, f32)>> =
            vec![Vec::new(); self.num_classes];

        for i in 0..num_anchors {
            let mut max_conf = 0.0f32;
            let mut max_class = 0usize;

            for c in 0..self.num_classes {
                let raw_score = out_data[(4 + c) * num_anchors + i];
                let conf = raw_score; 
                if conf > max_conf {
                    max_conf = conf;
                    max_class = c;
                }
            }

            if max_conf > self.conf_threshold {
                let cx = out_data[i];
                let cy = out_data[num_anchors + i];
                let w = out_data[2 * num_anchors + i];
                let h = out_data[3 * num_anchors + i];

                let x = (cx - w / 2.0) * scale_x;
                let y = (cy - h / 2.0) * scale_y;
                let width = w * scale_x;
                let height = h * scale_y;

                boxes_by_class[max_class].push((x, y, width, height, max_conf));
            }
        }

        for (class_id, boxes) in boxes_by_class.iter().enumerate() {
            if boxes.is_empty() { continue; }
            let mut nms_boxes = boxes.clone();
            nms_boxes.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());
            let mut keep = vec![true; nms_boxes.len()];

            for i in 0..nms_boxes.len() {
                if !keep[i] { continue; }
                for j in (i + 1)..nms_boxes.len() {
                    if !keep[j] { continue; }
                    let iou = self.compute_iou(&nms_boxes[i], &nms_boxes[j]);
                    if iou > self.iou_threshold { keep[j] = false; }
                }
            }

            for (idx, &(x, y, w, h, conf)) in nms_boxes.iter().enumerate() {
                if keep[idx] {
                    detections.push(YoloDetection {
                        class_id: class_id as i32,
                        confidence: conf,
                        x, y, width: w, height: h,
                    });
                }
            }
        }

        let py_list = PyList::empty(py);
        for det in detections {
            let py_det = Py::new(py, det)?;
            py_list.append(py_det)?;
        }

        Ok(py_list.into())
    }

    fn compute_iou(
        &self,
        box1: &(f32, f32, f32, f32, f32),
        box2: &(f32, f32, f32, f32, f32),
    ) -> f32 {
        let x1 = box1.0.max(box2.0);
        let y1 = box1.1.max(box2.1);
        let x2 = (box1.0 + box1.2).min(box2.0 + box2.2);
        let y2 = (box1.1 + box1.3).min(box2.1 + box2.3);

        let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        let area1 = box1.2 * box1.3;
        let area2 = box2.2 * box2.3;
        let union = area1 + area2 - intersection;

        if union == 0.0 {
            0.0
        } else {
            intersection / union
        }
    }
}
