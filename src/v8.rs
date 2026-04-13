//! Engine YOLOv8 Inference Implementation
//!
//! File này chứa logic thực thi riêng cho YOLOv8, bao gồm:
//! - Load model ONNX cho YOLOv8/v11
//! - Preprocessing (Kornia)
//! - Postprocessing & NMS cho YOLOv8
//! - Export kết quả ra định dạng Arrow

use arrow::array::{Array, Float32Array, Int32Array, StructArray};
use arrow::datatypes::{DataType, Field, Fields};
use log::{debug, info, warn};
use ndarray::{Array4, Axis, s};
use ort::ep::{CoreML, ExecutionProvider};
use ort::session::Session;
use ort::value::Value;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList};
use std::sync::Arc;
use std::time::Instant;

use crate::yolo::{YoloDetection, ModelConfig, YoloTask, ExecutionProviderType};
use crate::image_proc::draw_rect_native;

#[pyclass]
pub struct YoloV8Detector {
    session: Session,
    input_width: usize,
    input_height: usize,
    conf_threshold: f32,
    iou_threshold: f32,
    num_classes: usize,
    num_keypoints: usize,
    num_mask_coeffs: usize,
    #[pyo3(get)]
    pub is_cls_model: bool,
    #[pyo3(get)]
    pub is_obb_model: bool,
    pub last_preprocess_ms: f64,
    #[pyo3(get)]
    pub last_inference_ms: f64,
    #[pyo3(get)]
    pub last_nms_ms: f64,
    // Buffer tái sử dụng để tránh cấp phát bộ nhớ liên tục (640*640*3*4 bytes)
    input_tensor_buffer: Array4<f32>,
    #[pyo3(get)]
    pub ep: ExecutionProviderType,
}

#[pymethods]
impl YoloV8Detector {
    #[new]
    #[pyo3(signature = (model_path, conf_threshold=0.25, iou_threshold=0.45, execution_provider="coreml"))]
    fn new(model_path: &str, conf_threshold: f32, iou_threshold: f32, execution_provider: &str) -> PyResult<Self> {
        debug!("YoloV8Detector::new called with model: {}, execution_provider: {}", model_path, execution_provider);
        
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

        let class_ids = Int32Array::from(detections.iter().map(|d| d.class_id).collect::<Vec<_>>());
        let confidences =
            Float32Array::from(detections.iter().map(|d| d.confidence).collect::<Vec<_>>());
        let boxes_x = Float32Array::from(detections.iter().map(|d| d.x).collect::<Vec<_>>());
        let boxes_y = Float32Array::from(detections.iter().map(|d| d.y).collect::<Vec<_>>());
        let boxes_w = Float32Array::from(detections.iter().map(|d| d.width).collect::<Vec<_>>());
        let boxes_h = Float32Array::from(detections.iter().map(|d| d.height).collect::<Vec<_>>());

        let mut fields = Self::build_default_arrow_fields();
        let mut arrays = Self::build_default_arrow_arrays(class_ids, confidences, boxes_x, boxes_y, boxes_w, boxes_h);

        if self.num_keypoints > 0 {
            fields.push(Field::new(
                "keypoints",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                true,
            ));
            arrays.push(Self::build_list_array(&detections, |det| {
                det.keypoints.iter().flat_map(|&(x, y, conf)| [x, y, conf]).collect::<Vec<_>>().leak()
            }));
        }

        if self.num_mask_coeffs > 0 {
            fields.push(Field::new(
                "mask_coeffs",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                true,
            ));
            arrays.push(Self::build_list_array(&detections, |det| &det.mask_coeffs));
        }

        let struct_array =
            StructArray::try_new(Fields::from(fields), arrays, None).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Arrow error: {}", e))
            })?;

        let (arr_cap, sch_cap) = crate::ffi::export_to_python(py, struct_array.to_data())?;

        let (proto_arr, proto_sch) = if let Some(proto) = proto_flat {
            let proto_array = Float32Array::from(proto);
            let proto_data = proto_array.to_data();
            let (pa, ps) = crate::ffi::export_to_python(py, proto_data)?;
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
        
        // Tạo mutable slice từ con trỏ (BGR/RGB)
        let data = unsafe { std::slice::from_raw_parts_mut(data_ptr as *mut u8, width * height * 3) };

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

        // 4. Đóng gói kết quả Arrow (Giống detect_to_arrow)
        let class_ids = Int32Array::from(detections.iter().map(|d| d.class_id).collect::<Vec<_>>());
        let confidences = Float32Array::from(detections.iter().map(|d| d.confidence).collect::<Vec<_>>());
        let boxes_x = Float32Array::from(detections.iter().map(|d| d.x).collect::<Vec<_>>());
        let boxes_y = Float32Array::from(detections.iter().map(|d| d.y).collect::<Vec<_>>());
        let boxes_w = Float32Array::from(detections.iter().map(|d| d.width).collect::<Vec<_>>());
        let boxes_h = Float32Array::from(detections.iter().map(|d| d.height).collect::<Vec<_>>());

        let mut fields = Self::build_default_arrow_fields();
        let mut arrays = Self::build_default_arrow_arrays(class_ids, confidences, boxes_x, boxes_y, boxes_w, boxes_h);

        if self.num_keypoints > 0 {
            fields.push(Field::new("keypoints", DataType::List(Arc::new(Field::new("item", DataType::Float32, true))), true));
            arrays.push(Self::build_list_array(&detections, |det| {
                det.keypoints.iter().flat_map(|&(x, y, conf)| [x, y, conf]).collect::<Vec<_>>().leak()
            }));
        }

        if self.num_mask_coeffs > 0 {
            fields.push(Field::new("mask_coeffs", DataType::List(Arc::new(Field::new("item", DataType::Float32, true))), true));
            arrays.push(Self::build_list_array(&detections, |det| &det.mask_coeffs));
        }

        let struct_array = StructArray::try_new(Fields::from(fields), arrays, None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Arrow error: {}", e)))?;

        let (arr_cap, sch_cap) = crate::ffi::export_to_python(py, struct_array.to_data())?;
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
    // Helper functions to eliminate code duplication
    #[inline]
    fn build_default_arrow_fields() -> Vec<Field> {
        vec![
            Field::new("class_id", DataType::Int32, false),
            Field::new("confidence", DataType::Float32, false),
            Field::new("x", DataType::Float32, false),
            Field::new("y", DataType::Float32, false),
            Field::new("w", DataType::Float32, false),
            Field::new("h", DataType::Float32, false),
        ]
    }

    #[inline]
    fn build_default_arrow_arrays(class_ids: Int32Array, confidences: Float32Array, boxes_x: Float32Array, boxes_y: Float32Array, boxes_w: Float32Array, boxes_h: Float32Array) -> Vec<Arc<dyn Array>> {
        vec![
            Arc::new(class_ids),
            Arc::new(confidences),
            Arc::new(boxes_x),
            Arc::new(boxes_y),
            Arc::new(boxes_w),
            Arc::new(boxes_h),
        ]
    }

    #[inline]
    fn build_list_array<F, T>(detections: &[YoloDetection], mut extractor: F) -> Arc<dyn Array>
    where
        F: FnMut(&YoloDetection) -> &[T],
        T: Into<f32> + Copy,
    {
        let mut builder = arrow::array::ListBuilder::new(arrow::array::Float32Builder::new());
        for det in detections {
            for value in extractor(det) {
                builder.values().append_value((*value).into());
            }
            builder.append(true);
        }
        Arc::new(builder.finish())
    }

    #[inline]
    fn create_empty_detection(class_id: i32, confidence: f32) -> YoloDetection {
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
    fn run_detection_pipeline<'py>(&mut self, py: Python<'py>, numpy_array: &Bound<'py, PyAny>) -> PyResult<(Vec<YoloDetection>, Option<Vec<f32>>, usize, usize)> {
        let (width, height) = self.prepare_input(numpy_array)?;
        let (detections, proto_flat) = self.run_inference_internal(py, (width, height))?;
        Ok((detections, proto_flat, width, height))
    }

    #[inline]
    fn prepare_input(&mut self, numpy_array: &Bound<'_, PyAny>) -> PyResult<(usize, usize)> {
        let shape_obj = numpy_array.getattr("shape")?;
        let shape: (usize, usize, usize) = shape_obj.extract()?;
        let (height, width, _channels) = shape;

        let data_ptr = numpy_array
            .getattr("ctypes")?
            .getattr("data")?
            .extract::<usize>()?;

        let raw_data =
            unsafe { std::slice::from_raw_parts(data_ptr as *const u8, width * height * 3) };

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

    fn run_inference_internal(
        &mut self,
        py: Python,
        orig_dim: (usize, usize),
    ) -> PyResult<(Vec<YoloDetection>, Option<Vec<f32>>)> {
        let is_cls = self.is_cls_model;
        let num_classes = self.num_classes;
        let num_keypoints = self.num_keypoints;
        let num_mask_coeffs = self.num_mask_coeffs;
        let is_obb = self.is_obb_model;
        let iou_threshold = self.iou_threshold;
        let conf_threshold = self.conf_threshold;
        let input_width_f = self.input_width as f32;
        let input_height_f = self.input_height as f32;

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

        // Logging hiệu suất cực kỳ chi tiết cho M4 Pro
        if self.last_inference_ms > 0.1 {
            info!(
                "🚀 Perf Metrics [v8]: Pre={:.2}ms, Infer={:.2}ms, Total={:.2}ms | FPS: {:.1}",
                self.last_preprocess_ms,
                self.last_inference_ms,
                self.last_preprocess_ms + self.last_inference_ms,
                1000.0 / (self.last_preprocess_ms + self.last_inference_ms).max(0.1)
            );
        }

        let out_value = &outputs["output0"];
        let out_extract = out_value.try_extract_tensor::<f32>().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Extract error: {}", e))
        })?;

        let shape_usize: Vec<usize> = out_extract.0.iter().map(|&d| d as usize).collect();
        let out_data = ndarray::ArrayViewD::from_shape(shape_usize, out_extract.1).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("ArrayView error: {}", e))
        })?;
        let out_shape = out_data.shape();

        let proto_flat: Option<Vec<f32>> = if num_mask_coeffs > 0 {
            match outputs.get("output1") {
                Some(v) => {
                    let t = v.try_extract_tensor::<f32>().map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Proto error: {}", e))
                    })?;
                    Some(t.1.to_vec())
                }
                None => None,
            }
        } else {
            None
        };

        if is_cls {
            let cls_results = Self::decode_cls_output(out_extract.1, num_classes);
            return Ok((cls_results, proto_flat));
        }

        let num_anchors = out_shape[2];
        let scale_x = orig_dim.0 as f32 / input_width_f;
        let scale_y = orig_dim.1 as f32 / input_height_f;

        let t_nms = Instant::now();

        // Tối ưu layout: (1, 84, 8400) -> (84, 8400) -> (8400, 84)
        let out_data_2d = out_data.index_axis(Axis(0), 0).reversed_axes();
        let out_data_2d = out_data_2d.into_dimensionality::<ndarray::Ix2>().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Dimensionality error: {}", e))
        })?;

        let mut all_boxes: Vec<_> = Vec::with_capacity(128);
        for i in 0..num_anchors {
            let row = out_data_2d.row(i);
            let scores = row.slice(s![4..4 + num_classes]);

            // Vectorized confidence extraction (Fast Path)
            let mut max_conf = 0.0f32;
            let mut max_class = 0_usize;
            
            for (c, &conf) in scores.iter().enumerate() {
                if conf > max_conf {
                    max_conf = conf;
                    max_class = c;
                }
            }

            if max_conf <= conf_threshold {
                continue;
            }

            let cx = row[0];
            let cy = row[1];
            let w = row[2];
            let h = row[3];

            let mut keypoints = Vec::with_capacity(num_keypoints);
            let (final_x, final_y, final_w, final_h): (f32, f32, f32, f32) = if is_obb {
                let angle = row[4 + num_classes];
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let dx = w / 2.0 * cos_a;
                let dy = w / 2.0 * sin_a;
                let ex = -h / 2.0 * sin_a;
                let ey = h / 2.0 * cos_a;

                let pts = [
                    (cx - dx - ex, cy - dy - ey),
                    (cx + dx - ex, cy + dy - ey),
                    (cx + dx + ex, cy + dy + ey),
                    (cx - dx + ex, cy - dy + ey),
                ];

                for pt in &pts {
                    keypoints.push((pt.0 * scale_x, pt.1 * scale_y, 1.0));
                }

                let min_x = pts.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
                let min_y = pts.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
                let max_x = pts.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);
                let max_y = pts.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);

                (min_x * scale_x, min_y * scale_y, (max_x - min_x) * scale_x, (max_y - min_y) * scale_y)
            } else {
                ((cx - w / 2.0) * scale_x, (cy - h / 2.0) * scale_y, w * scale_x, h * scale_y)
            };

            let x = final_x.clamp(0.0, orig_dim.0 as f32);
            let y = final_y.clamp(0.0, orig_dim.1 as f32);
            let bbw = final_w.clamp(0.0, orig_dim.0 as f32);
            let bbh = final_h.clamp(0.0, orig_dim.1 as f32);

            if num_keypoints > 0 {
                let base_offset = 4 + num_classes;
                for kp_idx in 0..num_keypoints {
                    let kx = row[base_offset + kp_idx * 3] * scale_x;
                    let ky = row[base_offset + kp_idx * 3 + 1] * scale_y;
                    let kconf = row[base_offset + kp_idx * 3 + 2];
                    keypoints.push((kx, ky, kconf));
                }
            }

            let mut mask_coeffs = Vec::with_capacity(num_mask_coeffs);
            if num_mask_coeffs > 0 {
                let mc_base = 4 + num_classes;
                for m in 0..num_mask_coeffs {
                    mask_coeffs.push(row[mc_base + m]);
                }
            }

            all_boxes.push((x, y, bbw, bbh, max_conf, max_class, keypoints, mask_coeffs));
        }

        all_boxes.sort_unstable_by(|a, b| b.4.partial_cmp(&a.4).unwrap());
        let mut keep = vec![true; all_boxes.len()];
        let mut detections = Vec::with_capacity(all_boxes.len());

        for i in 0..all_boxes.len() {
            if !keep[i] { continue; }
            let (x, y, w, h, conf, class_id, ref kps, ref mcs) = all_boxes[i];
            detections.push(YoloDetection {
                class_id: class_id as i32,
                confidence: conf,
                x, y, width: w, height: h,
                keypoints: kps.clone(),
                mask_coeffs: mcs.clone(),
            });

            for j in (i + 1)..all_boxes.len() {
                if !keep[j] || all_boxes[j].5 != class_id { continue; }
                if Self::compute_iou_internal(&all_boxes[i], &all_boxes[j]) > iou_threshold {
                    keep[j] = false;
                }
            }
        }

        drop(out_data);
        drop(outputs);
        self.last_nms_ms = t_nms.elapsed().as_secs_f64() * 1000.0;

        Ok((detections, proto_flat))
    }

    fn compute_iou_internal(
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

    fn decode_cls_output(out_data: &[f32], _num_classes: usize) -> Vec<YoloDetection> {
        let max_val = out_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = out_data.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|x| x / sum).collect();

        let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        indexed
            .into_iter()
            .take(5)
            .map(|(idx, prob)| Self::create_empty_detection(idx as i32, prob))
            .collect()
    }
}
