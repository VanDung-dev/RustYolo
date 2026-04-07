//! ✅ Engine YOLOv8 Inference Native
//!
//! Toàn bộ logic AI chạy ở file này 100%:
//! - Load model ONNX
//! - Tăng tốc CoreML phần cứng Apple Silicon
//! - Preprocessing ảnh
//! - Inference ONNX Runtime
//! - Decode output tensor (Detection / Pose / Segmentation)
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
    pub keypoints: Vec<(f32, f32, f32)>,    // (x, y, conf) cho 17 điểm pose
    pub mask_coeffs: Vec<f32>,               // 32 coefficients cho segmentation
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
    pub last_preprocess_ms: f64,
    #[pyo3(get)]
    pub last_inference_ms: f64,
    #[pyo3(get)]
    pub last_nms_ms: f64,
}

#[pymethods]
impl YoloV8Detector {
    #[new]
    #[pyo3(signature = (model_path, conf_threshold=0.25, iou_threshold=0.45))]
    fn new(model_path: &str, conf_threshold: f32, iou_threshold: f32) -> PyResult<Self> {
        println!("DEB: YoloV8Detector::new called with model: {}", model_path);
        if !CoreML::default().is_available().unwrap_or(false) {
            println!("⚠️ CẢNH BÁO: CoreML không khả dụng. Đang lùi về CPU.");
        } else {
            println!("🚀 CoreML khả dụng! Đang kích hoạt tăng tốc phần cứng...");
        }

        let session = Session::builder()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create session builder: {}", e
            )))?
            .with_execution_providers([
                CoreML::default()
                    .with_subgraphs(true)
                    .with_compute_units(ort::execution_providers::coreml::ComputeUnits::All)
                    .build()
            ])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to enable CoreML: {}", e
            )))?
            .commit_from_file(model_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load model from {}: {}", model_path, e
            )))?;

        Ok(Self {
            session,
            input_width: 640,
            input_height: 640,
            conf_threshold,
            iou_threshold,
            num_classes: 80,
            num_keypoints: 0,
            num_mask_coeffs: 0,
            is_cls_model: false,
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

    /// ✅ Chạy inference và trả về kết quả Zero Copy qua Arrow Capsule
    ///
    /// Returns:
    ///   - Detection/Pose model: (array_capsule, schema_capsule, None, None)
    ///   - Seg model: (array_capsule, schema_capsule, proto_capsule, proto_schema_capsule)
    ///     trong đó proto là flat Float32Array shape [32*160*160]
    fn detect_to_arrow<'py>(
        &mut self,
        py: Python<'py>,
        numpy_array: &Bound<'py, pyo3::PyAny>,
    ) -> PyResult<(
        Bound<'py, PyCapsule>,
        Bound<'py, PyCapsule>,
        Py<pyo3::PyAny>,   // proto array capsule hoặc None
        Py<pyo3::PyAny>,   // proto schema capsule hoặc None
    )> {
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

        // ✅ Tiền xử lý ảnh
        let t_pre = Instant::now();
        let input_array = crate::image_proc::preprocess_image_kornia(
            raw_data, width, height, self.input_width, self.input_height,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Preprocessing failed: {}", e)
        ))?;
        self.last_preprocess_ms = t_pre.elapsed().as_secs_f64() * 1000.0;

        let (detections, proto_flat) =
            self.run_inference_internal(py, input_array, (width, height))?;

        // ── Xây dựng Arrow StructArray cho detections ──────────────────────
        let class_ids   = Int32Array::from(detections.iter().map(|d| d.class_id).collect::<Vec<_>>());
        let confidences = Float32Array::from(detections.iter().map(|d| d.confidence).collect::<Vec<_>>());
        let boxes_x     = Float32Array::from(detections.iter().map(|d| d.x).collect::<Vec<_>>());
        let boxes_y     = Float32Array::from(detections.iter().map(|d| d.y).collect::<Vec<_>>());
        let boxes_w     = Float32Array::from(detections.iter().map(|d| d.width).collect::<Vec<_>>());
        let boxes_h     = Float32Array::from(detections.iter().map(|d| d.height).collect::<Vec<_>>());

        let mut fields: Vec<Field> = vec![
            Field::new("class_id",   DataType::Int32,   false),
            Field::new("confidence", DataType::Float32, false),
            Field::new("x",          DataType::Float32, false),
            Field::new("y",          DataType::Float32, false),
            Field::new("w",          DataType::Float32, false),
            Field::new("h",          DataType::Float32, false),
        ];
        let mut arrays: Vec<Arc<dyn Array>> = vec![
            Arc::new(class_ids),
            Arc::new(confidences),
            Arc::new(boxes_x),
            Arc::new(boxes_y),
            Arc::new(boxes_w),
            Arc::new(boxes_h),
        ];

        // ── Keypoints (Pose model) ──────────────────────────────────────────
        if self.num_keypoints > 0 {
            fields.push(Field::new(
                "keypoints",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                true,
            ));
            let mut kp_builder =
                arrow::array::ListBuilder::new(arrow::array::Float32Builder::new());
            for det in &detections {
                for (x, y, conf) in &det.keypoints {
                    kp_builder.values().append_value(*x);
                    kp_builder.values().append_value(*y);
                    kp_builder.values().append_value(*conf);
                }
                kp_builder.append(true);
            }
            arrays.push(Arc::new(kp_builder.finish()));
        }

        // ── Mask coefficients (Seg model) ───────────────────────────────────
        if self.num_mask_coeffs > 0 {
            fields.push(Field::new(
                "mask_coeffs",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                true,
            ));
            let mut mc_builder =
                arrow::array::ListBuilder::new(arrow::array::Float32Builder::new());
            for det in &detections {
                for c in &det.mask_coeffs {
                    mc_builder.values().append_value(*c);
                }
                mc_builder.append(true);
            }
            arrays.push(Arc::new(mc_builder.finish()));
        }

        let struct_array = StructArray::try_new(
            Fields::from(fields),
            arrays,
            None,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Arrow error: {}", e)
        ))?;

        let (arr_cap, sch_cap) = crate::ffi::export_to_python(py, struct_array.to_data())?;

        // ── Proto tensor capsule (Seg model only) ────────────────────────────
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
            raw_data, width, height, self.input_width, self.input_height,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Preprocessing failed: {}", e)
        ))?;

        let (detections, _) = self.run_inference_internal(py, input_array, (width, height))?;

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

/// Internal methods
impl YoloV8Detector {
    /// Chạy inference và trả về (detections, Option<proto_flat>)
    fn run_inference_internal(
        &mut self,
        py: Python,
        input_array: Array4<f32>,
        orig_dim: (usize, usize),
    ) -> PyResult<(Vec<YoloDetection>, Option<Vec<f32>>)> {
        let input_tensor = Value::from_array(input_array).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create input tensor: {}", e
            ))
        })?;

        // ✅ Inference ONNX Runtime (GIL released)
        let t_infer = Instant::now();
        let outputs = py.detach(|| {
            self.session.run(ort::inputs![input_tensor])
        }).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Inference failed: {}", e))
        })?;
        self.last_inference_ms = t_infer.elapsed().as_secs_f64() * 1000.0;

        // ── Đọc output0 ─────────────────────────────────────────────────────
        let out_value = &outputs["output0"];
        let out_extract = out_value.try_extract_tensor::<f32>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Extract error: {}", e)))?;
        
        let out_shape = out_extract.0.to_vec();
        let out_data = out_extract.1.to_vec();

        // ── Đọc proto tensor (output1) sớm nếu có ─────────────────────
        let proto_flat: Option<Vec<f32>> = match outputs.get("output1") {
            Some(v) => {
                let t = v.try_extract_tensor::<f32>()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Proto error: {}", e)))?;
                Some(t.1.to_vec())
            }
            None => None,
        };

        // ✅ GIẢI PHÓNG MƯỢN TRÊN SELF SỚM NHẤT CÓ THỂ
        drop(outputs);

        let total_channels = out_shape[1] as usize;
        let ndim = out_shape.len();

        // ── Tự động phát hiện loại model ────────────────────────────────────
        // Classification: output0 shape là [1, 1000] (2D)
        if ndim == 2 {
            self.is_cls_model = true;
            self.num_classes = total_channels;
            self.num_keypoints = 0;
            self.num_mask_coeffs = 0;
            
            let cls_results = self.decode_cls_output(&out_data, total_channels);
            return Ok((cls_results, None));
        }


        self.is_cls_model = false;
        let num_anchors = out_shape[2] as usize;

        // Pose: total_channels == 56  (4 + 1 + 17*3)
        // Seg:  total_channels == 116 (4 + 80 + 32)
        // Det:  total_channels == 84  (4 + 80)
        let (num_classes, num_keypoints, num_mask_coeffs) = if total_channels == 56 {
            (1_usize, 17_usize, 0_usize)
        } else if total_channels > 84 && total_channels != 56 {
            // Seg: dư sau 4+80 = 80+4 là mask coefficients
            let nc = 80_usize;
            let nm = total_channels - 4 - nc;
            (nc, 0_usize, nm)
        } else {
            ((total_channels - 4), 0_usize, 0_usize)
        };

        self.num_classes     = num_classes;
        self.num_keypoints   = num_keypoints;
        self.num_mask_coeffs = num_mask_coeffs;

        let scale_x = orig_dim.0 as f32 / self.input_width as f32;
        let scale_y = orig_dim.1 as f32 / self.input_height as f32;
        let conf_threshold = self.conf_threshold;

        let t_nms = Instant::now();

        // ── Decode anchors ───────────────────────────────────────────────────
        let mut all_boxes: Vec<(f32, f32, f32, f32, f32, usize, Vec<(f32, f32, f32)>, Vec<f32>)> =
            Vec::with_capacity(128);

        for i in 0..num_anchors {
            let mut max_conf  = 0.0f32;
            let mut max_class = 0_usize;

            for c in 0..num_classes {
                let conf = out_data[(4 + c) * num_anchors + i];
                if conf > max_conf {
                    max_conf  = conf;
                    max_class = c;
                }
            }

            if max_conf <= conf_threshold {
                continue;
            }

            let cx = out_data[i];
            let cy = out_data[num_anchors + i];
            let w  = out_data[2 * num_anchors + i];
            let h  = out_data[3 * num_anchors + i];

            let x      = (cx - w / 2.0) * scale_x;
            let y      = (cy - h / 2.0) * scale_y;
            let bbw    = w * scale_x;
            let bbh    = h * scale_y;

            let x   = x.clamp(0.0, orig_dim.0 as f32);
            let y   = y.clamp(0.0, orig_dim.1 as f32);
            let bbw = bbw.clamp(0.0, orig_dim.0 as f32);
            let bbh = bbh.clamp(0.0, orig_dim.1 as f32);

            // ── Keypoints (Pose) ─────────────────────────────────────────────
            let mut keypoints = Vec::with_capacity(17);
            let output_len    = out_data.len();
            let base_offset   = (4 + num_classes) * num_anchors;

            if num_keypoints > 0
                && output_len >= base_offset + num_keypoints * 3 * num_anchors
            {
                for kp_idx in 0..num_keypoints {
                    let x_off = base_offset + kp_idx * 3 * num_anchors + i;
                    let y_off = base_offset + (kp_idx * 3 + 1) * num_anchors + i;
                    let c_off = base_offset + (kp_idx * 3 + 2) * num_anchors + i;

                    if x_off >= output_len || y_off >= output_len || c_off >= output_len {
                        keypoints.push((0.0, 0.0, 0.0));
                        continue;
                    }
                    let kx    = out_data[x_off] * scale_x;
                    let ky    = out_data[y_off] * scale_y;
                    let kconf = out_data[c_off];
                    keypoints.push((kx, ky, kconf));
                }
            }

            // ── Mask coefficients (Seg) ──────────────────────────────────────
            let mut mask_coeffs = Vec::with_capacity(num_mask_coeffs);
            if num_mask_coeffs > 0 {
                let mc_base = (4 + num_classes) * num_anchors;
                for m in 0..num_mask_coeffs {
                    let off = mc_base + m * num_anchors + i;
                    if off < output_len {
                        mask_coeffs.push(out_data[off]);
                    } else {
                        mask_coeffs.push(0.0);
                    }
                }
            }

            all_boxes.push((x, y, bbw, bbh, max_conf, max_class, keypoints, mask_coeffs));
        }

        // ── NMS ─────────────────────────────────────────────────────────────
        let iou_threshold = self.iou_threshold;
        all_boxes.sort_unstable_by(|a, b| b.4.partial_cmp(&a.4).unwrap());

        let mut keep       = vec![true; all_boxes.len()];
        let mut detections = Vec::with_capacity(all_boxes.len());

        for i in 0..all_boxes.len() {
            if !keep[i] { continue; }

            let (x, y, w, h, conf, class_id, ref kps, ref mcs) = all_boxes[i];
            detections.push(YoloDetection {
                class_id: class_id as i32,
                confidence: conf,
                x, y, width: w, height: h,
                keypoints:   kps.clone(),
                mask_coeffs: mcs.clone(),
            });

            for j in (i + 1)..all_boxes.len() {
                if !keep[j] || all_boxes[j].5 != class_id { continue; }
                if Self::compute_iou_internal(&all_boxes[i], &all_boxes[j]) > iou_threshold {
                    keep[j] = false;
                }
            }
        }

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
        let union  = area1 + area2 - intersection;

        if union == 0.0 { 0.0 } else { intersection / union }
    }

    /// Decode classification output: Softmax + Top-K
    fn decode_cls_output(&self, out_data: &[f32], _num_classes: usize) -> Vec<YoloDetection> {

        // 1. Softmax (numerically stable)
        let max_val = out_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = out_data.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|x| x / sum).collect();

        // 2. Lấy Top-5 classes
        let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate()
            .map(|(i, &p)| (i, p)).collect();
        
        // Sắp xếp giảm dần theo xác suất
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // 3. Trả về top 5 detections (không có box)
        indexed.into_iter().take(5).map(|(idx, prob)| {
            YoloDetection {
                class_id: idx as i32,
                confidence: prob,
                x: 0.0,
                y: 0.0,
                width: 0.0,
                height: 0.0,
                keypoints: vec![],
                mask_coeffs: vec![],
            }
        }).collect()
    }
}
