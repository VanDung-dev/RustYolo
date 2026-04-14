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
use log::{debug, info, warn};
use ndarray::Array4;
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
    // Buffer tái sử dụng để tránh cấp phát bộ nhớ liên tục
    input_tensor_buffer: Array4<f32>,
    #[pyo3(get)]
    pub ep: ExecutionProviderType,
}

struct YoloResultsV26 {
    detections: Vec<YoloDetection>,
    proto: Option<ndarray::ArrayD<f32>>,
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

        // 4. Đóng gói kết quả Arrow (Giống detect_to_arrow)
        let class_ids = Int32Array::from(detections.iter().map(|d: &YoloDetection| d.class_id).collect::<Vec<i32>>());
        let confidences = Float32Array::from(detections.iter().map(|d: &YoloDetection| d.confidence).collect::<Vec<f32>>());
        let boxes_x = Float32Array::from(detections.iter().map(|d: &YoloDetection| d.x).collect::<Vec<f32>>());
        let boxes_y = Float32Array::from(detections.iter().map(|d: &YoloDetection| d.y).collect::<Vec<f32>>());
        let boxes_w = Float32Array::from(detections.iter().map(|d: &YoloDetection| d.width).collect::<Vec<f32>>());
        let boxes_h = Float32Array::from(detections.iter().map(|d: &YoloDetection| d.height).collect::<Vec<f32>>());

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
            Arc::new(boxes_x), Arc::new(boxes_y),
            Arc::new(boxes_w), Arc::new(boxes_h),
        ];

        if self.num_keypoints > 0 {
            fields.push(Field::new("keypoints", DataType::List(Arc::new(Field::new("item", DataType::Float32, true))), true));
            let mut kp_builder = arrow::array::ListBuilder::new(arrow::array::Float32Builder::new());
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

        if self.num_mask_coeffs > 0 {
            fields.push(Field::new("mask_coeffs", DataType::List(Arc::new(Field::new("item", DataType::Float32, true))), true));
            let mut mc_builder = arrow::array::ListBuilder::new(arrow::array::Float32Builder::new());
            for det in &detections {
                for c in &det.mask_coeffs { mc_builder.values().append_value(*c); }
                mc_builder.append(true);
            }
            arrays.push(Arc::new(mc_builder.finish()));
        }

        let struct_array = StructArray::try_new(Fields::from(fields), arrays, None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Arrow error: {}", e)))?;

        let (arr_cap, sch_cap) = crate::ffi::export_to_python(py, struct_array.to_data())?;
        
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
    fn run_detection_pipeline<'py>(&mut self, py: Python<'py>, numpy_array: &Bound<'py, PyAny>) -> PyResult<(Vec<YoloDetection>, Option<ndarray::ArrayD<f32>>, usize, usize)> {
        let (width, height) = self.prepare_input(numpy_array)?;
        let results = self.run_inference_internal(py, (width, height))?;
        Ok((results.detections, results.proto, width, height))
    }

    fn prepare_input(&mut self, numpy_array: &Bound<'_, PyAny>) -> PyResult<(usize, usize)> {
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

    fn run_inference_internal(
        &mut self,
        py: Python,
        orig_dim: (usize, usize),
    ) -> PyResult<YoloResultsV26> {
        let conf_threshold = self.conf_threshold;
        let scale_x = orig_dim.0 as f32 / self.input_width as f32;
        let scale_y = orig_dim.1 as f32 / self.input_height as f32;

        let input_tensor = Value::from_array(self.input_tensor_buffer.clone()).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let t_infer = Instant::now();
        let outputs = py.detach(|| self.session.run(ort::inputs![input_tensor])).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        self.last_inference_ms = t_infer.elapsed().as_secs_f64() * 1000.0;

        // Logging hiệu suất cực kỳ chi tiết cho M4 Pro (YOLOv26)
        if self.last_inference_ms > 0.1 {
            info!(
                "🎯 Perf Metrics [v26]: Pre={:.2}ms, Infer={:.2}ms, Total={:.2}ms | FPS: {:.1}",
                self.last_preprocess_ms,
                self.last_inference_ms,
                self.last_preprocess_ms + self.last_inference_ms,
                1000.0 / (self.last_preprocess_ms + self.last_inference_ms).max(0.1)
            );
        }

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
