//! Cung cấp giao diện FaceToolbox cho Python thông qua PyO3 và truyền dữ liệu Arrow hiệu năng cao.
//! Dự án RustYolo - Hệ thống điểm danh thông minh sử dụng Rust và Python.

use pyo3::prelude::*;
use crate::face::align::norm_crop;
use crate::face::embedder::ArcFaceEmbedder;
use crate::face::detector::ScrfdDetector;
use image::{ImageBuffer, Rgb, DynamicImage};
use image::buffer::ConvertBuffer;
use std::sync::Mutex;
use std::sync::Arc;
use rayon::prelude::*;
use numpy::PyReadonlyArray3;

// Các import cho Arrow để truyền dữ liệu hiệu năng cao
use arrow::array::{Float32Array, StructArray, ListBuilder, Float32Builder, Array};
use arrow::datatypes::{DataType, Field, Fields};
use pyo3::types::PyCapsule;

#[pyclass]
pub struct FaceToolbox {
    embedder: Mutex<Option<ArcFaceEmbedder>>,
    detector: Mutex<Option<ScrfdDetector>>,
}

#[pymethods]
impl FaceToolbox {
    #[new]
    fn new() -> Self {
        Self { 
            embedder: Mutex::new(None),
            detector: Mutex::new(None),
        }
    }

    /// Nạp model ArcFace để trích xuất embedding
    #[pyo3(signature = (model_path, execution_provider="coreml"))]
    fn load_embedder(&self, model_path: String, execution_provider: &str) -> PyResult<()> {
        let ep = crate::yolo::ExecutionProviderType::from_str(execution_provider);
        let embedder = ArcFaceEmbedder::new(&model_path, &ep.get_dispatch())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let mut lock = self.embedder.lock().map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Lỗi khóa Mutex (Embedder)"))?;
        *lock = Some(embedder);
        Ok(())
    }

    /// Nạp model SCRFD để phát hiện khuôn mặt
    #[pyo3(signature = (model_path, input_size, execution_provider="coreml"))]
    fn load_detector(&self, model_path: String, input_size: (u32, u32), execution_provider: &str) -> PyResult<()> {
        let ep = crate::yolo::ExecutionProviderType::from_str(execution_provider);
        let detector = ScrfdDetector::new(&model_path, input_size, 0.5, 0.4, &ep.get_dispatch())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let mut lock = self.detector.lock().map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Lỗi khóa Mutex (Detector)"))?;
        *lock = Some(detector);
        Ok(())
    }

    /// Phát hiện khuôn mặt và trả về Arrow Capsules (Zero-copy)
    fn detect_faces_to_arrow<'py>(
        &self, 
        py: Python<'py>,
        image: PyReadonlyArray3<'_, u8>,
        score_threshold: Option<f32>
    ) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
        let mut lock = self.detector.lock().map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Lỗi khóa Mutex"))?;
        let detector = lock.as_mut()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Chưa nạp Detector"))?;

        if let Some(threshold) = score_threshold {
            detector.config.score_threshold = threshold;
        }

        let array = image.as_array();
        let height = array.shape()[0] as u32;
        let width = array.shape()[1] as u32;
        
        let slice = array.as_slice().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Mảng không liên tục (Non-contiguous)"))?;
        let img_buf = ImageBuffer::<Rgb<u8>, &[u8]>::from_raw(width, height, slice)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Kích thước ảnh không hợp lệ"))?;
        
        let rgb_vec_buf: ImageBuffer<Rgb<u8>, Vec<u8>> = img_buf.convert();
        let img = DynamicImage::ImageRgb8(rgb_vec_buf);
        let detections = detector.detect(&img)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let mut bboxes_x1 = Vec::with_capacity(detections.len());
        let mut bboxes_y1 = Vec::with_capacity(detections.len());
        let mut bboxes_x2 = Vec::with_capacity(detections.len());
        let mut bboxes_y2 = Vec::with_capacity(detections.len());
        let mut scores = Vec::with_capacity(detections.len());
        
        let mut lmark_builder = ListBuilder::new(Float32Builder::new());

        for d in detections {
            let abs_d = d.to_absolute(width, height);
            bboxes_x1.push(abs_d.bbox.x1);
            bboxes_y1.push(abs_d.bbox.y1);
            bboxes_x2.push(abs_d.bbox.x2);
            bboxes_y2.push(abs_d.bbox.y2);
            scores.push(abs_d.score);
            
            if let Some(lms) = abs_d.landmarks {
                for (lx, ly) in lms {
                    lmark_builder.values().append_value(lx);
                    lmark_builder.values().append_value(ly);
                }
                lmark_builder.append(true);
            } else {
                lmark_builder.append(false);
            }
        }

        let lmark_array = lmark_builder.finish();
        let fields = vec![
            Field::new("x1", DataType::Float32, false),
            Field::new("y1", DataType::Float32, false),
            Field::new("x2", DataType::Float32, false),
            Field::new("y2", DataType::Float32, false),
            Field::new("score", DataType::Float32, false),
            Field::new("landmarks", lmark_array.data_type().clone(), true),
        ];

        let arrays: Vec<Arc<dyn Array>> = vec![
            Arc::new(Float32Array::from(bboxes_x1)),
            Arc::new(Float32Array::from(bboxes_y1)),
            Arc::new(Float32Array::from(bboxes_x2)),
            Arc::new(Float32Array::from(bboxes_y2)),
            Arc::new(Float32Array::from(scores)),
            Arc::new(lmark_array),
        ];

        let struct_array = StructArray::try_new(Fields::from(fields), arrays, None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lỗi Arrow: {}", e)))?;

        crate::ffi::export_to_python(py, struct_array.to_data())
    }

    /// Trích xuất embedding cho hàng loạt khuôn mặt (Song song hóa qua Rayon)
    fn get_embeddings_batch_to_arrow<'py>(
        &self, 
        py: Python<'py>,
        image: PyReadonlyArray3<'_, u8>,
        all_landmarks: Vec<Vec<f32>>
    ) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
        if all_landmarks.is_empty() {
            let empty_array = Float32Array::from(Vec::<f32>::new());
            return crate::ffi::export_to_python(py, empty_array.to_data());
        }

        let array = image.as_array();
        let height = array.shape()[0] as u32;
        let width = array.shape()[1] as u32;
        let slice = array.as_slice().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Mảng không liên tục"))?;

        // Cân chỉnh khuôn mặt song song bằng Rayon
        let aligned_faces: Vec<ImageBuffer<Rgb<u8>, Vec<u8>>> = all_landmarks.into_par_iter().map(|landmarks_flat| {
            let img_buf = ImageBuffer::<Rgb<u8>, &[u8]>::from_raw(width, height, slice).unwrap();
            let mut lmarks = [(0.0, 0.0); 5];
            for i in 0..5 { 
                if i * 2 + 1 < landmarks_flat.len() {
                    lmarks[i] = (landmarks_flat[i*2], landmarks_flat[i*2+1]); 
                }
            }
            norm_crop(&img_buf, &lmarks, 112)
        }).collect();

        let mut lock = self.embedder.lock().map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Lỗi khóa Mutex"))?;
        let embedder = lock.as_mut()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Chưa nạp Embedder"))?;

        let embeddings = embedder.compute_embeddings_batch(&aligned_faces)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let flattened: Vec<f32> = embeddings.into_iter().flatten().collect();
        let arrow_array = Float32Array::from(flattened);

        crate::ffi::export_to_python(py, arrow_array.to_data())
    }

    /// Cân chỉnh (Align) một khuôn mặt đơn lẻ
    fn align_face(
        &self, 
        image: PyReadonlyArray3<'_, u8>, 
        landmarks: Vec<(f32, f32)>
    ) -> PyResult<Vec<u8>> {
        let array = image.as_array();
        let h = array.shape()[0] as u32;
        let w = array.shape()[1] as u32;
        let slice = array.as_slice().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Mảng không liên tục"))?;
        let img_buf = ImageBuffer::<Rgb<u8>, &[u8]>::from_raw(w, h, slice)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Kích thước không hợp lệ"))?;

        if landmarks.len() != 5 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Landmarks phải chứa đúng 5 điểm"));
        }
        let mut lmarks = [(0.0, 0.0); 5];
        for i in 0..5 { lmarks[i] = landmarks[i]; }

        let aligned = norm_crop(&img_buf, &lmarks, 112);
        Ok(aligned.into_raw())
    }

    /// Trích xuất embedding cho một ảnh khuôn mặt đã được crop 112x112
    fn get_embedding(&self, face_image_raw: Vec<u8>) -> PyResult<Vec<f32>> {
        let mut lock = self.embedder.lock().map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Lỗi khóa Mutex"))?;
        let embedder = lock.as_mut()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Chưa nạp Embedder"))?;

        let img_buf = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(112, 112, face_image_raw)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Kích thước ảnh khuôn mặt không hợp lệ (Phải là 112x112)"))?;

        let embedding = embedder.compute_embedding(&img_buf)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(embedding)
    }
}
