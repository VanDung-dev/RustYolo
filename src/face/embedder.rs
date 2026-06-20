//! Trích xuất các vector đặc trưng (Embedding) 512 chiều sử dụng mô hình ArcFace.
//! File này chứa mã nguồn dẫn xuất từ face_id-rs (https://github.com/RuurdBijlsma/face_id-rs) được cấp phép Apache 2.0.

#![allow(dead_code)]

use crate::face::error::FaceError;
use image::{ImageBuffer, Rgb};
use ndarray::{Array2, Array4, Axis, s};
use ort::ep::ExecutionProviderDispatch;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;

pub struct ArcFaceEmbedder {
    pub session: Session,
    pub input_name: String,
}

impl ArcFaceEmbedder {
    pub fn new(
        model_path: impl AsRef<Path>,
        with_execution_providers: &[ExecutionProviderDispatch],
    ) -> Result<Self, FaceError> {
        let session = Session::builder()?
            .with_execution_providers(with_execution_providers)?
            .commit_from_file(model_path)?;

        let input_name = session.inputs()[0].name().to_string();

        Ok(Self {
            session,
            input_name,
        })
    }

    /// Tính toán embedding cho một danh sách ảnh khuôn mặt (Batch)
    pub fn compute_embeddings_batch(
        &mut self,
        aligned_imgs: &[ImageBuffer<Rgb<u8>, Vec<u8>>],
    ) -> Result<Vec<Vec<f32>>, FaceError> {
        if aligned_imgs.is_empty() {
            return Ok(vec![]);
        }

        let input_tensor = Self::create_input_tensor_batch(aligned_imgs)?;
        let input_value = Value::from_array(input_tensor)?;

        let outputs = self
            .session
            .run(ort::inputs![&self.input_name => input_value])?;

        let mut output_tensor = outputs[0]
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality::<ndarray::Ix2>()?;

        let expected_batch_size = aligned_imgs.len();
        if output_tensor.shape()[0] != expected_batch_size {
            return Err(FaceError::Ort(format!(
                "Lỗi khớp Batch Size: mong đợi {expected_batch_size}, nhận được {}",
                output_tensor.shape()[0]
            )));
        }

        // Chuẩn hóa L2 cho các vector kết quả
        Self::l2_normalize_batch(&mut output_tensor);

        let batch_size = output_tensor.shape()[0];
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            results.push(output_tensor.slice(s![i, ..]).to_vec());
        }

        Ok(results)
    }

    /// Tạo input tensor 4D (N, C, H, W) từ danh sách ảnh
    fn create_input_tensor_batch(
        imgs: &[ImageBuffer<Rgb<u8>, Vec<u8>>],
    ) -> Result<Array4<f32>, FaceError> {
        let batch_size = imgs.len();
        let mut array = Array4::<f32>::zeros((batch_size, 3, 112, 112));

        let data = array
            .as_slice_memory_order_mut()
            .ok_or_else(|| FaceError::Ort("Không thể lấy slice để ghi dữ liệu".into()))?;

        let channel_stride = 112 * 112;
        for (batch_idx, img) in imgs.iter().enumerate() {
            let (w, h) = img.dimensions();
            if w != 112 || h != 112 {
                return Err(FaceError::InvalidModel(format!(
                    "ArcFace yêu cầu ảnh 112x112, nhận được {w}x{h}"
                )));
            }

            let raw = img.as_raw();
            let batch_offset = batch_idx * 3 * channel_stride;

            // Model ArcFace (insightface) yêu cầu normalization: (x - 127.5) / 127.5
            // Input là BGR (từ OpenCV), swap R↔B khi normalize
            for (i, chunk) in raw.chunks_exact(3).enumerate() {
                data[batch_offset + i] = (f32::from(chunk[2]) - 127.5) / 127.5;
                data[batch_offset + i + channel_stride] = (f32::from(chunk[1]) - 127.5) / 127.5;
                data[batch_offset + i + 2 * channel_stride] = (f32::from(chunk[0]) - 127.5) / 127.5;
            }
        }

        Ok(array)
    }

    /// Tính embedding cho một ảnh duy nhất
    pub fn compute_embedding(
        &mut self,
        aligned_img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> Result<Vec<f32>, FaceError> {
        let mut results = self.compute_embeddings_batch(std::slice::from_ref(aligned_img))?;
        results
            .pop()
            .ok_or_else(|| FaceError::Ort("Lỗi trích xuất embedding".into()))
    }

    pub fn create_input_tensor(
        img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> Result<Array4<f32>, FaceError> {
        Self::create_input_tensor_batch(std::slice::from_ref(img))
    }

    /// Chuẩn hóa L2 cho mảng embeddings (Matrix)
    pub fn l2_normalize_batch(embeddings: &mut Array2<f32>) {
        let view = embeddings.view();
        let sq_sums = (&view * &view).sum_axis(Axis(1));
        let inv_norms = sq_sums
            .mapv(|x| 1.0 / x.max(1e-12).sqrt())
            .insert_axis(Axis(1));
        *embeddings *= &inv_norms;
    }

    /// Chuẩn hóa L2 cho vector đơn
    pub fn l2_normalize(vec: &mut [f32]) {
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            let inv_norm = 1.0 / norm;
            for x in vec.iter_mut() {
                *x *= inv_norm;
            }
        }
    }

    /// Tính độ tương đồng Cosine (Dot product vì đã chuẩn hóa L2)
    #[must_use]
    pub fn compute_similarity(emb1: &[f32], emb2: &[f32]) -> f32 {
        emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum()
    }
}
