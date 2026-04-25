//! Định nghĩa các cấu trúc lỗi (FaceError) cho toàn bộ module xử lý khuôn mặt.
//! File này chứa mã nguồn dẫn xuất từ face_id-rs (https://github.com/RuurdBijlsma/face_id-rs) được cấp phép Apache 2.0.

#![allow(dead_code)]

use thiserror::Error;

#[derive(Error, Debug)]
pub enum FaceError {
    #[error("Lỗi IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("Lỗi hình ảnh: {0}")]
    Image(#[from] image::ImageError),
    #[error("Không thể lấy mutable slice: {0}")]
    FailedToGetMutableSlice(String),
    #[error("Lỗi ONNX Runtime: {0}")]
    Ort(String),
    #[error("Lỗi Mutex Poisoned: {0}")]
    MutexPoisoned(String),
    #[error("Lỗi NdArray: {0}")]
    NdArray(#[from] ndarray::ShapeError),
    #[error("Lỗi giải mã hình ảnh (Decode)")]
    Decode,
    #[error("Model không hợp lệ: {0}")]
    InvalidModel(String),
}

impl<T> From<ort::Error<T>> for FaceError {
    fn from(err: ort::Error<T>) -> Self {
        Self::Ort(err.to_string())
    }
}
