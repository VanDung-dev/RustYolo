//! Quản lý và xuất bản (export) các module con trong thư mục face.
//! File này chứa mã nguồn dẫn xuất từ face_id-rs (https://github.com/RuurdBijlsma/face_id-rs) được cấp phép Apache 2.0.

#![allow(dead_code)]

pub mod align;
pub mod detector;
pub mod embedder;
pub mod error;
pub mod ffi;

#[allow(unused_imports)]
pub use align::norm_crop;
#[allow(unused_imports)]
pub use detector::{ScrfdDetector, DetectedFace, BoundingBox};
#[allow(unused_imports)]
pub use embedder::ArcFaceEmbedder;
#[allow(unused_imports)]
pub use error::FaceError;
#[allow(unused_imports)]
pub use ffi::FaceToolbox;
