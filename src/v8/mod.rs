//! YOLOv8 Module
//!
//! Thư mục này chứa toàn bộ logic thực thi cho kiến trúc YOLOv8 và YOLOv11.
//! Cấu trúc được chia nhỏ thành các module để dễ dàng quản lý và mở rộng:
//! - `detector`: Quản lý Session và điều phối thực thi.
//! - `base`: Logic decode cho Object Detection tiêu chuẩn.
//! - `pose`: Logic decode cho Pose Estimation (Keypoints).
//! - `seg`: Logic decode cho Instance Segmentation.
//! - `obb`: Logic decode cho Oriented Bounding Box.
//! - `cls`: Logic decode cho Classification.

pub(crate) mod detector;
mod base;
mod pose;
mod seg;
mod obb;
mod cls;

pub use detector::YoloV8Detector;
