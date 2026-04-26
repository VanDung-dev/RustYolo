//! YOLOv26 Module (NMS-Free)
//!
//! Thư mục này chứa toàn bộ logic thực thi cho kiến trúc YOLOv26 và YOLOv10.
//! Đặc điểm của dòng này là "NMS-Free", việc decode đơn giản hơn và nhanh hơn.
//! - `detector`: Quản lý Session NMS-Free.
//! - `base`: Logic decode Detection.
//! - `pose`: Logic decode Pose Estimation.
//! - `seg`: Logic decode Segmentation.

mod detector;
mod base;
mod pose;
mod seg;

pub use detector::YoloV26Detector;
