//! Logic decode cho Object Detection YOLOv8 (Standard Bounding Boxes)

use crate::v8::detector::YoloV8Detector;
use crate::yolo::YoloDetection;
use ndarray::{Axis, s};
use pyo3::prelude::*;
use rayon::prelude::*;

impl YoloV8Detector {
    pub(crate) fn decode_base_v8(
        conf_threshold: f32,
        iou_threshold: f32,
        num_classes: usize,
        input_width: usize,
        input_height: usize,
        out_data: &ndarray::ArrayViewD<f32>,
        orig_dim: (usize, usize),
    ) -> PyResult<Vec<YoloDetection>> {
        let input_width_f = input_width as f32;
        let input_height_f = input_height as f32;

        let scale_x = orig_dim.0 as f32 / input_width_f;
        let scale_y = orig_dim.1 as f32 / input_height_f;

        // Tối ưu layout: (1, 84, 8400) -> (84, 8400) -> (8400, 84)
        let out_data_2d = out_data.index_axis(Axis(0), 0).reversed_axes();
        let out_data_2d = out_data_2d.into_dimensionality::<ndarray::Ix2>().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Dimensionality error: {}", e))
        })?;

        // Tối ưu hóa: Sử dụng Rayon để xử lý song song hàng ngàn anchors cùng lúc
        // Kết hợp lọc ngưỡng ngay trong lúc decode để giảm kích thước vector
        let mut all_boxes: Vec<_> = out_data_2d.axis_iter(Axis(0))
            .into_par_iter()
            .filter_map(|row| {
                let scores = row.slice(s![4..4 + num_classes]);
                
                // Tìm class có độ tự tin cao nhất
                let mut max_conf = 0.0f32;
                let mut max_class = 0_usize;
                for (c, &conf) in scores.iter().enumerate() {
                    if conf > max_conf {
                        max_conf = conf;
                        max_class = c;
                    }
                }

                if max_conf <= conf_threshold {
                    return None;
                }

                let cx = row[0];
                let cy = row[1];
                let w = row[2];
                let h = row[3];

                let x = ((cx - w / 2.0) * scale_x).clamp(0.0, orig_dim.0 as f32);
                let y = ((cy - h / 2.0) * scale_y).clamp(0.0, orig_dim.1 as f32);
                let bbw = (w * scale_x).clamp(0.0, orig_dim.0 as f32);
                let bbh = (h * scale_y).clamp(0.0, orig_dim.1 as f32);

                Some((x, y, bbw, bbh, max_conf, max_class, vec![], vec![]))
            })
            .collect();

        // NMS (Non-Maximum Suppression)
        all_boxes.sort_unstable_by(|a, b| b.4.partial_cmp(&a.4).unwrap());
        let mut keep = vec![true; all_boxes.len()];
        let mut detections = Vec::with_capacity(all_boxes.len());

        for i in 0..all_boxes.len() {
            if !keep[i] { continue; }
            let (x, y, w, h, conf, class_id, _, _) = all_boxes[i];
            detections.push(YoloDetection {
                class_id: class_id as i32,
                confidence: conf,
                x, y, width: w, height: h,
                keypoints: vec![],
                mask_coeffs: vec![],
            });

            // Tối ưu NMS: Chỉ so sánh với các box chưa bị loại và cùng class_id
            for j in (i + 1)..all_boxes.len() {
                if keep[j] && all_boxes[j].5 == class_id {
                    if Self::compute_iou_internal(&all_boxes[i], &all_boxes[j]) > iou_threshold {
                        keep[j] = false;
                    }
                }
            }
        }

        Ok(detections)
    }
}
