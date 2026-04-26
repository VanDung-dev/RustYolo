//! Logic decode cho YOLOv8 Instance Segmentation

use crate::v8::detector::YoloV8Detector;
use crate::yolo::{YoloDetection, YoloCommon};
use ndarray::{Axis, s};
use pyo3::prelude::*;

impl YoloV8Detector {
    pub(crate) fn decode_seg_v8(
        conf_threshold: f32,
        iou_threshold: f32,
        num_classes: usize,
        num_mask_coeffs: usize,
        input_width: usize,
        input_height: usize,
        out_data: &ndarray::ArrayViewD<f32>,
        orig_dim: (usize, usize),
    ) -> PyResult<Vec<YoloDetection>> {
        let input_width_f = input_width as f32;
        let input_height_f = input_height as f32;

        let out_shape = out_data.shape();
        let num_anchors = out_shape[2];
        let scale_x = orig_dim.0 as f32 / input_width_f;
        let scale_y = orig_dim.1 as f32 / input_height_f;

        let out_data_2d = out_data.index_axis(Axis(0), 0).reversed_axes();
        let out_data_2d = out_data_2d.into_dimensionality::<ndarray::Ix2>().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Dimensionality error: {}", e))
        })?;

        let mut all_boxes: Vec<(f32, f32, f32, f32, f32, usize, Vec<(f32, f32, f32)>, Vec<f32>)> = Vec::with_capacity(128);
        for i in 0..num_anchors {
            let row = out_data_2d.row(i);
            let scores = row.slice(s![4..4 + num_classes]);

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

            let x = ((cx - w / 2.0) * scale_x).clamp(0.0, orig_dim.0 as f32);
            let y = ((cy - h / 2.0) * scale_y).clamp(0.0, orig_dim.1 as f32);
            let bbw = (w * scale_x).clamp(0.0, orig_dim.0 as f32);
            let bbh = (h * scale_y).clamp(0.0, orig_dim.1 as f32);

            let mut mask_coeffs = Vec::with_capacity(num_mask_coeffs);
            let mc_base = 4 + num_classes;
            for m in 0..num_mask_coeffs {
                mask_coeffs.push(row[mc_base + m]);
            }

            all_boxes.push((x, y, bbw, bbh, max_conf, max_class, vec![], mask_coeffs));
        }

        // 2. Sắp xếp theo độ tin cậy giảm dần
        all_boxes.sort_unstable_by(|a, b| b.4.partial_cmp(&a.4).unwrap());

        // 3. NMS (Non-Maximum Suppression)
        let nms_boxes: Vec<(f32, f32, f32, f32)> = all_boxes.iter().map(|b| (b.0, b.1, b.2, b.3)).collect();
        let nms_classes: Vec<usize> = all_boxes.iter().map(|b| b.5).collect();
        let keep_indices = YoloCommon::perform_nms(&nms_boxes, &nms_classes, iou_threshold);

        let mut detections = Vec::with_capacity(keep_indices.len());
        for &idx in &keep_indices {
            let (x, y, w, h, conf, class_id, _, mcs) = &all_boxes[idx];
            detections.push(YoloDetection {
                class_id: *class_id as i32,
                confidence: *conf,
                x: *x, y: *y, width: *w, height: *h,
                keypoints: vec![],
                mask_coeffs: mcs.clone(),
            });
        }

        Ok(detections)
    }
}
