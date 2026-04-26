//! Logic decode cho YOLOv8 Pose Estimation (Keypoints)

use crate::v8::detector::YoloV8Detector;
use crate::yolo::{YoloDetection, YoloCommon};
use ndarray::{Axis, s};
use pyo3::prelude::*;
use rayon::prelude::*;

impl YoloV8Detector {
    pub(crate) fn decode_pose_v8(
        conf_threshold: f32,
        iou_threshold: f32,
        num_classes: usize,
        num_keypoints: usize,
        input_width: usize,
        input_height: usize,
        out_data: &ndarray::ArrayViewD<f32>,
        orig_dim: (usize, usize),
    ) -> PyResult<Vec<YoloDetection>> {
        let input_width_f = input_width as f32;
        let input_height_f = input_height as f32;

        let scale_x = orig_dim.0 as f32 / input_width_f;
        let scale_y = orig_dim.1 as f32 / input_height_f;

        let out_data_2d = YoloCommon::reshape_output_v8(out_data)?;

        let all_boxes: Vec<(f32, f32, f32, f32, f32, usize, Vec<(f32, f32, f32)>, Vec<f32>)> = out_data_2d.axis_iter(Axis(0))
            .into_par_iter()
            .filter_map(|row| {
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

                let mut keypoints = Vec::with_capacity(num_keypoints);
                let base_offset = 4 + num_classes;
                for kp_idx in 0..num_keypoints {
                    let kx = row[base_offset + kp_idx * 3] * scale_x;
                    let ky = row[base_offset + kp_idx * 3 + 1] * scale_y;
                    let kconf = row[base_offset + kp_idx * 3 + 2];
                    keypoints.push((kx, ky, kconf));
                }

                Some((x, y, bbw, bbh, max_conf, max_class, keypoints, vec![]))
            })
            .collect();

        Ok(YoloCommon::finalize_detections(all_boxes, iou_threshold))
    }
}

