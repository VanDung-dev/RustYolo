//! Logic decode cho YOLOv26 Pose Estimation

use crate::v26::detector::{YoloV26Detector, YoloResultsV26};
use crate::yolo::YoloDetection;
use pyo3::prelude::*;
use ort::session::SessionOutputs;

impl YoloV26Detector {
    pub(crate) fn decode_pose_v26(
        conf_threshold: f32,
        input_width: usize,
        input_height: usize,
        num_keypoints: usize,
        outputs: &SessionOutputs,
        orig_dim: (usize, usize),
    ) -> PyResult<YoloResultsV26> {
        let scale_x = orig_dim.0 as f32 / input_width as f32;
        let scale_y = orig_dim.1 as f32 / input_height as f32;

        let out_value = outputs.values().next().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No output found"))?;
        let out_extract = out_value.try_extract_tensor::<f32>().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let shape = out_extract.0.iter().map(|&d| d as usize).collect::<Vec<_>>();
        let out_data = ndarray::ArrayViewD::from_shape(shape, out_extract.1).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let num_detections = out_data.shape()[1];
        let mut detections = Vec::with_capacity(32);

        for i in 0..num_detections {
            let score = out_data[[0, i, 4]];
            if score < conf_threshold { continue; }

            let x1 = out_data[[0, i, 0]];
            let y1 = out_data[[0, i, 1]];
            let x2 = out_data[[0, i, 2]];
            let y2 = out_data[[0, i, 3]];
            let class_id = out_data[[0, i, 5]] as i32;

            let mut keypoints = Vec::new();
            for k in 0..num_keypoints {
                let kx = out_data[[0, i, 6 + k * 3]] * scale_x;
                let ky = out_data[[0, i, 7 + k * 3]] * scale_y;
                let kconf = out_data[[0, i, 8 + k * 3]];
                keypoints.push((kx, ky, kconf));
            }

            detections.push(YoloDetection {
                class_id, confidence: score,
                x: x1 * scale_x, y: y1 * scale_y,
                width: (x2 - x1) * scale_x, height: (y2 - y1) * scale_y,
                keypoints, mask_coeffs: vec![],
            });
        }

        Ok(YoloResultsV26 { detections, proto: None })
    }
}
