//! Logic decode cho Object Detection YOLOv26 (NMS-Free)

use crate::v26::detector::{YoloV26Detector, YoloResultsV26};
use crate::yolo::YoloDetection;
use pyo3::prelude::*;
use ort::session::SessionOutputs;
use rayon::prelude::*;
use ndarray::Axis;

impl YoloV26Detector {
    pub(crate) fn decode_base_v26(
        conf_threshold: f32,
        input_width: usize,
        input_height: usize,
        outputs: &SessionOutputs,
        orig_dim: (usize, usize),
    ) -> PyResult<YoloResultsV26> {
        let scale_x = orig_dim.0 as f32 / input_width as f32;
        let scale_y = orig_dim.1 as f32 / input_height as f32;

        let out_value = outputs.values().next().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No output found"))?;
        let out_extract = out_value.try_extract_tensor::<f32>().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let shape = out_extract.0.iter().map(|&d| d as usize).collect::<Vec<_>>();
        let out_data = ndarray::ArrayViewD::from_shape(shape, out_extract.1).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Tối ưu hóa: YOLOv26 (v10) thường có 300 anchors. Xử lý song song để giữ latency thấp nhất.
        let detections: Vec<YoloDetection> = out_data.index_axis(Axis(0), 0)
            .axis_iter(Axis(0))
            .into_par_iter()
            .filter_map(|row| {
                let score = row[4];
                if score < conf_threshold { return None; }

                let x1 = row[0];
                let y1 = row[1];
                let x2 = row[2];
                let y2 = row[3];
                let class_id = row[5] as i32;

                Some(YoloDetection {
                    class_id, confidence: score,
                    x: x1 * scale_x, y: y1 * scale_y,
                    width: (x2 - x1) * scale_x, height: (y2 - y1) * scale_y,
                    keypoints: vec![], mask_coeffs: vec![],
                })
            })
            .collect();

        Ok(YoloResultsV26 { detections, proto: None })
    }
}
