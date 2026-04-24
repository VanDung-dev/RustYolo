//! Logic decode cho YOLOv26 Classification

use crate::v26::detector::YoloV26Detector;
use crate::yolo::YoloDetection;

impl YoloV26Detector {
    pub(crate) fn decode_cls_v26(out_data: &[f32], _num_classes: usize) -> Vec<YoloDetection> {
        let max_val = out_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = out_data.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|x| x / sum).collect();
        let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.into_iter().take(5).map(|(idx, prob)| YoloDetection {
            class_id: idx as i32, confidence: prob,
            x: 0.0, y: 0.0, width: 0.0, height: 0.0,
            keypoints: vec![], mask_coeffs: vec![],
        }).collect()
    }
}
