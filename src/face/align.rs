//! Thực hiện cân chỉnh khuôn mặt (Umeyama transform) để đưa về tư thế chuẩn 112x112.
//! File này chứa mã nguồn dẫn xuất từ face_id-rs (https://github.com/RuurdBijlsma/face_id-rs) được cấp phép Apache 2.0.

#![allow(clippy::similar_names, clippy::many_single_char_names, dead_code)]

use image::{ImageBuffer, Rgb};
use nalgebra::{ArrayStorage, Matrix2, Matrix2x1, Matrix3, Matrix3x2};
use std::ops::Deref;

/// Tọa độ 5 điểm mốc chuẩn (Landmarks) cho ArcFace ở độ phân giải 112×112.
pub const ARCFACE_DST_112: [(f32, f32); 5] = [
    (38.2946, 51.6963), // mắt trái
    (73.5318, 51.5014), // mắt phải
    (56.0252, 71.7366), // đầu mũi
    (41.5493, 92.3655), // khóe miệng trái
    (70.7299, 92.2041), // khóe miệng phải
];

/// Thuật toán Umeyama để tính toán ma trận biến đổi affine (Affine Transformation Matrix)
/// Giúp xoay và căn chỉnh khuôn mặt về tư thế chuẩn.
pub fn umeyama<const R: usize>(src: &[(f32, f32); R], dst: &[(f32, f32); R]) -> Matrix3x2<f32> {
    let src_x_mean = src.iter().map(|v| v.0).sum::<f32>() / R as f32;
    let src_y_mean = src.iter().map(|v| v.1).sum::<f32>() / R as f32;
    let dst_x_mean = dst.iter().map(|v| v.0).sum::<f32>() / R as f32;
    let dst_y_mean = dst.iter().map(|v| v.1).sum::<f32>() / R as f32;

    let src_demean_s = ArrayStorage(src.map(|v| [v.0 - src_x_mean, v.1 - src_y_mean]));
    let dst_demean_s = ArrayStorage(dst.map(|v| [v.0 - dst_x_mean, v.1 - dst_y_mean]));
    let src_demean = nalgebra::Matrix::from_array_storage(src_demean_s);
    let dst_demean = nalgebra::Matrix::from_array_storage(dst_demean_s);

    let a = (dst_demean * src_demean.transpose()) / R as f32;
    let svd = a.svd(true, true);

    let mut d = [1f32; 2];
    if a.determinant() < 0.0 {
        d[1] = -1.0;
    }

    let mut t = Matrix2::<f32>::identity();
    let s = svd.singular_values;
    let u = svd.u.unwrap_or_else(Matrix2::identity);
    let v = svd.v_t.unwrap_or_else(Matrix2::identity);

    let rank = a.rank(1e-5);

    if rank == 0 {
        return Matrix3x2::identity();
    } else if rank == 1 {
        if u.determinant() * v.determinant() > 0.0 {
            u.mul_to(&v, &mut t);
        } else {
            d[1] = -1.0;
            let dg = Matrix2::new(d[0], 0.0, 0.0, d[1]);
            (u * dg).mul_to(&v, &mut t);
        }
    } else {
        let dg = Matrix2::new(d[0], 0.0, 0.0, d[1]);
        (u * dg).mul_to(&v, &mut t);
    }

    let d_dot_s = d[0].mul_add(s[0], d[1] * s[1]);
    let var_src = src_demean.remove_row(0).variance() + src_demean.remove_row(1).variance();
    let scale = d_dot_s / var_src;

    let dst_mean = Matrix2x1::new(dst_x_mean, dst_y_mean);
    let src_mean = Matrix2x1::new(src_x_mean, src_y_mean);
    let translation = dst_mean - scale * t * src_mean;

    let sr = t * scale;
    Matrix3x2::new(
        sr.m11,
        sr.m12,
        sr.m21,
        sr.m22,
        translation[0],
        translation[1],
    )
}

/// Thực hiện cân chỉnh khuôn mặt về kích thước chuẩn
#[must_use]
pub fn norm_crop<C>(
    img: &ImageBuffer<Rgb<u8>, C>,
    landmarks: &[(f32, f32); 5],
    image_size: u32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> 
where C: Deref<Target = [u8]> 
{
    let dst = scale_arcface_dst(image_size);
    let m = umeyama(landmarks, &dst);
    warp_affine(img, &m, image_size)
}

/// Tỉ lệ hóa tọa độ chuẩn theo kích thước ảnh mong muốn
fn scale_arcface_dst(image_size: u32) -> [(f32, f32); 5] {
    let ratio;
    let diff_x = if image_size % 112 == 0 {
        ratio = image_size as f32 / 112.0;
        0.0
    } else {
        ratio = image_size as f32 / 128.0;
        8.0 * ratio
    };
    ARCFACE_DST_112.map(|(x, y)| (x.mul_add(ratio, diff_x), y * ratio))
}

/// Áp dụng biến đổi Affine để tạo ảnh đầu ra
fn warp_affine<C>(
    img: &ImageBuffer<Rgb<u8>, C>,
    m: &Matrix3x2<f32>,
    output_size: u32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> 
where C: Deref<Target = [u8]>
{
    let mat = Matrix3::new(
        m[(0, 0)],
        m[(0, 1)],
        m[(2, 0)],
        m[(1, 0)],
        m[(1, 1)],
        m[(2, 1)],
        0.0,
        0.0,
        1.0,
    );

    let inv = mat.try_inverse().unwrap_or_else(Matrix3::identity);
    let ia = inv[(0, 0)];
    let ib = inv[(0, 1)];
    let itx = inv[(0, 2)];
    let ic = inv[(1, 0)];
    let id = inv[(1, 1)];
    let ity = inv[(1, 2)];

    let (orig_w, orig_h) = img.dimensions();
    let mut output = ImageBuffer::new(output_size, output_size);

    for py in 0..output_size {
        let py_f = py as f32;
        for px in 0..output_size {
            let px_f = px as f32;
            let sx = ia * px_f + ib * py_f + itx;
            let sy = ic * px_f + id * py_f + ity;
            output.put_pixel(px, py, bilinear_sample(img, sx, sy, orig_w, orig_h));
        }
    }
    output
}

/// Lấy mẫu nội suy song tuyến tính (Bilinear Interpolation)
#[inline]
fn bilinear_sample<C>(img: &ImageBuffer<Rgb<u8>, C>, x: f32, y: f32, w: u32, h: u32) -> Rgb<u8> 
where C: Deref<Target = [u8]>
{
    if x < 0.0 || y < 0.0 || x >= w as f32 || y >= h as f32 {
        return Rgb([0, 0, 0]);
    }
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let p00 = img.get_pixel(x0, y0);
    let p10 = img.get_pixel(x1, y0);
    let p01 = img.get_pixel(x0, y1);
    let p11 = img.get_pixel(x1, y1);

    Rgb([
        bilerp(p00[0], p10[0], p01[0], p11[0], fx, fy),
        bilerp(p00[1], p10[1], p01[1], p11[1], fx, fy),
        bilerp(p00[2], p10[2], p01[2], p11[2], fx, fy),
    ])
}

#[inline]
fn bilerp(c00: u8, c10: u8, c01: u8, c11: u8, fx: f32, fy: f32) -> u8 {
    let top = (f32::from(c10) - f32::from(c00)).mul_add(fx, f32::from(c00));
    let bot = (f32::from(c11) - f32::from(c01)).mul_add(fx, f32::from(c01));
    (bot - top).mul_add(fy, top) as u8
}
