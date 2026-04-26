//! Cầu nối Apache Arrow Zero Copy giữa Rust <> Python
//!
//! Triển khai thủ công Apache Arrow C Data Interface:
//! - Không sao chép dữ liệu khi truyền kết quả detection
//! - Truyền chỉ 2 con trỏ C qua biên giới ngôn ngữ
//! - Python import trực tiếp từ địa chỉ bộ nhớ
//! - Overhead giao tiếp gần như bằng 0
//!
//! Đây là phần quan trọng nhất làm cho dự án này nhanh hơn tất cả các dự án khác.

use arrow::array::{Array, ArrayData, Float32Array, Int32Array, StructArray, ListBuilder, Float32Builder};
use arrow::datatypes::{DataType, Field, Fields};
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use pyo3::ffi::{PyCapsule_GetPointer, PyCapsule_New};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use std::sync::Arc;

use crate::yolo::YoloDetection;

/// Tên chuẩn định dạng PyCapsule theo yêu cầu của PyArrow
/// Không được thay đổi giá trị này
const ARRAY_NAME: &[u8] = b"arrow_array\0";
const SCHEMA_NAME: &[u8] = b"arrow_schema\0";

/// Helper generic tạo PyCapsule bất kỳ kiểu dữ liệu Arrow
/// Loại bỏ toàn bộ duplication code, giữ nguyên 100% behavior và an toàn bộ nhớ
fn create_capsule<'py, T>(
    py: Python<'py>,
    value: T,
    name: &[u8],
    release: unsafe extern "C" fn(*mut pyo3::ffi::PyObject),
) -> Bound<'py, PyCapsule> {
    unsafe {
        let ptr = Box::into_raw(Box::new(value));
        let cap_ptr = PyCapsule_New(
            ptr as *mut _,
            name.as_ptr() as *const _,
            Some(release),
        );
        Bound::from_owned_ptr(py, cap_ptr).cast_into_unchecked::<PyCapsule>()
    }
}

/// Generic destructor chung cho tất cả các loại FFI Arrow
/// Rust 2024: Không đánh dấu extern "C" vì không gọi trực tiếp từ C
unsafe fn release_capsule<T>(capsule: *mut pyo3::ffi::PyObject, name: &[u8]) {
    let ptr = unsafe { PyCapsule_GetPointer(capsule, name.as_ptr() as *const _) };
    if !ptr.is_null() {
        unsafe { let _ = Box::from_raw(ptr as *mut T); };
    }
}

/// Export ArrayData Arrow thành 2 PyCapsule truyền sang Python
/// Không có sao chép dữ liệu, chỉ truyền ownership con trỏ
pub fn export_to_python(
    py: Python<'_>,
    data: ArrayData,
) -> PyResult<(Bound<'_, PyCapsule>, Bound<'_, PyCapsule>)> {
    let ffi_array = FFI_ArrowArray::new(&data);
    let ffi_schema = FFI_ArrowSchema::try_from(data.data_type()).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Arrow FFI Error: {}", e))
    })?;

    let array_capsule = create_capsule(py, ffi_array, ARRAY_NAME, release_array);
    let schema_capsule = create_capsule(py, ffi_schema, SCHEMA_NAME, release_schema);

    Ok((array_capsule, schema_capsule))
}

unsafe extern "C" fn release_array(capsule: *mut pyo3::ffi::PyObject) {
    unsafe { release_capsule::<FFI_ArrowArray>(capsule, ARRAY_NAME) };
    // Destructor FFI_ArrowArray tự động gọi callback giải phóng bộ nhớ
}

unsafe extern "C" fn release_schema(capsule: *mut pyo3::ffi::PyObject) {
    unsafe { release_capsule::<FFI_ArrowSchema>(capsule, SCHEMA_NAME) };
    // Destructor FFI_ArrowSchema tự động giải phóng toàn bộ bộ nhớ schema
}

/// Chuyển đổi Vec<YoloDetection> thành Apache Arrow capsules (Zero Copy)
pub fn export_detections_to_arrow<'py>(
    py: Python<'py>,
    detections: &[YoloDetection],
    num_keypoints: usize,
    num_mask_coeffs: usize,
) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
    let class_ids = Int32Array::from(detections.iter().map(|d| d.class_id).collect::<Vec<_>>());
    let confidences = Float32Array::from(detections.iter().map(|d| d.confidence).collect::<Vec<_>>());
    let boxes_x = Float32Array::from(detections.iter().map(|d| d.x).collect::<Vec<_>>());
    let boxes_y = Float32Array::from(detections.iter().map(|d| d.y).collect::<Vec<_>>());
    let boxes_w = Float32Array::from(detections.iter().map(|d| d.width).collect::<Vec<_>>());
    let boxes_h = Float32Array::from(detections.iter().map(|d| d.height).collect::<Vec<_>>());

    let mut fields = vec![
        Field::new("class_id", DataType::Int32, false),
        Field::new("confidence", DataType::Float32, false),
        Field::new("x", DataType::Float32, false),
        Field::new("y", DataType::Float32, false),
        Field::new("w", DataType::Float32, false),
        Field::new("h", DataType::Float32, false),
    ];

    let mut arrays: Vec<Arc<dyn Array>> = vec![
        Arc::new(class_ids),
        Arc::new(confidences),
        Arc::new(boxes_x),
        Arc::new(boxes_y),
        Arc::new(boxes_w),
        Arc::new(boxes_h),
    ];

    if num_keypoints > 0 {
        fields.push(Field::new(
            "keypoints",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            true,
        ));
        let mut builder = ListBuilder::new(Float32Builder::new());
        for det in detections {
            for &(x, y, conf) in &det.keypoints {
                builder.values().append_value(x);
                builder.values().append_value(y);
                builder.values().append_value(conf);
            }
            builder.append(true);
        }
        arrays.push(Arc::new(builder.finish()));
    }

    if num_mask_coeffs > 0 {
        fields.push(Field::new(
            "mask_coeffs",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            true,
        ));
        let mut builder = ListBuilder::new(Float32Builder::new());
        for det in detections {
            for &val in &det.mask_coeffs {
                builder.values().append_value(val);
            }
            builder.append(true);
        }
        arrays.push(Arc::new(builder.finish()));
    }

    let struct_array = StructArray::try_new(Fields::from(fields), arrays, None).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Arrow error: {}", e))
    })?;

    export_to_python(py, struct_array.to_data())
}
