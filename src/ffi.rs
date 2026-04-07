//! Cầu nối Apache Arrow Zero Copy giữa Rust <> Python
//!
//! Triển khai thủ công Apache Arrow C Data Interface:
//! - Không sao chép dữ liệu khi truyền kết quả detection
//! - Truyền chỉ 2 con trỏ C qua biên giới ngôn ngữ
//! - Python import trực tiếp từ địa chỉ bộ nhớ
//! - Overhead giao tiếp gần như bằng 0
//!
//! Đây là phần quan trọng nhất làm cho dự án này nhanh hơn tất cả các dự án khác.

use arrow::array::ArrayData;
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use pyo3::ffi::{PyCapsule_GetPointer, PyCapsule_New};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

/// Tên chuẩn định dạng PyCapsule theo yêu cầu của PyArrow
/// Không được thay đổi giá trị này
const ARRAY_NAME: &[u8] = b"arrow_array\0";
const SCHEMA_NAME: &[u8] = b"arrow_schema\0";

/// Export ArrayData Arrow thành 2 PyCapsule truyền sang Python
/// Không có sao chép dữ liệu, chỉ truyền ownership con trỏ
pub fn export_to_python<'py>(
    py: Python<'py>,
    data: ArrayData,
) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
    let ffi_array = FFI_ArrowArray::new(&data);
    let ffi_schema = FFI_ArrowSchema::try_from(data.data_type()).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Arrow FFI Error: {}", e))
    })?;

    let array_capsule = unsafe {
        let ptr = Box::into_raw(Box::new(ffi_array));
        let cap_ptr = PyCapsule_New(
            ptr as *mut _,
            ARRAY_NAME.as_ptr() as *const _,
            Some(release_array),
        );
        Bound::from_owned_ptr(py, cap_ptr).cast_into_unchecked::<PyCapsule>()
    };

    let schema_capsule = unsafe {
        let ptr = Box::into_raw(Box::new(ffi_schema));
        let cap_ptr = PyCapsule_New(
            ptr as *mut _,
            SCHEMA_NAME.as_ptr() as *const _,
            Some(release_schema),
        );
        Bound::from_owned_ptr(py, cap_ptr).cast_into_unchecked::<PyCapsule>()
    };

    Ok((array_capsule, schema_capsule))
}

unsafe extern "C" fn release_array(capsule: *mut pyo3::ffi::PyObject) {
    let ptr = unsafe { PyCapsule_GetPointer(capsule, ARRAY_NAME.as_ptr() as *const _) };
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr as *mut FFI_ArrowArray);
        };
        // Destructor FFI_ArrowArray tự động gọi callback giải phóng bộ nhớ
        // Không cần tự do thủ công các buffer bên trong
    }
}

unsafe extern "C" fn release_schema(capsule: *mut pyo3::ffi::PyObject) {
    let ptr = unsafe { PyCapsule_GetPointer(capsule, SCHEMA_NAME.as_ptr() as *const _) };
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr as *mut FFI_ArrowSchema);
        };
        // Destructor FFI_ArrowSchema tự động giải phóng toàn bộ bộ nhớ schema
    }
}
