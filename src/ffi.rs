//! Manual Arrow C Data Interface bridge for Zero-copy transfer (macOS ARM64).
//! Handles PyCapsules directly to bypass PyArrowType dependency issues.

use arrow::array::ArrayData;
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use pyo3::ffi::{PyCapsule_GetPointer, PyCapsule_New};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

/// Export an Arrow ArrayData to a pair of Python Capsules.
/// Static names for PyCapsules as required by PyArrow C Data Interface
const ARRAY_NAME: &[u8] = b"arrow_array\0";
const SCHEMA_NAME: &[u8] = b"arrow_schema\0";

/// Export an Arrow ArrayData to a pair of Python Capsules.
pub fn export_to_python<'py>(py: Python<'py>, data: ArrayData) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
    let ffi_array = FFI_ArrowArray::new(&data);
    let ffi_schema = FFI_ArrowSchema::try_from(data.data_type())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Arrow FFI Error: {}", e)))?;

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
        unsafe { let _ = Box::from_raw(ptr as *mut FFI_ArrowArray); };
        // The drop implementation of FFI_ArrowArray in arrow-rs 
        // calls the release callback if it's set.
    }
}

unsafe extern "C" fn release_schema(capsule: *mut pyo3::ffi::PyObject) {
    let ptr = unsafe { PyCapsule_GetPointer(capsule, SCHEMA_NAME.as_ptr() as *const _) };
    if !ptr.is_null() {
        unsafe { let _ = Box::from_raw(ptr as *mut FFI_ArrowSchema); };
        // The drop implementation of FFI_ArrowSchema in arrow-rs 
        // calls the release callback if it's set.
    }
}

