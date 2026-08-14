//! Minimal C ABI for the pinned-consumer adapter boundary.
//! No llama.cpp or GGML headers are used here.

use crate::{ConsumerModel, ConsumerTensorType};
use std::ffi::{c_char, CStr};

pub struct VbufMlConsumerHandle { model: ConsumerModel }

#[repr(C)]
pub struct VbufMlTensorInfo {
    pub representation: u8,
    pub rank: u8,
    pub dimensions: [u64; 16],
    pub payload: *const u8,
    pub payload_len: u64,
}

#[repr(C)]
pub struct VbufMlModelMetadataInfo {
    pub context_length: u64,
    pub embedding_length: u64,
    pub layer_count: u64,
    pub head_count: u64,
    pub kv_head_count: u64,
    pub key_head_dimension: u64,
    pub value_head_dimension: u64,
    pub feed_forward_length: u64,
    pub normalization_epsilon: f64,
    pub rope_theta: f64,
}

const OK: u32 = 0;
const INVALID_ARGUMENT: u32 = 1;
const VALIDATION_ERROR: u32 = 2;
const BUFFER_TOO_SMALL: u32 = 3;

/// # Safety
/// `path` must point to a valid NUL-terminated UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_open(path: *const c_char) -> *mut VbufMlConsumerHandle {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if path.is_null() { return std::ptr::null_mut(); }
        let Ok(path) = CStr::from_ptr(path).to_str() else { return std::ptr::null_mut(); };
        let Ok(model) = ConsumerModel::open(path) else { return std::ptr::null_mut(); };
        Box::into_raw(Box::new(VbufMlConsumerHandle { model }))
    })).unwrap_or(std::ptr::null_mut())
}

/// # Safety
/// `handle` must be a pointer returned by `vbuf_ml_consumer_open` and not
/// previously closed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_close(handle: *mut VbufMlConsumerHandle) {
    if !handle.is_null() { unsafe { drop(Box::from_raw(handle)); } }
}

/// # Safety
/// Pointers must be valid for writes when non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_tensor_count(handle: *const VbufMlConsumerHandle, count: *mut u64) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() || count.is_null() { return INVALID_ARGUMENT; }
        match (*handle).model.tensor_count() { Ok(value) => { *count = value as u64; OK }, Err(_) => VALIDATION_ERROR }
    })).unwrap_or(VALIDATION_ERROR)
}

/// # Safety
/// `handle`, `info`, and `name_buffer` must be valid for the duration of the
/// call; `name_buffer` must have `name_capacity` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_tensor_info(handle: *const VbufMlConsumerHandle, index: u64, info: *mut VbufMlTensorInfo, name_buffer: *mut c_char, name_capacity: usize) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() || info.is_null() || name_buffer.is_null() { return INVALID_ARGUMENT; }
        let model = &(*handle).model; let Ok(Some(name)) = model.tensor_name(index as usize) else { return VALIDATION_ERROR; }; let Ok(Some(shape)) = model.tensor_shape(index as usize) else { return VALIDATION_ERROR; }; let Ok(Some(kind)) = model.tensor_type(index as usize) else { return VALIDATION_ERROR; }; let Ok(Some(payload)) = model.tensor_payload(index as usize) else { return VALIDATION_ERROR; };
        let bytes = name.as_bytes(); if bytes.len().checked_add(1).is_none_or(|needed| needed > name_capacity) { return BUFFER_TOO_SMALL; }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), name_buffer.cast::<u8>(), bytes.len()); *name_buffer.add(bytes.len()) = 0;
        if shape.len() > 16 { return VALIDATION_ERROR; }
        (*info).representation = match kind { ConsumerTensorType::F32 => 0, ConsumerTensorType::Bf16 => 1, ConsumerTensorType::Q8_0 => 2 }; (*info).rank = shape.len() as u8; (*info).dimensions = [0; 16]; (&mut (*info).dimensions)[..shape.len()].copy_from_slice(&shape); (*info).payload = payload.as_ptr(); (*info).payload_len = payload.len() as u64; OK
    })).unwrap_or(VALIDATION_ERROR)
}

/// # Safety
/// `handle` and `info` must be valid pointers for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_metadata(handle: *const VbufMlConsumerHandle, info: *mut VbufMlModelMetadataInfo) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() || info.is_null() { return INVALID_ARGUMENT; }
        let Ok(value) = (*handle).model.model_metadata() else { return VALIDATION_ERROR; };
        *info = VbufMlModelMetadataInfo { context_length: value.context_length, embedding_length: value.embedding_length, layer_count: value.layer_count, head_count: value.head_count, kv_head_count: value.kv_head_count, key_head_dimension: value.key_head_dimension, value_head_dimension: value.value_head_dimension, feed_forward_length: value.feed_forward_length, normalization_epsilon: value.normalization_epsilon, rope_theta: value.rope_theta }; OK
    })).unwrap_or(VALIDATION_ERROR)
}
