//! Minimal C ABI for the pinned-consumer adapter boundary.
//! No llama.cpp or GGML headers are used here.

use crate::{BorrowedModel, ConsumerTensorType, RuntimeTokenizerIndexes};
use std::ffi::{c_char, CStr};

pub struct VbufMlConsumerHandle {
    // Borrowed tables/indexes are declared before the owner so they drop first.
    runtime_indexes: std::sync::OnceLock<RuntimeTokenizerIndexes<'static>>,
    token_views: std::sync::OnceLock<Vec<VbufMlTokenView>>,
    merge_views: std::sync::OnceLock<Vec<VbufMlMergeView>>,
    tensor_views: std::sync::OnceLock<Vec<VbufMlTensorView>>,
    model: BorrowedModel,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VbufMlTokenView {
    pub text: *const c_char,
    pub text_len: u64,
    pub score: f32,
    pub token_type: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VbufMlMergeView { pub left: u64, pub right: u64 }

#[repr(C)]
pub struct VbufMlTokenArrays {
    pub text: *const u8,
    pub text_len: u64,
    pub offsets: *const u8,
    pub offset_count: u64,
    pub types: *const u8,
    pub type_bytes: u64,
    pub scores: *const u8,
    pub score_bytes: u64,
    pub token_count: u64,
}

#[repr(C)]
pub struct VbufMlMergeArrays {
    pub left: *const u8,
    pub right: *const u8,
    pub merge_count: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VbufMlTensorView {
    pub name: *const c_char,
    pub name_len: u64,
    pub representation: u8,
    pub rank: u8,
    pub dimensions: *const u64,
    pub payload: *const u8,
    pub payload_len: u64,
}

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
        let Ok(model) = BorrowedModel::open(path) else { return std::ptr::null_mut(); };
        Box::into_raw(Box::new(VbufMlConsumerHandle { model, token_views: std::sync::OnceLock::new(), merge_views: std::sync::OnceLock::new(), tensor_views: std::sync::OnceLock::new(), runtime_indexes: std::sync::OnceLock::new() }))
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
pub unsafe extern "C" fn vbuf_ml_consumer_tensor_physical_range(handle: *const VbufMlConsumerHandle, index: u64, offset: *mut u64, length: *mut u64) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() || offset.is_null() || length.is_null() { return INVALID_ARGUMENT; }
        let Some(tensor) = (*handle).model.view.directory.tensors().get(index as usize) else { return VALIDATION_ERROR; };
        *offset = tensor.range.offset(); *length = tensor.range.length(); OK
    })).unwrap_or(VALIDATION_ERROR)
}

/// # Safety
/// `handle`, `info`, and `name_buffer` must be valid for the duration of the call.
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

/// # Safety
/// `handle` and `count` must be valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_token_count(handle: *const VbufMlConsumerHandle, count: *mut u64) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() || count.is_null() { return INVALID_ARGUMENT; }
        match (*handle).model.tokenizer_count() { Ok(value) => { *count = value; OK }, Err(_) => VALIDATION_ERROR }
    })).unwrap_or(VALIDATION_ERROR)
}

/// # Safety
/// `handle` and `value` must be valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_token_type(handle: *const VbufMlConsumerHandle, index: u64, value: *mut i32) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() || value.is_null() { return INVALID_ARGUMENT; }
        match (*handle).model.token_type(index) { Ok(Some(result)) => { *value = result as i32; OK }, _ => VALIDATION_ERROR }
    })).unwrap_or(VALIDATION_ERROR)
}

/// Return validated canonical tokenizer arrays. No per-token table is built.
/// # Safety
/// `handle`, `arrays` must be valid pointers for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_token_arrays(handle: *const VbufMlConsumerHandle, arrays: *mut VbufMlTokenArrays) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() || arrays.is_null() { return INVALID_ARGUMENT; }
        let tokenizer = &(*handle).model.view().tokenizer;
        let scores = tokenizer.score_bytes().unwrap_or(&[]); let types = tokenizer.type_bytes().unwrap_or(&[]);
        *arrays = VbufMlTokenArrays { text: tokenizer.text_bytes().as_ptr(), text_len: tokenizer.text_bytes().len() as u64, offsets: tokenizer.offset_bytes().as_ptr(), offset_count: (tokenizer.offset_bytes().len() / 8) as u64, types: types.as_ptr(), type_bytes: types.len() as u64, scores: scores.as_ptr(), score_bytes: scores.len() as u64, token_count: tokenizer.token_count() }; OK
    })).unwrap_or(VALIDATION_ERROR)
}

/// Build the reusable runtime-local indexes on first use. The indexes borrow
/// validated canonical bytes and are retained by the mmap-owning handle.
/// # Safety
/// `handle` must be a valid handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_runtime_indexes(handle: *const VbufMlConsumerHandle) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() { return INVALID_ARGUMENT; }
        let handle = &*handle;
        let _ = handle.runtime_indexes.get_or_init(|| RuntimeTokenizerIndexes::build(&handle.model.view().tokenizer).expect("validated tokenizer runtime index"));
        OK
    })).unwrap_or(VALIDATION_ERROR)
}

/// Resolve a token byte span through the runtime-local borrowed-key index.
/// # Safety
/// `handle`, `bytes`, and `token_id` must be valid; `bytes` must contain
/// `length` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_token_id(handle: *const VbufMlConsumerHandle, bytes: *const u8, length: usize, token_id: *mut u32) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() || bytes.is_null() || token_id.is_null() { return INVALID_ARGUMENT; }
        let handle = &*handle; let indexes = handle.runtime_indexes.get_or_init(|| RuntimeTokenizerIndexes::build(&handle.model.view().tokenizer).expect("validated tokenizer runtime index"));
        let key = std::slice::from_raw_parts(bytes, length); match indexes.token.lookup(key) { Some(value) => { *token_id = value; OK }, None => VALIDATION_ERROR }
    })).unwrap_or(VALIDATION_ERROR)
}

/// Resolve a numeric merge pair through the packed runtime-local index.
/// # Safety
/// `handle` and `rank` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_merge_rank(handle: *const VbufMlConsumerHandle, left: u64, right: u64, rank: *mut u32) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() || rank.is_null() { return INVALID_ARGUMENT; }
        let handle = &*handle; let indexes = handle.runtime_indexes.get_or_init(|| RuntimeTokenizerIndexes::build(&handle.model.view().tokenizer).expect("validated tokenizer runtime index")); match indexes.merge.lookup(left, right) { Some(value) => { *rank = value; OK }, None => VALIDATION_ERROR }
    })).unwrap_or(VALIDATION_ERROR)
}

/// Return validated canonical numeric merge arrays. No per-merge table is built.
/// # Safety
/// `handle`, `arrays` must be valid pointers for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_merge_arrays(handle: *const VbufMlConsumerHandle, arrays: *mut VbufMlMergeArrays) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() || arrays.is_null() { return INVALID_ARGUMENT; }
        let tokenizer = &(*handle).model.view().tokenizer;
        let left = tokenizer.merge_left_bytes().unwrap_or(&[]); let right = tokenizer.merge_right_bytes().unwrap_or(&[]);
        *arrays = VbufMlMergeArrays { left: left.as_ptr(), right: right.as_ptr(), merge_count: tokenizer.merge_count() }; OK
    })).unwrap_or(VALIDATION_ERROR)
}

/// Return one immutable legacy token table view. Pointers remain valid until close.
/// # Safety
/// `handle`, `views`, and `count` must be valid pointers for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_token_views(handle: *const VbufMlConsumerHandle, views: *mut *const VbufMlTokenView, count: *mut u64) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() || views.is_null() || count.is_null() { return INVALID_ARGUMENT; }
        let handle = &*handle;
        let table = handle.token_views.get_or_init(|| (0..handle.model.view().tokenizer.token_count()).filter_map(|i| handle.model.view().tokenizer.token_text(i).map(|text| VbufMlTokenView { text: text.as_ptr().cast(), text_len: text.len() as u64, score: handle.model.view().tokenizer.score(i).unwrap_or(0.0) as f32, token_type: handle.model.view().tokenizer.token_type(i).unwrap_or(1) as i32 })).collect());
        *views = table.as_ptr(); *count = table.len() as u64; OK
    })).unwrap_or(VALIDATION_ERROR)
}

/// # Safety
/// `handle`, `views`, and `count` must be valid pointers for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_merge_views(handle: *const VbufMlConsumerHandle, views: *mut *const VbufMlMergeView, count: *mut u64) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() || views.is_null() || count.is_null() { return INVALID_ARGUMENT; }
        let handle = &*handle;
        let table = handle.merge_views.get_or_init(|| (0..handle.model.view().tokenizer.merge_count()).filter_map(|i| handle.model.view().tokenizer.merge_pair(i).map(|(left, right)| VbufMlMergeView { left, right })).collect());
        *views = table.as_ptr(); *count = table.len() as u64; OK
    })).unwrap_or(VALIDATION_ERROR)
}

/// # Safety
/// `handle`, `views`, and `count` must be valid pointers for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_tensor_views(handle: *const VbufMlConsumerHandle, views: *mut *const VbufMlTensorView, count: *mut u64) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() || views.is_null() || count.is_null() { return INVALID_ARGUMENT; }
        let handle = &*handle;
        let table = handle.tensor_views.get_or_init(|| handle.model.view().directory.tensors().iter().map(|tensor| VbufMlTensorView { name: tensor.name.as_ptr().cast(), name_len: tensor.name.len() as u64, representation: match tensor.representation { crate::TensorRepresentation::CanonicalPrimitive => 0, crate::TensorRepresentation::Bf16 => 1, crate::TensorRepresentation::GgmlQ8_0 => 2 }, rank: tensor.dimensions.len() as u8, dimensions: tensor.dimensions.as_ptr(), payload: tensor.range.bytes().as_ptr(), payload_len: tensor.range.bytes().len() as u64 }).collect());
        *views = table.as_ptr(); *count = table.len() as u64; OK
    })).unwrap_or(VALIDATION_ERROR)
}

/// # Safety
/// `handle` and `buffer` must be valid pointers for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_architecture(handle: *const VbufMlConsumerHandle, buffer: *mut c_char, capacity: usize) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() || buffer.is_null() { return INVALID_ARGUMENT; }
        let Ok(metadata) = (*handle).model.model_metadata() else { return VALIDATION_ERROR; };
        let bytes = metadata.architecture.as_bytes();
        if bytes.len().checked_add(1).is_none_or(|needed| needed > capacity) { return BUFFER_TOO_SMALL; }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.cast(), bytes.len()); *buffer.add(bytes.len()) = 0; OK
    })).unwrap_or(VALIDATION_ERROR)
}

/// Copy a token's UTF-8 bytes, including a trailing NUL. The buffer is caller-owned.
/// # Safety
/// `handle` and `buffer` must be valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_token_text(handle: *const VbufMlConsumerHandle, index: u64, buffer: *mut c_char, capacity: usize) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() || buffer.is_null() { return INVALID_ARGUMENT; }
        let Ok(Some(text)) = (*handle).model.token_text(index) else { return VALIDATION_ERROR; };
        let bytes = text.as_bytes();
        if bytes.len().checked_add(1).is_none_or(|needed| needed > capacity) { return BUFFER_TOO_SMALL; }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.cast::<u8>(), bytes.len());
        *buffer.add(bytes.len()) = 0;
        OK
    })).unwrap_or(VALIDATION_ERROR)
}

/// # Safety
/// `handle` and `count` must be valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_merge_count(handle: *const VbufMlConsumerHandle, count: *mut u64) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() || count.is_null() { return INVALID_ARGUMENT; }
        match (*handle).model.merge_count() { Ok(value) => { *count = value; OK }, Err(_) => VALIDATION_ERROR }
    })).unwrap_or(VALIDATION_ERROR)
}

/// # Safety
/// `handle`, `left`, and `right` must be valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_merge_pair(handle: *const VbufMlConsumerHandle, index: u64, left: *mut u64, right: *mut u64) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() || left.is_null() || right.is_null() { return INVALID_ARGUMENT; }
        match (*handle).model.merge_pair(index) { Ok(Some((l, r))) => { *left = l; *right = r; OK }, _ => VALIDATION_ERROR }
    })).unwrap_or(VALIDATION_ERROR)
}

/// Copy the chat template as UTF-8 with a trailing NUL.
/// # Safety
/// `handle` and `buffer` must be valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_chat_template(handle: *const VbufMlConsumerHandle, buffer: *mut c_char, capacity: usize) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() || buffer.is_null() { return INVALID_ARGUMENT; }
        let Ok(Some(template)) = (*handle).model.chat_template() else { return VALIDATION_ERROR; };
        let bytes = template.as_bytes(); if bytes.len().checked_add(1).is_none_or(|needed| needed > capacity) { return BUFFER_TOO_SMALL; }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.cast::<u8>(), bytes.len()); *buffer.add(bytes.len()) = 0; OK
    })).unwrap_or(VALIDATION_ERROR)
}

/// # Safety
/// `handle` and `value` must be valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_token_score(handle: *const VbufMlConsumerHandle, index: u64, value: *mut f32) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() || value.is_null() { return INVALID_ARGUMENT; }
        match (*handle).model.token_score(index) { Ok(Some(score)) => { *value = score as f32; OK }, _ => VALIDATION_ERROR }
    })).unwrap_or(VALIDATION_ERROR)
}

/// # Safety
/// `handle` and `value` must be valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_special_token(handle: *const VbufMlConsumerHandle, kind: u8, value: *mut u64) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() || value.is_null() { return INVALID_ARGUMENT; }
        match (*handle).model.special_token(kind) { Ok(Some(id)) => { *value = id; OK }, _ => VALIDATION_ERROR }
    })).unwrap_or(VALIDATION_ERROR)
}

/// `value` is 0 or 1. An absent optional field returns VALIDATION_ERROR.
/// # Safety
/// `handle` and `value` must be valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_ml_consumer_add_bos(handle: *const VbufMlConsumerHandle, value: *mut bool) -> u32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() || value.is_null() { return INVALID_ARGUMENT; }
        match (*handle).model.add_bos() { Ok(Some(result)) => { *value = result; OK }, _ => VALIDATION_ERROR }
    })).unwrap_or(VALIDATION_ERROR)
}
