use std::ffi::CString;
use std::path::PathBuf;
use vbuf_ml::consumer_ffi::{vbuf_ml_consumer_add_bos, vbuf_ml_consumer_close, vbuf_ml_consumer_merge_arrays, vbuf_ml_consumer_merge_count, vbuf_ml_consumer_merge_pair, vbuf_ml_consumer_merge_rank, vbuf_ml_consumer_open, vbuf_ml_consumer_runtime_indexes, vbuf_ml_consumer_tensor_count, vbuf_ml_consumer_tensor_info, vbuf_ml_consumer_tensor_physical_range, vbuf_ml_consumer_token_arrays, vbuf_ml_consumer_token_id, vbuf_ml_consumer_token_text, VbufMlMergeArrays, VbufMlTensorInfo, VbufMlTokenArrays};

#[test]
fn c_bridge_projects_tokenizer_and_tensor_views() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../research-models/Qwen3-0.6B-BF16.vbuf");
    if !path.exists() { return; }
    let path = CString::new(path.to_str().unwrap()).unwrap();
    let handle = unsafe { vbuf_ml_consumer_open(path.as_ptr()) };
    assert!(!handle.is_null());
    let mut tensor_count = 0;
    assert_eq!(unsafe { vbuf_ml_consumer_tensor_count(handle, &mut tensor_count) }, 0);
    assert_eq!(tensor_count, 311);
    let mut name = vec![0i8; 4096];
    let mut info = VbufMlTensorInfo { representation: 0, rank: 0, dimensions: [0; 16], payload: std::ptr::null(), payload_len: 0 };
    assert_eq!(unsafe { vbuf_ml_consumer_tensor_info(handle, 0, &mut info, name.as_mut_ptr(), name.len()) }, 0);
    assert!(info.payload_len > 0);
    let mut physical_offset = 0; let mut physical_length = 0;
    assert_eq!(unsafe { vbuf_ml_consumer_tensor_physical_range(handle, 0, &mut physical_offset, &mut physical_length) }, 0);
    assert!(physical_offset > 0 && physical_length == info.payload_len);
    let mut token = vec![0i8; 4097];
    assert_eq!(unsafe { vbuf_ml_consumer_token_text(handle, 0, token.as_mut_ptr(), token.len()) }, 0);
    let mut token_arrays = VbufMlTokenArrays { text: std::ptr::null(), text_len: 0, offsets: std::ptr::null(), offset_count: 0, types: std::ptr::null(), type_bytes: 0, scores: std::ptr::null(), score_bytes: 0, token_count: 0 };
    assert_eq!(unsafe { vbuf_ml_consumer_token_arrays(handle, &mut token_arrays) }, 0);
    assert_eq!(token_arrays.token_count, 151_936);
    assert_eq!(token_arrays.offset_count, token_arrays.token_count + 1);
    assert!(token_arrays.text_len > 1_000_000 && !token_arrays.text.is_null());
    assert_eq!(unsafe { vbuf_ml_consumer_runtime_indexes(handle) }, 0);
    let mut token_id = u32::MAX;
    let first_end = unsafe { u64::from_le_bytes(*(token_arrays.offsets.cast::<[u8; 8]>().add(1))) } as usize;
    let first = unsafe { std::slice::from_raw_parts(token_arrays.text, first_end) };
    assert_eq!(unsafe { vbuf_ml_consumer_token_id(handle, first.as_ptr(), first.len(), &mut token_id) }, 0);
    assert_eq!(token_id, 0);
    let mut merges = 0;
    assert_eq!(unsafe { vbuf_ml_consumer_merge_count(handle, &mut merges) }, 0);
    assert_eq!(merges, 151_387);
    let mut left = 0; let mut right = 0;
    assert_eq!(unsafe { vbuf_ml_consumer_merge_pair(handle, 0, &mut left, &mut right) }, 0);
    let mut merge_arrays = VbufMlMergeArrays { left: std::ptr::null(), right: std::ptr::null(), merge_count: 0 };
    assert_eq!(unsafe { vbuf_ml_consumer_merge_arrays(handle, &mut merge_arrays) }, 0);
    assert_eq!(merge_arrays.merge_count, 151_387);
    assert!(!merge_arrays.left.is_null() && !merge_arrays.right.is_null());
    let mut expected_left = 0; let mut expected_right = 0;
    assert_eq!(unsafe { vbuf_ml_consumer_merge_pair(handle, 2, &mut expected_left, &mut expected_right) }, 0);
    let mut rank = u32::MAX;
    assert_eq!(unsafe { vbuf_ml_consumer_merge_rank(handle, expected_left, expected_right, &mut rank) }, 0);
    assert_eq!(rank, 2);
    let mut add_bos = true;
    assert_eq!(unsafe { vbuf_ml_consumer_add_bos(handle, &mut add_bos) }, 0);
    assert!(!add_bos);
    unsafe { vbuf_ml_consumer_close(handle); }
}
