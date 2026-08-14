use std::ffi::CString;
use std::path::PathBuf;
use vbuf_ml::consumer_ffi::{vbuf_ml_consumer_add_bos, vbuf_ml_consumer_close, vbuf_ml_consumer_merge_count, vbuf_ml_consumer_merge_pair, vbuf_ml_consumer_open, vbuf_ml_consumer_tensor_count, vbuf_ml_consumer_tensor_info, vbuf_ml_consumer_token_text, VbufMlTensorInfo};

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
    let mut token = vec![0i8; 4097];
    assert_eq!(unsafe { vbuf_ml_consumer_token_text(handle, 0, token.as_mut_ptr(), token.len()) }, 0);
    let mut merges = 0;
    assert_eq!(unsafe { vbuf_ml_consumer_merge_count(handle, &mut merges) }, 0);
    assert_eq!(merges, 151_387);
    let mut left = 0; let mut right = 0;
    assert_eq!(unsafe { vbuf_ml_consumer_merge_pair(handle, 0, &mut left, &mut right) }, 0);
    let mut add_bos = true;
    assert_eq!(unsafe { vbuf_ml_consumer_add_bos(handle, &mut add_bos) }, 0);
    assert!(!add_bos);
    unsafe { vbuf_ml_consumer_close(handle); }
}
