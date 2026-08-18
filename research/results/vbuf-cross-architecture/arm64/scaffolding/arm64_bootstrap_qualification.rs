use memmap2::Mmap;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::ffi::CString;
use vbuf_core::v06::parse_v06;
use vbuf_ml::consumer_ffi::{vbuf_ml_consumer_close, vbuf_ml_consumer_open_metadata, vbuf_ml_consumer_tensor_info_with_bytes, VbufMlTensorInfo};
use vbuf_ml::{parse_source_profile, BorrowedModelView, Bootstrap, MetadataValue};

fn digest_view(view: &BorrowedModelView<'_>) -> String {
    let mut digest = Sha256::new();
    for field in view.metadata.fields() {
        digest.update((field.key as u16).to_le_bytes());
        match &field.value {
            MetadataValue::Text(value) => { digest.update([1]); digest.update((value.len() as u64).to_le_bytes()); digest.update(value.as_bytes()); }
            MetadataValue::Unsigned(value) => { digest.update([2]); digest.update(value.to_le_bytes()); }
            MetadataValue::Float(value) => { digest.update([3]); digest.update(value.to_le_bytes()); }
        }
    }
    for tensor in view.directory.tensors() {
        digest.update((tensor.name.len() as u64).to_le_bytes()); digest.update(tensor.name.as_bytes());
        digest.update([tensor.representation as u8]); digest.update((tensor.dimensions.len() as u64).to_le_bytes());
        for dimension in &tensor.dimensions { digest.update(dimension.to_le_bytes()); }
        digest.update(tensor.payload.offset().to_le_bytes()); digest.update(tensor.payload.length().to_le_bytes());
    }
    digest.update(view.tokenizer.token_count().to_le_bytes());
    for index in 0..view.tokenizer.token_count() {
        let text = view.tokenizer.token_text(index).unwrap_or("").as_bytes(); digest.update((text.len() as u64).to_le_bytes()); digest.update(text);
        digest.update(view.tokenizer.token_type(index).unwrap_or(u64::MAX).to_le_bytes());
        digest.update(view.tokenizer.score(index).unwrap_or(f64::NAN).to_le_bytes());
    }
    digest.update(view.tokenizer.merge_count().to_le_bytes());
    for index in 0..view.tokenizer.merge_count() { let pair = view.tokenizer.merge_pair(index).unwrap_or((u64::MAX, u64::MAX)); digest.update(pair.0.to_le_bytes()); digest.update(pair.1.to_le_bytes()); }
    for (kind, value) in view.tokenizer.specials() { digest.update([*kind as u8]); digest.update(value.to_le_bytes()); }
    digest.update([u8::from(view.tokenizer.add_bos().unwrap_or(false))]);
    if let Some(template) = view.tokenizer.chat_template() { digest.update((template.len() as u64).to_le_bytes()); digest.update(template.as_bytes()); }
    format!("{:x}", digest.finalize())
}

fn main() {
    let path = std::env::args().nth(1).expect("bootstrap path");
    let file = File::open(path).expect("open bootstrap");
    let mapping = unsafe { Mmap::map(&file).expect("map bootstrap") };
    let validated = parse_v06(&mapping).expect("canonical parse");
    let bootstrap = Bootstrap::discover(&validated).expect("bootstrap");
    let profile = parse_source_profile(&validated, &bootstrap).expect("source profile").expect("source profile role");
    let view = BorrowedModelView::parse_with_sources(&mapping, &profile.registry, &[]).expect("semantic parse");
    let selected = &view.directory.tensors()[22];
    let end = selected.payload.offset().checked_add(selected.payload.length()).expect("u64 end overflow");
    assert!(selected.payload.offset() > u32::MAX as u64);
    assert!(usize::try_from(selected.payload.length()).is_ok());
    let range_path = std::env::args().nth(2).expect("materialized range path");
    let materialized = std::fs::read(range_path).expect("read materialized range");
    let bootstrap_c = CString::new(std::env::args().nth(1).unwrap()).unwrap();
    let handle = unsafe { vbuf_ml_consumer_open_metadata(bootstrap_c.as_ptr()) };
    assert!(!handle.is_null());
    let mut info = VbufMlTensorInfo { representation: 0, rank: 0, dimensions: [0; 16], payload: std::ptr::null(), payload_len: 0 };
    let mut name = vec![0u8; 256];
    let status = unsafe { vbuf_ml_consumer_tensor_info_with_bytes(handle, 22, materialized.as_ptr(), materialized.len() as u64, &mut info, name.as_mut_ptr().cast(), name.len()) };
    assert_eq!(status, 0);
    assert_eq!(info.payload_len, materialized.len() as u64);
    assert!(!info.payload.is_null());
    unsafe { vbuf_ml_consumer_close(handle); }
    println!("{{\"bootstrap_bytes\":{},\"blocks\":{},\"architecture\":\"{}\",\"tensors\":{},\"tokens\":{},\"merges\":{},\"discovery_digest\":\"{}\",\"payload_bytes_fetched_during_discovery\":0,\"full_source_opened_during_discovery\":false,\"selected_tensor_ordinal\":22,\"source_id\":{},\"logical_offset\":{},\"logical_length\":{},\"logical_end\":{},\"usize_truncation\":false,\"materialized_bytes\":{},\"materialized_pointer_bound\":true}}", mapping.len(), validated.blocks().len(), view.metadata.architecture().unwrap_or(""), view.directory.tensors().len(), view.tokenizer.token_count(), view.tokenizer.merge_count(), digest_view(&view), selected.payload.source_id().value(), selected.payload.offset(), selected.payload.length(), end, materialized.len());
}
