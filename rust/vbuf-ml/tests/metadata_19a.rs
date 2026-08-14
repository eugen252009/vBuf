use vbuf_ml::{encode_metadata_payload, MetadataEntry, ModelMetadataKey};

#[test]
fn qwen3_attention_semantics_are_optional_profile_keys() {
    assert_eq!(ModelMetadataKey::from_id(9), Some(ModelMetadataKey::KVHeadCount));
    assert_eq!(ModelMetadataKey::from_id(10), Some(ModelMetadataKey::KeyHeadDimension));
    assert_eq!(ModelMetadataKey::from_id(11), Some(ModelMetadataKey::ValueHeadDimension));
    assert!(!ModelMetadataKey::KVHeadCount.is_required());
    assert!(!ModelMetadataKey::KeyHeadDimension.is_required());
    assert!(!ModelMetadataKey::ValueHeadDimension.is_required());
    let bytes = encode_metadata_payload(&[
        MetadataEntry::new(9, false, 10, 0),
        MetadataEntry::new(10, false, 11, 0),
        MetadataEntry::new(11, false, 12, 0),
    ]).unwrap();
    assert_eq!(u16::from_le_bytes(bytes[12..14].try_into().unwrap()), 3);
}
