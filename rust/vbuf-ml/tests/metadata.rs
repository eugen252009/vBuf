use std::io::Cursor;
use vbuf_core::v06::parse_v06;
use vbuf_core::writer::{BlockOptions, VBufV06Writer};
use vbuf_ml::bootstrap::{encode_payload as encode_bootstrap, Bootstrap, BootstrapEntry, BOOTSTRAP_KEY_ID};
use vbuf_ml::metadata::{encode_payload, MetadataEntry, ModelMetadataKey};
use vbuf_ml::{MlErrorCode, ModelMetadata};

fn model_file(metadata: &[u8], context: u32) -> Vec<u8> {
    let mut writer = VBufV06Writer::new_known_size(Cursor::new(Vec::new()), 3).unwrap();
    writer.write_opaque(BlockOptions::array(BOOTSTRAP_KEY_ID), &encode_bootstrap(&[
        BootstrapEntry::new(1, true, 12, 0),
        BootstrapEntry::new(2, true, 10, 0),
    ]).unwrap()).unwrap();
    writer.write_opaque(BlockOptions::array(10), metadata).unwrap();
    writer.write_opaque(BlockOptions::array(12), b"directory").unwrap();
    writer.write_opaque(BlockOptions::array(20), b"llama").unwrap();
    writer.write_u32(BlockOptions::scalar(21), &[context]).unwrap();
    writer.write_u32(BlockOptions::scalar(22), &[4096]).unwrap();
    writer.write_u32(BlockOptions::scalar(23), &[32]).unwrap();
    writer.write_u32(BlockOptions::scalar(24), &[32]).unwrap();
    writer.finish().unwrap().into_inner()
}

fn required_metadata() -> Vec<u8> {
    encode_payload(&[
        MetadataEntry::new(1, true, 20, 0),
        MetadataEntry::new(2, true, 21, 0),
        MetadataEntry::new(3, true, 22, 0),
        MetadataEntry::new(4, true, 23, 0),
        MetadataEntry::new(5, true, 24, 0),
    ]).unwrap()
}

fn parse_metadata(metadata: &[u8], context: u32) -> Result<ModelMetadata<'static>, MlErrorCode> {
    let bytes: &'static [u8] = Box::leak(model_file(metadata, context).into_boxed_slice());
    let validated = parse_v06(bytes).map_err(|error| MlErrorCode::Canonical(error.code))?;
    let bootstrap = Bootstrap::discover(&validated).map_err(|error| error.code)?;
    ModelMetadata::parse(&validated, &bootstrap).map_err(|error| error.code)
}

#[test]
fn canonical_metadata_fixture_is_self_contained() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/minimal-model-metadata.vbuf");
    let bytes: &'static [u8] = Box::leak(std::fs::read(path).unwrap().into_boxed_slice());
    let validated = parse_v06(bytes).unwrap();
    let bootstrap = Bootstrap::discover(&validated).unwrap();
    assert_eq!(ModelMetadata::parse(&validated, &bootstrap).unwrap().architecture(), Some("llama"));
}

#[test]
fn required_model_metadata_reuses_canonical_values() {
    let metadata = parse_metadata(&required_metadata(), 2048).unwrap();
    assert_eq!(metadata.architecture(), Some("llama"));
    assert_eq!(metadata.unsigned(ModelMetadataKey::ContextLength), Some(2048));
    assert_eq!(metadata.unsigned(ModelMetadataKey::LayerCount), Some(32));
    assert_eq!(metadata.fields().len(), 5);
    assert_eq!(metadata.get(ModelMetadataKey::ContextLength).unwrap().range.bytes().len(), 4);
}

#[test]
fn optional_unknown_and_required_metadata_rules_are_deterministic() {
    let extra = MetadataEntry::new(99, false, 20, 0);
    let mut entries = vec![
        MetadataEntry::new(1, true, 20, 0), MetadataEntry::new(2, true, 21, 0),
        MetadataEntry::new(3, true, 22, 0), MetadataEntry::new(4, true, 23, 0),
        MetadataEntry::new(5, true, 24, 0), extra,
    ];
    let optional = encode_payload(&entries).unwrap();
    assert_eq!(parse_metadata(&optional, 1).unwrap().fields().len(), 5);
    entries[5] = MetadataEntry::new(99, true, 20, 0);
    let required = encode_payload(&entries).unwrap();
    assert_eq!(parse_metadata(&required, 1).unwrap_err(), MlErrorCode::UnsupportedMetadataKey);
}

#[test]
fn wrong_type_missing_required_and_out_of_range_values_fail() {
    let mut wrong_type = required_metadata();
    // Entry for ContextLength is the second 8-byte entry; replace its generic reference with the text block.
    wrong_type[16 + 8 + 4..16 + 8 + 6].copy_from_slice(&20u16.to_le_bytes());
    assert_eq!(parse_metadata(&wrong_type, 1).unwrap_err(), MlErrorCode::MetadataTypeMismatch);

    let missing = encode_payload(&[
        MetadataEntry::new(1, true, 20, 0), MetadataEntry::new(2, true, 21, 0),
        MetadataEntry::new(3, true, 22, 0), MetadataEntry::new(4, true, 23, 0),
    ]).unwrap();
    assert_eq!(parse_metadata(&missing, 1).unwrap_err(), MlErrorCode::MissingRequiredMetadata);
    assert_eq!(parse_metadata(&required_metadata(), 0).unwrap_err(), MlErrorCode::MetadataValueOutOfRange);
}

#[test]
fn malformed_and_duplicate_metadata_are_rejected() {
    let metadata = required_metadata();
    assert_eq!(parse_metadata(&metadata[..metadata.len() - 1], 1).unwrap_err(), MlErrorCode::MalformedModelMetadata);
    let duplicate = [MetadataEntry::new(1, true, 20, 0), MetadataEntry::new(1, true, 20, 0)];
    assert_eq!(encode_payload(&duplicate).unwrap_err().code, MlErrorCode::DuplicateMetadataKey);
}
