use std::io::Cursor;
use vbuf_core::v06::{parse_v06, V06Semantic};
use vbuf_core::writer::{BlockOptions, VBufV06Writer};
use vbuf_ml::bootstrap::{encode_payload as encode_bootstrap, BootstrapEntry, BOOTSTRAP_KEY_ID};
use vbuf_ml::tensor_directory::{encode_payload, TensorEntry};
use vbuf_ml::{representation_contract, Bootstrap, MlErrorCode, TensorDirectory, TensorRepresentation};

fn file(directory: &[u8], tensor_values: &[(&[u8], u16, u64)]) -> Vec<u8> {
    let mut writer = VBufV06Writer::new_known_size(Cursor::new(Vec::new()), 3).unwrap();
    writer.write_opaque(BlockOptions::array(BOOTSTRAP_KEY_ID), &encode_bootstrap(&[
        BootstrapEntry::new(1, true, 10, 0),
        BootstrapEntry::new(2, true, 11, 0),
    ]).unwrap()).unwrap();
    writer.write_opaque(BlockOptions::array(10), directory).unwrap();
    writer.write_opaque(BlockOptions::array(11), b"model").unwrap();
    for (payload, key, count) in tensor_values {
        writer.write_block(BlockOptions::array(*key), V06Semantic::Float, 32, *count, payload).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

fn valid_directory() -> Vec<u8> {
    encode_payload(&[
        TensorEntry::new("layer.0", vec![2, 2], 100, 0),
        TensorEntry::new("scalar", vec![], 101, 0),
    ]).unwrap()
}

fn parsed(directory: &[u8], tensors: &[(&[u8], u16, u64)]) -> Result<TensorDirectory<'static>, MlErrorCode> {
    let bytes = file(directory, tensors);
    // The helper intentionally leaks only test data so the returned checked
    // ranges have the same lifetime as the parsed canonical input.
    let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let validated = parse_v06(bytes).map_err(|error| MlErrorCode::Canonical(error.code))?;
    let bootstrap = Bootstrap::discover(&validated).map_err(|error| error.code)?;
    TensorDirectory::parse(&validated, &bootstrap).map_err(|error| error.code)
}

#[test]
fn canonical_primitive_contract_has_no_duplicate_packed_layout() {
    let contract = representation_contract(TensorRepresentation::CanonicalPrimitive);
    assert_eq!(contract.logical_elements_per_block, None);
    assert_eq!(contract.physical_bytes_per_block, None);
    assert_eq!(contract.required_payload_alignment, 1);
}

#[test]
fn canonical_fixture_resolves_tensor_directory_through_bootstrap() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/minimal-tensor-directory.vbuf");
    let bytes = std::fs::read(path).unwrap();
    let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let validated = parse_v06(bytes).unwrap();
    let bootstrap = Bootstrap::discover(&validated).unwrap();
    let directory = TensorDirectory::parse(&validated, &bootstrap).unwrap();
    assert_eq!(directory.get("layer.0").unwrap().dimensions, vec![2, 2]);
    assert_eq!(directory.get("scalar").unwrap().range.as_ref().unwrap().bytes().len(), 4);
}

#[test]
fn directory_round_trip_reuses_canonical_primitive_semantics() {
    let f32_values = [1.0f32, 2.0, 3.0, 4.0];
    let scalar = [5.0f32];
    let directory = valid_directory();
    let tensor_directory = parsed(&directory, &[
        (bytemuck_bytes(&f32_values), 100, 4),
        (bytemuck_bytes(&scalar), 101, 1),
    ]).unwrap();
    assert_eq!(tensor_directory.tensors().len(), 2);
    assert_eq!(tensor_directory.tensors()[0].name, "layer.0");
    assert_eq!(tensor_directory.get("scalar").unwrap().dimensions, Vec::<u64>::new());
    assert_eq!(tensor_directory.get("layer.0").unwrap().range.as_ref().unwrap().bytes().len(), 16);
}

fn bytemuck_bytes(values: &[f32]) -> &[u8] {
    // The test values are native-independent because only their byte length is
    // consumed by the canonical writer; use explicit little-endian conversion.
    let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
    Box::leak(bytes.into_boxed_slice())
}

#[test]
fn names_are_sorted_and_duplicates_are_rejected() {
    let entries = [TensorEntry::new("z", vec![1], 100, 0), TensorEntry::new("a", vec![1], 101, 0)];
    let encoded = encode_payload(&entries).unwrap();
    assert_eq!(&encoded[20 + 10..20 + 11], b"a");
    let duplicate = [TensorEntry::new("same", vec![1], 100, 0), TensorEntry::new("same", vec![1], 101, 0)];
    assert_eq!(encode_payload(&duplicate).unwrap_err().code, MlErrorCode::DuplicateTensorName);
}

#[test]
fn shape_and_reference_validation_fail_closed() {
    let mismatch = encode_payload(&[TensorEntry::new("bad", vec![3], 100, 0)]).unwrap();
    assert_eq!(parsed(&mismatch, &[(bytemuck_bytes(&[1.0, 2.0]), 100, 2)]).unwrap_err(), MlErrorCode::TensorRepresentationMismatch);

    let missing = encode_payload(&[TensorEntry::new("missing", vec![1], 999, 0)]).unwrap();
    assert_eq!(parsed(&missing, &[(bytemuck_bytes(&[1.0]), 100, 1)]).unwrap_err(), MlErrorCode::TensorReferenceMissing);

    let overflow = encode_payload(&[TensorEntry::new("overflow", vec![u64::MAX, 2], 100, 0)]);
    assert_eq!(overflow.unwrap_err().code, MlErrorCode::ShapeOverflow);
}

#[test]
fn malformed_truncated_reserved_and_unknown_representation_are_rejected() {
    let original = valid_directory();

    let truncated = &original[..original.len() - 1];
    assert_eq!(parsed(truncated, &[(bytemuck_bytes(&[1.0, 2.0, 3.0, 4.0]), 100, 4), (bytemuck_bytes(&[5.0]), 101, 1)]).unwrap_err(), MlErrorCode::MalformedTensorDirectory);

    let mut reserved = original.clone();
    reserved[20 + 8] = 1;
    assert_eq!(parsed(&reserved, &[(bytemuck_bytes(&[1.0, 2.0, 3.0, 4.0]), 100, 4), (bytemuck_bytes(&[5.0]), 101, 1)]).unwrap_err(), MlErrorCode::MalformedTensorDirectory);

    let mut unknown = original.clone();
    unknown[20 + 3] = 99;
    assert_eq!(parsed(&unknown, &[(bytemuck_bytes(&[1.0, 2.0, 3.0, 4.0]), 100, 4), (bytemuck_bytes(&[5.0]), 101, 1)]).unwrap_err(), MlErrorCode::UnsupportedTensorRepresentation);

    let mut invalid_name = original.clone();
    invalid_name[30] = 0xff;
    assert_eq!(parsed(&invalid_name, &[(bytemuck_bytes(&[1.0, 2.0, 3.0, 4.0]), 100, 4), (bytemuck_bytes(&[5.0]), 101, 1)]).unwrap_err(), MlErrorCode::InvalidTensorName);

    let mut invalid_rank = original;
    invalid_rank[22] = 17;
    assert_eq!(parsed(&invalid_rank, &[(bytemuck_bytes(&[1.0, 2.0, 3.0, 4.0]), 100, 4), (bytemuck_bytes(&[5.0]), 101, 1)]).unwrap_err(), MlErrorCode::InvalidRank);
}
