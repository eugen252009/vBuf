use std::io::Cursor;
use vbuf_core::v06::parse_v06;
use vbuf_core::writer::{BlockOptions, VBufV06Writer};
use vbuf_ml::MlErrorCode;
use vbuf_ml::bootstrap::{
    BOOTSTRAP_KEY_ID, Bootstrap, BootstrapEntry, encode_payload as encode_bootstrap,
};
use vbuf_ml::integrity::{IntegrityEntry, IntegrityMetadata, digest_payload, encode_payload};

fn file(integrity: Option<&[u8]>, target: &[u8]) -> Vec<u8> {
    let mut writer = VBufV06Writer::new_known_size(Cursor::new(Vec::new()), 3).unwrap();
    let mut bootstrap_entries = vec![
        BootstrapEntry::new(1, true, 12, 0),
        BootstrapEntry::new(2, true, 10, 0),
    ];
    if integrity.is_some() {
        bootstrap_entries.push(BootstrapEntry::new(4, false, 11, 0));
    }
    writer
        .write_opaque(
            BlockOptions::array(BOOTSTRAP_KEY_ID),
            &encode_bootstrap(&bootstrap_entries).unwrap(),
        )
        .unwrap();
    writer
        .write_opaque(BlockOptions::array(10), b"model")
        .unwrap();
    writer
        .write_opaque(BlockOptions::array(12), b"directory")
        .unwrap();
    if let Some(payload) = integrity {
        writer
            .write_opaque(BlockOptions::array(11), payload)
            .unwrap();
    }
    writer
        .write_opaque(BlockOptions::array(20), target)
        .unwrap();
    writer.finish().unwrap().into_inner()
}

fn parse_integrity(
    integrity: Option<&[u8]>,
    target: &[u8],
) -> Result<Option<IntegrityMetadata<'static>>, MlErrorCode> {
    let bytes: &'static [u8] = Box::leak(file(integrity, target).into_boxed_slice());
    let validated = parse_v06(bytes).map_err(|error| MlErrorCode::Canonical(error.code))?;
    let bootstrap = Bootstrap::discover(&validated).map_err(|error| error.code)?;
    IntegrityMetadata::discover(&validated, &bootstrap).map_err(|error| error.code)
}

#[test]
fn optional_integrity_is_absent_without_changing_model_validity() {
    assert!(parse_integrity(None, b"payload").unwrap().is_none());
}

#[test]
fn payload_only_sha256_verification_is_targeted_and_direct() {
    let digest = digest_payload(b"payload");
    let payload = encode_payload(&[IntegrityEntry::new(20, 0, digest)]).unwrap();
    let index = parse_integrity(Some(&payload), b"payload")
        .unwrap()
        .unwrap();
    assert_eq!(index.records().len(), 1);
    assert_eq!(index.records()[0].range.bytes(), b"payload");
    index.verify_target(20, 0).unwrap();
    assert_eq!(
        index.verify_target(21, 0).unwrap_err().code,
        MlErrorCode::NoIntegrityAvailable
    );
}

#[test]
fn mutation_fails_integrity_but_not_canonical_structure() {
    let digest = digest_payload(b"payload");
    let integrity = encode_payload(&[IntegrityEntry::new(20, 0, digest)]).unwrap();
    let mut bytes = file(Some(&integrity), b"payload");
    let target_offset = {
        let validated = parse_v06(&bytes).unwrap();
        validated
            .blocks()
            .iter()
            .find(|block| block.key_id == 20)
            .unwrap()
            .payload_start as usize
    };
    bytes[target_offset] ^= 1;
    let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let validated = parse_v06(bytes).unwrap();
    let bootstrap = Bootstrap::discover(&validated).unwrap();
    let index = IntegrityMetadata::discover(&validated, &bootstrap)
        .unwrap()
        .unwrap();
    assert_eq!(
        index.verify_target(20, 0).unwrap_err().code,
        MlErrorCode::IntegrityMismatch
    );
}

#[test]
fn malformed_duplicate_and_unknown_integrity_records_fail_closed() {
    let digest = digest_payload(b"payload");
    let duplicate = [
        IntegrityEntry::new(20, 0, digest),
        IntegrityEntry::new(20, 0, digest),
    ];
    assert_eq!(
        encode_payload(&duplicate).unwrap_err().code,
        MlErrorCode::DuplicateIntegrityTarget
    );

    let valid = encode_payload(&[IntegrityEntry::new(20, 0, digest)]).unwrap();
    assert_eq!(
        parse_integrity(Some(&valid[..valid.len() - 1]), b"payload").unwrap_err(),
        MlErrorCode::MalformedIntegrityMetadata
    );
    let mut unknown = valid;
    unknown[20 + 4] = 99;
    assert_eq!(
        parse_integrity(Some(&unknown), b"payload").unwrap_err(),
        MlErrorCode::UnsupportedIntegrityAlgorithm
    );
}
