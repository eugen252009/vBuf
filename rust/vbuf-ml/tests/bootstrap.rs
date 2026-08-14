use std::io::Cursor;
use std::path::Path;
use vbuf_core::v06::{parse_v06, V06Semantic};
use vbuf_core::writer::{BlockOptions, VBufV06Writer};
use vbuf_ml::bootstrap::{encode_payload, BootstrapEntry, BOOTSTRAP_KEY_ID};
use vbuf_ml::{Bootstrap, MlErrorCode, RegionRole};

fn stream(entries: &[BootstrapEntry]) -> Vec<u8> {
    let mut writer = VBufV06Writer::new_known_size(Cursor::new(Vec::new()), 3).unwrap();
    writer.write_opaque(BlockOptions::array(BOOTSTRAP_KEY_ID), &encode_payload(entries).unwrap()).unwrap();
    // The referenced values deliberately use generic representations. The ML
    // bootstrap supplies roles, not duplicate semantic/width/count fields.
    writer.write_f32(BlockOptions::array(10), &[1.0, 2.0]).unwrap();
    writer.write_opaque(BlockOptions::array(11), b"metadata").unwrap();
    writer.write_opaque(BlockOptions::array(12), b"tokens").unwrap();
    writer.finish().unwrap().into_inner()
}

fn required_entries() -> [BootstrapEntry; 3] {
    [
        BootstrapEntry::new(RegionRole::TensorDirectory as u16, true, 10, 0),
        BootstrapEntry::new(RegionRole::ModelMetadata as u16, true, 11, 0),
        BootstrapEntry::new(RegionRole::TokenizerMetadata as u16, false, 12, 0),
    ]
}

#[test]
fn minimal_bootstrap_maps_roles_to_generic_ranges_without_retyping_values() {
    let bytes = stream(&required_entries());
    let validated = parse_v06(&bytes).unwrap();
    assert_eq!(validated.blocks()[1].semantic, V06Semantic::Float);
    assert_eq!(validated.blocks()[1].bit_width, 32);
    let bootstrap = Bootstrap::discover(&validated).unwrap();
    assert_eq!(bootstrap.profile_version(), 1);
    assert_eq!(bootstrap.region(RegionRole::TensorDirectory).unwrap().range.bytes().len(), 8);
    assert_eq!(bootstrap.region(RegionRole::ModelMetadata).unwrap().range.bytes(), b"metadata");
    assert_eq!(bootstrap.region(RegionRole::TokenizerMetadata).unwrap().range.bytes(), b"tokens");
    assert_eq!(bootstrap.region(RegionRole::TensorDirectory).unwrap().block_index, 1);
}

#[test]
fn unknown_optional_roles_are_skipped_but_unknown_required_roles_fail() {
    let known = required_entries();
    let mut optional = known.to_vec();
    optional.push(BootstrapEntry::new(99, false, 12, 0));
    assert_eq!(Bootstrap::discover(&parse_v06(&stream(&optional)).unwrap()).unwrap().regions().len(), 3);

    let mut required = known.to_vec();
    required.push(BootstrapEntry::new(99, true, 12, 0));
    assert_eq!(Bootstrap::discover(&parse_v06(&stream(&required)).unwrap()).unwrap_err().code, MlErrorCode::UnknownRequiredRole);
}

#[test]
fn missing_duplicate_and_bad_references_fail_deterministically() {
    let mut missing = required_entries();
    missing[0] = BootstrapEntry::new(RegionRole::TensorDirectory as u16, true, 99, 0);
    assert_eq!(Bootstrap::discover(&parse_v06(&stream(&missing)).unwrap()).unwrap_err().code, MlErrorCode::ReferencedRegionMissing);

    let duplicate = [
        BootstrapEntry::new(1, true, 10, 0),
        BootstrapEntry::new(1, true, 10, 0),
        BootstrapEntry::new(2, true, 11, 0),
    ];
    assert_eq!(Bootstrap::discover(&parse_v06(&stream(&duplicate)).unwrap()).unwrap_err().code, MlErrorCode::DuplicateRole);

    let no_model = [BootstrapEntry::new(1, true, 10, 0)];
    assert_eq!(Bootstrap::discover(&parse_v06(&stream(&no_model)).unwrap()).unwrap_err().code, MlErrorCode::MissingRequiredRole);
}

#[test]
fn unsupported_or_truncated_bootstrap_is_an_ml_error_after_canonical_validation() {
    let mut truncated_writer = VBufV06Writer::new_known_size(Cursor::new(Vec::new()), 3).unwrap();
    let encoded = encode_payload(&required_entries()).unwrap();
    truncated_writer.write_opaque(BlockOptions::array(BOOTSTRAP_KEY_ID), &encoded[..15]).unwrap();
    let truncated_bytes = truncated_writer.finish().unwrap().into_inner();
    let truncated_validated = parse_v06(&truncated_bytes).unwrap();
    assert_eq!(Bootstrap::discover(&truncated_validated).unwrap_err().code, MlErrorCode::MalformedBootstrap);

    let mut bytes = stream(&required_entries());
    let validated = parse_v06(&bytes).unwrap();
    let start = validated.blocks()[0].payload_start as usize;
    bytes[start + 8..start + 10].copy_from_slice(&2u16.to_le_bytes());
    let validated = parse_v06(&bytes).unwrap();
    assert_eq!(Bootstrap::discover(&validated).unwrap_err().code, MlErrorCode::UnsupportedProfileVersion);

    let mut truncated = stream(&required_entries());
    let validated = parse_v06(&truncated).unwrap();
    let payload_start = validated.blocks()[0].payload_start as usize;
    truncated[payload_start + 15] = 0xff;
    // The canonical payload remains valid; the profile parser rejects its reserved byte.
    let validated = parse_v06(&truncated).unwrap();
    assert_eq!(Bootstrap::discover(&validated).unwrap_err().code, MlErrorCode::InvalidBootstrapFlags);
}

#[test]
fn checked_fixture_can_be_read_from_the_descendant_fixture_directory() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/minimal-bootstrap.vbuf");
    let bytes = std::fs::read(path).unwrap();
    let validated = parse_v06(&bytes).unwrap();
    let bootstrap = Bootstrap::discover(&validated).unwrap();
    assert_eq!(bootstrap.regions().len(), 3);
}
