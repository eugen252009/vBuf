use std::io::Cursor;
use vbuf_core::v06::{parse_v06, V06Physical, V06Semantic};
use vbuf_ml::{encode_nested_payload, Bootstrap, BootstrapEntry, LayoutClass, NestedDirectory, NestedEntry, PlacementRequest, RegionRole};

fn opaque(key: u16, order: u64, bytes: &[u8]) -> PlacementRequest<'_> {
    PlacementRequest { class: LayoutClass::Auxiliary, order, key_id: key, semantic: V06Semantic::Opaque, physical: V06Physical::Array, bit_width: 8, count: bytes.len() as u64, payload_alignment: 8, payload: bytes }
}

#[test]
fn inline_child_is_canonically_validated_and_resolved() {
    let child = vbuf_ml::write_known_size(Cursor::new(Vec::new()), 3, &[opaque(0x7000, 0, b"child")]).unwrap().into_inner();
    let nested = encode_nested_payload(&[NestedEntry { name: "child".into(), key_id: 0x7000, occurrence: 0, child_offset: 0, child_length: child.len() as u64 }]).unwrap();
    let bootstrap = vbuf_ml::bootstrap::encode_payload(&[
        BootstrapEntry::new(RegionRole::TensorDirectory as u16, true, 0x7001, 0),
        BootstrapEntry::new(RegionRole::ModelMetadata as u16, true, 0x7002, 0),
        BootstrapEntry::new(RegionRole::NestedDirectory as u16, false, 0x7003, 0),
    ]).unwrap();
    let parent = vbuf_ml::write_known_size(Cursor::new(Vec::new()), 3, &[
        opaque(vbuf_ml::bootstrap::BOOTSTRAP_KEY_ID, 0, &bootstrap),
        opaque(0x7001, 1, b"tensor-directory"),
        opaque(0x7002, 2, b"metadata"),
        opaque(0x7003, 3, &nested),
        opaque(0x7000, 4, &child),
    ]).unwrap().into_inner();
    let validated = parse_v06(&parent).unwrap();
    let discovered = Bootstrap::discover(&validated).unwrap();
    let directory = NestedDirectory::parse(&validated, &discovered).unwrap();
    assert_eq!(directory.children().len(), 1);
    assert_eq!(directory.get("child").unwrap().validated.blocks().len(), 1);
}

#[test]
fn nested_payload_rejects_empty_ranges() {
    let error = encode_nested_payload(&[NestedEntry { name: "empty".into(), key_id: 1, occurrence: 0, child_offset: 0, child_length: 0 }]).unwrap_err();
    assert_eq!(error.code, vbuf_ml::MlErrorCode::NestedChildRangeInvalid);
}
