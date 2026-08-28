use std::io::Cursor;
use vbuf_core::nested::NestedV06;
use vbuf_core::v06::{V06Semantic, parse_v06};
use vbuf_core::writer::{BlockOptions, VBufV06Writer};

fn canonical_child() -> Vec<u8> {
    let mut writer = VBufV06Writer::new_known_size(Cursor::new(Vec::new()), 3).unwrap();
    writer
        .write_block(
            BlockOptions::array(7),
            V06Semantic::Opaque,
            8,
            3,
            b"abc",
        )
        .unwrap();
    writer.finish().unwrap().into_inner()
}

#[test]
fn nested_root_is_validated_in_a_parent_payload() {
    let child = canonical_child();
    let mut writer = VBufV06Writer::new_known_size(Cursor::new(Vec::new()), 3).unwrap();
    writer
        .write_block(
            BlockOptions::array(9),
            V06Semantic::Opaque,
            8,
            child.len() as u64,
            &child,
        )
        .unwrap();
    let parent_bytes = writer.finish().unwrap().into_inner();
    let parent = parse_v06(&parent_bytes).unwrap();
    let nested = NestedV06::from_parent_payload(&parent, 0, 0, child.len() as u64).unwrap();
    assert_eq!(nested.range().bytes(), child.as_slice());
    assert_eq!(nested.validated().blocks()[0].key_id, 7);
}

#[test]
fn nested_root_rejects_truncation_and_empty_ranges() {
    let child = canonical_child();
    let mut writer = VBufV06Writer::new_known_size(Cursor::new(Vec::new()), 3).unwrap();
    writer
        .write_block(
            BlockOptions::array(9),
            V06Semantic::Opaque,
            8,
            child.len() as u64,
            &child,
        )
        .unwrap();
    let parent_bytes = writer.finish().unwrap().into_inner();
    let parent = parse_v06(&parent_bytes).unwrap();
    assert!(NestedV06::from_parent_payload(&parent, 0, 0, 0).is_err());
    assert!(NestedV06::from_parent_payload(&parent, 0, 0, child.len() as u64 - 1).is_err());
}
