use std::io::Cursor;
use std::path::Path;
use vbuf_core::v06::parse_v06;
use vbuf_core::writer::{BlockOptions, VBufV06Writer};

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/v06")
            .join(name),
    )
    .unwrap()
}

#[test]
fn block_payload_and_refined_ranges_are_checked_and_borrowed() {
    let bytes = fixture("valid-basic.vbuf");
    let parsed = parse_v06(&bytes).unwrap();
    let physical = parsed.block_range(0).unwrap();
    assert_eq!(
        (physical.offset(), physical.length(), physical.end()),
        (24, 20, 44)
    );
    assert_eq!(physical.bytes(), &bytes[24..44]);
    let payload = parsed.payload_range(0).unwrap();
    assert_eq!(
        (payload.offset(), payload.length(), payload.end()),
        (32, 12, 44)
    );
    assert_eq!(payload.refine(4, 8).unwrap().bytes(), &bytes[36..44]);
    assert_eq!(parsed.payload_subrange(0, 12, 0).unwrap().bytes(), &[]);
    assert!(parsed.payload_subrange(0, 12, 1).is_err());
    assert!(parsed.payload_subrange(0, u64::MAX, 1).is_err());
    assert!(parsed.block_range(99).is_err());
}

#[test]
fn zero_length_and_partial_final_ranges_are_valid() {
    let zero_bytes = fixture("valid-zero-array.vbuf");
    let zero = parse_v06(&zero_bytes).unwrap();
    let range = zero.payload_range(0).unwrap();
    assert_eq!((range.length(), range.offset(), range.end()), (0, 32, 32));

    let partial_bytes = fixture("valid-basic.vbuf");
    let partial = parse_v06(&partial_bytes).unwrap();
    let final_range = partial.block_range(0).unwrap();
    assert_eq!(final_range.end(), partial.header().data_region_end);
}

#[test]
fn continuation_and_duplicate_ids_keep_independent_ranges() {
    let chain_bytes = fixture("valid-duplicate-chain.vbuf");
    let parsed = parse_v06(&chain_bytes).unwrap();
    assert_eq!(parsed.blocks().len(), 3);
    for index in 0..3 {
        assert!(!parsed.payload_range(index).unwrap().bytes().is_empty());
        assert!(parsed.block_range(index).unwrap().end() <= parsed.header().data_region_end);
    }
    let duplicate_bytes = fixture("valid-duplicate-unchained.vbuf");
    let duplicate = parse_v06(&duplicate_bytes).unwrap();
    assert_ne!(
        duplicate.payload_range(0).unwrap().offset(),
        duplicate.payload_range(1).unwrap().offset()
    );
}

#[test]
fn wire_valid_range_is_host_checked_at_view_boundary() {
    let mut writer = VBufV06Writer::new_known_size(Cursor::new(Vec::new()), 3).unwrap();
    writer.write_u8(BlockOptions::array(1), &[1, 2, 3]).unwrap();
    let bytes = writer.finish().unwrap().into_inner();
    let parsed = parse_v06(&bytes).unwrap();
    let range = parsed.payload_range(0).unwrap();
    assert_eq!(range.bytes(), &[1, 2, 3]);
    assert!(range.require_alignment(8).is_ok());
    assert!(range.require_alignment(3).is_err());
}
