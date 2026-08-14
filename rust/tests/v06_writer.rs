use std::io::Cursor;
use std::path::Path;
use vbuf_core::v06::{V06Physical, parse_v06};
use vbuf_core::writer::{BlockOptions, VBufV06Writer, WriterError};

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/v06")
            .join(name),
    )
    .unwrap()
}

fn options(key_id: u16) -> BlockOptions {
    BlockOptions::array(key_id)
}

fn write_primitive_fixture() -> Vec<u8> {
    let cursor = Cursor::new(Vec::new());
    let mut writer = VBufV06Writer::new_known_size(cursor, 4).unwrap();
    writer.write_u8(options(10), &[1, 2, 255]).unwrap();
    writer.write_u16(options(11), &[0x1234, 0xabcd]).unwrap();
    writer.write_u32(options(12), &[1, 0xdeadbeef]).unwrap();
    writer
        .write_u64(options(13), &[1, 0x0102030405060708])
        .unwrap();
    writer.write_i8(options(14), &[-1, 2]).unwrap();
    writer.write_i16(options(15), &[-2, 3]).unwrap();
    writer.write_i32(options(16), &[-3, 4]).unwrap();
    writer.write_i64(options(17), &[-4, 5]).unwrap();
    writer.write_f32(options(18), &[1.5, -2.25]).unwrap();
    writer.write_f64(options(19), &[3.5, -4.75]).unwrap();
    writer.write_opaque(options(20), b"abc").unwrap();
    writer.write_u32(BlockOptions::scalar(21), &[42]).unwrap();
    let mut first = options(22);
    first.continuation = true;
    writer.write_opaque(first, b"A").unwrap();
    writer.write_opaque(options(22), b"B").unwrap();
    writer.finish().unwrap().into_inner()
}

#[test]
fn rust_writer_matches_the_cross_language_canonical_fixture() {
    let actual = write_primitive_fixture();
    assert_eq!(actual, fixture("valid-writer-primitives.vbuf"));
    let parsed = parse_v06(&actual).unwrap();
    assert_eq!(parsed.blocks().len(), 14);
    assert!(parsed.blocks()[12].continuation);
    assert!(!parsed.blocks()[13].continuation);
    assert_eq!(parsed.u8_view(10, 0).unwrap(), &[1, 2, 255]);
    assert_eq!(parsed.u16_view(11, 0).unwrap(), &[0x1234, 0xabcd]);
    assert_eq!(parsed.u32_view(12, 0).unwrap(), &[1, 0xdeadbeef]);
    assert_eq!(parsed.u64_view(13, 0).unwrap(), &[1, 0x0102030405060708]);
    assert_eq!(parsed.i8_view(14, 0).unwrap(), &[-1, 2]);
    assert_eq!(parsed.i16_view(15, 0).unwrap(), &[-2, 3]);
    assert_eq!(parsed.i32_view(16, 0).unwrap(), &[-3, 4]);
    assert_eq!(parsed.i64_view(17, 0).unwrap(), &[-4, 5]);
    assert_eq!(parsed.f32_view(18, 0).unwrap(), &[1.5, -2.25]);
    assert_eq!(parsed.f64_view(19, 0).unwrap(), &[3.5, -4.75]);
    let opaque = &parsed.blocks()[10];
    assert_eq!(
        &actual
            [opaque.payload_start as usize..(opaque.payload_start + opaque.payload_len) as usize],
        b"abc"
    );
}

#[test]
fn extended_count_and_partial_final_region_are_canonical() {
    let cursor = Cursor::new(Vec::new());
    let mut writer = VBufV06Writer::new_known_size(cursor, 3).unwrap();
    writer.write_u8(options(1), &vec![7; 65536]).unwrap();
    let bytes = writer.finish().unwrap().into_inner();
    let parsed = parse_v06(&bytes).unwrap();
    assert_eq!(parsed.blocks()[0].count, 65536);
    let anchor = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
    assert_ne!(anchor & (1 << 9), 0);
    assert_eq!((anchor >> 48) & 0xffff, 0);
}

#[test]
fn indefinite_writer_keeps_zero_size_and_is_valid_at_finish() {
    let cursor = Cursor::new(Vec::new());
    let mut writer = VBufV06Writer::new_indefinite(cursor, 3).unwrap();
    writer.write_u16(options(1), &[1, 2]).unwrap();
    let bytes = writer.finish().unwrap().into_inner();
    assert_eq!(bytes[9], 1);
    assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 0);
    assert!(parse_v06(&bytes).unwrap().header().indefinite);
}

#[test]
fn continuation_must_point_to_a_matching_next_block_and_terminate() {
    let cursor = Cursor::new(Vec::new());
    let mut writer = VBufV06Writer::new_known_size(cursor, 3).unwrap();
    let mut first = options(7);
    first.continuation = true;
    writer.write_u8(first, &[1]).unwrap();
    assert!(matches!(
        writer.write_u8(options(8), &[2]),
        Err(WriterError::ContinuationMismatch)
    ));
    writer.write_u8(options(7), &[2]).unwrap();
    assert!(parse_v06(&writer.finish().unwrap().into_inner()).is_ok());

    let cursor = Cursor::new(Vec::new());
    let mut writer = VBufV06Writer::new_known_size(cursor, 3).unwrap();
    writer.write_u8(first, &[1]).unwrap();
    assert!(matches!(
        writer.finish(),
        Err(WriterError::UnterminatedContinuation)
    ));
}

#[test]
fn interrupted_stream_policy_is_unambiguous() {
    let mut indefinite_bytes = Cursor::new(Vec::new());
    {
        let mut writer = VBufV06Writer::new_indefinite(&mut indefinite_bytes, 3).unwrap();
        writer.write_u8(options(1), &[1, 2]).unwrap();
    }
    assert!(parse_v06(indefinite_bytes.get_ref()).is_ok());

    let mut known_bytes = Cursor::new(Vec::new());
    {
        let mut writer = VBufV06Writer::new_known_size(&mut known_bytes, 3).unwrap();
        writer.write_u8(options(1), &[1, 2]).unwrap();
    }
    assert!(parse_v06(known_bytes.get_ref()).is_err());
}

#[test]
fn empty_array_and_stricter_payload_alignment_are_canonical() {
    let cursor = Cursor::new(Vec::new());
    let mut writer = VBufV06Writer::new_known_size(cursor, 4).unwrap();
    writer.write_u8(options(1), &[]).unwrap();
    let mut aligned = options(2);
    aligned.payload_shift = 2;
    writer.write_u32(aligned, &[7, 8]).unwrap();
    let bytes = writer.finish().unwrap().into_inner();
    let parsed = parse_v06(&bytes).unwrap();
    assert_eq!(parsed.blocks()[0].count, 0);
    assert_eq!(parsed.blocks()[1].payload_alignment, 64);
    assert_eq!(parsed.blocks()[1].payload_start % 64, 0);
}

#[test]
fn invalid_geometry_and_scalar_count_fail_before_bytes_are_committed() {
    assert!(matches!(
        VBufV06Writer::new_known_size(Cursor::new(Vec::new()), 2),
        Err(WriterError::InvalidBaseShift)
    ));
    assert!(matches!(
        VBufV06Writer::new_known_size(Cursor::new(vec![0]), 3),
        Err(WriterError::NonEmptySink)
    ));
    let cursor = Cursor::new(Vec::new());
    let mut writer = VBufV06Writer::new_known_size(cursor, 8).unwrap();
    let mut invalid_shift = options(1);
    invalid_shift.payload_shift = 56;
    assert!(matches!(
        writer.write_u8(invalid_shift, &[1]),
        Err(WriterError::InvalidPayloadShift)
    ));
    let scalar = BlockOptions {
        key_id: 2,
        physical: V06Physical::Scalar,
        continuation: false,
        payload_shift: 0,
    };
    assert!(matches!(
        writer.write_u8(scalar, &[]),
        Err(WriterError::InvalidRepresentation)
    ));
}
