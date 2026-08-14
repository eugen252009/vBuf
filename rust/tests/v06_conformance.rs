use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use vbuf_core::v06::{V06ErrorCode, VBufV06, parse_v06};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/v06")
}

#[test]
fn every_cross_language_fixture_has_the_declared_outcome() {
    let dir = fixture_dir();
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("read fixture directory")
        .map(|entry| entry.expect("fixture entry").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "vbuf")
        })
        .collect();
    files.sort();
    assert_eq!(files.len(), 42);
    for path in files {
        let name = path.file_name().unwrap().to_string_lossy();
        let expected = name.starts_with("valid-");
        let bytes = std::fs::read(&path).expect("read fixture");
        let result = parse_v06(&bytes);
        assert_eq!(result.is_ok(), expected, "fixture {name}: {result:?}");
    }
}

#[test]
fn validated_descriptors_precede_typed_views() {
    let dir = fixture_dir();
    let instance = VBufV06::open(dir.join("valid-basic.vbuf")).expect("valid basic fixture");
    assert_eq!(instance.header().base_step, 8);
    assert_eq!(instance.header().data_region_start, 24);
    assert_eq!(instance.blocks().len(), 1);
    let block = &instance.blocks()[0];
    assert_eq!(
        (block.block_start, block.payload_start, block.payload_len),
        (24, 32, 12)
    );
    assert_eq!(
        instance.u32_view(1, 0).expect("validated u32 view"),
        &[1, 2, 3]
    );
    assert_eq!(
        instance.f32_view(1, 0).expect_err("semantic mismatch").code,
        V06ErrorCode::TypeMismatch
    );
}

#[test]
fn forward_continuation_preserves_physical_occurrences() {
    let bytes = std::fs::read(fixture_dir().join("valid-duplicate-chain.vbuf")).unwrap();
    let parsed = parse_v06(&bytes).unwrap();
    assert_eq!(parsed.blocks().len(), 3);
    assert!(parsed.blocks()[0].continuation);
    assert!(parsed.blocks()[1].continuation);
    assert!(!parsed.blocks()[2].continuation);
    assert_eq!(parsed.u8_view(7, 0).unwrap(), b"A");
    assert_eq!(parsed.u8_view(7, 1).unwrap(), b"B");
    assert_eq!(parsed.u8_view(7, 2).unwrap(), b"C");
}

#[test]
fn legal_empty_and_partial_final_regions_are_accepted() {
    let empty = std::fs::read(fixture_dir().join("valid-empty.vbuf")).unwrap();
    let empty = parse_v06(&empty).unwrap();
    assert!(empty.blocks().is_empty());

    let zero = std::fs::read(fixture_dir().join("valid-zero-array.vbuf")).unwrap();
    let zero = parse_v06(&zero).unwrap();
    assert_eq!(zero.blocks()[0].count, 0);
    assert!(zero.u8_view(2, 0).unwrap().is_empty());

    let partial = std::fs::read(fixture_dir().join("valid-basic.vbuf")).unwrap();
    let partial = parse_v06(&partial).unwrap();
    assert_ne!(
        partial.header().data_region_end % partial.header().base_step,
        0
    );
}

#[test]
fn native_typed_view_checks_actual_source_alignment() {
    let bytes = std::fs::read(fixture_dir().join("valid-basic.vbuf")).unwrap();
    let mut shifted = vec![0u8];
    shifted.extend_from_slice(&bytes);
    let parsed = parse_v06(&shifted[1..]).expect("wire offsets remain canonical");
    assert_eq!(
        parsed
            .u32_view(1, 0)
            .expect_err("actual native address is unaligned")
            .code,
        V06ErrorCode::Misaligned
    );
}

#[test]
fn preserved_float64_truncation_is_rejected_before_view_lookup() {
    let bytes = std::fs::read(fixture_dir().join("payload-truncated-element.vbuf")).unwrap();
    let error = parse_v06(&bytes).expect_err("two declared Float64 values cannot fit");
    assert_eq!(error.code, V06ErrorCode::TruncatedBlock);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn sparse_range_larger_than_four_gib_is_checked_without_allocation() {
    let path = std::env::temp_dir().join(format!("vbuf-v06-sparse-{}.vbuf", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    let data_size = (1u64 << 32) + 8;
    let mut header = [0u8; 24];
    header[0..4].copy_from_slice(b"VBUF");
    header[4..8].copy_from_slice(&0x0006_0000u32.to_le_bytes());
    header[8] = 3;
    header[10..12].copy_from_slice(&24u16.to_le_bytes());
    header[16..24].copy_from_slice(&data_size.to_le_bytes());
    file.write_all(&header).unwrap();
    file.seek(SeekFrom::Start(24 + data_size - 1)).unwrap();
    file.write_all(&[0]).unwrap();
    drop(file);

    let error = VBufV06::open(&path).expect_err("zero sparse data cannot form an anchor");
    assert_eq!(error.code, V06ErrorCode::InvalidAnchor);
    std::fs::remove_file(path).unwrap();
}
