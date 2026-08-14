use std::io::Cursor;
use std::path::{Path, PathBuf};
use vbuf_core::{VBufInstance, VBufWriter};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("tests/fixtures/v05")
        .join(name)
}

#[test]
#[cfg(target_endian = "little")]
fn current_rust_writer_bytes_are_preserved() {
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = VBufWriter::new(&mut cursor, 16).expect("legacy writer header");
        writer
            .write_column(1, &[1u32, 2, 3])
            .expect("legacy u32 column");
        writer
            .write_column(2, &[500u16, 1000])
            .expect("legacy u16 column");
    }

    let expected = std::fs::read(fixture("current-rust.vbuf")).expect("read fixture");
    assert_eq!(cursor.into_inner(), expected);
}

#[test]
fn rust_reader_reads_rust_and_typescript_legacy_fixtures() {
    let rust = VBufInstance::open(fixture("current-rust.vbuf").to_str().unwrap())
        .expect("open Rust fixture");
    assert_eq!(rust.get_as::<u32>(1), Some(&[1, 2, 3][..]));
    assert_eq!(rust.get_as::<u16>(2), Some(&[500, 1000][..]));

    let typescript = VBufInstance::open(fixture("current-typescript.vbuf").to_str().unwrap())
        .expect("open TypeScript fixture");
    assert_eq!(typescript.get_as::<i32>(1), Some(&[1, 2, 3][..]));
    assert_eq!(typescript.get_as::<f64>(2), Some(&[1.5, -2.25][..]));
}

#[test]
fn rust_legacy_malformed_behavior_is_preserved_as_evidence() {
    assert!(VBufInstance::open(fixture("short.vbuf").to_str().unwrap()).is_err());
    assert!(VBufInstance::open(fixture("bad-magic.vbuf").to_str().unwrap()).is_err());

    // Legacy open validates only the global header. Legacy typed lookup checks
    // the full requested range and returns None for this truncated payload.
    // Step 4 must turn this fixture into mandatory v0.6 open/access rejection.
    let truncated = VBufInstance::open(fixture("truncated-typescript.vbuf").to_str().unwrap())
        .expect("legacy opener accepts the intact global header");
    assert!(truncated.get_as::<f64>(2).is_none());
}
