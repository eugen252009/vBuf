use std::io::Cursor;
use vbuf_core::v06::parse_v06;
use vbuf_core::writer::{BlockOptions, VBufV06Writer};
use vbuf_ml::bootstrap::{encode_payload as encode_bootstrap, Bootstrap, BootstrapEntry, BOOTSTRAP_KEY_ID};
use vbuf_ml::tokenizer::{encode_payload, TokenizerEntry, TokenizerKind};
use vbuf_ml::{MlErrorCode, SpecialToken, TokenizerMetadata};

fn tokenizer_file(payload: &[u8], pool: &[u8], offsets: &[u64], scores: &[f32], types: &[u8], eos: u32) -> Vec<u8> {
    let mut writer = VBufV06Writer::new_known_size(Cursor::new(Vec::new()), 3).unwrap();
    writer.write_opaque(BlockOptions::array(BOOTSTRAP_KEY_ID), &encode_bootstrap(&[
        BootstrapEntry::new(1, true, 12, 0),
        BootstrapEntry::new(2, true, 10, 0),
        BootstrapEntry::new(3, false, 11, 0),
    ]).unwrap()).unwrap();
    writer.write_opaque(BlockOptions::array(10), b"model").unwrap();
    writer.write_opaque(BlockOptions::array(11), payload).unwrap();
    writer.write_opaque(BlockOptions::array(12), b"directory").unwrap();
    writer.write_opaque(BlockOptions::array(30), pool).unwrap();
    writer.write_u64(BlockOptions::array(31), offsets).unwrap();
    writer.write_f32(BlockOptions::array(32), scores).unwrap();
    writer.write_u8(BlockOptions::array(33), types).unwrap();
    writer.write_u32(BlockOptions::scalar(34), &[0]).unwrap();
    writer.write_u32(BlockOptions::scalar(35), &[eos]).unwrap();
    writer.finish().unwrap().into_inner()
}

fn valid_payload() -> Vec<u8> {
    encode_payload(TokenizerKind::VocabularyOnly, &[
        TokenizerEntry::new(1, true, 30, 0),
        TokenizerEntry::new(2, true, 31, 0),
        TokenizerEntry::new(3, false, 32, 0),
        TokenizerEntry::new(4, false, 33, 0),
        TokenizerEntry::new(5, false, 34, 0),
        TokenizerEntry::new(6, false, 35, 0),
    ]).unwrap()
}

fn parse_tokenizer(payload: &[u8], pool: &[u8], offsets: &[u64], scores: &[f32], types: &[u8], eos: u32) -> Result<TokenizerMetadata<'static>, MlErrorCode> {
    let bytes: &'static [u8] = Box::leak(tokenizer_file(payload, pool, offsets, scores, types, eos).into_boxed_slice());
    let validated = parse_v06(bytes).map_err(|error| MlErrorCode::Canonical(error.code))?;
    let bootstrap = Bootstrap::discover(&validated).map_err(|error| error.code)?;
    TokenizerMetadata::parse(&validated, &bootstrap).map_err(|error| error.code)
}

#[test]
fn canonical_tokenizer_fixture_is_self_contained() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/minimal-tokenizer.vbuf");
    let bytes: &'static [u8] = Box::leak(std::fs::read(path).unwrap().into_boxed_slice());
    let validated = parse_v06(bytes).unwrap();
    let bootstrap = Bootstrap::discover(&validated).unwrap();
    let tokenizer = TokenizerMetadata::parse(&validated, &bootstrap).unwrap();
    assert_eq!(tokenizer.token_text(1), Some("world"));
}

#[test]
fn tokenizer_views_are_direct_canonical_ranges() {
    let tokenizer = parse_tokenizer(&valid_payload(), b"helloworld", &[0, 5, 10], &[0.5, -0.25], &[1, 2], 1).unwrap();
    assert_eq!(tokenizer.kind, TokenizerKind::VocabularyOnly);
    assert_eq!(tokenizer.token_count(), 2);
    assert_eq!(tokenizer.token_text(0), Some("hello"));
    assert_eq!(tokenizer.token_text(1), Some("world"));
    assert_eq!(tokenizer.score(0), Some(0.5));
    assert_eq!(tokenizer.token_type(1), Some(2));
    assert!(tokenizer.specials().contains(&(SpecialToken::Bos, 0)));
    assert!(tokenizer.specials().contains(&(SpecialToken::Eos, 1)));
}

#[test]
fn tokenizer_parallel_arrays_and_special_ids_are_checked() {
    assert_eq!(parse_tokenizer(&valid_payload(), b"helloworld", &[0, 5, 10], &[0.5], &[1, 2], 1).unwrap_err(), MlErrorCode::TokenizerArrayLengthMismatch);
    assert_eq!(parse_tokenizer(&valid_payload(), b"helloworld", &[0, 5, 10], &[0.5, -0.25], &[1, 2], 2).unwrap_err(), MlErrorCode::InvalidSpecialTokenId);
    assert_eq!(parse_tokenizer(&valid_payload(), b"hello\xff", &[0, 5, 6], &[0.5, -0.25], &[1, 2], 1).unwrap_err(), MlErrorCode::InvalidTokenText);
    assert_eq!(parse_tokenizer(&valid_payload(), b"helloworld", &[0, 6, 5], &[0.5, -0.25], &[1, 2], 1).unwrap_err(), MlErrorCode::InvalidTokenText);
}

#[test]
fn tokenizer_header_and_role_errors_fail_closed() {
    let mut unknown_kind = valid_payload();
    unknown_kind[12] = 99;
    assert_eq!(parse_tokenizer(&unknown_kind, b"helloworld", &[0, 5, 10], &[0.5, -0.25], &[1, 2], 1).unwrap_err(), MlErrorCode::UnsupportedTokenizerKind);

    let mut truncated = valid_payload();
    truncated.pop();
    assert_eq!(parse_tokenizer(&truncated, b"helloworld", &[0, 5, 10], &[0.5, -0.25], &[1, 2], 1).unwrap_err(), MlErrorCode::MalformedTokenizerMetadata);

    let duplicate = [TokenizerEntry::new(1, true, 30, 0), TokenizerEntry::new(1, true, 30, 0)];
    assert_eq!(encode_payload(TokenizerKind::VocabularyOnly, &duplicate).unwrap_err().code, MlErrorCode::DuplicateTokenizerRole);
}
