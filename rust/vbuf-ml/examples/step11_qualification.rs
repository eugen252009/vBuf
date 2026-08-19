//! Step 11 qualification harness. This is experimental evidence tooling, not
//! a tokenizer engine or a portable wire feature.
use memmap2::Mmap;
use std::fs::{self, File};
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;
use vbuf_core::v06::parse_v06;
use vbuf_core::writer::{BlockOptions, VBufV06Writer};
use vbuf_ml::bootstrap::{BOOTSTRAP_KEY_ID, BootstrapEntry, encode_payload as encode_bootstrap};
use vbuf_ml::metadata::{MetadataEntry, encode_payload as encode_metadata};
use vbuf_ml::tensor_directory::{TensorEntry, encode_payload as encode_directory};
use vbuf_ml::tokenizer::{TokenizerEntry, TokenizerKind, encode_payload as encode_tokenizer};
use vbuf_ml::{Bootstrap, ModelMetadata, TensorDirectory, TokenizerMetadata};

const TOKEN_COUNT: usize = 1024;
const TOKEN_BYTES: usize = 4096;
const AUX_BYTES: usize = 8 * 1024 * 1024;

fn build_fixture(path: &Path) {
    let bootstrap = encode_bootstrap(&[
        BootstrapEntry::new(1, true, 12, 0),
        BootstrapEntry::new(2, true, 10, 0),
        BootstrapEntry::new(3, false, 11, 0),
    ])
    .unwrap();
    let metadata = encode_metadata(&[
        MetadataEntry::new(1, true, 20, 0),
        MetadataEntry::new(2, true, 21, 0),
        MetadataEntry::new(3, true, 22, 0),
        MetadataEntry::new(4, true, 23, 0),
        MetadataEntry::new(5, true, 24, 0),
    ])
    .unwrap();
    let directory =
        encode_directory(&[TensorEntry::new("layer.0.weight", vec![1], 60, 0)]).unwrap();
    let tokenizer = encode_tokenizer(
        TokenizerKind::VocabularyOnly,
        &[
            TokenizerEntry::new(1, true, 31, 0),
            TokenizerEntry::new(2, true, 32, 0),
            TokenizerEntry::new(3, false, 33, 0),
            TokenizerEntry::new(4, false, 34, 0),
        ],
    )
    .unwrap();
    let pool = vec![b'a'; TOKEN_COUNT * TOKEN_BYTES];
    let offsets: Vec<u64> = (0..=TOKEN_COUNT)
        .map(|index| (index * TOKEN_BYTES) as u64)
        .collect();
    let scores = vec![0.5f32; TOKEN_COUNT];
    let types = vec![1u8; TOKEN_COUNT];
    let aux = vec![0x5au8; AUX_BYTES];
    let mut writer = VBufV06Writer::new_known_size(std::io::Cursor::new(Vec::new()), 3).unwrap();
    writer
        .write_opaque(BlockOptions::array(BOOTSTRAP_KEY_ID), &bootstrap)
        .unwrap();
    writer
        .write_opaque(BlockOptions::array(10), &metadata)
        .unwrap();
    writer
        .write_opaque(BlockOptions::array(12), &directory)
        .unwrap();
    writer
        .write_opaque(BlockOptions::array(11), &tokenizer)
        .unwrap();
    writer
        .write_opaque(BlockOptions::array(20), b"llama")
        .unwrap();
    writer.write_u32(BlockOptions::scalar(21), &[2048]).unwrap();
    writer.write_u32(BlockOptions::scalar(22), &[4096]).unwrap();
    writer.write_u32(BlockOptions::scalar(23), &[32]).unwrap();
    writer.write_u32(BlockOptions::scalar(24), &[32]).unwrap();
    writer.write_f32(BlockOptions::array(60), &[1.0]).unwrap();
    writer.write_opaque(BlockOptions::array(31), &pool).unwrap();
    writer.write_u64(BlockOptions::array(32), &offsets).unwrap();
    writer.write_f32(BlockOptions::array(33), &scores).unwrap();
    writer.write_u8(BlockOptions::array(34), &types).unwrap();
    writer.write_opaque(BlockOptions::array(50), &aux).unwrap();
    let bytes = writer.finish().unwrap().into_inner();
    fs::write(path, bytes).unwrap();
}

fn faults() -> (u64, u64) {
    let text = fs::read_to_string("/proc/self/stat").unwrap();
    let fields: Vec<&str> = text
        .rsplit_once(") ")
        .unwrap()
        .1
        .split_whitespace()
        .collect();
    (fields[7].parse().unwrap(), fields[9].parse().unwrap())
}

fn touch(bytes: &[u8]) -> u64 {
    let mut sum = 0u64;
    for page in bytes.chunks(4096) {
        sum = sum.wrapping_add(u64::from(page[0]));
    }
    black_box(sum)
}

fn record(
    scenario: &str,
    phase: &str,
    start: Instant,
    before: (u64, u64),
    bytes: u64,
    checksum: u64,
) {
    let after = faults();
    println!(
        "{scenario},{phase},{},{},{},{bytes},{checksum}",
        start.elapsed().as_nanos(),
        after.0.saturating_sub(before.0),
        after.1.saturating_sub(before.1)
    );
}

fn run(path: &Path, eager: bool) {
    let file = File::open(path).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    let before = faults();
    let start = Instant::now();
    let validated = parse_v06(&mmap).unwrap();
    record(
        if eager { "eager" } else { "lazy" },
        "T1_canonical",
        start,
        before,
        0,
        validated.blocks().len() as u64,
    );
    let before = faults();
    let start = Instant::now();
    let bootstrap = Bootstrap::discover(&validated).unwrap();
    record(
        if eager { "eager" } else { "lazy" },
        "T2_bootstrap",
        start,
        before,
        bootstrap
            .regions()
            .iter()
            .map(|region| region.range.length())
            .sum(),
        bootstrap.regions().len() as u64,
    );
    let before = faults();
    let start = Instant::now();
    let model = ModelMetadata::parse(&validated, &bootstrap).unwrap();
    let tensor = TensorDirectory::parse(&validated, &bootstrap).unwrap();
    let tensor_bytes = tensor
        .tensors()
        .iter()
        .filter_map(|entry| entry.range.as_ref())
        .map(|range| range.length())
        .sum::<u64>();
    record(
        if eager { "eager" } else { "lazy" },
        "T3_model_tensor",
        start,
        before,
        model
            .fields()
            .iter()
            .map(|field| field.range.length())
            .sum::<u64>()
            + tensor_bytes,
        tensor.tensors().len() as u64,
    );
    if eager {
        let before = faults();
        let start = Instant::now();
        let tokenizer = TokenizerMetadata::parse(&validated, &bootstrap).unwrap();
        record(
            "eager",
            "T4_tokenizer_control",
            start,
            before,
            (TOKEN_COUNT * TOKEN_BYTES + (TOKEN_COUNT + 1) * 8) as u64,
            tokenizer.token_count(),
        );
        let before = faults();
        let start = Instant::now();
        let mut checksum = 0u64;
        for index in 0..tokenizer.token_count() {
            checksum = checksum.wrapping_add(tokenizer.token_text(index).unwrap().len() as u64);
            checksum = checksum.wrapping_add(tokenizer.score(index).unwrap_or_default() as u64);
            checksum = checksum.wrapping_add(tokenizer.token_type(index).unwrap_or_default());
        }
        record(
            "eager",
            "T5_tokenizer_payload",
            start,
            before,
            (TOKEN_COUNT * TOKEN_BYTES + TOKEN_COUNT * 4 + TOKEN_COUNT) as u64,
            checksum,
        );
        let block = validated.block(50, 0).unwrap();
        let range = validated
            .payload_range(
                validated
                    .blocks()
                    .iter()
                    .position(|candidate| candidate.block_start == block.block_start)
                    .unwrap(),
            )
            .unwrap();
        let before = faults();
        let start = Instant::now();
        let checksum = touch(range.bytes());
        record(
            "eager",
            "T6_auxiliary_cold",
            start,
            before,
            range.length(),
            checksum,
        );
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("vbuf-ml-step11-qualification.vbuf"));
    build_fixture(&path);
    println!("scenario,phase,nanoseconds,minor_faults,major_faults,explicit_bytes,checksum");
    run(&path, false);
    run(&path, true);
    eprintln!(
        "fixture={} bytes={}",
        path.display(),
        fs::metadata(&path).unwrap().len()
    );
}
