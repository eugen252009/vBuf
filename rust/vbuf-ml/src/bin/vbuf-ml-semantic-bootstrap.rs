use memmap2::Mmap;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::File;
use std::io::Cursor;
use std::path::PathBuf;
use vbuf_core::v06::{parse_v06, V06Physical, V06Semantic};
use vbuf_core::writer::{BlockOptions, VBufV06Writer};
use vbuf_ml::bootstrap::{encode_payload as encode_bootstrap, BootstrapEntry, BOOTSTRAP_KEY_ID};
use vbuf_ml::{encode_source_profile, parse_source_profile, BorrowedModelView, Bootstrap, ModelMetadata, RegionRole, SourceDescriptor, SourceHash, SourceId, SourceLocator, SourceRegistry, TensorDirectory, TensorRef, TokenizerMetadata};

const SOURCE_METADATA_KEY: u16 = 0xEFFE;

fn sha256(bytes: &[u8]) -> Vec<u8> { Sha256::digest(bytes).to_vec() }

fn build(source_bytes: &[u8], metadata_size: u64, locator: &str) -> Result<Vec<u8>, String> {
    let validated = parse_v06(source_bytes).map_err(|e| e.to_string())?;
    let bootstrap = Bootstrap::discover(&validated).map_err(|e| e.to_string())?;
    let _metadata = ModelMetadata::parse(&validated, &bootstrap).map_err(|e| e.to_string())?;
    let directory = TensorDirectory::parse(&validated, &bootstrap).map_err(|e| e.to_string())?;
    let _tokenizer = TokenizerMetadata::parse(&validated, &bootstrap).map_err(|e| e.to_string())?;
    let tensor_blocks: HashSet<usize> = directory.tensors().iter().map(|tensor| tensor.block_index).collect();
    let self_source = SourceDescriptor::self_artifact(metadata_size);
    let authoritative = SourceDescriptor {
        id: SourceId::new(1),
        declared_size: Some(source_bytes.len() as u64),
        locator: if locator.starts_with("http://") || locator.starts_with("https://") { SourceLocator::Http(locator.to_owned()) } else { SourceLocator::File(locator.to_owned()) },
        hashes: vec![SourceHash { algorithm: 1, value: sha256(source_bytes) }],
    };
    let registry = SourceRegistry::new(vec![self_source, authoritative.clone()]).map_err(|e| format!("source registry: {e:?}"))?;
    let bindings: Vec<(u16, u16, TensorRef)> = directory.tensors().iter().map(|tensor| {
        TensorRef::new(&authoritative, tensor.payload.offset(), tensor.payload.length()).map(|reference| (tensor.key_id, tensor.occurrence, reference)).map_err(|e| format!("tensor source reference: {e}"))
    }).collect::<Result<_, _>>()?;
    let source_profile = encode_source_profile(&registry, &bindings).map_err(|e| e.to_string())?;
    let mut profile_entries: Vec<BootstrapEntry> = bootstrap.regions().iter().map(|region| BootstrapEntry::new(region.role as u16, region.role.is_required(), region.key_id, region.occurrence)).collect();
    profile_entries.push(BootstrapEntry::new(RegionRole::SourceMetadata as u16, false, SOURCE_METADATA_KEY, 0));
    let bootstrap_payload = encode_bootstrap(&profile_entries).map_err(|e| e.to_string())?;
    let original_bootstrap_index = validated.blocks().iter().position(|block| block.key_id == BOOTSTRAP_KEY_ID).ok_or("bootstrap block is absent")?;
    let mut writer = VBufV06Writer::new_known_size(Cursor::new(Vec::new()), validated.header().base_shift).map_err(|e| e.to_string())?;
    for (index, block) in validated.blocks().iter().enumerate() {
        if index == original_bootstrap_index {
            writer.write_block(BlockOptions::array(BOOTSTRAP_KEY_ID), V06Semantic::Opaque, 8, bootstrap_payload.len() as u64, &bootstrap_payload).map_err(|e| e.to_string())?;
        } else if tensor_blocks.contains(&index) {
            if block.physical != V06Physical::Array { return Err("tensor placeholder is not an array".into()); }
            writer.write_block(BlockOptions::array(block.key_id), block.semantic, block.bit_width, 0, &[]).map_err(|e| e.to_string())?;
        } else {
            let payload = validated.payload_range(index).map_err(|e| e.to_string())?.bytes();
            let options = BlockOptions { key_id: block.key_id, physical: block.physical, continuation: false, payload_shift: 0 };
            writer.write_block(options, block.semantic, block.bit_width, block.count, payload).map_err(|e| e.to_string())?;
        }
    }
    writer.write_block(BlockOptions::array(SOURCE_METADATA_KEY), V06Semantic::Opaque, 8, source_profile.len() as u64, &source_profile).map_err(|e| e.to_string())?;
    Ok(writer.finish().map_err(|e| e.to_string())?.into_inner())
}

fn main() {
    let source_path = PathBuf::from(std::env::args().nth(1).expect("authoritative source path"));
    let target_path = PathBuf::from(std::env::args().nth(2).expect("semantic bootstrap output path"));
    let locator = std::env::args().nth(3).unwrap_or_else(|| source_path.to_string_lossy().into_owned());
    let file = File::open(&source_path).expect("open source");
    let mapping = unsafe { Mmap::map(&file).expect("map source") };
    let first = build(&mapping, 0, &locator).expect("build semantic bootstrap");
    let output = build(&mapping, first.len() as u64, &locator).expect("rebuild semantic bootstrap");
    assert_eq!(first.len(), output.len(), "self-source size changed profile geometry");
    std::fs::write(&target_path, &output).expect("write semantic bootstrap");
    let target_file = File::open(&target_path).expect("open semantic bootstrap");
    let target_mapping = unsafe { Mmap::map(&target_file).expect("map semantic bootstrap") };
    let target_validated = parse_v06(&target_mapping).expect("semantic bootstrap canonical validation");
    let target_bootstrap = Bootstrap::discover(&target_validated).expect("semantic bootstrap profile");
    let profile = parse_source_profile(&target_validated, &target_bootstrap).expect("source profile parse").expect("source profile role");
    BorrowedModelView::parse_with_sources(&target_mapping, &profile.registry, &[]).expect("semantic bootstrap semantic parse");
    println!("source_bytes={} bootstrap_bytes={} target={}", mapping.len(), output.len(), target_path.display());
}
