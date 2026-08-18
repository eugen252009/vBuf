use memmap2::Mmap;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::path::PathBuf;
use vbuf_core::v06::parse_v06;
use vbuf_ml::{execute_plan_with_sources, parse_source_profile, select_tensor_ordinals, BorrowedModelView, Coalescing, PositionedFileSource, RegionRole, SourceId, SourceSet};

fn digest_view(view: &BorrowedModelView<'_>) -> String {
    let mut digest = Sha256::new();
    for field in view.metadata.fields() {
        digest.update((field.key as u16).to_le_bytes());
        match &field.value {
            vbuf_ml::MetadataValue::Text(value) => { digest.update([1]); digest.update((value.len() as u64).to_le_bytes()); digest.update(value.as_bytes()); }
            vbuf_ml::MetadataValue::Unsigned(value) => { digest.update([2]); digest.update(value.to_le_bytes()); }
            vbuf_ml::MetadataValue::Float(value) => { digest.update([3]); digest.update(value.to_le_bytes()); }
        }
    }
    for tensor in view.directory.tensors() {
        digest.update((tensor.name.len() as u64).to_le_bytes()); digest.update(tensor.name.as_bytes());
        digest.update([tensor.representation as u8]); digest.update((tensor.dimensions.len() as u64).to_le_bytes());
        for dimension in &tensor.dimensions { digest.update(dimension.to_le_bytes()); }
        digest.update(tensor.payload.offset().to_le_bytes()); digest.update(tensor.payload.length().to_le_bytes());
    }
    digest.update(view.tokenizer.token_count().to_le_bytes());
    for index in 0..view.tokenizer.token_count() {
        let text = view.tokenizer.token_text(index).unwrap_or("").as_bytes(); digest.update((text.len() as u64).to_le_bytes()); digest.update(text);
        digest.update(view.tokenizer.token_type(index).unwrap_or(u64::MAX).to_le_bytes());
        digest.update(view.tokenizer.score(index).unwrap_or(f64::NAN).to_le_bytes());
    }
    digest.update(view.tokenizer.merge_count().to_le_bytes());
    for index in 0..view.tokenizer.merge_count() { let pair = view.tokenizer.merge_pair(index).unwrap_or((u64::MAX, u64::MAX)); digest.update(pair.0.to_le_bytes()); digest.update(pair.1.to_le_bytes()); }
    for (kind, value) in view.tokenizer.specials() { digest.update([*kind as u8]); digest.update(value.to_le_bytes()); }
    digest.update([u8::from(view.tokenizer.add_bos().unwrap_or(false))]);
    if let Some(template) = view.tokenizer.chat_template() { digest.update((template.len() as u64).to_le_bytes()); digest.update(template.as_bytes()); }
    format!("{:x}", digest.finalize())
}

fn sha256(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }

fn main() {
    let full_path = PathBuf::from(std::env::args().nth(1).expect("full artifact"));
    let bootstrap_path = PathBuf::from(std::env::args().nth(2).expect("semantic bootstrap"));
    let full_file = File::open(&full_path).expect("open full");
    let full_map = unsafe { Mmap::map(&full_file).expect("map full") };
    let bootstrap_file = File::open(&bootstrap_path).expect("open bootstrap");
    let bootstrap_map = unsafe { Mmap::map(&bootstrap_file).expect("map bootstrap") };
    let full_view = BorrowedModelView::parse(&full_map).expect("full semantic parse");
    let bootstrap_validated = parse_v06(&bootstrap_map).expect("bootstrap canonical parse");
    let bootstrap_profile = parse_source_profile(&bootstrap_validated, &vbuf_ml::Bootstrap::discover(&bootstrap_validated).expect("bootstrap profile" )).expect("source profile parse").expect("source profile role");
    let bootstrap_view = BorrowedModelView::parse_with_sources(&bootstrap_map, &bootstrap_profile.registry, &[]).expect("bootstrap semantic parse");
    let full_digest = digest_view(&full_view);
    let bootstrap_digest = digest_view(&bootstrap_view);
    let source = PositionedFileSource::open(&full_path).expect("open range source");
    let names = [0, full_view.directory.tensors().len() / 2, full_view.directory.tensors().len() - 1];
    let mut ranges = Vec::new();
    for ordinal in names.into_iter().chain((0..full_view.directory.tensors().len()).filter(|index| full_view.directory.tensors()[*index].payload.offset() > u32::MAX as u64).take(1)) {
        let full_tensor = &full_view.directory.tensors()[ordinal];
        let selected = select_tensor_ordinals(&bootstrap_view.directory, &[ordinal]).expect("select tensor");
        let plan = vbuf_ml::ReadPlan::build(&selected, Coalescing::None).expect("build range plan");
        let loaded = execute_plan_with_sources(&SourceSet::new(vec![(SourceId::new(1), &source)]), &plan).expect("read external tensor");
        let external = loaded.target_bytes(&plan.targets()[0]).expect("external bytes");
        let full_bytes = &full_map[full_tensor.payload.range().host_range(full_map.len()).expect("full range")];
        ranges.push(format!("{{\"ordinal\":{},\"offset\":{},\"length\":{},\"type\":{},\"dimensions\":{:?},\"full_sha256\":\"{}\",\"external_sha256\":\"{}\",\"match\":{}}}", ordinal, full_tensor.payload.offset(), full_tensor.payload.length(), full_tensor.representation as u8, full_tensor.dimensions, sha256(full_bytes), sha256(external), full_bytes == external));
    }
    let region_size = |role| bootstrap_view.bootstrap.region(role).map_or(0, |region| region.range.length());
    let bootstrap_payload_bytes = bootstrap_validated.blocks().iter().find(|block| block.key_id == vbuf_ml::bootstrap::BOOTSTRAP_KEY_ID).map_or(0, |block| block.payload_len);
    println!("{{\"full_digest\":\"{}\",\"bootstrap_digest\":\"{}\",\"discovery_match\":{},\"bootstrap_bytes\":{},\"blocks\":{},\"semantic_regions\":{{\"bootstrap\":{},\"model_metadata\":{},\"tensor_directory\":{},\"tokenizer\":{},\"source_metadata\":{}}},\"ranges\":[{}]}}", full_digest, bootstrap_digest, full_digest == bootstrap_digest, bootstrap_map.len(), bootstrap_validated.blocks().len(), bootstrap_payload_bytes, region_size(RegionRole::ModelMetadata), region_size(RegionRole::TensorDirectory), region_size(RegionRole::TokenizerMetadata), region_size(RegionRole::SourceMetadata), ranges.join(","));
}
