use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::time::Instant;
use vbuf_core::v06::{parse_v06, ValidatedV06};
use vbuf_ml::{Bootstrap, ModelMetadata, RegionRole, TensorDirectory, TokenizerMetadata};

fn u16le(b: &[u8], o: usize) -> u16 { u16::from_le_bytes(b[o..o + 2].try_into().unwrap()) }
fn u32le(b: &[u8], o: usize) -> u32 { u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) }
fn resolve(blocks: &[vbuf_core::v06::V06Block], key: u16, occurrence: u16) -> usize {
    blocks.iter().enumerate().filter(|(_, b)| b.key_id == key).nth(occurrence as usize).map(|(i, _)| i).unwrap()
}
fn referenced_indices(validated: &ValidatedV06<'_>, bootstrap: &Bootstrap<'_>) -> (HashMap<usize, String>, u64, u64, u64, u64, u64, u64, u64) {
    let blocks = validated.blocks(); let mut categories = HashMap::new();
    let boot = blocks.iter().position(|b| b.key_id == 0xF000).unwrap(); categories.insert(boot, "bootstrap".to_string());
    let payload = validated.payload_range(boot).unwrap().bytes();
    let count = usize::from(u16le(payload, 12));
    let mut role_blocks = HashMap::new();
    for i in 0..count { let o = 16 + i * 16; let role = u16le(payload, o); let key = u16le(payload, o + 4); let occ = u16le(payload, o + 6); let idx = resolve(blocks, key, occ); role_blocks.insert(role, idx); categories.insert(idx, format!("role_{role}")); }
    let metadata = *role_blocks.get(&(RegionRole::ModelMetadata as u16)).unwrap();
    let directory = *role_blocks.get(&(RegionRole::TensorDirectory as u16)).unwrap();
    let tokenizer = *role_blocks.get(&(RegionRole::TokenizerMetadata as u16)).unwrap();
    let mut metadata_refs = 0u64; let mut tensor_refs = 0u64; let mut token_text = 0u64; let mut token_offsets = 0u64; let mut token_scores = 0u64; let mut token_types = 0u64; let mut merges = 0u64; let mut chat = 0u64;
    let md = validated.payload_range(metadata).unwrap().bytes();
    for i in 0..usize::from(u16le(md, 12)) { let o = 16 + i * 8; categories.insert(resolve(blocks, u16le(md, o + 4), u16le(md, o + 6)), "model_metadata_value".into()); metadata_refs += 1; }
    let td = validated.payload_range(directory).unwrap().bytes(); let mut cursor = 20; let tensor_count = u32le(td, 12);
    for _ in 0..tensor_count { let name_len = usize::from(u16le(td, cursor)); let rank = usize::from(td[cursor + 2]); let key = u16le(td, cursor + 4); let occ = u16le(td, cursor + 6); categories.insert(resolve(blocks, key, occ), "tensor_payload".into()); tensor_refs += 1; cursor += 10 + name_len + rank * 8; }
    let tk = validated.payload_range(tokenizer).unwrap().bytes(); let tk_count = usize::from(u16le(tk, 14));
    for i in 0..tk_count { let o = 20 + i * 12; let role = u16le(tk, o); let idx = resolve(blocks, u16le(tk, o + 4), u16le(tk, o + 6)); let name = match role { 1 => { token_text += 1; "token_text" }, 2 => { token_offsets += 1; "token_offsets" }, 3 => { token_scores += 1; "token_scores" }, 4 => { token_types += 1; "token_types" }, 9 | 10 => { merges += 1; "merge_ids" }, 14 => { chat += 1; "chat_template" }, _ => "tokenizer_control" }; categories.insert(idx, name.into()); }
    let _ = (bootstrap, metadata_refs, tensor_refs, token_text, token_offsets, token_scores, token_types, merges, chat);
    (categories, metadata_refs, tensor_refs, token_text, token_offsets, token_scores, token_types, merges)
}
fn ns(start: Instant) -> u128 { start.elapsed().as_nanos() }
fn main() {
    let args: Vec<String> = std::env::args().collect(); if args.len() < 2 { eprintln!("usage: vbuf-nano-runtime-audit FILE [RUNS]"); std::process::exit(2); }
    let runs: usize = args.get(2).and_then(|x| x.parse().ok()).unwrap_or(20);
    let file = File::open(&args[1]).unwrap(); let mmap = unsafe { Mmap::map(&file).unwrap() };
    let validated = parse_v06(&mmap).unwrap(); let bootstrap = Bootstrap::discover(&validated).unwrap(); let metadata = ModelMetadata::parse(&validated, &bootstrap).unwrap(); let directory = TensorDirectory::parse(&validated, &bootstrap).unwrap(); let tokenizer = TokenizerMetadata::parse(&validated, &bootstrap).unwrap();
    let h = validated.header(); let structural_bytes: u64 = (h.data_region_start - u64::from(h.header_size)) + validated.blocks().iter().map(|b| b.payload_start - b.block_start).sum::<u64>(); let slots = h.data_region_size.div_ceil(h.base_step); let nano_bytes = slots.div_ceil(8); let (categories, metadata_refs, tensor_refs, _, _, _, _, merge_refs) = referenced_indices(&validated, &bootstrap);
    let mut counts: HashMap<&str, usize> = HashMap::new(); let mut category_bytes: HashMap<&str, u64> = HashMap::new();
    for (index, category) in &categories { *counts.entry(category).or_default() += 1; *category_bytes.entry(category).or_default() += validated.blocks()[*index].payload_len; }
    let tensor_name_bytes: usize = directory.tensors().iter().map(|t| t.name.len()).sum(); let tensor_dimension_count: usize = directory.tensors().iter().map(|t| t.dimensions.len()).sum();
    println!("GEOMETRY,file_bytes={},base_step={},data_region_bytes={},slots={},nano_bytes={},blocks={},set_bits={},continuations={},metadata_fields={},tensor_entries={},token_count={},merge_count={},metadata_value_blocks={},tensor_payload_blocks={},token_text_blocks={},token_offset_blocks={},token_score_blocks={},token_type_blocks={},merge_id_blocks={},category_other_blocks={},token_text_bytes={},token_offsets_bytes={},token_scores_bytes={},token_types_bytes={},merge_id_bytes={},tensor_payload_bytes={},metadata_value_bytes={},bootstrap_bytes={},tensor_directory_bytes={},tokenizer_metadata_bytes={},tensor_name_bytes={},tensor_dimension_count={}", mmap.len(), h.base_step, h.data_region_size, slots, nano_bytes, validated.blocks().len(), categories.len(), validated.blocks().iter().filter(|b| b.continuation).count(), metadata.fields().len(), directory.tensors().len(), tokenizer.token_count(), tokenizer.merge_count(), metadata_refs, tensor_refs, counts.get("token_text").copied().unwrap_or(0), counts.get("token_offsets").copied().unwrap_or(0), counts.get("token_scores").copied().unwrap_or(0), counts.get("token_types").copied().unwrap_or(0), merge_refs, validated.blocks().len().saturating_sub(categories.len()), category_bytes.get("token_text").copied().unwrap_or(0), category_bytes.get("token_offsets").copied().unwrap_or(0), category_bytes.get("token_scores").copied().unwrap_or(0), category_bytes.get("token_types").copied().unwrap_or(0), category_bytes.get("merge_ids").copied().unwrap_or(0), category_bytes.get("tensor_payload").copied().unwrap_or(0), category_bytes.get("model_metadata_value").copied().unwrap_or(0), category_bytes.get("bootstrap").copied().unwrap_or(0), category_bytes.get("role_1").copied().unwrap_or(0), category_bytes.get("role_3").copied().unwrap_or(0), tensor_name_bytes, tensor_dimension_count);
    let names: Vec<String> = directory.tensors().iter().map(|t| t.name.clone()).collect(); let mut scratch = 0u64;
    for sample in 0..runs {
        let start = Instant::now(); let parsed = parse_v06(&mmap).unwrap(); scratch ^= parsed.blocks().iter().map(|b| b.block_start).sum::<u64>(); println!("SAMPLE,canonical_parse,{sample},{},{},{}", ns(start), parsed.blocks().len(), structural_bytes);
        let start = Instant::now(); let b = Bootstrap::discover(&parsed).unwrap(); let m = ModelMetadata::parse(&parsed, &b).unwrap(); let d = TensorDirectory::parse(&parsed, &b).unwrap(); let t = TokenizerMetadata::parse(&parsed, &b).unwrap(); scratch ^= m.fields().len() as u64 ^ d.tensors().len() as u64 ^ t.token_count(); println!("SAMPLE,semantic_parse_all,{sample},{},4,0", ns(start));
        let start = Instant::now(); let mut bits = vec![0u8; nano_bytes as usize]; for block in validated.blocks() { let slot = (block.block_start - h.data_region_start) / h.base_step; bits[(slot / 8) as usize] |= 1 << (slot % 8); } println!("SAMPLE,nano_reconstruct,{sample},{},{},{},{}", ns(start), validated.blocks().len(), nano_bytes, 0);
        let start = Instant::now(); let mut found = 0u64; for (byte_i, byte) in bits.iter().enumerate() { let mut v = *byte; while v != 0 { found += 1; v &= v - 1; } scratch ^= byte_i as u64; } println!("SAMPLE,nano_setbit_iter_existing,{sample},{},{},{},{}", ns(start), found, nano_bytes, found);
        let start = Instant::now(); let mut visited = 0u64; for block in validated.blocks() { visited += block.block_start; } scratch ^= visited; println!("SAMPLE,sequential_block_traversal,{sample},{},{},{},{}", ns(start), validated.blocks().len(), validated.blocks().len() * 8, visited);
        let start = Instant::now(); let mut hits = 0u64; for name in &names { if directory.get(name).is_some() { hits += 1; } } scratch ^= hits; println!("SAMPLE,directory_binary_lookup_all_tensors,{sample},{},{},{},{}", ns(start), hits, names.len() * 8, hits);
        let start = Instant::now(); let mut hits = 0u64; for (i, _) in names.iter().enumerate() { hits += directory.tensors().get(i).map_or(0, |_| 1); } scratch ^= hits; println!("SAMPLE,dense_ordinal_view_all_tensors,{sample},{},{},{},{}", ns(start), hits, names.len() * 8, hits);
    }
    eprintln!("scratch={scratch}");
}
