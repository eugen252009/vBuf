use std::{collections::HashMap, env, time::Instant};
use memmap2::Mmap;
use std::fs::File;
use vbuf_core::v06::parse_v06;
use vbuf_ml::{BorrowedModel, BorrowedModelView, Bootstrap, ConsumerModel, ModelMetadata, TensorDirectory, TokenizerMetadata};

fn ns(start: Instant) -> u128 { start.elapsed().as_nanos() }
fn main() {
    let path = env::args().nth(1).expect("model path");
    let runs: usize = env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(10);
    let file = File::open(&path).unwrap(); let mapping = unsafe { Mmap::map(&file).unwrap() };
    println!("artifact,operation,sample,nanos,entries,bytes,allocating");
    for sample in 0..runs {
        let start = Instant::now(); let validated = parse_v06(&mapping).unwrap();
        println!("model,canonical_validation,{sample},{},{},{},false", ns(start), validated.blocks().len(), 0);
        let start = Instant::now(); let view = BorrowedModelView::parse(&mapping).unwrap();
        println!("model,borrowed_view_establishment,{sample},{},{},{},false", ns(start), view.directory.tensors().len(), view.tokenizer.text_bytes().len() + view.tokenizer.offset_bytes().len());
        let start = Instant::now(); let mut physical = Vec::with_capacity(validated.blocks().len()); for block in validated.blocks() { physical.push((block.block_start, block.key_id, block.payload_start, block.payload_end)); }
        println!("index,canonical_descriptors,{sample},{},{},{},true", ns(start), physical.len(), physical.len() * std::mem::size_of::<(u64, u16, u64, u64)>());
        let start = Instant::now(); let slots = validated.header().data_region_size.div_ceil(validated.header().base_step); let mut nano = vec![0u8; slots.div_ceil(8) as usize]; for block in validated.blocks() { let slot = (block.block_start - validated.header().data_region_start) / validated.header().base_step; nano[(slot / 8) as usize] |= 1 << (slot % 8); } let mut minimal_headers = HashMap::with_capacity(validated.blocks().len()); for block in validated.blocks() { minimal_headers.insert((block.block_start - validated.header().data_region_start) / validated.header().base_step, (block.key_id, block.payload_start)); } let mut derived = 0usize; for (byte_index, byte) in nano.iter().enumerate() { let mut bits = *byte; while bits != 0 { let bit = bits.trailing_zeros() as u64; let slot = byte_index as u64 * 8 + bit; if minimal_headers.contains_key(&slot) { derived += 1; } bits &= bits - 1; } }
        println!("index,nano_plus_minimal_headers,{sample},{},{},{},true", ns(start), derived, nano.len());
        let start = Instant::now(); let mut token_index: HashMap<Vec<u8>, u32> = HashMap::with_capacity(view.tokenizer.token_count() as usize);
        for i in 0..view.tokenizer.token_count() { token_index.insert(view.tokenizer.token_text(i).unwrap().as_bytes().to_vec(), i as u32); }
        println!("model,eager_token_index,{sample},{},{},{},true", ns(start), token_index.len(), token_index.keys().map(Vec::len).sum::<usize>());
        let start = Instant::now(); let mut merge_index: HashMap<(u64,u64), u32> = HashMap::with_capacity(view.tokenizer.merge_count() as usize);
        for i in 0..view.tokenizer.merge_count() { merge_index.insert(view.tokenizer.merge_pair(i).unwrap(), i as u32); }
        println!("model,eager_merge_index,{sample},{},{},{},true", ns(start), merge_index.len(), merge_index.len() * std::mem::size_of::<((u64,u64),u32)>());
        let start = Instant::now(); let _owned = BorrowedModel::open(&path).unwrap();
        println!("model,borrowed_model_open_total,{sample},{},{},{},false", ns(start), 0, 0);
        let start = Instant::now(); let _snapshot = ConsumerModel::open(&path).unwrap();
        println!("model,owned_consumer_open_total,{sample},{},{},{},true", ns(start), 0, 0);
        let start = Instant::now(); let _ = (Bootstrap::discover(&validated).unwrap(), ModelMetadata::parse(&validated, &Bootstrap::discover(&validated).unwrap()).unwrap(), TensorDirectory::parse(&validated, &Bootstrap::discover(&validated).unwrap()).unwrap(), TokenizerMetadata::parse(&validated, &Bootstrap::discover(&validated).unwrap()).unwrap());
        println!("model,legacy_semantic_parse_control,{sample},{},{},{},true", ns(start), 0, 0);
        let start = Instant::now(); let mut semantic = HashMap::with_capacity(view.directory.tensors().len()); for (i, tensor) in view.directory.tensors().iter().enumerate() { semantic.insert(tensor.name.as_str(), i); }
        println!("index,semantic_directory,{sample},{},{},{},true", ns(start), semantic.len(), view.directory.tensors().len());
    }
}
