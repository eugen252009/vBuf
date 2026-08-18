use memmap2::Mmap;
use std::fs::File;
use std::hint::black_box;
use std::time::Instant;
use vbuf_core::v06::parse_v06;
use vbuf_ml::{parse_source_profile, BorrowedModelView, Bootstrap};

fn main() {
    let bootstrap_path = std::env::args().nth(1).expect("bootstrap");
    let range_path = std::env::args().nth(2).expect("range");
    let bootstrap_file = File::open(bootstrap_path).unwrap();
    let bootstrap_map = unsafe { Mmap::map(&bootstrap_file).unwrap() };
    let validated = parse_v06(&bootstrap_map).unwrap();
    let mut discovery_ns = Vec::new();
    for _ in 0..10 {
        let start = Instant::now();
        let bootstrap = Bootstrap::discover(&validated).unwrap();
        let profile = parse_source_profile(&validated, &bootstrap).unwrap().unwrap();
        let view = BorrowedModelView::parse_with_sources(&bootstrap_map, &profile.registry, &[]).unwrap();
        black_box((view.directory.tensors().len(), view.tokenizer.token_count()));
        discovery_ns.push(start.elapsed().as_nanos());
    }
    let bootstrap = Bootstrap::discover(&validated).unwrap();
    let profile = parse_source_profile(&validated, &bootstrap).unwrap().unwrap();
    let view = BorrowedModelView::parse_with_sources(&bootstrap_map, &profile.registry, &[]).unwrap();
    let start = Instant::now();
    let mut lookup_sink = 0u64;
    for _ in 0..1_000_000 {
        let tensor = &view.directory.tensors()[22];
        lookup_sink = lookup_sink.wrapping_add(tensor.payload.offset()).wrapping_add(tensor.payload.length());
        black_box(tensor.payload.source_id());
    }
    black_box(lookup_sink);
    let lookup_ns = start.elapsed().as_nanos();
    let bytes = std::fs::read(range_path).unwrap();
    let mut materialization_ns = Vec::new();
    for _ in 0..10 {
        let start = Instant::now();
        let copy = bytes.clone();
        black_box(copy);
        materialization_ns.push(start.elapsed().as_nanos());
    }
    println!("{{\"discovery_ns\":{:?},\"tensor_lookup_iterations\":1000000,\"tensor_lookup_total_ns\":{},\"materialization_ns\":{:?},\"materialized_bytes\":{}}}", discovery_ns, lookup_ns, materialization_ns, bytes.len());
}
