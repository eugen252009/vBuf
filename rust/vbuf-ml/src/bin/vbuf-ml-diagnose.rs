use memmap2::Mmap;
use std::path::PathBuf;
use std::time::Instant;
use vbuf_core::v06::parse_v06;
use vbuf_ml::{Bootstrap, ConsumerModel, ModelMetadata, TensorDirectory, TokenizerMetadata};

fn main() {
    let path = PathBuf::from(std::env::args().nth(1).expect("vbuf path"));
    let started = Instant::now();
    let file = std::fs::File::open(&path).expect("open");
    let map_start = Instant::now();
    let mapping = unsafe { Mmap::map(&file).expect("mmap") };
    let map_us = map_start.elapsed().as_secs_f64() * 1e6;
    let parse_start = Instant::now();
    let validated = parse_v06(&mapping).expect("canonical validation");
    let parse_us = parse_start.elapsed().as_secs_f64() * 1e6;
    let bootstrap_start = Instant::now();
    let bootstrap = Bootstrap::discover(&validated).expect("bootstrap");
    let bootstrap_us = bootstrap_start.elapsed().as_secs_f64() * 1e6;
    let metadata_start = Instant::now();
    let _metadata = ModelMetadata::parse(&validated, &bootstrap).expect("metadata");
    let metadata_us = metadata_start.elapsed().as_secs_f64() * 1e6;
    let directory_start = Instant::now();
    let directory = TensorDirectory::parse(&validated, &bootstrap).expect("directory");
    let directory_us = directory_start.elapsed().as_secs_f64() * 1e6;
    let tokenizer_start = Instant::now();
    let tokenizer = TokenizerMetadata::parse(&validated, &bootstrap).expect("tokenizer");
    let tokenizer_us = tokenizer_start.elapsed().as_secs_f64() * 1e6;
    let descriptor_start = Instant::now();
    let consumer = ConsumerModel::open(&path).expect("consumer");
    let descriptor_us = descriptor_start.elapsed().as_secs_f64() * 1e6;
    println!(
        "{{\"phase\":\"vbuf_diagnostic\",\"file_bytes\":{},\"blocks\":{},\"tensors\":{},\"tokens\":{},\"merges\":{},\"map_us\":{:.3},\"canonical_us\":{:.3},\"bootstrap_us\":{:.3},\"model_metadata_us\":{:.3},\"tensor_directory_us\":{:.3},\"tokenizer_metadata_us\":{:.3},\"consumer_open_total_us\":{:.3},\"descriptor_tensors\":{},\"elapsed_us\":{:.3}}}",
        mapping.len(),
        validated.blocks().len(),
        directory.tensors().len(),
        tokenizer.token_count(),
        tokenizer.merge_count(),
        map_us,
        parse_us,
        bootstrap_us,
        metadata_us,
        directory_us,
        tokenizer_us,
        descriptor_us,
        consumer.tensor_count().unwrap(),
        started.elapsed().as_secs_f64() * 1e6
    );
}
