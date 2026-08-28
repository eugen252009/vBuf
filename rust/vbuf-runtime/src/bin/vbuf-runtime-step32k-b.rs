#[cfg(feature = "cuda")]
#[allow(dead_code)]
mod step32e {
    include!("vbuf-runtime-step32e-block.rs");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), String> {
    step32e::run_cuda_progressive(std::env::args().skip(1).collect())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("vbuf-runtime-step32k-b requires --features cuda");
    std::process::exit(2);
}
