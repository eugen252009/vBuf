//! Step 32F progressive real-layer qualification entry point.

#[allow(dead_code)]
mod block_runner {
    include!("vbuf-runtime-step32e-block.rs");
}

fn main() -> Result<(), String> {
    block_runner::run_progressive(std::env::args().skip(1).collect())
}
