//! Step 32I retained-KV one-token decode qualification entry point.

#[allow(dead_code)]
mod block_runner {
    include!("vbuf-runtime-step32e-block.rs");
}

fn main() -> Result<(), String> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.len() != 5 {
        return Err(
            "usage: step32i <sidecar> <payload> <checkpoint> <manifest> <qualification-text>"
                .into(),
        );
    }
    block_runner::run_retained_kv(arguments)
}
