//! Step 32J repeated greedy autoregressive generation qualification entry point.

#[allow(dead_code)]
mod block_runner {
    include!("vbuf-runtime-step32e-block.rs");
}

fn main() -> Result<(), String> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.len() != 6 && arguments.len() != 7 {
        return Err(
            "usage: step32j <sidecar> <payload> <checkpoint> <manifest> <qualification-text> <max-new-tokens>"
                .into(),
        );
    }
    block_runner::run_repeated_generation(arguments)
}
