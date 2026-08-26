//! Step 32H real persisted-token input to full-stack logits qualification.

#[allow(dead_code)]
mod block_runner {
    include!("vbuf-runtime-step32e-block.rs");
}

fn main() -> Result<(), String> {
    let mut arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.len() != 5 {
        return Err(
            "usage: step32h <sidecar> <payload> <checkpoint> <manifest> <qualification-text>"
                .into(),
        );
    }
    let text = arguments.pop().expect("validated text argument");
    arguments.extend(["0".into(), "46".into(), "0".into(), text]);
    block_runner::run_progressive(arguments)
}
