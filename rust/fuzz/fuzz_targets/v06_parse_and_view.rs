#![no_main]

use libfuzzer_sys::fuzz_target;
use vbuf_core::v06::{V06Semantic, parse_v06};

fuzz_target!(|data: &[u8]| {
    let Ok(parsed) = parse_v06(data) else {
        return;
    };
    for (index, block) in parsed.blocks().iter().enumerate() {
        let occurrence = parsed.blocks()[..index]
            .iter()
            .filter(|previous| previous.key_id == block.key_id)
            .count();
        match (block.semantic, block.bit_width) {
            (V06Semantic::Unsigned, 8) => {
                let _ = parsed.u8_view(block.key_id, occurrence);
            }
            (V06Semantic::Unsigned, 16) => {
                let _ = parsed.u16_view(block.key_id, occurrence);
            }
            (V06Semantic::Unsigned, 32) => {
                let _ = parsed.u32_view(block.key_id, occurrence);
            }
            (V06Semantic::Unsigned, 64) => {
                let _ = parsed.u64_view(block.key_id, occurrence);
            }
            (V06Semantic::Signed, 8) => {
                let _ = parsed.i8_view(block.key_id, occurrence);
            }
            (V06Semantic::Signed, 16) => {
                let _ = parsed.i16_view(block.key_id, occurrence);
            }
            (V06Semantic::Signed, 32) => {
                let _ = parsed.i32_view(block.key_id, occurrence);
            }
            (V06Semantic::Signed, 64) => {
                let _ = parsed.i64_view(block.key_id, occurrence);
            }
            (V06Semantic::Float, 32) => {
                let _ = parsed.f32_view(block.key_id, occurrence);
            }
            (V06Semantic::Float, 64) => {
                let _ = parsed.f64_view(block.key_id, occurrence);
            }
            (V06Semantic::Opaque, 8) => {
                let _ = parsed.opaque_bytes(block.key_id, occurrence);
            }
            _ => unreachable!("validated parser admitted an unsupported representation"),
        }
    }
});
