//! Downstream vBuf-ML profile boundary.
//!
//! This crate intentionally contains no tensor, model, tokenizer, quantization,
//! backend, or runtime semantics. It may interpret bytes only after canonical
//! vBuf validation and checked generic range construction. Generic vBuf does
//! not depend on this crate.
//!
//! The base remains a small compositional binary vocabulary: efficient
//! composition is preferred over maximal base functionality.

use vbuf_layout::CheckedRange;

/// Minimal boundary smoke helper for future profile code.
///
/// This reports the size of a range already validated by a generic container;
/// it does not parse, reinterpret, or authorize the range.
pub fn validated_range_len(range: &CheckedRange<'_>) -> usize {
    range.bytes().len()
}
