//! Profile-local tensor representation contracts.
//!
//! The packed contracts in this module are deliberately limited to the exact
//! representations present in the pinned Step-17 evidence corpus.  Their
//! external meanings are documented in `docs/vbuf-ml/step17-upstream-representation-qualification.md`.

use crate::error::{MlError, MlErrorCode};
use vbuf_core::v06::{V06Block, V06Physical, V06Semantic};

pub const Q8_0_BLOCK_ELEMENTS: u64 = 32;
pub const Q8_0_BLOCK_BYTES: u64 = 34;
pub const BF16_BYTES_PER_ELEMENT: u64 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TensorRepresentation {
    CanonicalPrimitive = 0,
    Bf16 = 1,
    GgmlQ8_0 = 2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CanonicalStorage {
    Primitive,
    OpaqueBytes,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RepresentationContract {
    pub id: TensorRepresentation,
    pub canonical_storage: CanonicalStorage,
    /// `None` means that the canonical primitive descriptor determines size.
    pub logical_elements_per_block: Option<u64>,
    /// `None` means that the canonical primitive descriptor determines size.
    pub physical_bytes_per_block: Option<u64>,
    pub required_payload_alignment: u64,
    pub ggml_type_id: Option<u32>,
}

pub const fn representation_contract(id: TensorRepresentation) -> RepresentationContract {
    match id {
        TensorRepresentation::CanonicalPrimitive => RepresentationContract {
            id,
            canonical_storage: CanonicalStorage::Primitive,
            logical_elements_per_block: None,
            physical_bytes_per_block: None,
            required_payload_alignment: 1,
            ggml_type_id: Some(0),
        },
        TensorRepresentation::Bf16 => RepresentationContract {
            id,
            canonical_storage: CanonicalStorage::OpaqueBytes,
            logical_elements_per_block: Some(1),
            physical_bytes_per_block: Some(BF16_BYTES_PER_ELEMENT),
            required_payload_alignment: 1,
            ggml_type_id: Some(30),
        },
        TensorRepresentation::GgmlQ8_0 => RepresentationContract {
            id,
            canonical_storage: CanonicalStorage::OpaqueBytes,
            logical_elements_per_block: Some(Q8_0_BLOCK_ELEMENTS),
            physical_bytes_per_block: Some(Q8_0_BLOCK_BYTES),
            required_payload_alignment: 1,
            ggml_type_id: Some(8),
        },
    }
}

pub fn representation_name(id: TensorRepresentation) -> &'static str {
    match id {
        TensorRepresentation::CanonicalPrimitive => "CanonicalPrimitive",
        TensorRepresentation::Bf16 => "BF16",
        TensorRepresentation::GgmlQ8_0 => "GGML_Q8_0",
    }
}

pub fn representation_from_id(id: u8) -> Result<TensorRepresentation, MlError> {
    match id {
        0 => Ok(TensorRepresentation::CanonicalPrimitive),
        1 => Ok(TensorRepresentation::Bf16),
        2 => Ok(TensorRepresentation::GgmlQ8_0),
        _ => Err(MlError::new(MlErrorCode::UnsupportedTensorRepresentation, "unsupported tensor representation")),
    }
}

pub fn logical_elements(dimensions: &[u64]) -> Result<u64, MlError> {
    dimensions.iter().try_fold(1u64, |product, dimension| product.checked_mul(*dimension)).ok_or_else(|| MlError::new(MlErrorCode::ShapeOverflow, "tensor shape product overflows u64"))
}

pub fn expected_payload_bytes(representation: TensorRepresentation, dimensions: &[u64]) -> Result<u64, MlError> {
    let elements = logical_elements(dimensions)?;
    match representation {
        TensorRepresentation::CanonicalPrimitive => Err(MlError::new(MlErrorCode::TensorRepresentationMismatch, "canonical primitive size depends on its descriptor")),
        TensorRepresentation::Bf16 => elements.checked_mul(BF16_BYTES_PER_ELEMENT).ok_or_else(|| MlError::new(MlErrorCode::RepresentationArithmeticOverflow, "BF16 payload size overflows u64")),
        TensorRepresentation::GgmlQ8_0 => {
            let row_width = *dimensions.first().ok_or_else(|| MlError::new(MlErrorCode::InvalidQuantizedShape, "Q8_0 requires a row dimension"))?;
            if row_width == 0 || row_width % Q8_0_BLOCK_ELEMENTS != 0 {
                return Err(MlError::new(MlErrorCode::InvalidQuantizedShape, "Q8_0 row width must be divisible by 32"));
            }
            let rows = dimensions[1..].iter().try_fold(1u64, |product, dimension| product.checked_mul(*dimension)).ok_or_else(|| MlError::new(MlErrorCode::InvalidQuantizedBlockCount, "Q8_0 row count overflows u64"))?;
            let blocks_per_row = row_width / Q8_0_BLOCK_ELEMENTS;
            rows.checked_mul(blocks_per_row).and_then(|blocks| blocks.checked_mul(Q8_0_BLOCK_BYTES)).ok_or_else(|| MlError::new(MlErrorCode::RepresentationArithmeticOverflow, "Q8_0 payload size overflows u64"))
        }
    }
}

/// Interpret the explicitly stored BF16 bits without relying on a host ABI.
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

pub fn validate_tensor_representation(
    representation: TensorRepresentation,
    dimensions: &[u64],
    block: &V06Block,
) -> Result<(), MlError> {
    let logical_elements = logical_elements(dimensions)?;
    match representation {
        TensorRepresentation::CanonicalPrimitive => {
            if !matches!(block.semantic, V06Semantic::Unsigned | V06Semantic::Signed | V06Semantic::Float) || !block.bit_width.is_multiple_of(8) || block.count != logical_elements {
                return Err(MlError::new(MlErrorCode::TensorRepresentationMismatch, "canonical primitive does not match tensor shape"));
            }
            let expected_bytes = logical_elements.checked_mul(u64::from(block.bit_width / 8)).ok_or_else(|| MlError::new(MlErrorCode::RepresentationArithmeticOverflow, "tensor payload size overflows u64"))?;
            if block.payload_len != expected_bytes {
                return Err(MlError::new(MlErrorCode::TensorPayloadSizeMismatch, "tensor payload length does not match representation"));
            }
        }
        TensorRepresentation::Bf16 | TensorRepresentation::GgmlQ8_0 => {
            if block.semantic != V06Semantic::Opaque || block.physical != V06Physical::Array || block.bit_width != 8 || block.count != expected_payload_bytes(representation, dimensions)? {
                return Err(MlError::new(MlErrorCode::TensorRepresentationMismatch, "packed representation requires an opaque byte array"));
            }
            let expected_bytes = expected_payload_bytes(representation, dimensions)?;
            if block.payload_len != expected_bytes {
                return Err(MlError::new(MlErrorCode::TensorPayloadSizeMismatch, "packed tensor payload length does not match representation"));
            }
        }
    }
    Ok(())
}
