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
pub const Q4_0_BLOCK_ELEMENTS: u64 = 32;
pub const Q4_0_BLOCK_BYTES: u64 = 18;
pub const Q2_K_BLOCK_ELEMENTS: u64 = 256;
pub const Q2_K_BLOCK_BYTES: u64 = 84;
pub const IQ1_S_BLOCK_ELEMENTS: u64 = 256;
pub const IQ1_S_BLOCK_BYTES: u64 = 50;
pub const Q4_K_BLOCK_ELEMENTS: u64 = 256;
pub const Q4_K_BLOCK_BYTES: u64 = 144;
pub const IQ4_NL_BLOCK_ELEMENTS: u64 = 32;
pub const IQ4_NL_BLOCK_BYTES: u64 = 18;
pub const IQ4_XS_BLOCK_ELEMENTS: u64 = 256;
pub const IQ4_XS_BLOCK_BYTES: u64 = 136;
pub const Q3_K_BLOCK_ELEMENTS: u64 = 256;
pub const Q3_K_BLOCK_BYTES: u64 = 110;
pub const IQ2_XXS_BLOCK_ELEMENTS: u64 = 256;
pub const IQ2_XXS_BLOCK_BYTES: u64 = 66;
pub const IQ2_XS_BLOCK_ELEMENTS: u64 = 256;
pub const IQ2_XS_BLOCK_BYTES: u64 = 74;
pub const IQ2_S_BLOCK_ELEMENTS: u64 = 256;
pub const IQ2_S_BLOCK_BYTES: u64 = 82;
pub const Q5_K_BLOCK_ELEMENTS: u64 = 256;
pub const Q5_K_BLOCK_BYTES: u64 = 176;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
#[allow(non_camel_case_types)]
pub enum TensorRepresentation {
    CanonicalPrimitive = 0,
    Bf16 = 1,
    GgmlQ8_0 = 2,
    GgmlQ4_0 = 3,
    GgmlQ2_K = 4,
    GgmlIQ1_S = 5,
    GgmlQ4_K = 6,
    GgmlIQ4_NL = 7,
    GgmlIQ4_XS = 8,
    GgmlQ3_K = 9,
    GgmlIQ2_XXS = 10,
    GgmlIQ2_XS = 11,
    GgmlIQ2_S = 12,
    GgmlQ5_K = 13,
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
        TensorRepresentation::GgmlQ4_0 => RepresentationContract {
            id, canonical_storage: CanonicalStorage::OpaqueBytes,
            logical_elements_per_block: Some(Q4_0_BLOCK_ELEMENTS), physical_bytes_per_block: Some(Q4_0_BLOCK_BYTES),
            required_payload_alignment: 1, ggml_type_id: Some(2),
        },
        TensorRepresentation::GgmlQ2_K => RepresentationContract {
            id, canonical_storage: CanonicalStorage::OpaqueBytes,
            logical_elements_per_block: Some(Q2_K_BLOCK_ELEMENTS), physical_bytes_per_block: Some(Q2_K_BLOCK_BYTES),
            required_payload_alignment: 1, ggml_type_id: Some(10),
        },
        TensorRepresentation::GgmlIQ1_S => RepresentationContract {
            id, canonical_storage: CanonicalStorage::OpaqueBytes,
            logical_elements_per_block: Some(IQ1_S_BLOCK_ELEMENTS), physical_bytes_per_block: Some(IQ1_S_BLOCK_BYTES),
            required_payload_alignment: 1, ggml_type_id: Some(19),
        },
        TensorRepresentation::GgmlQ4_K => RepresentationContract {
            id, canonical_storage: CanonicalStorage::OpaqueBytes,
            logical_elements_per_block: Some(Q4_K_BLOCK_ELEMENTS), physical_bytes_per_block: Some(Q4_K_BLOCK_BYTES),
            required_payload_alignment: 1, ggml_type_id: Some(12),
        },
        TensorRepresentation::GgmlIQ4_NL => RepresentationContract {
            id, canonical_storage: CanonicalStorage::OpaqueBytes,
            logical_elements_per_block: Some(IQ4_NL_BLOCK_ELEMENTS), physical_bytes_per_block: Some(IQ4_NL_BLOCK_BYTES),
            required_payload_alignment: 1, ggml_type_id: Some(20),
        },
        TensorRepresentation::GgmlIQ4_XS => RepresentationContract {
            id, canonical_storage: CanonicalStorage::OpaqueBytes,
            logical_elements_per_block: Some(IQ4_XS_BLOCK_ELEMENTS), physical_bytes_per_block: Some(IQ4_XS_BLOCK_BYTES),
            required_payload_alignment: 1, ggml_type_id: Some(23),
        },
        TensorRepresentation::GgmlQ3_K => RepresentationContract {
            id, canonical_storage: CanonicalStorage::OpaqueBytes,
            logical_elements_per_block: Some(Q3_K_BLOCK_ELEMENTS), physical_bytes_per_block: Some(Q3_K_BLOCK_BYTES),
            required_payload_alignment: 1, ggml_type_id: Some(11),
        },
        TensorRepresentation::GgmlIQ2_XXS => RepresentationContract {
            id, canonical_storage: CanonicalStorage::OpaqueBytes,
            logical_elements_per_block: Some(IQ2_XXS_BLOCK_ELEMENTS), physical_bytes_per_block: Some(IQ2_XXS_BLOCK_BYTES),
            required_payload_alignment: 1, ggml_type_id: Some(16),
        },
        TensorRepresentation::GgmlIQ2_XS => RepresentationContract {
            id, canonical_storage: CanonicalStorage::OpaqueBytes,
            logical_elements_per_block: Some(IQ2_XS_BLOCK_ELEMENTS), physical_bytes_per_block: Some(IQ2_XS_BLOCK_BYTES),
            required_payload_alignment: 1, ggml_type_id: Some(17),
        },
        TensorRepresentation::GgmlIQ2_S => RepresentationContract {
            id, canonical_storage: CanonicalStorage::OpaqueBytes,
            logical_elements_per_block: Some(IQ2_S_BLOCK_ELEMENTS), physical_bytes_per_block: Some(IQ2_S_BLOCK_BYTES),
            required_payload_alignment: 1, ggml_type_id: Some(22),
        },
        TensorRepresentation::GgmlQ5_K => RepresentationContract {
            id, canonical_storage: CanonicalStorage::OpaqueBytes,
            logical_elements_per_block: Some(Q5_K_BLOCK_ELEMENTS), physical_bytes_per_block: Some(Q5_K_BLOCK_BYTES),
            required_payload_alignment: 1, ggml_type_id: Some(13),
        },
    }
}

pub fn representation_name(id: TensorRepresentation) -> &'static str {
    match id {
        TensorRepresentation::CanonicalPrimitive => "CanonicalPrimitive",
        TensorRepresentation::Bf16 => "BF16",
        TensorRepresentation::GgmlQ8_0 => "GGML_Q8_0",
        TensorRepresentation::GgmlQ4_0 => "GGML_Q4_0",
        TensorRepresentation::GgmlQ2_K => "GGML_Q2_K",
        TensorRepresentation::GgmlIQ1_S => "GGML_IQ1_S",
        TensorRepresentation::GgmlQ4_K => "GGML_Q4_K",
        TensorRepresentation::GgmlIQ4_NL => "GGML_IQ4_NL",
        TensorRepresentation::GgmlIQ4_XS => "GGML_IQ4_XS",
        TensorRepresentation::GgmlQ3_K => "GGML_Q3_K",
        TensorRepresentation::GgmlIQ2_XXS => "GGML_IQ2_XXS",
        TensorRepresentation::GgmlIQ2_XS => "GGML_IQ2_XS",
        TensorRepresentation::GgmlIQ2_S => "GGML_IQ2_S",
        TensorRepresentation::GgmlQ5_K => "GGML_Q5_K",
    }
}

pub fn representation_from_id(id: u8) -> Result<TensorRepresentation, MlError> {
    match id {
        0 => Ok(TensorRepresentation::CanonicalPrimitive),
        1 => Ok(TensorRepresentation::Bf16),
        2 => Ok(TensorRepresentation::GgmlQ8_0),
        3 => Ok(TensorRepresentation::GgmlQ4_0),
        4 => Ok(TensorRepresentation::GgmlQ2_K),
        5 => Ok(TensorRepresentation::GgmlIQ1_S),
        6 => Ok(TensorRepresentation::GgmlQ4_K),
        7 => Ok(TensorRepresentation::GgmlIQ4_NL),
        8 => Ok(TensorRepresentation::GgmlIQ4_XS),
        9 => Ok(TensorRepresentation::GgmlQ3_K),
        10 => Ok(TensorRepresentation::GgmlIQ2_XXS),
        11 => Ok(TensorRepresentation::GgmlIQ2_XS),
        12 => Ok(TensorRepresentation::GgmlIQ2_S),
        13 => Ok(TensorRepresentation::GgmlQ5_K),
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
        TensorRepresentation::GgmlQ8_0 | TensorRepresentation::GgmlQ4_0 | TensorRepresentation::GgmlQ2_K | TensorRepresentation::GgmlIQ1_S | TensorRepresentation::GgmlQ4_K | TensorRepresentation::GgmlIQ4_NL | TensorRepresentation::GgmlIQ4_XS | TensorRepresentation::GgmlQ3_K | TensorRepresentation::GgmlIQ2_XXS | TensorRepresentation::GgmlIQ2_XS | TensorRepresentation::GgmlIQ2_S | TensorRepresentation::GgmlQ5_K => {
            let row_width = *dimensions.first().ok_or_else(|| MlError::new(MlErrorCode::InvalidQuantizedShape, "Q8_0 requires a row dimension"))?;
            let (block_elements, block_bytes, label) = match representation {
                TensorRepresentation::GgmlQ8_0 => (Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES, "Q8_0"),
                TensorRepresentation::GgmlQ4_0 => (Q4_0_BLOCK_ELEMENTS, Q4_0_BLOCK_BYTES, "Q4_0"),
                TensorRepresentation::GgmlQ2_K => (Q2_K_BLOCK_ELEMENTS, Q2_K_BLOCK_BYTES, "Q2_K"),
                TensorRepresentation::GgmlIQ1_S => (IQ1_S_BLOCK_ELEMENTS, IQ1_S_BLOCK_BYTES, "IQ1_S"),
                TensorRepresentation::GgmlQ4_K => (Q4_K_BLOCK_ELEMENTS, Q4_K_BLOCK_BYTES, "Q4_K"),
                TensorRepresentation::GgmlIQ4_NL => (IQ4_NL_BLOCK_ELEMENTS, IQ4_NL_BLOCK_BYTES, "IQ4_NL"),
                TensorRepresentation::GgmlIQ4_XS => (IQ4_XS_BLOCK_ELEMENTS, IQ4_XS_BLOCK_BYTES, "IQ4_XS"),
                TensorRepresentation::GgmlQ3_K => (Q3_K_BLOCK_ELEMENTS, Q3_K_BLOCK_BYTES, "Q3_K"),
                TensorRepresentation::GgmlIQ2_XXS => (IQ2_XXS_BLOCK_ELEMENTS, IQ2_XXS_BLOCK_BYTES, "IQ2_XXS"),
                TensorRepresentation::GgmlIQ2_XS => (IQ2_XS_BLOCK_ELEMENTS, IQ2_XS_BLOCK_BYTES, "IQ2_XS"),
                TensorRepresentation::GgmlIQ2_S => (IQ2_S_BLOCK_ELEMENTS, IQ2_S_BLOCK_BYTES, "IQ2_S"),
                TensorRepresentation::GgmlQ5_K => (Q5_K_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES, "Q5_K"),
                _ => unreachable!(),
            };
            if row_width == 0 || row_width % block_elements != 0 {
                return Err(MlError::new(MlErrorCode::InvalidQuantizedShape, "quantized row width is not divisible by its block width"));
            }
            let rows = dimensions[1..].iter().try_fold(1u64, |product, dimension| product.checked_mul(*dimension)).ok_or_else(|| MlError::new(MlErrorCode::InvalidQuantizedBlockCount, "Q8_0 row count overflows u64"))?;
            let blocks_per_row = row_width / block_elements;
            let _ = label;
            rows.checked_mul(blocks_per_row).and_then(|blocks| blocks.checked_mul(block_bytes)).ok_or_else(|| MlError::new(MlErrorCode::RepresentationArithmeticOverflow, "quantized payload size overflows u64"))
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
        TensorRepresentation::Bf16 | TensorRepresentation::GgmlQ8_0 | TensorRepresentation::GgmlQ4_0 | TensorRepresentation::GgmlQ2_K | TensorRepresentation::GgmlIQ1_S | TensorRepresentation::GgmlQ4_K | TensorRepresentation::GgmlIQ4_NL | TensorRepresentation::GgmlIQ4_XS | TensorRepresentation::GgmlQ3_K | TensorRepresentation::GgmlIQ2_XXS | TensorRepresentation::GgmlIQ2_XS | TensorRepresentation::GgmlIQ2_S | TensorRepresentation::GgmlQ5_K => {
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

/// Validates tensor geometry against an external payload length while the
/// metadata artifact contains only a zero-length canonical placeholder block.
pub fn validate_external_tensor_representation(
    representation: TensorRepresentation,
    dimensions: &[u64],
    block: &V06Block,
    payload_len: u64,
) -> Result<(), MlError> {
    let logical_elements = logical_elements(dimensions)?;
    match representation {
        TensorRepresentation::CanonicalPrimitive => {
            if !matches!(block.semantic, V06Semantic::Unsigned | V06Semantic::Signed | V06Semantic::Float) || !block.bit_width.is_multiple_of(8) {
                return Err(MlError::new(MlErrorCode::TensorRepresentationMismatch, "canonical external primitive does not match tensor geometry"));
            }
            let expected = logical_elements.checked_mul(u64::from(block.bit_width / 8)).ok_or_else(|| MlError::new(MlErrorCode::RepresentationArithmeticOverflow, "external tensor payload size overflows u64"))?;
            if payload_len != expected { return Err(MlError::new(MlErrorCode::TensorPayloadSizeMismatch, "external canonical payload length does not match representation")); }
        }
        TensorRepresentation::Bf16 | TensorRepresentation::GgmlQ8_0 | TensorRepresentation::GgmlQ4_0 | TensorRepresentation::GgmlQ2_K | TensorRepresentation::GgmlIQ1_S | TensorRepresentation::GgmlQ4_K | TensorRepresentation::GgmlIQ4_NL | TensorRepresentation::GgmlIQ4_XS | TensorRepresentation::GgmlQ3_K | TensorRepresentation::GgmlIQ2_XXS | TensorRepresentation::GgmlIQ2_XS | TensorRepresentation::GgmlIQ2_S | TensorRepresentation::GgmlQ5_K => {
            if block.semantic != V06Semantic::Opaque || block.physical != V06Physical::Array || block.bit_width != 8 {
                return Err(MlError::new(MlErrorCode::TensorRepresentationMismatch, "packed external representation requires an opaque byte array"));
            }
            let expected = expected_payload_bytes(representation, dimensions)?;
            if payload_len != expected { return Err(MlError::new(MlErrorCode::TensorPayloadSizeMismatch, "external packed payload length does not match representation")); }
        }
    }
    Ok(())
}
