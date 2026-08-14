//! Profile-local tensor representation contracts.
//!
//! Profile 0.1 intentionally contains only the generic primitive contract.
//! Quantized IDs cannot be selected until a first target and pinned upstream
//! layout revision are recorded; unknown IDs therefore fail closed.

use crate::error::{MlError, MlErrorCode};
use vbuf_core::v06::{V06Semantic, V06Block};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TensorRepresentation {
    CanonicalPrimitive = 0,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RepresentationContract {
    pub id: TensorRepresentation,
    /// `None` means that the canonical primitive descriptor determines size.
    pub logical_elements_per_block: Option<u64>,
    /// `None` means that the canonical primitive descriptor determines size.
    pub physical_bytes_per_block: Option<u64>,
    pub required_payload_alignment: u64,
}

pub const fn representation_contract(id: TensorRepresentation) -> RepresentationContract {
    RepresentationContract {
        id,
        logical_elements_per_block: None,
        physical_bytes_per_block: None,
        required_payload_alignment: 1,
    }
}

pub fn representation_from_id(id: u8) -> Result<TensorRepresentation, MlError> {
    match id {
        0 => Ok(TensorRepresentation::CanonicalPrimitive),
        _ => Err(MlError::new(MlErrorCode::UnsupportedTensorRepresentation, "unsupported tensor representation")),
    }
}

pub fn validate_tensor_representation(
    representation: TensorRepresentation,
    dimensions: &[u64],
    block: &V06Block,
) -> Result<(), MlError> {
    let logical_elements = dimensions.iter().try_fold(1u64, |product, dimension| product.checked_mul(*dimension)).ok_or_else(|| MlError::new(MlErrorCode::ShapeOverflow, "tensor shape product overflows u64"))?;
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
    }
    Ok(())
}
