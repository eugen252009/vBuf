//! Downstream vBuf-ML profile boundary.
//!
//! This crate consumes generic values that have already been validated by
//! canonical vBuf. It assigns ML-local roles and relationships; it does not
//! invent primitive types or override physical validity.
//!
//! The base remains a small compositional binary vocabulary: efficient
//! composition is preferred over maximal base functionality.

pub mod bootstrap;
pub mod error;
pub mod region_roles;
pub mod tensor_directory;
pub mod metadata;
pub mod tokenizer;
pub mod representations;
pub mod layout;
pub mod integrity;
pub mod range_loading;

pub use bootstrap::{Bootstrap, BootstrapEntry, SemanticRegion};
pub use error::{MlError, MlErrorCode};
pub use region_roles::RegionRole;
pub use tensor_directory::{TensorDescriptor, TensorDirectory, TensorEntry};
pub use representations::{bf16_bits_to_f32, expected_payload_bytes, logical_elements, representation_contract, representation_from_id, representation_name, validate_tensor_representation, CanonicalStorage, RepresentationContract, TensorRepresentation, BF16_BYTES_PER_ELEMENT, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS};
pub use layout::{canonical_payload_alignment, payload_shift, write_indefinite, write_known_size, LayoutClass, LayoutError, LayoutPlan, PlacementRequest, PlannedBlock};
pub use integrity::{digest_payload, encode_payload as encode_integrity_payload, IntegrityAlgorithm, IntegrityEntry, IntegrityMetadata, IntegrityRecord};
pub use range_loading::{execute_plan, select_tensor_names, select_tensor_ordinals, Coalescing, LoadedPlan, LoadedRead, MmapSource, PhysicalRange, PositionedFileSource, PlannedTarget, RangeLoadError, ReadPlan, RangeSource, SelectedTensor};
pub use metadata::{encode_payload as encode_metadata_payload, MetadataEntry, MetadataField, MetadataValue, ModelMetadata, ModelMetadataKey};
pub use tokenizer::{PreTokenizer, SpecialToken, TokenizerEntry, TokenizerKind, TokenizerMetadata, TokenizerModel, TokenizerRole};

/// Boundary smoke helper for future profile code. It does not parse or
/// authorize the range.
pub fn validated_range_len(range: &vbuf_layout::CheckedRange<'_>) -> usize {
    range.bytes().len()
}
