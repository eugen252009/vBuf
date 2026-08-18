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
pub mod source;
pub mod consumer;
pub mod consumer_ffi;
pub mod runtime_tokenizer;
pub mod nested;
pub mod moe;
pub mod qwen_moe;
pub mod deepseek_moe;

pub use bootstrap::{Bootstrap, BootstrapEntry, SemanticRegion};
pub use error::{MlError, MlErrorCode};
pub use region_roles::RegionRole;
pub use tensor_directory::{TensorDescriptor, TensorDirectory, TensorEntry};
pub use representations::{bf16_bits_to_f32, expected_payload_bytes, logical_elements, representation_contract, representation_from_id, representation_name, validate_external_tensor_representation, validate_tensor_representation, CanonicalStorage, RepresentationContract, TensorRepresentation, BF16_BYTES_PER_ELEMENT, IQ1_S_BLOCK_BYTES, IQ1_S_BLOCK_ELEMENTS, IQ2_S_BLOCK_BYTES, IQ2_S_BLOCK_ELEMENTS, IQ2_XXS_BLOCK_BYTES, IQ2_XXS_BLOCK_ELEMENTS, IQ2_XS_BLOCK_BYTES, IQ2_XS_BLOCK_ELEMENTS, IQ4_NL_BLOCK_BYTES, IQ4_NL_BLOCK_ELEMENTS, IQ4_XS_BLOCK_BYTES, IQ4_XS_BLOCK_ELEMENTS, Q2_K_BLOCK_BYTES, Q2_K_BLOCK_ELEMENTS, Q3_K_BLOCK_BYTES, Q3_K_BLOCK_ELEMENTS, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS};
pub use layout::{canonical_payload_alignment, payload_shift, write_indefinite, write_known_size, LayoutClass, LayoutError, LayoutPlan, PlacementRequest, PlannedBlock};
pub use integrity::{digest_payload, encode_payload as encode_integrity_payload, IntegrityAlgorithm, IntegrityEntry, IntegrityMetadata, IntegrityRecord};
pub use range_loading::{execute_plan, execute_plan_with_sources, select_tensor_names, select_tensor_ordinals, Coalescing, LoadedPlan, LoadedRead, MmapSource, PhysicalRange, PositionedFileSource, PlannedTarget, RangeLoadError, ReadPlan, RangeSource, SelectedTensor, SourceResolver, SourceSet};
pub use source::{encode_profile as encode_source_profile, parse_profile as parse_source_profile, CheckedSourceRange, PersistentSourceMetadata, SourceDescriptor, SourceHash, SourceId, SourceLocator, SourceRangeError, SourceRegistry, SourceRegistryError, TensorRef};
pub use metadata::{encode_payload as encode_metadata_payload, MetadataEntry, MetadataField, MetadataValue, ModelMetadata, ModelMetadataKey};
pub use tokenizer::{PreTokenizer, SpecialToken, TokenizerEntry, TokenizerKind, TokenizerMetadata, TokenizerModel, TokenizerRole};
pub use consumer::{BorrowedModel, BorrowedModelView, ConsumerModel, ConsumerModelMetadata, ConsumerTensorType};
pub use runtime_tokenizer::{MergeRankIndex, RuntimeTokenizerIndexes, TokenIndex};
pub use nested::{encode_payload as encode_nested_payload, NestedChild, NestedDirectory, NestedEntry};
pub use moe::{encode_payload as encode_moe_payload, MoeDirectory, MoeEntry, MoeLoaderKind, MoeParameters};
pub use qwen_moe::QwenMoELoader;
pub use deepseek_moe::DeepSeekMoELoader;

/// Boundary smoke helper for future profile code. It does not parse or
/// authorize the range.
pub fn validated_range_len(range: &vbuf_layout::CheckedRange<'_>) -> usize {
    range.bytes().len()
}
