//! Downstream vBuf-ML profile boundary.
//!
//! This crate consumes generic values that have already been validated by
//! canonical vBuf. It assigns ML-local roles and relationships; it does not
//! invent primitive types or override physical validity.
//!
//! The base remains a small compositional binary vocabulary: efficient
//! composition is preferred over maximal base functionality.

pub mod bootstrap;
pub mod consumer;
pub mod consumer_ffi;
pub mod deepseek_moe;
pub mod error;
pub mod integrity;
pub mod layout;
pub mod metadata;
pub mod moe;
pub mod nested;
pub mod qwen_moe;
pub mod range_loading;
pub mod region_roles;
pub mod representations;
pub mod runtime_tokenizer;
pub mod source;
pub mod tensor_directory;
pub mod tokenizer;

pub use bootstrap::{Bootstrap, BootstrapEntry, SemanticRegion};
pub use consumer::{
    BorrowedModel, BorrowedModelView, ConsumerModel, ConsumerModelMetadata, ConsumerTensorType,
};
pub use deepseek_moe::DeepSeekMoELoader;
pub use error::{MlError, MlErrorCode};
pub use integrity::{
    IntegrityAlgorithm, IntegrityEntry, IntegrityMetadata, IntegrityRecord, digest_payload,
    encode_payload as encode_integrity_payload,
};
pub use layout::{
    LayoutClass, LayoutError, LayoutPlan, PlacementRequest, PlannedBlock,
    canonical_payload_alignment, payload_shift, write_indefinite, write_known_size,
};
pub use metadata::{
    MetadataEntry, MetadataField, MetadataValue, ModelMetadata, ModelMetadataKey,
    encode_payload as encode_metadata_payload,
};
pub use moe::{
    MoeDirectory, MoeEntry, MoeLoaderKind, MoeParameters, encode_payload as encode_moe_payload,
};
pub use nested::{
    NestedChild, NestedDirectory, NestedEntry, encode_payload as encode_nested_payload,
};
pub use qwen_moe::QwenMoELoader;
pub use range_loading::{
    Coalescing, LoadedPlan, LoadedRead, MmapSource, PhysicalRange, PlannedTarget,
    PositionedFileSource, RangeLoadError, RangeSource, ReadPlan, SelectedTensor, SourceResolver,
    SourceSet, execute_plan, execute_plan_with_sources, select_tensor_names,
    select_tensor_ordinals,
};
pub use region_roles::RegionRole;
pub use representations::{
    BF16_BYTES_PER_ELEMENT, CanonicalStorage, IQ1_S_BLOCK_BYTES, IQ1_S_BLOCK_ELEMENTS,
    IQ2_S_BLOCK_BYTES, IQ2_S_BLOCK_ELEMENTS, IQ2_XS_BLOCK_BYTES, IQ2_XS_BLOCK_ELEMENTS,
    IQ2_XXS_BLOCK_BYTES, IQ2_XXS_BLOCK_ELEMENTS, IQ4_NL_BLOCK_BYTES, IQ4_NL_BLOCK_ELEMENTS,
    IQ4_XS_BLOCK_BYTES, IQ4_XS_BLOCK_ELEMENTS, Q2_K_BLOCK_BYTES, Q2_K_BLOCK_ELEMENTS,
    Q3_K_BLOCK_BYTES, Q3_K_BLOCK_ELEMENTS, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS, Q4_K_BLOCK_BYTES,
    Q4_K_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES,
    Q6_K_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS, RepresentationContract,
    TensorRepresentation, bf16_bits_to_f32, expected_payload_bytes, logical_elements,
    representation_contract, representation_from_id, representation_name,
    validate_external_tensor_representation, validate_tensor_representation,
};
pub use runtime_tokenizer::{MergeRankIndex, RuntimeTokenizerIndexes, TokenIndex};
pub use source::{
    CheckedSourceRange, PersistentSourceMetadata, SourceDescriptor, SourceHash, SourceId,
    SourceLocator, SourceRangeError, SourceRegistry, SourceRegistryError, TensorRef,
    encode_profile as encode_source_profile, parse_profile as parse_source_profile,
};
pub use tensor_directory::{TensorDescriptor, TensorDirectory, TensorEntry};
pub use tokenizer::{
    PreTokenizer, SpecialToken, TokenizerEntry, TokenizerKind, TokenizerMetadata, TokenizerModel,
    TokenizerRole,
};

/// Boundary smoke helper for future profile code. It does not parse or
/// authorize the range.
pub fn validated_range_len(range: &vbuf_layout::CheckedRange<'_>) -> usize {
    range.bytes().len()
}
