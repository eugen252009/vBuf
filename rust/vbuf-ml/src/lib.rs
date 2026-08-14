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

pub use bootstrap::{Bootstrap, BootstrapEntry, SemanticRegion};
pub use error::{MlError, MlErrorCode};
pub use region_roles::RegionRole;
pub use tensor_directory::{TensorDescriptor, TensorDirectory, TensorEntry, TensorRepresentation};
pub use metadata::{MetadataEntry, MetadataField, MetadataValue, ModelMetadata, ModelMetadataKey};
pub use tokenizer::{SpecialToken, TokenizerEntry, TokenizerKind, TokenizerMetadata, TokenizerRole};

/// Boundary smoke helper for future profile code. It does not parse or
/// authorize the range.
pub fn validated_range_len(range: &vbuf_layout::CheckedRange<'_>) -> usize {
    range.bytes().len()
}
