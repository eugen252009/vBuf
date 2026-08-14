//! Validated consumer-facing descriptor bridge.
//!
//! This is deliberately not a llama.cpp runtime or tokenizer implementation.
//! It keeps canonical validation and checked-range provenance on the Rust side
//! while exposing only semantic tensor/tokenizer access needed by a future
//! pinned-consumer adapter.

use crate::{Bootstrap, ModelMetadata, TensorDirectory, TensorRepresentation, TokenizerMetadata};
use crate::error::{MlError, MlErrorCode};
use memmap2::Mmap;
use vbuf_core::v06::{parse_v06, ValidatedV06};

#[derive(Debug)]
struct Views<'a> {
    validated: ValidatedV06<'a>,
    bootstrap: Bootstrap<'a>,
    metadata: ModelMetadata<'a>,
    directory: TensorDirectory<'a>,
    tokenizer: TokenizerMetadata<'a>,
}

#[derive(Debug)]
pub struct ConsumerModel {
    mapping: Mmap,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ConsumerTensorType {
    F32 = 0,
    Bf16 = 1,
    Q8_0 = 2,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConsumerModelMetadata {
    pub architecture: String,
    pub context_length: u64,
    pub embedding_length: u64,
    pub layer_count: u64,
    pub head_count: u64,
    pub kv_head_count: u64,
    pub key_head_dimension: u64,
    pub value_head_dimension: u64,
    pub feed_forward_length: u64,
    pub normalization_epsilon: f64,
    pub rope_theta: f64,
}

impl ConsumerModel {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, MlError> {
        let file = std::fs::File::open(path).map_err(|_| MlError::new(MlErrorCode::MissingTokenizerReference, "consumer model file cannot be read"))?;
        let mapping = unsafe { Mmap::map(&file).map_err(|_| MlError::new(MlErrorCode::MissingTokenizerReference, "consumer model mapping cannot be created"))? };
        let model = Self { mapping };
        model.views()?;
        Ok(model)
    }

    fn views(&self) -> Result<Views<'_>, MlError> {
        let validated = parse_v06(&self.mapping)?;
        let bootstrap = Bootstrap::discover(&validated)?;
        let metadata = ModelMetadata::parse(&validated, &bootstrap)?;
        let directory = TensorDirectory::parse(&validated, &bootstrap)?;
        let tokenizer = TokenizerMetadata::parse(&validated, &bootstrap)?;
        Ok(Views { validated, bootstrap, metadata, directory, tokenizer })
    }

    pub fn tensor_count(&self) -> Result<usize, MlError> { Ok(self.views()?.directory.tensors().len()) }

    pub fn tensor_name(&self, index: usize) -> Result<Option<String>, MlError> {
        Ok(self.views()?.directory.tensors().get(index).map(|tensor| tensor.name.clone()))
    }

    pub fn tensor_shape(&self, index: usize) -> Result<Option<Vec<u64>>, MlError> {
        Ok(self.views()?.directory.tensors().get(index).map(|tensor| tensor.dimensions.clone()))
    }

    pub fn tensor_type(&self, index: usize) -> Result<Option<ConsumerTensorType>, MlError> {
        Ok(self.views()?.directory.tensors().get(index).map(|tensor| match tensor.representation {
            TensorRepresentation::CanonicalPrimitive => ConsumerTensorType::F32,
            TensorRepresentation::Bf16 => ConsumerTensorType::Bf16,
            TensorRepresentation::GgmlQ8_0 => ConsumerTensorType::Q8_0,
        }))
    }

    /// Returns a slice whose lifetime is tied to this model owner. No raw file
    /// offset is exposed to the caller.
    pub fn tensor_payload(&self, index: usize) -> Result<Option<&[u8]>, MlError> {
        let views = self.views()?;
        Ok(views.directory.tensors().get(index).map(|tensor| tensor.range.bytes()))
    }

    pub fn model_metadata(&self) -> Result<ConsumerModelMetadata, MlError> {
        let views = self.views()?;
        let value = |key| views.metadata.unsigned(key).ok_or_else(|| MlError::new(MlErrorCode::MissingRequiredMetadata, "consumer model metadata value is absent"));
        Ok(ConsumerModelMetadata {
            architecture: views.metadata.architecture().ok_or_else(|| MlError::new(MlErrorCode::MissingRequiredMetadata, "consumer architecture is absent"))?.to_owned(),
            context_length: value(crate::ModelMetadataKey::ContextLength)?,
            embedding_length: value(crate::ModelMetadataKey::EmbeddingLength)?,
            layer_count: value(crate::ModelMetadataKey::LayerCount)?,
            head_count: value(crate::ModelMetadataKey::HeadCount)?,
            kv_head_count: value(crate::ModelMetadataKey::KVHeadCount)?,
            key_head_dimension: value(crate::ModelMetadataKey::KeyHeadDimension)?,
            value_head_dimension: value(crate::ModelMetadataKey::ValueHeadDimension)?,
            feed_forward_length: value(crate::ModelMetadataKey::FeedForwardLength)?,
            normalization_epsilon: views.metadata.float(crate::ModelMetadataKey::NormalizationEpsilon).ok_or_else(|| MlError::new(MlErrorCode::MissingRequiredMetadata, "consumer normalization epsilon is absent"))?,
            rope_theta: views.metadata.float(crate::ModelMetadataKey::RopeTheta).ok_or_else(|| MlError::new(MlErrorCode::MissingRequiredMetadata, "consumer RoPE theta is absent"))?,
        })
    }

    pub fn tokenizer_count(&self) -> Result<u64, MlError> { Ok(self.views()?.tokenizer.token_count()) }
    pub fn token_text(&self, index: u64) -> Result<Option<String>, MlError> { Ok(self.views()?.tokenizer.token_text(index).map(ToOwned::to_owned)) }
    pub fn token_type(&self, index: u64) -> Result<Option<u64>, MlError> { Ok(self.views()?.tokenizer.token_type(index)) }
    pub fn merge_count(&self) -> Result<u64, MlError> { Ok(self.views()?.tokenizer.merge_count()) }
    pub fn merge_pair(&self, index: u64) -> Result<Option<(u64, u64)>, MlError> { Ok(self.views()?.tokenizer.merge_pair(index)) }
    pub fn add_bos(&self) -> Result<Option<bool>, MlError> { Ok(self.views()?.tokenizer.add_bos()) }
    pub fn chat_template(&self) -> Result<Option<String>, MlError> { Ok(self.views()?.tokenizer.chat_template().map(ToOwned::to_owned)) }

    pub fn is_validated(&self) -> Result<bool, MlError> {
        let views = self.views()?;
        let _ = views.bootstrap.profile_version();
        let _ = views.validated.blocks().len();
        Ok(true)
    }
}
