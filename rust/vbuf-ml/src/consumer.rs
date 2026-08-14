//! Validated, mmap-backed consumer-facing descriptor bridge.
//!
//! Canonical parsing happens once at open. The resulting semantic snapshot is
//! owned by the handle; payload bytes remain borrowed from its mmap.

use crate::{Bootstrap, ModelMetadata, TensorDirectory, TensorRepresentation, TokenizerMetadata};
use crate::error::{MlError, MlErrorCode};
use memmap2::Mmap;
use std::sync::OnceLock;
use vbuf_core::v06::{parse_v06, ValidatedV06};

#[derive(Debug)]
struct Views<'a> {
    validated: ValidatedV06<'a>,
    bootstrap: Bootstrap<'a>,
    metadata: ModelMetadata<'a>,
    directory: TensorDirectory<'a>,
    tokenizer: TokenizerMetadata<'a>,
}

#[derive(Clone, Debug)]
struct TensorSnapshot {
    name: String,
    dimensions: Vec<u64>,
    kind: ConsumerTensorType,
    offset: u64,
    length: u64,
}

#[derive(Clone, Debug)]
struct Snapshot {
    tensors: Vec<TensorSnapshot>,
    metadata: ConsumerModelMetadata,
    token_text: Vec<String>,
    token_types: Vec<Option<u64>>,
    token_scores: Vec<Option<f64>>,
    merges: Vec<(u64, u64)>,
    special_tokens: [Option<u64>; 4],
    add_bos: Option<bool>,
    chat_template: Option<String>,
}

#[derive(Debug)]
pub struct ConsumerModel {
    mapping: Mmap,
    snapshot: OnceLock<Snapshot>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ConsumerTensorType { F32 = 0, Bf16 = 1, Q8_0 = 2 }

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
        let model = Self { mapping, snapshot: OnceLock::new() };
        model.snapshot()?;
        Ok(model)
    }

    fn parse_views(&self) -> Result<Views<'_>, MlError> {
        let validated = parse_v06(&self.mapping)?;
        let bootstrap = Bootstrap::discover(&validated)?;
        let metadata = ModelMetadata::parse(&validated, &bootstrap)?;
        let directory = TensorDirectory::parse(&validated, &bootstrap)?;
        let tokenizer = TokenizerMetadata::parse(&validated, &bootstrap)?;
        Ok(Views { validated, bootstrap, metadata, directory, tokenizer })
    }

    fn build_snapshot(&self) -> Result<Snapshot, MlError> {
        let views = self.parse_views()?;
        let value = |key| views.metadata.unsigned(key).ok_or_else(|| MlError::new(MlErrorCode::MissingRequiredMetadata, "consumer model metadata value is absent"));
        let metadata = ConsumerModelMetadata {
            architecture: views.metadata.architecture().ok_or_else(|| MlError::new(MlErrorCode::MissingRequiredMetadata, "consumer architecture is absent"))?.to_owned(),
            context_length: value(crate::ModelMetadataKey::ContextLength)?, embedding_length: value(crate::ModelMetadataKey::EmbeddingLength)?, layer_count: value(crate::ModelMetadataKey::LayerCount)?, head_count: value(crate::ModelMetadataKey::HeadCount)?,
            kv_head_count: value(crate::ModelMetadataKey::KVHeadCount)?, key_head_dimension: value(crate::ModelMetadataKey::KeyHeadDimension)?, value_head_dimension: value(crate::ModelMetadataKey::ValueHeadDimension)?, feed_forward_length: value(crate::ModelMetadataKey::FeedForwardLength)?,
            normalization_epsilon: views.metadata.float(crate::ModelMetadataKey::NormalizationEpsilon).ok_or_else(|| MlError::new(MlErrorCode::MissingRequiredMetadata, "consumer normalization epsilon is absent"))?,
            rope_theta: views.metadata.float(crate::ModelMetadataKey::RopeTheta).ok_or_else(|| MlError::new(MlErrorCode::MissingRequiredMetadata, "consumer RoPE theta is absent"))?,
        };
        let tensors = views.directory.tensors().iter().map(|tensor| TensorSnapshot { name: tensor.name.clone(), dimensions: tensor.dimensions.clone(), kind: match tensor.representation { TensorRepresentation::CanonicalPrimitive => ConsumerTensorType::F32, TensorRepresentation::Bf16 => ConsumerTensorType::Bf16, TensorRepresentation::GgmlQ8_0 => ConsumerTensorType::Q8_0 }, offset: tensor.range.offset(), length: tensor.range.length() }).collect();
        let token_count = views.tokenizer.token_count();
        let mut token_text = Vec::with_capacity(usize::try_from(token_count).map_err(|_| MlError::new(MlErrorCode::TokenizerArrayLengthMismatch, "vocabulary exceeds host range"))?);
        let mut token_types = Vec::with_capacity(token_text.capacity()); let mut token_scores = Vec::with_capacity(token_text.capacity());
        for index in 0..token_count { token_text.push(views.tokenizer.token_text(index).ok_or_else(|| MlError::new(MlErrorCode::InvalidTokenText, "token text snapshot failed"))?.to_owned()); token_types.push(views.tokenizer.token_type(index)); token_scores.push(views.tokenizer.score(index)); }
        let mut merges = Vec::with_capacity(usize::try_from(views.tokenizer.merge_count()).unwrap_or(0));
        for index in 0..views.tokenizer.merge_count() { merges.push(views.tokenizer.merge_pair(index).ok_or_else(|| MlError::new(MlErrorCode::InvalidMergeTable, "merge snapshot failed"))?); }
        let mut special_tokens = [None; 4];
        for (kind, wanted) in [(0, crate::SpecialToken::Bos), (1, crate::SpecialToken::Eos), (2, crate::SpecialToken::Unk), (3, crate::SpecialToken::Pad)] { special_tokens[kind] = views.tokenizer.specials().iter().find(|(token, _)| *token == wanted).map(|(_, value)| *value); }
        let _ = views.bootstrap.profile_version(); let _ = views.validated.blocks().len();
        Ok(Snapshot { tensors, metadata, token_text, token_types, token_scores, merges, special_tokens, add_bos: views.tokenizer.add_bos(), chat_template: views.tokenizer.chat_template().map(ToOwned::to_owned) })
    }

    fn snapshot(&self) -> Result<&Snapshot, MlError> {
        if self.snapshot.get().is_none() { let _ = self.snapshot.set(self.build_snapshot()?); }
        self.snapshot.get().ok_or_else(|| MlError::new(MlErrorCode::MalformedTensorDirectory, "consumer snapshot is unavailable"))
    }
    pub fn tensor_count(&self) -> Result<usize, MlError> { Ok(self.snapshot()?.tensors.len()) }
    pub fn tensor_name(&self, index: usize) -> Result<Option<String>, MlError> { Ok(self.snapshot()?.tensors.get(index).map(|t| t.name.clone())) }
    pub fn tensor_shape(&self, index: usize) -> Result<Option<Vec<u64>>, MlError> { Ok(self.snapshot()?.tensors.get(index).map(|t| t.dimensions.clone())) }
    pub fn tensor_type(&self, index: usize) -> Result<Option<ConsumerTensorType>, MlError> { Ok(self.snapshot()?.tensors.get(index).map(|t| t.kind)) }
    pub fn tensor_payload(&self, index: usize) -> Result<Option<&[u8]>, MlError> { let tensor = self.snapshot()?.tensors.get(index); Ok(tensor.map(|t| { let start = usize::try_from(t.offset).expect("validated offset fits host"); let end = start + usize::try_from(t.length).expect("validated length fits host"); &self.mapping[start..end] })) }
    pub fn model_metadata(&self) -> Result<ConsumerModelMetadata, MlError> { Ok(self.snapshot()?.metadata.clone()) }
    pub fn tokenizer_count(&self) -> Result<u64, MlError> { Ok(self.snapshot()?.token_text.len() as u64) }
    pub fn token_text(&self, index: u64) -> Result<Option<String>, MlError> { Ok(self.snapshot()?.token_text.get(usize::try_from(index).unwrap_or(usize::MAX)).cloned()) }
    pub fn token_type(&self, index: u64) -> Result<Option<u64>, MlError> { Ok(self.snapshot()?.token_types.get(usize::try_from(index).unwrap_or(usize::MAX)).copied().flatten()) }
    pub fn token_score(&self, index: u64) -> Result<Option<f64>, MlError> { Ok(self.snapshot()?.token_scores.get(usize::try_from(index).unwrap_or(usize::MAX)).copied().flatten()) }
    pub fn special_token(&self, kind: u8) -> Result<Option<u64>, MlError> { Ok(self.snapshot()?.special_tokens.get(usize::from(kind)).copied().flatten()) }
    pub fn merge_count(&self) -> Result<u64, MlError> { Ok(self.snapshot()?.merges.len() as u64) }
    pub fn merge_pair(&self, index: u64) -> Result<Option<(u64, u64)>, MlError> { Ok(self.snapshot()?.merges.get(usize::try_from(index).unwrap_or(usize::MAX)).copied()) }
    pub fn add_bos(&self) -> Result<Option<bool>, MlError> { Ok(self.snapshot()?.add_bos) }
    pub fn chat_template(&self) -> Result<Option<String>, MlError> { Ok(self.snapshot()?.chat_template.clone()) }
    pub fn is_validated(&self) -> Result<bool, MlError> { self.snapshot().map(|_| true) }
}
