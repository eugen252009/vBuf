//! Validated, mmap-backed consumer-facing descriptor bridge.
//!
//! Canonical parsing happens once at open. The resulting semantic snapshot is
//! owned by the handle; payload bytes remain borrowed from its mmap.

use crate::error::{MlError, MlErrorCode};
use crate::{
    Bootstrap, DeepSeekMoELoader, ModelMetadata, ModelMetadataKey, MoeDirectory, NestedDirectory,
    QwenMoELoader, SourceRegistry, TensorDirectory, TensorRef, TensorRepresentation,
    TokenizerMetadata, parse_source_profile,
};
use memmap2::Mmap;
use std::sync::OnceLock;
use vbuf_core::v06::{ValidatedV06, parse_v06};

#[derive(Debug)]
struct Views<'a> {
    validated: ValidatedV06<'a>,
    bootstrap: Bootstrap<'a>,
    metadata: ModelMetadata<'a>,
    directory: TensorDirectory<'a>,
    tokenizer: TokenizerMetadata<'a>,
}

#[derive(Clone, Debug)]
pub(crate) struct TensorSnapshot {
    pub(crate) name: String,
    pub(crate) dimensions: Vec<u64>,
    pub(crate) kind: ConsumerTensorType,
    pub(crate) offset: u64,
    pub(crate) length: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct Snapshot {
    pub(crate) tensors: Vec<TensorSnapshot>,
    pub(crate) metadata: ConsumerModelMetadata,
    pub(crate) token_text: Vec<String>,
    pub(crate) token_types: Vec<Option<u64>>,
    pub(crate) token_scores: Vec<Option<f64>>,
    pub(crate) merges: Vec<(u64, u64)>,
    special_tokens: [Option<u64>; 4],
    add_bos: Option<bool>,
    chat_template: Option<String>,
}

#[derive(Debug)]
pub struct ConsumerModel {
    pub(crate) mapping: Mmap,
    snapshot: OnceLock<Snapshot>,
}

/// Validated semantic views whose arrays remain borrowed from the supplied bytes.
/// This is the runtime-facing counterpart to the owned `ConsumerModel` convenience
/// API; it deliberately exposes no per-token or per-merge object table.
#[derive(Debug)]
pub struct BorrowedModelView<'a> {
    pub validated: ValidatedV06<'a>,
    pub bootstrap: Bootstrap<'a>,
    pub metadata: ModelMetadata<'a>,
    pub directory: TensorDirectory<'a>,
    pub tokenizer: TokenizerMetadata<'a>,
    pub quantization: Option<crate::F8QuantizationDirectory>,
    pub nested: Option<NestedDirectory<'a>>,
    pub moe: Option<MoeDirectory>,
}

impl<'a> BorrowedModelView<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, MlError> {
        let sources =
            SourceRegistry::new(vec![SourceRegistry::self_descriptor(bytes.len() as u64)])
                .map_err(|_| {
                    MlError::new(
                        MlErrorCode::MalformedTensorDirectory,
                        "self source registry is invalid",
                    )
                })?;
        Self::parse_with_sources(bytes, &sources, &[])
    }

    pub fn parse_with_sources(
        bytes: &'a [u8],
        sources: &SourceRegistry,
        external: &[(u16, u16, TensorRef)],
    ) -> Result<Self, MlError> {
        if sources
            .get(crate::SourceId::SELF)
            .and_then(|source| source.declared_size)
            != Some(bytes.len() as u64)
        {
            return Err(MlError::new(
                MlErrorCode::TensorReferenceMissing,
                "self source size does not match metadata artifact",
            ));
        }
        let validated = parse_v06(bytes)?;
        let bootstrap = Bootstrap::discover(&validated)?;
        let persistent = parse_source_profile(&validated, &bootstrap)?;
        if let Some(profile) = persistent.as_ref() {
            for descriptor in profile.registry.descriptors() {
                let runtime = sources.get(descriptor.id).ok_or_else(|| {
                    MlError::new(
                        MlErrorCode::TensorReferenceMissing,
                        "persistent source is absent from runtime registry",
                    )
                })?;
                if runtime.declared_size != descriptor.declared_size {
                    return Err(MlError::new(
                        MlErrorCode::TensorReferenceMissing,
                        "persistent source size differs from runtime registry",
                    ));
                }
            }
        }
        let metadata = ModelMetadata::parse(&validated, &bootstrap)?;
        let persistent_bindings = persistent
            .as_ref()
            .map_or(external, |profile| profile.bindings.as_slice());
        let directory = TensorDirectory::parse_with_sources(
            &validated,
            &bootstrap,
            sources,
            persistent_bindings,
        )?;
        let tokenizer = TokenizerMetadata::parse(&validated, &bootstrap)?;
        let quantization = crate::F8QuantizationDirectory::parse(&validated, &bootstrap)?;
        if let Some(quantization) = quantization.as_ref() {
            quantization.validate_against(&directory)?;
        }
        let nested = if bootstrap
            .region(crate::RegionRole::NestedDirectory)
            .is_some()
        {
            Some(NestedDirectory::parse(&validated, &bootstrap)?)
        } else {
            None
        };
        let moe = if bootstrap.region(crate::RegionRole::MoeDirectory).is_some() {
            let nested = nested.as_ref().ok_or_else(|| {
                MlError::new(
                    MlErrorCode::MoeExpertReferenceMissing,
                    "MoE directory requires a nested directory",
                )
            })?;
            Some(MoeDirectory::parse(&validated, &bootstrap, nested)?)
        } else {
            None
        };
        Ok(Self {
            validated,
            bootstrap,
            metadata,
            directory,
            tokenizer,
            nested,
            moe,
            quantization,
        })
    }
}

/// Owned mmap lifetime plus borrowed semantic views. The views use a fixed
/// lifetime internally only after `BorrowedModelView::parse` has validated the
/// mapping; the mapping field outlives the view and is never exposed mutably.
#[derive(Debug)]
pub struct BorrowedModel {
    // Declared first so the borrowed view is dropped before the mapping.
    pub(crate) view: BorrowedModelView<'static>,
    pub(crate) _mapping: Mmap,
}

impl BorrowedModel {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, MlError> {
        let file = std::fs::File::open(path).map_err(|_| {
            MlError::new(
                MlErrorCode::MissingTokenizerReference,
                "borrowed model file cannot be read",
            )
        })?;
        let mapping = unsafe {
            Mmap::map(&file).map_err(|_| {
                MlError::new(
                    MlErrorCode::MissingTokenizerReference,
                    "borrowed model mapping cannot be created",
                )
            })?
        };
        let view = BorrowedModelView::parse(&mapping)?;
        // SAFETY: `view` contains immutable slices into `mapping`, which is
        // stored in the same owner and dropped after the view. No mutable
        // access to the mapping is possible through this type.
        let view = unsafe {
            std::mem::transmute::<BorrowedModelView<'_>, BorrowedModelView<'static>>(view)
        };
        Ok(Self {
            _mapping: mapping,
            view,
        })
    }

    pub fn open_with_sources(
        path: impl AsRef<std::path::Path>,
        sources: &crate::SourceRegistry,
        external: &[(u16, u16, TensorRef)],
    ) -> Result<Self, MlError> {
        let file = std::fs::File::open(path).map_err(|_| {
            MlError::new(
                MlErrorCode::MissingTokenizerReference,
                "borrowed model file cannot be read",
            )
        })?;
        let mapping = unsafe {
            Mmap::map(&file).map_err(|_| {
                MlError::new(
                    MlErrorCode::MissingTokenizerReference,
                    "borrowed model mapping cannot be created",
                )
            })?
        };
        let view = BorrowedModelView::parse_with_sources(&mapping, sources, external)?;
        let view = unsafe {
            std::mem::transmute::<BorrowedModelView<'_>, BorrowedModelView<'static>>(view)
        };
        Ok(Self {
            _mapping: mapping,
            view,
        })
    }

    pub fn open_persistent(path: impl AsRef<std::path::Path>) -> Result<Self, MlError> {
        let file = std::fs::File::open(path).map_err(|_| {
            MlError::new(
                MlErrorCode::MissingTokenizerReference,
                "borrowed model file cannot be read",
            )
        })?;
        let mapping = unsafe {
            Mmap::map(&file).map_err(|_| {
                MlError::new(
                    MlErrorCode::MissingTokenizerReference,
                    "borrowed model mapping cannot be created",
                )
            })?
        };
        let validated = parse_v06(&mapping)?;
        let bootstrap = Bootstrap::discover(&validated)?;
        let profile = crate::parse_source_profile(&validated, &bootstrap)?.ok_or_else(|| {
            MlError::new(
                MlErrorCode::TensorReferenceMissing,
                "persistent source profile is absent",
            )
        })?;
        let view = BorrowedModelView::parse_with_sources(&mapping, &profile.registry, &[])?;
        let view = unsafe {
            std::mem::transmute::<BorrowedModelView<'_>, BorrowedModelView<'static>>(view)
        };
        Ok(Self {
            _mapping: mapping,
            view,
        })
    }

    pub fn view(&self) -> &BorrowedModelView<'static> {
        &self.view
    }
    pub fn nested_child_count(&self) -> usize {
        self.view
            .nested
            .as_ref()
            .map_or(0, |directory| directory.children().len())
    }
    pub fn moe_parameters(&self) -> Option<crate::MoeParameters> {
        self.view.moe.as_ref().map(MoeDirectory::parameters)
    }
    pub fn moe_loader_kind(&self) -> crate::MoeLoaderKind {
        crate::MoeLoaderKind::for_architecture(self.view.metadata.architecture().unwrap_or(""))
    }
    pub fn qwen_moe_loader(&self) -> Result<QwenMoELoader, MlError> {
        let moe = self.view.moe.as_ref().ok_or_else(|| {
            MlError::new(
                MlErrorCode::MoeDirectoryMissing,
                "model has no MoE directory",
            )
        })?;
        QwenMoELoader::from_model(
            self.view.metadata.architecture().unwrap_or(""),
            &self.view.metadata,
            &self.view.directory,
            moe,
        )
    }
    pub fn deepseek_moe_loader(&self) -> Result<DeepSeekMoELoader, MlError> {
        let moe = self.view.moe.as_ref().ok_or_else(|| {
            MlError::new(
                MlErrorCode::MoeDirectoryMissing,
                "model has no MoE directory",
            )
        })?;
        DeepSeekMoELoader::from_model(
            self.view.metadata.architecture().unwrap_or(""),
            &self.view.metadata,
            &self.view.directory,
            moe,
        )
    }
    pub fn is_validated(&self) -> bool {
        true
    }
    pub fn tensor_count(&self) -> Result<usize, MlError> {
        Ok(self.view.directory.tensors().len())
    }
    pub fn tensor_name(&self, index: usize) -> Result<Option<String>, MlError> {
        Ok(self
            .view
            .directory
            .tensors()
            .get(index)
            .map(|t| t.name.clone()))
    }
    pub fn tensor_shape(&self, index: usize) -> Result<Option<Vec<u64>>, MlError> {
        Ok(self
            .view
            .directory
            .tensors()
            .get(index)
            .map(|t| t.dimensions.clone()))
    }
    pub fn tensor_type(&self, index: usize) -> Result<Option<ConsumerTensorType>, MlError> {
        Ok(self
            .view
            .directory
            .tensors()
            .get(index)
            .map(|t| match t.representation {
                TensorRepresentation::CanonicalPrimitive => ConsumerTensorType::F32,
                TensorRepresentation::Bf16 => ConsumerTensorType::Bf16,
                TensorRepresentation::GgmlQ8_0 => ConsumerTensorType::Q8_0,
                TensorRepresentation::GgmlQ4_0 => ConsumerTensorType::Q4_0,
                TensorRepresentation::GgmlQ2_K => ConsumerTensorType::Q2_K,
                TensorRepresentation::GgmlIQ1_S => ConsumerTensorType::IQ1_S,
                TensorRepresentation::GgmlQ4_K => ConsumerTensorType::Q4_K,
                TensorRepresentation::GgmlIQ4_NL => ConsumerTensorType::IQ4_NL,
                TensorRepresentation::GgmlIQ4_XS => ConsumerTensorType::IQ4_XS,
                TensorRepresentation::GgmlQ3_K => ConsumerTensorType::Q3_K,
                TensorRepresentation::GgmlIQ2_XXS => ConsumerTensorType::IQ2_XXS,
                TensorRepresentation::GgmlIQ2_XS => ConsumerTensorType::IQ2_XS,
                TensorRepresentation::GgmlIQ2_S => ConsumerTensorType::IQ2_S,
                TensorRepresentation::GgmlQ5_K => ConsumerTensorType::Q5_K,
                TensorRepresentation::GgmlQ6_K => ConsumerTensorType::Q6_K,
                TensorRepresentation::F8_E4M3 => ConsumerTensorType::F8_E4M3,
            }))
    }
    pub fn tensor_payload(&self, index: usize) -> Result<Option<&[u8]>, MlError> {
        Ok(self
            .view
            .directory
            .tensors()
            .get(index)
            .and_then(|t| t.range.as_ref().map(|range| range.bytes())))
    }

    /// Materializes one validated FP8 tensor into the existing F32 execution
    /// representation. Only the selected weight and its scale tensor are read.
    pub fn materialize_f8_tensor(&self, name: &str) -> Result<Option<Vec<f32>>, MlError> {
        let Some(weight) = self.view.directory.get(name) else {
            return Ok(None);
        };
        if weight.representation != TensorRepresentation::F8_E4M3 {
            return Ok(None);
        }
        let quantization = self.view.quantization.as_ref().ok_or_else(|| {
            MlError::new(
                MlErrorCode::QuantizationMetadataMissing,
                "FP8 tensor has no quantization provenance",
            )
        })?;
        let entry = quantization.get(name).ok_or_else(|| {
            MlError::new(
                MlErrorCode::InvalidQuantizationMetadata,
                "FP8 tensor has no scale association",
            )
        })?;
        let scale = self.view.directory.get(&entry.scale_name).ok_or_else(|| {
            MlError::new(
                MlErrorCode::InvalidQuantizationMetadata,
                "FP8 scale tensor is absent",
            )
        })?;
        let weight_bytes = weight.range.as_ref().ok_or_else(|| {
            MlError::new(
                MlErrorCode::TensorReferenceMissing,
                "FP8 weight payload is not locally materialized",
            )
        })?;
        let scale_bytes = scale.range.as_ref().ok_or_else(|| {
            MlError::new(
                MlErrorCode::TensorReferenceMissing,
                "FP8 scale payload is not locally materialized",
            )
        })?;
        let shape = [weight.dimensions[0], weight.dimensions[1]];
        Ok(Some(crate::dequantize_f8_e4m3(
            weight_bytes.bytes(),
            shape,
            scale_bytes.bytes(),
            [entry.block_rows, entry.block_columns],
        )?))
    }
    pub fn model_metadata(&self) -> Result<ConsumerModelMetadata, MlError> {
        let m = &self.view.metadata;
        let u = |key| {
            m.unsigned(key).ok_or_else(|| {
                MlError::new(
                    MlErrorCode::MissingRequiredMetadata,
                    "borrowed metadata value is absent",
                )
            })
        };
        Ok(ConsumerModelMetadata {
            architecture: m
                .architecture()
                .ok_or_else(|| {
                    MlError::new(
                        MlErrorCode::MissingRequiredMetadata,
                        "borrowed architecture is absent",
                    )
                })?
                .to_owned(),
            context_length: u(ModelMetadataKey::ContextLength)?,
            embedding_length: u(ModelMetadataKey::EmbeddingLength)?,
            layer_count: u(ModelMetadataKey::LayerCount)?,
            head_count: u(ModelMetadataKey::HeadCount)?,
            kv_head_count: u(ModelMetadataKey::KVHeadCount)?,
            key_head_dimension: u(ModelMetadataKey::KeyHeadDimension)?,
            value_head_dimension: u(ModelMetadataKey::ValueHeadDimension)?,
            feed_forward_length: u(ModelMetadataKey::FeedForwardLength)?,
            normalization_epsilon: m.float(ModelMetadataKey::NormalizationEpsilon).ok_or_else(
                || {
                    MlError::new(
                        MlErrorCode::MissingRequiredMetadata,
                        "borrowed epsilon is absent",
                    )
                },
            )?,
            rope_theta: m.float(ModelMetadataKey::RopeTheta).ok_or_else(|| {
                MlError::new(
                    MlErrorCode::MissingRequiredMetadata,
                    "borrowed rope theta is absent",
                )
            })?,
        })
    }
    pub fn tokenizer_count(&self) -> Result<u64, MlError> {
        Ok(self.view.tokenizer.token_count())
    }
    pub fn token_text(&self, index: u64) -> Result<Option<String>, MlError> {
        Ok(self.view.tokenizer.token_text(index).map(ToOwned::to_owned))
    }
    pub fn token_type(&self, index: u64) -> Result<Option<u64>, MlError> {
        Ok(self.view.tokenizer.token_type(index))
    }
    pub fn token_score(&self, index: u64) -> Result<Option<f64>, MlError> {
        Ok(self.view.tokenizer.score(index))
    }
    pub fn special_token(&self, kind: u8) -> Result<Option<u64>, MlError> {
        let wanted = [
            crate::SpecialToken::Bos,
            crate::SpecialToken::Eos,
            crate::SpecialToken::Unk,
            crate::SpecialToken::Pad,
        ]
        .get(usize::from(kind));
        Ok(wanted.and_then(|w| {
            self.view
                .tokenizer
                .specials()
                .iter()
                .find(|(k, _)| k == w)
                .map(|(_, id)| *id)
        }))
    }
    pub fn merge_count(&self) -> Result<u64, MlError> {
        Ok(self.view.tokenizer.merge_count())
    }
    pub fn merge_pair(&self, index: u64) -> Result<Option<(u64, u64)>, MlError> {
        Ok(self.view.tokenizer.merge_pair(index))
    }
    pub fn add_bos(&self) -> Result<Option<bool>, MlError> {
        Ok(self.view.tokenizer.add_bos())
    }
    pub fn chat_template(&self) -> Result<Option<String>, MlError> {
        Ok(self.view.tokenizer.chat_template().map(ToOwned::to_owned))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
#[allow(non_camel_case_types)]
pub enum ConsumerTensorType {
    F32 = 0,
    Bf16 = 1,
    Q8_0 = 2,
    Q4_0 = 3,
    Q2_K = 4,
    IQ1_S = 5,
    Q4_K = 6,
    IQ4_NL = 7,
    IQ4_XS = 8,
    Q3_K = 9,
    IQ2_XXS = 10,
    IQ2_XS = 11,
    IQ2_S = 12,
    Q5_K = 13,
    Q6_K = 14,
    F8_E4M3 = 15,
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
        let file = std::fs::File::open(path).map_err(|_| {
            MlError::new(
                MlErrorCode::MissingTokenizerReference,
                "consumer model file cannot be read",
            )
        })?;
        let mapping = unsafe {
            Mmap::map(&file).map_err(|_| {
                MlError::new(
                    MlErrorCode::MissingTokenizerReference,
                    "consumer model mapping cannot be created",
                )
            })?
        };
        let model = Self {
            mapping,
            snapshot: OnceLock::new(),
        };
        model.snapshot()?;
        Ok(model)
    }

    fn parse_views(&self) -> Result<Views<'_>, MlError> {
        let validated = parse_v06(&self.mapping)?;
        let bootstrap = Bootstrap::discover(&validated)?;
        let metadata = ModelMetadata::parse(&validated, &bootstrap)?;
        let directory = TensorDirectory::parse(&validated, &bootstrap)?;
        let tokenizer = TokenizerMetadata::parse(&validated, &bootstrap)?;
        Ok(Views {
            validated,
            bootstrap,
            metadata,
            directory,
            tokenizer,
        })
    }

    fn build_snapshot(&self) -> Result<Snapshot, MlError> {
        let views = self.parse_views()?;
        let value = |key| {
            views.metadata.unsigned(key).ok_or_else(|| {
                MlError::new(
                    MlErrorCode::MissingRequiredMetadata,
                    "consumer model metadata value is absent",
                )
            })
        };
        let metadata = ConsumerModelMetadata {
            architecture: views
                .metadata
                .architecture()
                .ok_or_else(|| {
                    MlError::new(
                        MlErrorCode::MissingRequiredMetadata,
                        "consumer architecture is absent",
                    )
                })?
                .to_owned(),
            context_length: value(crate::ModelMetadataKey::ContextLength)?,
            embedding_length: value(crate::ModelMetadataKey::EmbeddingLength)?,
            layer_count: value(crate::ModelMetadataKey::LayerCount)?,
            head_count: value(crate::ModelMetadataKey::HeadCount)?,
            kv_head_count: value(crate::ModelMetadataKey::KVHeadCount)?,
            key_head_dimension: value(crate::ModelMetadataKey::KeyHeadDimension)?,
            value_head_dimension: value(crate::ModelMetadataKey::ValueHeadDimension)?,
            feed_forward_length: value(crate::ModelMetadataKey::FeedForwardLength)?,
            normalization_epsilon: views
                .metadata
                .float(crate::ModelMetadataKey::NormalizationEpsilon)
                .ok_or_else(|| {
                    MlError::new(
                        MlErrorCode::MissingRequiredMetadata,
                        "consumer normalization epsilon is absent",
                    )
                })?,
            rope_theta: views
                .metadata
                .float(crate::ModelMetadataKey::RopeTheta)
                .ok_or_else(|| {
                    MlError::new(
                        MlErrorCode::MissingRequiredMetadata,
                        "consumer RoPE theta is absent",
                    )
                })?,
        };
        let tensors = views
            .directory
            .tensors()
            .iter()
            .map(|tensor| TensorSnapshot {
                name: tensor.name.clone(),
                dimensions: tensor.dimensions.clone(),
                kind: match tensor.representation {
                    TensorRepresentation::CanonicalPrimitive => ConsumerTensorType::F32,
                    TensorRepresentation::Bf16 => ConsumerTensorType::Bf16,
                    TensorRepresentation::GgmlQ8_0 => ConsumerTensorType::Q8_0,
                    TensorRepresentation::GgmlQ4_0 => ConsumerTensorType::Q4_0,
                    TensorRepresentation::GgmlQ2_K => ConsumerTensorType::Q2_K,
                    TensorRepresentation::GgmlIQ1_S => ConsumerTensorType::IQ1_S,
                    TensorRepresentation::GgmlQ4_K => ConsumerTensorType::Q4_K,
                    TensorRepresentation::GgmlIQ4_NL => ConsumerTensorType::IQ4_NL,
                    TensorRepresentation::GgmlIQ4_XS => ConsumerTensorType::IQ4_XS,
                    TensorRepresentation::GgmlQ3_K => ConsumerTensorType::Q3_K,
                    TensorRepresentation::GgmlIQ2_XXS => ConsumerTensorType::IQ2_XXS,
                    TensorRepresentation::GgmlIQ2_XS => ConsumerTensorType::IQ2_XS,
                    TensorRepresentation::GgmlIQ2_S => ConsumerTensorType::IQ2_S,
                    TensorRepresentation::GgmlQ5_K => ConsumerTensorType::Q5_K,
                    TensorRepresentation::GgmlQ6_K => ConsumerTensorType::Q6_K,
                    TensorRepresentation::F8_E4M3 => ConsumerTensorType::F8_E4M3,
                },
                offset: tensor.payload.offset(),
                length: tensor.payload.length(),
            })
            .collect();
        let token_count = views.tokenizer.token_count();
        let mut token_text = Vec::with_capacity(usize::try_from(token_count).map_err(|_| {
            MlError::new(
                MlErrorCode::TokenizerArrayLengthMismatch,
                "vocabulary exceeds host range",
            )
        })?);
        let mut token_types = Vec::with_capacity(token_text.capacity());
        let mut token_scores = Vec::with_capacity(token_text.capacity());
        for index in 0..token_count {
            token_text.push(
                views
                    .tokenizer
                    .token_text(index)
                    .ok_or_else(|| {
                        MlError::new(MlErrorCode::InvalidTokenText, "token text snapshot failed")
                    })?
                    .to_owned(),
            );
            token_types.push(views.tokenizer.token_type(index));
            token_scores.push(views.tokenizer.score(index));
        }
        let mut merges =
            Vec::with_capacity(usize::try_from(views.tokenizer.merge_count()).unwrap_or(0));
        for index in 0..views.tokenizer.merge_count() {
            merges.push(views.tokenizer.merge_pair(index).ok_or_else(|| {
                MlError::new(MlErrorCode::InvalidMergeTable, "merge snapshot failed")
            })?);
        }
        let mut special_tokens = [None; 4];
        for (kind, wanted) in [
            (0, crate::SpecialToken::Bos),
            (1, crate::SpecialToken::Eos),
            (2, crate::SpecialToken::Unk),
            (3, crate::SpecialToken::Pad),
        ] {
            special_tokens[kind] = views
                .tokenizer
                .specials()
                .iter()
                .find(|(token, _)| *token == wanted)
                .map(|(_, value)| *value);
        }
        let _ = views.bootstrap.profile_version();
        let _ = views.validated.blocks().len();
        Ok(Snapshot {
            tensors,
            metadata,
            token_text,
            token_types,
            token_scores,
            merges,
            special_tokens,
            add_bos: views.tokenizer.add_bos(),
            chat_template: views.tokenizer.chat_template().map(ToOwned::to_owned),
        })
    }

    pub(crate) fn snapshot(&self) -> Result<&Snapshot, MlError> {
        if self.snapshot.get().is_none() {
            let _ = self.snapshot.set(self.build_snapshot()?);
        }
        self.snapshot.get().ok_or_else(|| {
            MlError::new(
                MlErrorCode::MalformedTensorDirectory,
                "consumer snapshot is unavailable",
            )
        })
    }
    pub fn tensor_count(&self) -> Result<usize, MlError> {
        Ok(self.snapshot()?.tensors.len())
    }
    pub fn tensor_name(&self, index: usize) -> Result<Option<String>, MlError> {
        Ok(self.snapshot()?.tensors.get(index).map(|t| t.name.clone()))
    }
    pub fn tensor_shape(&self, index: usize) -> Result<Option<Vec<u64>>, MlError> {
        Ok(self
            .snapshot()?
            .tensors
            .get(index)
            .map(|t| t.dimensions.clone()))
    }
    pub fn tensor_type(&self, index: usize) -> Result<Option<ConsumerTensorType>, MlError> {
        Ok(self.snapshot()?.tensors.get(index).map(|t| t.kind))
    }
    pub fn tensor_payload(&self, index: usize) -> Result<Option<&[u8]>, MlError> {
        let tensor = self.snapshot()?.tensors.get(index);
        Ok(tensor.map(|t| {
            let start = usize::try_from(t.offset).expect("validated offset fits host");
            let end = start + usize::try_from(t.length).expect("validated length fits host");
            &self.mapping[start..end]
        }))
    }
    pub fn model_metadata(&self) -> Result<ConsumerModelMetadata, MlError> {
        Ok(self.snapshot()?.metadata.clone())
    }
    pub fn tokenizer_count(&self) -> Result<u64, MlError> {
        Ok(self.snapshot()?.token_text.len() as u64)
    }
    pub fn token_text(&self, index: u64) -> Result<Option<String>, MlError> {
        Ok(self
            .snapshot()?
            .token_text
            .get(usize::try_from(index).unwrap_or(usize::MAX))
            .cloned())
    }
    pub fn token_type(&self, index: u64) -> Result<Option<u64>, MlError> {
        Ok(self
            .snapshot()?
            .token_types
            .get(usize::try_from(index).unwrap_or(usize::MAX))
            .copied()
            .flatten())
    }
    pub fn token_score(&self, index: u64) -> Result<Option<f64>, MlError> {
        Ok(self
            .snapshot()?
            .token_scores
            .get(usize::try_from(index).unwrap_or(usize::MAX))
            .copied()
            .flatten())
    }
    pub fn special_token(&self, kind: u8) -> Result<Option<u64>, MlError> {
        Ok(self
            .snapshot()?
            .special_tokens
            .get(usize::from(kind))
            .copied()
            .flatten())
    }
    pub fn merge_count(&self) -> Result<u64, MlError> {
        Ok(self.snapshot()?.merges.len() as u64)
    }
    pub fn merge_pair(&self, index: u64) -> Result<Option<(u64, u64)>, MlError> {
        Ok(self
            .snapshot()?
            .merges
            .get(usize::try_from(index).unwrap_or(usize::MAX))
            .copied())
    }
    pub fn add_bos(&self) -> Result<Option<bool>, MlError> {
        Ok(self.snapshot()?.add_bos)
    }
    pub fn chat_template(&self) -> Result<Option<String>, MlError> {
        Ok(self.snapshot()?.chat_template.clone())
    }
    pub fn is_validated(&self) -> Result<bool, MlError> {
        self.snapshot().map(|_| true)
    }
}
