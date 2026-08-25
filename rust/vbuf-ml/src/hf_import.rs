//! Streaming Hugging Face Safetensors importer.
//!
//! The importer is deliberately split into a metadata-only planner and a
//! range executor. Safetensors is never a runtime source: the executor writes
//! source bytes into the already planned canonical vBuf payload ranges.

use crate::bootstrap::{
    BOOTSTRAP_KEY_ID, Bootstrap, BootstrapEntry, encode_payload as encode_bootstrap,
};
use crate::layout::{LayoutClass, LayoutPlan, PlacementLengthRequest};
use crate::metadata::{MetadataEntry, ModelMetadataKey, encode_payload as encode_metadata};
use crate::quantization::{F8QuantizationEntry, encode_payload as encode_quantization};
use crate::region_roles::RegionRole;
use crate::representations::{TensorRepresentation, logical_elements};
use crate::tensor_directory::{TensorDirectory, TensorEntry, encode_payload as encode_directory};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::Digest;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Duration;
use vbuf_core::v06::{V06Physical, V06Semantic, parse_v06};

pub const DEFAULT_STAGING_BYTES: u64 = 64 * 1024 * 1024;
pub const DEFAULT_INFLIGHT_BYTES: u64 = DEFAULT_STAGING_BYTES;
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
const COALESCE_GAP_BYTES: u64 = 64 * 1024;
const SAFETY_MARGIN_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SourceIdentity {
    pub repository: String,
    pub requested_revision: String,
    pub resolved_revision: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ShardPlan {
    pub id: String,
    pub file_name: String,
    pub size: u64,
    pub header_length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TensorPlan {
    pub name: String,
    pub source_shard: String,
    pub source_begin: u64,
    pub source_end: u64,
    pub source_length: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub representation: Option<TensorRepresentationId>,
    pub destination_offset: u64,
    pub destination_block_start: u64,
    pub destination_length: u64,
    pub destination_alignment: u64,
    pub destination_occurrence: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TensorRepresentationId {
    CanonicalPrimitive,
    Bf16,
    F8E4M3,
}

impl TensorRepresentationId {
    fn runtime(self) -> TensorRepresentation {
        match self {
            Self::CanonicalPrimitive => TensorRepresentation::CanonicalPrimitive,
            Self::Bf16 => TensorRepresentation::Bf16,
            Self::F8E4M3 => TensorRepresentation::F8_E4M3,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PlanReport {
    pub source: SourceIdentity,
    pub shards: Vec<ShardPlan>,
    pub tensors: Vec<TensorPlan>,
    pub source_payload_bytes: u64,
    pub final_vbuf_payload_bytes: u64,
    pub final_vbuf_bytes: u64,
    pub metadata_bytes: u64,
    pub bootstrap_bytes: u64,
    pub alignment_padding_bytes: u64,
    pub largest_tensor: u64,
    pub dtype_distribution: BTreeMap<String, u64>,
    pub destination_alignment: u64,
    pub estimated_staging_bytes: u64,
    pub unsupported_dtypes: Vec<String>,
    pub unsupported_semantics: Vec<String>,
    pub all_destination_offsets_known: bool,
    pub can_execute: bool,
    pub expert_bank_repack_required: bool,
    pub plan_digest: String,
    pub controls: Vec<ControlPlan>,
    pub metadata_files: Vec<String>,
    pub quantization: Vec<F8QuantizationEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ControlPlan {
    pub block_start: u64,
    pub payload_offset: u64,
    pub bytes: Vec<u8>,
    pub semantic: u8,
    pub physical: u8,
    pub bit_width: u16,
    pub count: u64,
    pub key_id: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RangeResponse {
    pub status: u16,
    pub content_range: Option<(u64, u64, Option<u64>)>,
    pub body: Vec<u8>,
}

pub trait RepositoryTransport {
    fn metadata(&mut self, name: &str) -> Result<Option<Vec<u8>>, String>;
    fn range(&mut self, shard: &str, start: u64, end: u64) -> Result<RangeResponse, String>;
}

#[derive(Clone, Debug)]
pub struct RepositoryMetadata {
    pub source: SourceIdentity,
    pub config: Vec<u8>,
    pub index: Option<Vec<u8>>,
    pub metadata_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct SourceTensor {
    pub name: String,
    pub shard: String,
    pub begin: u64,
    pub end: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub representation: Option<TensorRepresentationId>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ConversionState {
    version: u32,
    plan_digest: String,
    source: SourceIdentity,
    destination_size: u64,
    completed: BTreeSet<String>,
}

#[derive(Clone, Debug)]
pub struct ImportOptions {
    pub staging_bytes: u64,
    pub max_retries: u32,
    pub parallel_requests: usize,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            staging_bytes: DEFAULT_STAGING_BYTES,
            max_retries: 3,
            parallel_requests: 1,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ImportStats {
    pub planning_requests: u64,
    pub planning_bytes: u64,
    pub payload_requests: u64,
    pub payload_requested_bytes: u64,
    pub payload_returned_bytes: u64,
    pub destination_written_bytes: u64,
    pub source_overfetch_bytes: u64,
    pub resume_skipped_bytes: u64,
    pub resume_downloaded_bytes: u64,
}

#[derive(Clone, Debug)]
pub enum ImportError {
    Invalid(String),
    Io(String),
    Transport(String),
    Range(String),
    Space { free: u64, required: u64 },
    IdentityMismatch(String),
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(v) => write!(f, "invalid Safetensors import: {v}"),
            Self::Io(v) => write!(f, "Safetensors import I/O error: {v}"),
            Self::Transport(v) => write!(f, "Safetensors transport error: {v}"),
            Self::Range(v) => write!(f, "Safetensors range error: {v}"),
            Self::Space { free, required } => {
                write!(f, "insufficient space: free={free} required={required}")
            }
            Self::IdentityMismatch(v) => write!(f, "conversion identity mismatch: {v}"),
        }
    }
}

impl std::error::Error for ImportError {}

fn checked_add(a: u64, b: u64, label: &str) -> Result<u64, ImportError> {
    a.checked_add(b)
        .ok_or_else(|| ImportError::Invalid(format!("{label} overflows u64")))
}

fn checked_mul(a: u64, b: u64, label: &str) -> Result<u64, ImportError> {
    a.checked_mul(b)
        .ok_or_else(|| ImportError::Invalid(format!("{label} overflows u64")))
}

fn parse_dtype(
    value: &str,
    shape: &[u64],
) -> Result<(Option<TensorRepresentationId>, u64), ImportError> {
    let elements = if shape.is_empty() {
        1
    } else {
        logical_elements(shape).map_err(|e| ImportError::Invalid(e.to_string()))?
    };
    let (representation, bytes_per_element) = match value {
        "F32" => (Some(TensorRepresentationId::CanonicalPrimitive), 4),
        "BF16" => (Some(TensorRepresentationId::Bf16), 2),
        "F8_E4M3" => (Some(TensorRepresentationId::F8E4M3), 1),
        "F16" | "F64" | "I8" | "I16" | "I32" | "I64" | "U8" | "U16" | "U32" | "BOOL"
        | "F8_E5M2" => (None, 0),
        _ => (None, 0),
    };
    let expected = checked_mul(
        elements,
        bytes_per_element,
        "Safetensors tensor byte length",
    )?;
    Ok((representation, expected))
}

pub fn parse_safetensors_header(
    shard_id: impl Into<String>,
    file_size: u64,
    first_eight: &[u8],
    header: &[u8],
) -> Result<Vec<SourceTensor>, ImportError> {
    if first_eight.len() != 8 || header.len() > usize::try_from(MAX_HEADER_BYTES).unwrap() {
        return Err(ImportError::Invalid(
            "invalid Safetensors header framing".into(),
        ));
    }
    let shard_id = shard_id.into();
    let header_length = u64::from_le_bytes(first_eight.try_into().unwrap());
    if header_length == 0 || header_length > MAX_HEADER_BYTES {
        return Err(ImportError::Invalid(
            "Safetensors header length is outside bounds".into(),
        ));
    }
    if header_length != u64::try_from(header.len()).unwrap() {
        return Err(ImportError::Invalid(
            "Safetensors header length mismatch".into(),
        ));
    }
    let json = std::str::from_utf8(header)
        .map_err(|_| ImportError::Invalid("Safetensors header is not UTF-8".into()))?;
    let object = serde_json::from_str::<Value>(json)
        .map_err(|e| ImportError::Invalid(format!("Safetensors header JSON: {e}")))?
        .as_object()
        .cloned()
        .ok_or_else(|| ImportError::Invalid("Safetensors header is not an object".into()))?;
    let data_start = checked_add(8, header_length, "Safetensors data start")?;
    if data_start > file_size {
        return Err(ImportError::Invalid(
            "Safetensors header exceeds file size".into(),
        ));
    }
    let mut tensors = Vec::new();
    let mut names = BTreeSet::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        if !names.insert(name.clone()) {
            return Err(ImportError::Invalid(
                "duplicate Safetensors tensor name".into(),
            ));
        }
        let object = value
            .as_object()
            .ok_or_else(|| ImportError::Invalid("tensor metadata is not an object".into()))?;
        let dtype = object
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| ImportError::Invalid("tensor dtype is missing".into()))?
            .to_string();
        let shape_value = object
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| ImportError::Invalid("tensor shape is missing".into()))?;
        let mut shape = Vec::with_capacity(shape_value.len());
        for dimension in shape_value {
            let dimension = dimension.as_u64().ok_or_else(|| {
                ImportError::Invalid("tensor shape dimension is not a non-negative integer".into())
            })?;
            shape.push(dimension);
        }
        let offsets = object
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| ImportError::Invalid("tensor data_offsets is missing".into()))?;
        if offsets.len() != 2 {
            return Err(ImportError::Invalid(
                "tensor data_offsets must contain two values".into(),
            ));
        }
        let begin = offsets[0]
            .as_u64()
            .ok_or_else(|| ImportError::Invalid("tensor data offset is invalid".into()))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| ImportError::Invalid("tensor data offset is invalid".into()))?;
        if begin > end {
            return Err(ImportError::Invalid(
                "tensor data_offsets are reversed".into(),
            ));
        }
        let source_begin = checked_add(data_start, begin, "Safetensors source offset")?;
        let source_end = checked_add(data_start, end, "Safetensors source end")?;
        if source_end > file_size {
            return Err(ImportError::Invalid(
                "tensor range exceeds Safetensors file".into(),
            ));
        }
        let (representation, expected) = parse_dtype(&dtype, &shape)?;
        let length = end - begin;
        if representation.is_some() && expected != length {
            return Err(ImportError::Invalid(format!(
                "tensor {name} byte length does not match dtype and shape"
            )));
        }
        tensors.push(SourceTensor {
            name,
            shard: shard_id.clone(),
            begin: source_begin,
            end: source_end,
            dtype,
            shape,
            representation,
        });
    }
    tensors.sort_by(|a, b| {
        (a.shard.as_str(), a.begin, a.name.as_str()).cmp(&(
            b.shard.as_str(),
            b.begin,
            b.name.as_str(),
        ))
    });
    let mut previous_end = data_start;
    for tensor in &tensors {
        if tensor.begin < previous_end {
            return Err(ImportError::Invalid(
                "Safetensors tensor ranges overlap".into(),
            ));
        }
        previous_end = tensor.end;
    }
    Ok(tensors)
}

fn index_map(index: Option<&Value>) -> Result<Option<BTreeMap<String, String>>, ImportError> {
    let Some(index) = index else { return Ok(None) };
    let map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| ImportError::Invalid("Safetensors index has no weight_map".into()))?;
    let mut result = BTreeMap::new();
    for (name, shard) in map {
        let shard = shard
            .as_str()
            .ok_or_else(|| ImportError::Invalid("Safetensors index shard is not text".into()))?;
        if result.insert(name.clone(), shard.to_string()).is_some() {
            return Err(ImportError::Invalid(
                "duplicate tensor ownership in Safetensors index".into(),
            ));
        }
    }
    Ok(Some(result))
}

pub fn build_plan(
    metadata: RepositoryMetadata,
    index_json: Option<&[u8]>,
    shard_headers: Vec<(ShardPlan, Vec<u8>, Vec<u8>)>,
    staging_bytes: u64,
) -> Result<PlanReport, ImportError> {
    if staging_bytes == 0 {
        return Err(ImportError::Invalid(
            "staging limit must be non-zero".into(),
        ));
    }
    let config: Value = serde_json::from_slice(&metadata.config)
        .map_err(|e| ImportError::Invalid(format!("config.json: {e}")))?;
    let index_value = index_json
        .map(|bytes| serde_json::from_slice::<Value>(bytes))
        .transpose()
        .map_err(|e| ImportError::Invalid(format!("model.safetensors.index.json: {e}")))?;
    let ownership = index_map(index_value.as_ref())?;
    let mut sources = Vec::new();
    let mut shard_plans = Vec::new();
    let mut shard_names = BTreeSet::new();
    for (shard, first_eight, header) in shard_headers {
        shard_plans.push(shard.clone());
        if !shard_names.insert(shard.file_name.clone()) {
            return Err(ImportError::Invalid("duplicate Safetensors shard".into()));
        }
        let mut tensors =
            parse_safetensors_header(shard.file_name.clone(), shard.size, &first_eight, &header)?;
        for tensor in &mut tensors {
            if let Some(ownership) = ownership.as_ref() {
                match ownership.get(&tensor.name) {
                    Some(owner) if owner == &shard.file_name => {}
                    Some(_) => {
                        return Err(ImportError::Invalid(format!(
                            "index/header shard disagreement for {}",
                            tensor.name
                        )));
                    }
                    None => {
                        return Err(ImportError::Invalid(format!(
                            "header tensor {} is absent from index",
                            tensor.name
                        )));
                    }
                }
            }
        }
        sources.extend(tensors);
    }
    if let Some(ownership) = ownership.as_ref() {
        for name in ownership.keys() {
            if !sources.iter().any(|tensor| &tensor.name == name) {
                return Err(ImportError::Invalid(format!(
                    "indexed tensor {name} is absent from shard headers"
                )));
            }
        }
    }
    let mut names = BTreeSet::new();
    let mut dtype_distribution = BTreeMap::new();
    let mut unsupported_dtypes = BTreeSet::new();
    let mut source_payload_bytes = 0u64;
    let mut tensor_plans = Vec::new();
    for source in sources {
        if !names.insert(source.name.clone()) {
            return Err(ImportError::Invalid(
                "duplicate tensor name across Safetensors shards".into(),
            ));
        }
        let length = source.end - source.begin;
        source_payload_bytes = checked_add(source_payload_bytes, length, "source payload total")?;
        *dtype_distribution.entry(source.dtype.clone()).or_insert(0) += 1;
        if source.representation.is_none() {
            unsupported_dtypes.insert(source.dtype.clone());
        }
        tensor_plans.push(TensorPlan {
            name: source.name,
            source_shard: source.shard,
            source_begin: source.begin,
            source_end: source.end,
            source_length: length,
            dtype: source.dtype,
            shape: source.shape,
            representation: source.representation,
            destination_offset: 0,
            destination_length: length,
            destination_alignment: 0,
            destination_block_start: 0,
            destination_occurrence: 0,
        });
    }
    tensor_plans.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
    let mut quantization = Vec::new();
    for tensor in &tensor_plans {
        if tensor.representation != Some(TensorRepresentationId::F8E4M3) {
            continue;
        }
        if tensor.shape.len() != 2 {
            return Err(ImportError::Invalid(format!(
                "FP8 tensor {} must be a 2-D matrix",
                tensor.name
            )));
        }
        let scale_name = [
            format!("{}_scale_inv", tensor.name),
            format!("{}_scale", tensor.name),
        ]
        .into_iter()
        .find(|candidate| tensor_plans.iter().any(|tensor| tensor.name == *candidate))
        .ok_or_else(|| {
            ImportError::Invalid(format!(
                "FP8 tensor {} has no semantic scale tensor",
                tensor.name
            ))
        })?;
        let scale = tensor_plans
            .iter()
            .find(|candidate| candidate.name == scale_name)
            .expect("scale name was selected from tensor plans");
        if scale.representation != Some(TensorRepresentationId::CanonicalPrimitive)
            || scale.shape.len() != 2
        {
            return Err(ImportError::Invalid(format!(
                "FP8 scale tensor {} must be an F32 matrix",
                scale.name
            )));
        }
        let rows = tensor.shape[0];
        let columns = tensor.shape[1];
        let block_size = fp8_block_size(&config, [rows, columns])?;
        let expected_scale_shape = [
            rows.div_ceil(block_size[0]),
            columns.div_ceil(block_size[1]),
        ];
        let scale_shape_matches = scale.shape.as_slice() == expected_scale_shape
            || (block_size == [1, columns] && scale.shape.as_slice() == [rows]);
        if !scale_shape_matches {
            return Err(ImportError::Invalid(format!(
                "FP8 scale tensor {} shape does not match configured block geometry",
                scale.name
            )));
        }
        quantization.push(F8QuantizationEntry {
            weight_name: tensor.name.clone(),
            scale_name,
            block_rows: block_size[0],
            block_columns: block_size[1],
        });
    }
    let quantization_directory =
        encode_quantization(&quantization).map_err(|e| ImportError::Invalid(e.to_string()))?;
    let architecture = config
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|v| v.first())
        .and_then(Value::as_str)
        .unwrap_or("");
    let mut unsupported_semantics = Vec::new();
    if architecture.is_empty() {
        unsupported_semantics.push("config.json has no declarative architecture".into());
    }
    if architecture.to_ascii_lowercase().contains("moe")
        || architecture.to_ascii_lowercase().contains("deepseek")
    {
        unsupported_semantics
            .push("expert-bank provenance requires an architecture profile".into());
    }
    let base_shift = 4u8;
    let alignment = 1u64 << base_shift;
    let metadata_values = config_metadata(&config)?;
    let metadata_entries = metadata_values
        .iter()
        .map(|(key, _, _, _)| {
            MetadataEntry::new(
                *key,
                ModelMetadataKey::from_id(*key).is_some_and(ModelMetadataKey::is_required),
                0x0400 + *key,
                0,
            )
        })
        .collect::<Vec<_>>();
    let metadata_directory =
        encode_metadata(&metadata_entries).map_err(|e| ImportError::Invalid(e.to_string()))?;
    let directory_entries = tensor_plans
        .iter()
        .enumerate()
        .map(|(index, tensor)| {
            Ok(TensorEntry {
                name: tensor.name.clone(),
                dimensions: tensor.shape.clone(),
                representation: tensor
                    .representation
                    .map(TensorRepresentationId::runtime)
                    .unwrap_or(TensorRepresentation::CanonicalPrimitive),
                key_id: 0x0200,
                occurrence: u16::try_from(index)
                    .map_err(|_| ImportError::Invalid("tensor occurrence exceeds u16".into()))?,
            })
        })
        .collect::<Result<Vec<_>, ImportError>>()?;
    let directory =
        encode_directory(&directory_entries).map_err(|e| ImportError::Invalid(e.to_string()))?;
    let bootstrap = encode_bootstrap(&[
        BootstrapEntry::new(RegionRole::TensorDirectory as u16, true, 0x0201, 0),
        BootstrapEntry::new(RegionRole::ModelMetadata as u16, true, 0x0202, 0),
        BootstrapEntry::new(RegionRole::QuantizationMetadata as u16, false, 0x0203, 0),
    ])
    .map_err(|e| ImportError::Invalid(e.to_string()))?;
    let mut requests = Vec::new();
    requests.push(PlacementLengthRequest {
        class: LayoutClass::Bootstrap,
        order: 0,
        key_id: BOOTSTRAP_KEY_ID,
        semantic: V06Semantic::Opaque,
        physical: V06Physical::Array,
        bit_width: 8,
        count: bootstrap.len() as u64,
        payload_alignment: alignment,
        payload_len: bootstrap.len() as u64,
    });
    for (key, _, _, bytes) in &metadata_values {
        requests.push(PlacementLengthRequest {
            class: LayoutClass::ModelMetadata,
            order: 100 + u64::from(*key),
            key_id: 0x0400 + *key,
            semantic: if ModelMetadataKey::from_id(*key)
                .is_some_and(|v| matches!(v, ModelMetadataKey::Architecture))
            {
                V06Semantic::Opaque
            } else if ModelMetadataKey::from_id(*key).is_some_and(|v| {
                matches!(
                    v,
                    ModelMetadataKey::NormalizationEpsilon | ModelMetadataKey::RopeTheta
                )
            }) {
                V06Semantic::Float
            } else {
                V06Semantic::Unsigned
            },
            physical: if ModelMetadataKey::from_id(*key)
                .is_some_and(|v| matches!(v, ModelMetadataKey::Architecture))
            {
                V06Physical::Array
            } else {
                V06Physical::Scalar
            },
            bit_width: if ModelMetadataKey::from_id(*key)
                .is_some_and(|v| matches!(v, ModelMetadataKey::Architecture))
            {
                8
            } else if ModelMetadataKey::from_id(*key).is_some_and(|v| {
                matches!(
                    v,
                    ModelMetadataKey::NormalizationEpsilon | ModelMetadataKey::RopeTheta
                )
            }) {
                64
            } else {
                64
            },
            count: if ModelMetadataKey::from_id(*key)
                .is_some_and(|v| matches!(v, ModelMetadataKey::Architecture))
            {
                bytes.len() as u64
            } else {
                1
            },
            payload_alignment: alignment,
            payload_len: bytes.len() as u64,
        });
    }
    requests.push(PlacementLengthRequest {
        class: LayoutClass::ModelMetadata,
        order: 1,
        key_id: 0x0202,
        semantic: V06Semantic::Opaque,
        physical: V06Physical::Array,
        bit_width: 8,
        count: metadata_directory.len() as u64,
        payload_alignment: alignment,
        payload_len: metadata_directory.len() as u64,
    });
    requests.push(PlacementLengthRequest {
        class: LayoutClass::TensorDirectory,
        order: 2,
        key_id: 0x0201,
        semantic: V06Semantic::Opaque,
        physical: V06Physical::Array,
        bit_width: 8,
        count: directory.len() as u64,
        payload_alignment: alignment,
        payload_len: directory.len() as u64,
    });
    requests.push(PlacementLengthRequest {
        class: LayoutClass::Auxiliary,
        order: 3,
        key_id: 0x0203,
        semantic: V06Semantic::Opaque,
        physical: V06Physical::Array,
        bit_width: 8,
        count: quantization_directory.len() as u64,
        payload_alignment: alignment,
        payload_len: quantization_directory.len() as u64,
    });
    for (index, tensor) in tensor_plans.iter().enumerate() {
        let (semantic, width, count) = match tensor.representation {
            Some(TensorRepresentationId::CanonicalPrimitive) => (
                V06Semantic::Float,
                32,
                logical_elements(&tensor.shape).map_err(|e| ImportError::Invalid(e.to_string()))?,
            ),
            Some(TensorRepresentationId::Bf16) => (V06Semantic::Opaque, 8, tensor.source_length),
            Some(TensorRepresentationId::F8E4M3) => (V06Semantic::Opaque, 8, tensor.source_length),
            None => (V06Semantic::Opaque, 8, tensor.source_length),
        };
        requests.push(PlacementLengthRequest {
            class: LayoutClass::TensorPayload,
            order: 1_000_000 + index as u64,
            key_id: 0x0200,
            semantic,
            physical: V06Physical::Array,
            bit_width: width,
            count,
            payload_alignment: alignment,
            payload_len: tensor.source_length,
        });
    }
    let layout = LayoutPlan::build_lengths(&requests, base_shift)
        .map_err(|e| ImportError::Invalid(e.to_string()))?;
    let tensor_count = tensor_plans.len();
    let tensor_request_start = requests.len() - tensor_count;
    for (index, tensor) in tensor_plans.iter_mut().enumerate() {
        let request_index = tensor_request_start + index;
        let entry = layout
            .entries()
            .iter()
            .find(|entry| entry.request_index == request_index)
            .ok_or_else(|| ImportError::Invalid("tensor placement is missing".into()))?;
        tensor.destination_offset = entry.payload_start;
        tensor.destination_block_start = entry.block_start;
        tensor.destination_alignment = alignment;
        tensor.destination_occurrence = u16::try_from(index)
            .map_err(|_| ImportError::Invalid("tensor occurrence exceeds u16".into()))?;
    }
    let mut controls = Vec::new();
    let mut ordered_control_payloads = Vec::new();
    let bootstrap_bytes = bootstrap.len() as u64;
    let metadata_bytes = metadata_directory.len() as u64;
    let directory_bytes = directory.len() as u64;
    let quantization_bytes = quantization_directory.len() as u64;
    ordered_control_payloads.push((
        0usize,
        bootstrap.clone(),
        V06Semantic::Opaque,
        V06Physical::Array,
        8u16,
        requests[0].count,
        BOOTSTRAP_KEY_ID,
    ));
    let mut request_cursor = 1usize;
    for (_key, _, _, bytes) in &metadata_values {
        ordered_control_payloads.push((
            request_cursor,
            bytes.clone(),
            requests[request_cursor].semantic,
            requests[request_cursor].physical,
            requests[request_cursor].bit_width,
            requests[request_cursor].count,
            requests[request_cursor].key_id,
        ));
        request_cursor += 1;
    }
    ordered_control_payloads.push((
        request_cursor,
        metadata_directory.clone(),
        requests[request_cursor].semantic,
        requests[request_cursor].physical,
        requests[request_cursor].bit_width,
        requests[request_cursor].count,
        requests[request_cursor].key_id,
    ));
    request_cursor += 1;
    ordered_control_payloads.push((
        request_cursor,
        directory.clone(),
        requests[request_cursor].semantic,
        requests[request_cursor].physical,
        requests[request_cursor].bit_width,
        requests[request_cursor].count,
        requests[request_cursor].key_id,
    ));
    request_cursor += 1;
    ordered_control_payloads.push((
        request_cursor,
        quantization_directory.clone(),
        requests[request_cursor].semantic,
        requests[request_cursor].physical,
        requests[request_cursor].bit_width,
        requests[request_cursor].count,
        requests[request_cursor].key_id,
    ));
    for (request_index, bytes, semantic, physical, width, count, key_id) in ordered_control_payloads
    {
        let entry = layout
            .entries()
            .iter()
            .find(|entry| entry.request_index == request_index)
            .ok_or_else(|| ImportError::Invalid("control placement is missing".into()))?;
        controls.push(ControlPlan {
            block_start: entry.block_start,
            payload_offset: entry.payload_start,
            bytes,
            semantic: semantic as u8,
            physical: physical as u8,
            bit_width: width,
            count,
            key_id,
        });
    }
    if let Some(index) = index_value
        .as_ref()
        .and_then(|value| value.get("metadata"))
        .and_then(Value::as_object)
        .and_then(|value| value.get("total_size"))
        .and_then(Value::as_u64)
        && index != source_payload_bytes
    {
        unsupported_semantics.push(
            "Safetensors index total_size differs from header-derived payload total; headers are authoritative".into(),
        );
    }
    let mut plan = PlanReport {
        source: metadata.source,
        shards: shard_plans,
        tensors: tensor_plans,
        source_payload_bytes,
        final_vbuf_payload_bytes: source_payload_bytes,
        final_vbuf_bytes: layout.final_size(),
        metadata_bytes: metadata_bytes + directory_bytes + quantization_bytes,
        bootstrap_bytes,
        alignment_padding_bytes: layout.padding_bytes(),
        largest_tensor: 0,
        dtype_distribution,
        destination_alignment: alignment,
        estimated_staging_bytes: staging_bytes,
        unsupported_dtypes: unsupported_dtypes.into_iter().collect(),
        unsupported_semantics,
        all_destination_offsets_known: true,
        can_execute: false,
        expert_bank_repack_required: false,
        plan_digest: String::new(),
        controls,
        metadata_files: Vec::new(),
        quantization,
    };
    plan.largest_tensor = plan
        .tensors
        .iter()
        .map(|t| t.source_length)
        .max()
        .unwrap_or(0);
    plan.can_execute = plan.unsupported_dtypes.is_empty() && plan.largest_tensor <= staging_bytes;
    plan.plan_digest = digest_plan(&plan)?;
    Ok(plan)
}

fn fp8_block_size(config: &Value, shape: [u64; 2]) -> Result<[u64; 2], ImportError> {
    let value = config.get("weight_block_size").or_else(|| {
        config
            .get("quantization_config")
            .and_then(|v| v.get("weight_block_size"))
    });
    if let Some(value) = value {
        let values = value
            .as_array()
            .ok_or_else(|| ImportError::Invalid("weight_block_size must be an array".into()))?;
        if values.len() != 2 {
            return Err(ImportError::Invalid(
                "weight_block_size must contain exactly two dimensions".into(),
            ));
        }
        let rows = values[0]
            .as_u64()
            .filter(|value| *value != 0)
            .ok_or_else(|| {
                ImportError::Invalid("weight_block_size rows must be non-zero".into())
            })?;
        let columns = values[1]
            .as_u64()
            .filter(|value| *value != 0)
            .ok_or_else(|| {
                ImportError::Invalid("weight_block_size columns must be non-zero".into())
            })?;
        return Ok([rows, columns]);
    }
    let channel_strategy = config
        .get("quantization_config")
        .and_then(|value| value.get("config_groups"))
        .and_then(Value::as_object)
        .is_some_and(|groups| {
            groups.values().any(|group| {
                group
                    .get("weights")
                    .and_then(|weights| weights.get("strategy"))
                    .and_then(Value::as_str)
                    == Some("channel")
            })
        });
    if channel_strategy {
        return Ok([1, shape[1]]);
    }
    Err(ImportError::Invalid(
        "F8_E4M3 requires config-driven weight_block_size or channel strategy".into(),
    ))
}

fn config_metadata(config: &Value) -> Result<Vec<(u16, String, u8, Vec<u8>)>, ImportError> {
    let architecture = config
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|v| v.first())
        .and_then(Value::as_str)
        .ok_or_else(|| ImportError::Invalid("config.json architectures is required".into()))?;
    let mut result = vec![(
        1,
        architecture.to_string(),
        1,
        architecture.as_bytes().to_vec(),
    )];
    let unsigned = [
        (2, "max_position_embeddings"),
        (3, "hidden_size"),
        (4, "num_hidden_layers"),
        (5, "num_attention_heads"),
        (6, "intermediate_size"),
        (9, "num_key_value_heads"),
    ];
    for (key, name) in unsigned {
        if let Some(value) = config.get(name).and_then(Value::as_u64) {
            result.push((key, name.into(), 2, value.to_le_bytes().to_vec()));
        }
    }
    for (key, name) in [(7, "rms_norm_eps"), (8, "rope_theta")] {
        if let Some(value) = config.get(name).and_then(Value::as_f64) {
            result.push((key, name.into(), 3, value.to_le_bytes().to_vec()));
        }
    }
    result.sort_by_key(|v| v.0);
    Ok(result)
}

fn digest_plan(plan: &PlanReport) -> Result<String, ImportError> {
    let bytes = serde_json::to_vec(plan).map_err(|e| ImportError::Invalid(e.to_string()))?;
    let digest = sha2::Sha256::digest(bytes);
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}

pub fn serialized_plan(plan: &PlanReport) -> Result<Vec<u8>, ImportError> {
    serde_json::to_vec_pretty(plan).map_err(|e| ImportError::Invalid(e.to_string()))
}

#[derive(Clone, Debug, Default)]
pub struct PlanningStats {
    pub requests: u64,
    pub bytes: u64,
    pub metadata_files: Vec<String>,
}

pub fn plan_remote(
    transport: &mut HfTransport,
    requested_revision: &str,
    repository: &str,
    staging_bytes: u64,
) -> Result<(PlanReport, PlanningStats), ImportError> {
    let resolved_revision = transport
        .resolve_revision()
        .map_err(ImportError::Transport)?;
    transport.revision = resolved_revision.clone();
    let mut stats = PlanningStats::default();
    let config = transport
        .metadata("config.json")
        .map_err(ImportError::Transport)?
        .ok_or_else(|| ImportError::Invalid("config.json is missing".into()))?;
    stats.requests += 1;
    stats.bytes += config.len() as u64;
    let index = transport
        .metadata("model.safetensors.index.json")
        .map_err(ImportError::Transport)?;
    if let Some(bytes) = &index {
        stats.requests += 1;
        stats.bytes += bytes.len() as u64;
    }
    for name in [
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ] {
        if let Some(bytes) = transport.metadata(name).map_err(ImportError::Transport)? {
            stats.requests += 1;
            stats.bytes += bytes.len() as u64;
            stats.metadata_files.push(name.to_string());
        }
    }
    let shard_names = if let Some(bytes) = &index {
        let value: Value = serde_json::from_slice(bytes)
            .map_err(|e| ImportError::Invalid(format!("Safetensors index: {e}")))?;
        index_map(Some(&value))
            .map_err(|e| e)?
            .ok_or_else(|| ImportError::Invalid("Safetensors index has no weight_map".into()))?
            .values()
            .cloned()
            .collect::<BTreeSet<_>>()
    } else {
        BTreeSet::from(["model.safetensors".to_string()])
    };
    let mut shard_headers = Vec::new();
    for name in shard_names {
        let first = transport
            .range(&name, 0, 8)
            .map_err(ImportError::Transport)?;
        stats.requests += 1;
        stats.bytes += first.body.len() as u64;
        if first.status != 206 || first.body.len() != 8 {
            return Err(ImportError::Range(
                "Safetensors header prefix was not a bounded 206 response".into(),
            ));
        }
        let header_length = u64::from_le_bytes(first.body[..8].try_into().unwrap());
        if header_length == 0 || header_length > MAX_HEADER_BYTES {
            return Err(ImportError::Invalid(
                "Safetensors header length is outside bounds".into(),
            ));
        }
        let size = first
            .content_range
            .and_then(|(_, _, total)| total)
            .ok_or_else(|| {
                ImportError::Range("header response did not publish source size".into())
            })?;
        let header_end = checked_add(8, header_length, "Safetensors header range")?;
        let response = transport
            .range(&name, 8, header_end)
            .map_err(ImportError::Transport)?;
        stats.requests += 1;
        stats.bytes += response.body.len() as u64;
        if response.status != 206
            || response.body.len() as u64 != header_length
            || !response
                .content_range
                .is_some_and(|v| v.0 == 8 && v.1 == header_end)
        {
            return Err(ImportError::Range(
                "Safetensors header range framing mismatch".into(),
            ));
        }
        shard_headers.push((
            ShardPlan {
                id: name.clone(),
                file_name: name,
                size,
                header_length,
            },
            first.body,
            response.body,
        ));
    }
    let metadata = RepositoryMetadata {
        source: SourceIdentity {
            repository: repository.into(),
            requested_revision: requested_revision.into(),
            resolved_revision,
        },
        config,
        index: index.clone(),
        metadata_bytes: stats.bytes,
    };
    let mut plan = build_plan(metadata, index.as_deref(), shard_headers, staging_bytes)?;
    plan.source.resolved_revision = transport.revision.clone();
    plan.metadata_files = stats.metadata_files.clone();
    plan.plan_digest.clear();
    plan.plan_digest = digest_plan(&plan)?;
    Ok((plan, stats))
}

pub fn space_gate(
    free_bytes: u64,
    final_vbuf_bytes: u64,
    staging_bytes: u64,
) -> Result<(), ImportError> {
    let required = checked_add(
        final_vbuf_bytes,
        checked_add(staging_bytes, SAFETY_MARGIN_BYTES, "space safety margin")?,
        "required free space",
    )?;
    if free_bytes < required {
        Err(ImportError::Space {
            free: free_bytes,
            required,
        })
    } else {
        Ok(())
    }
}

fn write_at(file: &File, offset: u64, bytes: &[u8]) -> Result<(), ImportError> {
    let mut written = 0usize;
    while written < bytes.len() {
        let n = file
            .write_at(&bytes[written..], offset + written as u64)
            .map_err(|e| ImportError::Io(e.to_string()))?;
        if n == 0 {
            return Err(ImportError::Io("zero-length destination write".into()));
        }
        written += n;
    }
    Ok(())
}

fn anchor(
    semantic: V06Semantic,
    physical: V06Physical,
    width: u16,
    count: u64,
    key: u16,
) -> Result<[u8; 16], ImportError> {
    let extended = count > 65535;
    let mut value = semantic as u64
        | ((physical as u64) << 4)
        | (u64::from(extended) << 9)
        | ((u64::from(key)) << 16)
        | ((u64::from(width)) << 32)
        | ((if extended { 0 } else { count }) << 48);
    let mut output = [0u8; 16];
    output[..8].copy_from_slice(&value.to_le_bytes());
    if extended {
        output[8..16].copy_from_slice(&count.to_le_bytes());
    }
    value = 0;
    let _ = value;
    Ok(output)
}

fn initialize_partial(
    path: &Path,
    plan: &PlanReport,
    controls: &[(u64, u64, Vec<u8>, V06Semantic, V06Physical, u16, u64, u16)],
) -> Result<(), ImportError> {
    let file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| ImportError::Io(e.to_string()))?;
    file.set_len(plan.final_vbuf_bytes)
        .map_err(|e| ImportError::Io(e.to_string()))?;
    let mut header = [0u8; 24];
    header[..4].copy_from_slice(b"VBUF");
    header[4..8].copy_from_slice(&0x0006_0000u32.to_le_bytes());
    header[8] = 4;
    header[10..12].copy_from_slice(&24u16.to_le_bytes());
    header[16..24].copy_from_slice(&(plan.final_vbuf_bytes - 32).to_le_bytes());
    write_at(&file, 0, &header)?;
    for (block_start, payload_offset, bytes, semantic, physical, width, count, key) in controls {
        let encoded = anchor(*semantic, *physical, *width, *count, *key)?;
        write_at(&file, *block_start, &encoded[..8])?;
        if *count > 65535 {
            write_at(&file, *block_start + 8, &encoded[8..])?;
        }
        write_at(&file, *payload_offset, bytes)?;
    }
    file.sync_data()
        .map_err(|e| ImportError::Io(e.to_string()))?;
    Ok(())
}

fn state_path(output: &Path) -> PathBuf {
    output.with_extension(format!(
        "{}convert-state",
        output
            .extension()
            .and_then(|v| v.to_str())
            .map(|v| format!("{v}."))
            .unwrap_or_default()
    ))
}

fn partial_path(output: &Path) -> PathBuf {
    output.with_extension(format!(
        "{}partial",
        output
            .extension()
            .and_then(|v| v.to_str())
            .map(|v| format!("{v}."))
            .unwrap_or_default()
    ))
}

fn persist_state(path: &Path, state: &ConversionState) -> Result<(), ImportError> {
    let temporary = path.with_extension("convert-state.tmp");
    let bytes = serde_json::to_vec(state).map_err(|e| ImportError::Io(e.to_string()))?;
    std::fs::write(&temporary, bytes).map_err(|e| ImportError::Io(e.to_string()))?;
    std::fs::rename(temporary, path).map_err(|e| ImportError::Io(e.to_string()))
}

pub fn execute_plan<T: RepositoryTransport>(
    transport: &mut T,
    plan: &PlanReport,
    output: &Path,
    options: &ImportOptions,
) -> Result<ImportStats, ImportError> {
    if !plan.can_execute {
        return Err(ImportError::Invalid("plan is not executable".into()));
    }
    if plan.largest_tensor > options.staging_bytes {
        return Err(ImportError::Invalid(
            "largest tensor exceeds staging limit".into(),
        ));
    }
    if output.exists() {
        return Err(ImportError::IdentityMismatch(
            "final output already exists".into(),
        ));
    }
    let partial = partial_path(output);
    let state_file = state_path(output);
    let mut state = if state_file.exists() {
        let bytes = std::fs::read(&state_file).map_err(|e| ImportError::Io(e.to_string()))?;
        serde_json::from_slice::<ConversionState>(&bytes)
            .map_err(|e| ImportError::IdentityMismatch(e.to_string()))?
    } else {
        ConversionState {
            version: 1,
            plan_digest: plan.plan_digest.clone(),
            source: plan.source.clone(),
            destination_size: plan.final_vbuf_bytes,
            completed: BTreeSet::new(),
        }
    };
    if state.plan_digest != plan.plan_digest
        || state.source != plan.source
        || state.destination_size != plan.final_vbuf_bytes
    {
        return Err(ImportError::IdentityMismatch(
            "plan, source revision, or destination layout differs".into(),
        ));
    }
    if !partial.exists() {
        let controls = plan
            .controls
            .iter()
            .map(|control| {
                (
                    control.block_start,
                    control.payload_offset,
                    control.bytes.clone(),
                    match control.semantic {
                        0 => V06Semantic::Unsigned,
                        1 => V06Semantic::Float,
                        2 => V06Semantic::Signed,
                        _ => V06Semantic::Opaque,
                    },
                    match control.physical {
                        0 => V06Physical::Scalar,
                        _ => V06Physical::Array,
                    },
                    control.bit_width,
                    control.count,
                    control.key_id,
                )
            })
            .chain(plan.tensors.iter().map(|tensor| {
                let (semantic, width, count) = match tensor.representation {
                    Some(TensorRepresentationId::CanonicalPrimitive) => (
                        V06Semantic::Float,
                        32,
                        logical_elements(&tensor.shape).unwrap_or(0),
                    ),
                    _ => (V06Semantic::Opaque, 8, tensor.source_length),
                };
                (
                    tensor.destination_block_start,
                    tensor.destination_offset,
                    Vec::new(),
                    semantic,
                    V06Physical::Array,
                    width,
                    count,
                    0x0200,
                )
            }))
            .collect::<Vec<_>>();
        initialize_partial(&partial, plan, &controls)?;
    } else if !state_file.exists() {
        return Err(ImportError::IdentityMismatch(
            "partial artifact has no conversion state".into(),
        ));
    }
    if !state_file.exists() {
        persist_state(&state_file, &state)?;
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&partial)
        .map_err(|e| ImportError::Io(e.to_string()))?;
    let mut stats = ImportStats::default();
    let mut by_shard: BTreeMap<&str, Vec<&TensorPlan>> = BTreeMap::new();
    for tensor in &plan.tensors {
        by_shard
            .entry(&tensor.source_shard)
            .or_default()
            .push(tensor);
    }
    for tensors in by_shard.values_mut() {
        tensors.sort_by_key(|tensor| tensor.source_begin);
        let mut index = 0;
        while index < tensors.len() {
            while index < tensors.len() && state.completed.contains(&tensors[index].name) {
                stats.resume_skipped_bytes += tensors[index].source_length;
                index += 1;
            }
            if index == tensors.len() {
                break;
            }
            let first = index;
            let mut end = tensors[index].source_end;
            index += 1;
            while index < tensors.len() && !state.completed.contains(&tensors[index].name) {
                let candidate = tensors[index].source_end;
                if candidate - tensors[first].source_begin > options.staging_bytes
                    || tensors[index].source_begin.saturating_sub(end) > COALESCE_GAP_BYTES
                {
                    break;
                }
                end = candidate;
                index += 1;
            }
            let start = tensors[first].source_begin;
            let response = retry_range(
                transport,
                tensors[first].source_shard.as_str(),
                start,
                end,
                options.max_retries,
            )?;
            stats.payload_requests += 1;
            stats.payload_requested_bytes += end - start;
            stats.payload_returned_bytes += response.body.len() as u64;
            let expected = end - start;
            let tensor_bytes = tensors[first..index]
                .iter()
                .try_fold(0u64, |total, tensor| {
                    total.checked_add(tensor.source_length)
                })
                .ok_or_else(|| {
                    ImportError::Invalid("window payload accounting overflows u64".into())
                })?;
            stats.source_overfetch_bytes += expected.saturating_sub(tensor_bytes);
            if response.body.len() as u64 != expected {
                return Err(ImportError::Range("short range response".into()));
            }
            let response_end = checked_add(start, expected, "response end")?;
            if response.status != 206
                || !response
                    .content_range
                    .is_some_and(|value| value.0 == start && value.1 == response_end)
            {
                return Err(ImportError::Range("range response framing mismatch".into()));
            }
            for tensor in &tensors[first..index] {
                let local_start = usize::try_from(tensor.source_begin - start)
                    .map_err(|_| ImportError::Invalid("range local offset exceeds host".into()))?;
                let local_end =
                    local_start
                        .checked_add(usize::try_from(tensor.source_length).map_err(|_| {
                            ImportError::Invalid("tensor length exceeds host".into())
                        })?)
                        .ok_or_else(|| ImportError::Invalid("range scatter overflow".into()))?;
                let bytes = response.body.get(local_start..local_end).ok_or_else(|| {
                    ImportError::Range("scatter range is outside response".into())
                })?;
                write_at(&file, tensor.destination_offset, bytes)?;
                stats.destination_written_bytes += tensor.destination_length;
                stats.resume_downloaded_bytes += tensor.source_length;
            }
            file.sync_data()
                .map_err(|e| ImportError::Io(e.to_string()))?;
            for tensor in &tensors[first..index] {
                state.completed.insert(tensor.name.clone());
            }
            persist_state(&state_file, &state)?;
        }
    }
    if state.completed.len() != plan.tensors.len() {
        return Err(ImportError::Invalid(
            "conversion state is incomplete".into(),
        ));
    }
    validate_final(&partial, plan)?;
    std::fs::rename(&partial, output).map_err(|e| ImportError::Io(e.to_string()))?;
    let _ = std::fs::remove_file(&state_file);
    Ok(stats)
}

#[derive(Clone)]
struct ParallelWindow {
    shard: String,
    start: u64,
    end: u64,
    tensors: Vec<ParallelTensor>,
}

#[derive(Clone)]
struct ParallelTensor {
    name: String,
    source_begin: u64,
    source_length: u64,
    destination_offset: u64,
    destination_length: u64,
}

fn add_stats(total: &mut ImportStats, part: &ImportStats) {
    total.planning_requests += part.planning_requests;
    total.planning_bytes += part.planning_bytes;
    total.payload_requests += part.payload_requests;
    total.payload_requested_bytes += part.payload_requested_bytes;
    total.payload_returned_bytes += part.payload_returned_bytes;
    total.destination_written_bytes += part.destination_written_bytes;
    total.source_overfetch_bytes += part.source_overfetch_bytes;
    total.resume_skipped_bytes += part.resume_skipped_bytes;
    total.resume_downloaded_bytes += part.resume_downloaded_bytes;
}

fn parallel_window<T: RepositoryTransport>(
    transport: &mut T,
    file: &File,
    window: ParallelWindow,
    max_retries: u32,
) -> Result<(ImportStats, Vec<String>), ImportError> {
    let response = retry_range(
        transport,
        &window.shard,
        window.start,
        window.end,
        max_retries,
    )?;
    let expected = window.end - window.start;
    if response.body.len() as u64 != expected {
        return Err(ImportError::Range("short range response".into()));
    }
    let response_end = checked_add(window.start, expected, "response end")?;
    if response.status != 206
        || !response
            .content_range
            .is_some_and(|value| value.0 == window.start && value.1 == response_end)
    {
        return Err(ImportError::Range("range response framing mismatch".into()));
    }
    let tensor_bytes = window
        .tensors
        .iter()
        .try_fold(0u64, |total, tensor| {
            total.checked_add(tensor.source_length)
        })
        .ok_or_else(|| ImportError::Invalid("window payload accounting overflows u64".into()))?;
    let mut stats = ImportStats {
        payload_requests: 1,
        payload_requested_bytes: expected,
        payload_returned_bytes: expected,
        source_overfetch_bytes: expected.saturating_sub(tensor_bytes),
        ..ImportStats::default()
    };
    let mut completed = Vec::with_capacity(window.tensors.len());
    for tensor in &window.tensors {
        let local_start = usize::try_from(tensor.source_begin - window.start)
            .map_err(|_| ImportError::Invalid("range local offset exceeds host".into()))?;
        let local_end = local_start
            .checked_add(
                usize::try_from(tensor.source_length)
                    .map_err(|_| ImportError::Invalid("tensor length exceeds host".into()))?,
            )
            .ok_or_else(|| ImportError::Invalid("range scatter overflow".into()))?;
        let bytes = response
            .body
            .get(local_start..local_end)
            .ok_or_else(|| ImportError::Range("scatter range is outside response".into()))?;
        write_at(file, tensor.destination_offset, bytes)?;
        stats.destination_written_bytes += tensor.destination_length;
        stats.resume_downloaded_bytes += tensor.source_length;
        completed.push(tensor.name.clone());
    }
    file.sync_data()
        .map_err(|e| ImportError::Io(e.to_string()))?;
    Ok((stats, completed))
}

pub fn execute_plan_parallel<T>(
    transport: &T,
    plan: &PlanReport,
    output: &Path,
    options: &ImportOptions,
) -> Result<ImportStats, ImportError>
where
    T: RepositoryTransport + Clone + Send + 'static,
{
    if options.parallel_requests < 2 {
        return Err(ImportError::Invalid(
            "parallel request count must be at least two".into(),
        ));
    }
    if !plan.can_execute || plan.largest_tensor > options.staging_bytes {
        return Err(ImportError::Invalid(
            "plan is not executable under staging limit".into(),
        ));
    }
    if output.exists() {
        return Err(ImportError::IdentityMismatch(
            "final output already exists".into(),
        ));
    }
    let partial = partial_path(output);
    let state_file = state_path(output);
    let mut state = if state_file.exists() {
        let bytes = std::fs::read(&state_file).map_err(|e| ImportError::Io(e.to_string()))?;
        serde_json::from_slice::<ConversionState>(&bytes)
            .map_err(|e| ImportError::IdentityMismatch(e.to_string()))?
    } else {
        ConversionState {
            version: 1,
            plan_digest: plan.plan_digest.clone(),
            source: plan.source.clone(),
            destination_size: plan.final_vbuf_bytes,
            completed: BTreeSet::new(),
        }
    };
    if state.plan_digest != plan.plan_digest
        || state.source != plan.source
        || state.destination_size != plan.final_vbuf_bytes
    {
        return Err(ImportError::IdentityMismatch(
            "plan, source revision, or destination layout differs".into(),
        ));
    }
    if !partial.exists() {
        let controls = plan
            .controls
            .iter()
            .map(|control| {
                (
                    control.block_start,
                    control.payload_offset,
                    control.bytes.clone(),
                    match control.semantic {
                        0 => V06Semantic::Unsigned,
                        1 => V06Semantic::Float,
                        2 => V06Semantic::Signed,
                        _ => V06Semantic::Opaque,
                    },
                    match control.physical {
                        0 => V06Physical::Scalar,
                        _ => V06Physical::Array,
                    },
                    control.bit_width,
                    control.count,
                    control.key_id,
                )
            })
            .chain(plan.tensors.iter().map(|tensor| {
                let (semantic, width, count) = match tensor.representation {
                    Some(TensorRepresentationId::CanonicalPrimitive) => (
                        V06Semantic::Float,
                        32,
                        logical_elements(&tensor.shape).unwrap_or(0),
                    ),
                    _ => (V06Semantic::Opaque, 8, tensor.source_length),
                };
                (
                    tensor.destination_block_start,
                    tensor.destination_offset,
                    Vec::new(),
                    semantic,
                    V06Physical::Array,
                    width,
                    count,
                    0x0200,
                )
            }))
            .collect::<Vec<_>>();
        initialize_partial(&partial, plan, &controls)?;
    } else if !state_file.exists() {
        return Err(ImportError::IdentityMismatch(
            "partial artifact has no conversion state".into(),
        ));
    }
    if !state_file.exists() {
        persist_state(&state_file, &state)?;
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&partial)
        .map_err(|e| ImportError::Io(e.to_string()))?;
    let mut total = ImportStats::default();
    for tensor in &plan.tensors {
        if state.completed.contains(&tensor.name) {
            total.resume_skipped_bytes += tensor.source_length;
        }
    }
    let mut by_shard: BTreeMap<&str, Vec<&TensorPlan>> = BTreeMap::new();
    for tensor in &plan.tensors {
        if !state.completed.contains(&tensor.name) {
            by_shard
                .entry(&tensor.source_shard)
                .or_default()
                .push(tensor);
        }
    }
    let mut windows = Vec::new();
    for tensors in by_shard.values_mut() {
        tensors.sort_by_key(|tensor| tensor.source_begin);
        let mut index = 0;
        while index < tensors.len() {
            let first = index;
            let mut end = tensors[index].source_end;
            index += 1;
            while index < tensors.len()
                && tensors[index].source_end - tensors[first].source_begin <= options.staging_bytes
                && tensors[index].source_begin.saturating_sub(end) <= COALESCE_GAP_BYTES
            {
                end = tensors[index].source_end;
                index += 1;
            }
            windows.push(ParallelWindow {
                shard: tensors[first].source_shard.clone(),
                start: tensors[first].source_begin,
                end,
                tensors: tensors[first..index]
                    .iter()
                    .map(|tensor| ParallelTensor {
                        name: tensor.name.clone(),
                        source_begin: tensor.source_begin,
                        source_length: tensor.source_length,
                        destination_offset: tensor.destination_offset,
                        destination_length: tensor.destination_length,
                    })
                    .collect(),
            });
        }
    }
    let mut index = 0;
    while index < windows.len() {
        let end = (index + options.parallel_requests).min(windows.len());
        let batch = &windows[index..end];
        let results = std::thread::scope(|scope| {
            let handles = batch
                .iter()
                .cloned()
                .map(|window| {
                    let worker_transport = transport.clone();
                    let worker_file = file
                        .try_clone()
                        .map_err(|e| ImportError::Io(e.to_string()))?;
                    Ok(scope.spawn(move || {
                        let mut worker_transport = worker_transport;
                        parallel_window(
                            &mut worker_transport,
                            &worker_file,
                            window,
                            options.max_retries,
                        )
                    }))
                })
                .collect::<Result<Vec<_>, ImportError>>()?;
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .map_err(|_| ImportError::Io("parallel importer worker panicked".into()))?
                })
                .collect::<Result<Vec<_>, ImportError>>()
        })?;
        for (stats, completed) in results {
            add_stats(&mut total, &stats);
            state.completed.extend(completed);
            persist_state(&state_file, &state)?;
        }
        index = end;
    }
    if state.completed.len() != plan.tensors.len() {
        return Err(ImportError::Invalid(
            "conversion state is incomplete".into(),
        ));
    }
    validate_final(&partial, plan)?;
    std::fs::rename(&partial, output).map_err(|e| ImportError::Io(e.to_string()))?;
    let _ = std::fs::remove_file(&state_file);
    Ok(total)
}

fn retry_range<T: RepositoryTransport>(
    transport: &mut T,
    shard: &str,
    start: u64,
    end: u64,
    max_retries: u32,
) -> Result<RangeResponse, ImportError> {
    let mut last = None;
    for attempt in 0..=max_retries {
        match transport.range(shard, start, end) {
            Ok(response) if response.status == 206 && response.body.len() as u64 == end - start => {
                return Ok(response);
            }
            Ok(response) => {
                last = Some(ImportError::Range(format!(
                    "invalid response status={} bytes={}",
                    response.status,
                    response.body.len()
                )))
            }
            Err(error) => last = Some(ImportError::Transport(error)),
        }
        if attempt < max_retries {
            std::thread::sleep(Duration::from_millis(10 * (u64::from(attempt) + 1)));
        }
    }
    Err(last.unwrap_or_else(|| ImportError::Transport("range request failed".into())))
}

fn validate_final(path: &Path, plan: &PlanReport) -> Result<(), ImportError> {
    let file = File::open(path).map_err(|e| ImportError::Io(e.to_string()))?;
    let bytes = unsafe { memmap2::Mmap::map(&file).map_err(|e| ImportError::Io(e.to_string()))? };
    let validated = parse_v06(&bytes).map_err(|e| ImportError::Invalid(e.to_string()))?;
    if validated.bytes().len() as u64 != plan.final_vbuf_bytes {
        return Err(ImportError::Invalid(
            "final vBuf size differs from plan".into(),
        ));
    }
    let bootstrap =
        Bootstrap::discover(&validated).map_err(|e| ImportError::Invalid(e.to_string()))?;
    let directory = TensorDirectory::parse(&validated, &bootstrap)
        .map_err(|e| ImportError::Invalid(e.to_string()))?;
    if directory.tensors().len() != plan.tensors.len() {
        return Err(ImportError::Invalid(
            "final tensor directory count differs from plan".into(),
        ));
    }
    let quantization = crate::F8QuantizationDirectory::parse(&validated, &bootstrap)
        .map_err(|e| ImportError::Invalid(e.to_string()))?
        .ok_or_else(|| {
            ImportError::Invalid("final artifact has no quantization metadata".into())
        })?;
    quantization
        .validate_against(&directory)
        .map_err(|e| ImportError::Invalid(e.to_string()))?;
    if quantization.entries() != plan.quantization.as_slice() {
        return Err(ImportError::Invalid(
            "final quantization metadata differs from plan".into(),
        ));
    }
    Ok(())
}

fn parse_content_range(value: &str) -> Option<(u64, u64, Option<u64>)> {
    let value = value.strip_prefix("bytes ")?;
    let (range, total) = value.split_once('/')?;
    let (start, end) = range.split_once('-')?;
    Some((
        start.parse().ok()?,
        end.parse::<u64>().ok()?.checked_add(1)?,
        if total == "*" {
            None
        } else {
            Some(total.parse().ok()?)
        },
    ))
}

/// Minimal Hugging Face transport. Tokens are read only from `HF_TOKEN` and
/// are never included in plan/state files or diagnostics.
#[derive(Clone)]
pub struct HfTransport {
    agent: ureq::Agent,
    base_url: String,
    repository: String,
    revision: String,
    token: Option<String>,
}

impl HfTransport {
    pub fn new(repository: impl Into<String>, revision: impl Into<String>) -> Self {
        let config = ureq::Agent::config_builder()
            .max_redirects(0)
            .max_redirects_will_error(false)
            .http_status_as_error(false)
            .accept_encoding("")
            .build();
        Self {
            agent: ureq::Agent::new_with_config(config),
            base_url: "https://huggingface.co".into(),
            repository: repository.into(),
            revision: revision.into(),
            token: std::env::var("HF_TOKEN").ok(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
    fn request(&self, name: &str, range: Option<(u64, u64)>) -> Result<RangeResponse, String> {
        let mut url = format!(
            "{}/{}/resolve/{}/{}",
            self.base_url, self.repository, self.revision, name
        );
        for _ in 0..=5 {
            let mut request = self.agent.get(&url);
            if url.starts_with(&self.base_url) {
                if let Some(token) = &self.token {
                    request = request.header("Authorization", format!("Bearer {token}"));
                }
            }
            if let Some((start, end)) = range {
                request = request.header(
                    "Range",
                    format!("bytes={start}-{}", end.checked_sub(1).ok_or("empty range")?),
                );
            }
            let response = request.call().map_err(|e| format!("{url}: {e}"))?;
            let status = response.status().as_u16();
            if (300..400).contains(&status) {
                let location = response
                    .headers()
                    .get("location")
                    .and_then(|v| v.to_str().ok())
                    .ok_or("redirect has no Location")?;
                url = if location.starts_with("http://") || location.starts_with("https://") {
                    location.to_string()
                } else {
                    format!("{}{}", self.base_url, location)
                };
                continue;
            }
            let content_range = response
                .headers()
                .get("content-range")
                .and_then(|v| v.to_str().ok())
                .and_then(parse_content_range);
            if range.is_some() && status != 206 {
                return Ok(RangeResponse {
                    status,
                    content_range,
                    body: Vec::new(),
                });
            }
            let limit = range
                .map(|(start, end)| end.saturating_sub(start))
                .map(|length| length.saturating_add(1))
                .unwrap_or(MAX_HEADER_BYTES);
            let body = response
                .into_body()
                .into_with_config()
                .limit(limit)
                .read_to_vec()
                .map_err(|e| format!("{url}: {e}"))?;
            return Ok(RangeResponse {
                status,
                content_range,
                body,
            });
        }
        Err("too many HTTP redirects".into())
    }

    fn resolve_revision(&self) -> Result<String, String> {
        if self.revision.len() == 40 && self.revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Ok(self.revision.clone());
        }
        let url = format!(
            "https://huggingface.co/api/models/{}/revision/{}",
            self.repository, self.revision
        );
        let mut request = self.agent.get(&url);
        if let Some(token) = &self.token {
            request = request.header("Authorization", format!("Bearer {token}"));
        }
        let response = request.call().map_err(|e| e.to_string())?;
        let body = response
            .into_body()
            .read_to_vec()
            .map_err(|e| e.to_string())?;
        let value: Value = serde_json::from_slice(&body).map_err(|e| e.to_string())?;
        value
            .get("sha")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| "Hugging Face revision response has no sha".into())
    }
}

impl RepositoryTransport for HfTransport {
    fn metadata(&mut self, name: &str) -> Result<Option<Vec<u8>>, String> {
        match self.request(name, None) {
            Ok(response) if response.status == 200 => Ok(Some(response.body)),
            Ok(_) => Ok(None),
            Err(error) if error.contains("http status: 404") => Ok(None),
            Err(error) => Err(error),
        }
    }
    fn range(&mut self, shard: &str, start: u64, end: u64) -> Result<RangeResponse, String> {
        self.request(shard, Some((start, end)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(dtype: &str, shape: &[u64], begin: u64, end: u64) -> Vec<u8> {
        serde_json::json!({"x":{"dtype":dtype,"shape":shape,"data_offsets":[begin,end]}})
            .to_string()
            .into_bytes()
    }

    #[test]
    fn safetensors_offsets_include_header_and_length_prefix() {
        let json = header("F32", &[2], 0, 8);
        let first = (json.len() as u64).to_le_bytes();
        let tensors =
            parse_safetensors_header("a", 8 + json.len() as u64 + 8, &first, &json).unwrap();
        assert_eq!(tensors[0].begin, 8 + json.len() as u64);
        assert_eq!(tensors[0].end, 8 + json.len() as u64 + 8);
    }

    #[test]
    fn malformed_overlap_and_shape_are_rejected() {
        let json = serde_json::json!({
            "a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},
            "b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}
        })
        .to_string()
        .into_bytes();
        let first = (json.len() as u64).to_le_bytes();
        assert!(parse_safetensors_header("a", 8 + json.len() as u64 + 12, &first, &json).is_err());
    }

    #[test]
    fn space_gate_and_u64_alignment_are_checked() {
        assert!(space_gate(u64::MAX, u64::MAX, 1).is_err());
        let value = (1u64 << 62) + 1;
        let aligned = (value + 15) & !15;
        assert_eq!(aligned, (1u64 << 62) + 16);
    }
}
