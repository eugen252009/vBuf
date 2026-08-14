use crate::bootstrap::Bootstrap;
use crate::error::{MlError, MlErrorCode};
use crate::region_roles::RegionRole;
use vbuf_core::v06::{V06Physical, V06Semantic, ValidatedV06};
use vbuf_layout::CheckedRange;

pub const METADATA_MAGIC: [u8; 8] = *b"VBMLMD\0\0";
pub const METADATA_VERSION: u16 = 1;
pub const MAX_METADATA_BYTES: usize = 4096;
pub const MAX_METADATA_ENTRIES: usize = 256;
const HEADER_BYTES: usize = 16;
const ENTRY_BYTES: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub enum ModelMetadataKey {
    Architecture = 1,
    ContextLength = 2,
    EmbeddingLength = 3,
    LayerCount = 4,
    HeadCount = 5,
    FeedForwardLength = 6,
    NormalizationEpsilon = 7,
    RopeTheta = 8,
    KVHeadCount = 9,
    KeyHeadDimension = 10,
    ValueHeadDimension = 11,
}

impl ModelMetadataKey {
    pub const fn from_id(id: u16) -> Option<Self> {
        match id {
            1 => Some(Self::Architecture),
            2 => Some(Self::ContextLength),
            3 => Some(Self::EmbeddingLength),
            4 => Some(Self::LayerCount),
            5 => Some(Self::HeadCount),
            6 => Some(Self::FeedForwardLength),
            7 => Some(Self::NormalizationEpsilon),
            8 => Some(Self::RopeTheta),
            9 => Some(Self::KVHeadCount),
            10 => Some(Self::KeyHeadDimension),
            11 => Some(Self::ValueHeadDimension),
            _ => None,
        }
    }

    pub const fn is_required(self) -> bool {
        matches!(self, Self::Architecture | Self::ContextLength | Self::EmbeddingLength | Self::LayerCount | Self::HeadCount)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetadataEntry {
    pub key_id: u16,
    pub required: bool,
    pub generic_key_id: u16,
    pub occurrence: u16,
}

impl MetadataEntry {
    pub const fn new(key_id: u16, required: bool, generic_key_id: u16, occurrence: u16) -> Self {
        Self { key_id, required, generic_key_id, occurrence }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MetadataValue {
    Text(String),
    Unsigned(u64),
    Float(f64),
}

#[derive(Debug)]
pub struct MetadataField<'a> {
    pub key: ModelMetadataKey,
    pub value: MetadataValue,
    pub generic_key_id: u16,
    pub occurrence: u16,
    pub block_index: usize,
    pub range: CheckedRange<'a>,
}

#[derive(Debug)]
pub struct ModelMetadata<'a> {
    fields: Vec<MetadataField<'a>>,
}

impl<'a> ModelMetadata<'a> {
    pub fn parse(validated: &ValidatedV06<'a>, bootstrap: &Bootstrap<'a>) -> Result<Self, MlError> {
        let region = bootstrap.region(RegionRole::ModelMetadata).ok_or_else(|| MlError::new(MlErrorCode::MissingModelMetadata, "bootstrap has no model metadata role"))?;
        let region_block = validated.blocks().get(region.block_index).ok_or_else(|| MlError::new(MlErrorCode::MissingModelMetadata, "model metadata block index is absent"))?;
        if region_block.semantic != V06Semantic::Opaque || region_block.physical != V06Physical::Array || region_block.bit_width != 8 || region_block.continuation {
            return Err(MlError::new(MlErrorCode::MalformedModelMetadata, "model metadata must be a non-continuing opaque byte array"));
        }
        let entries = parse_payload(region.range.bytes())?;
        let mut fields = Vec::with_capacity(entries.len());
        let mut previous = 0u16;
        let mut seen = [false; 12];
        for (position, entry) in entries.into_iter().enumerate() {
            if position > 0 && entry.key_id <= previous {
                return Err(MlError::new(if entry.key_id == previous { MlErrorCode::DuplicateMetadataKey } else { MlErrorCode::MalformedModelMetadata }, "model metadata keys must be strictly sorted"));
            }
            previous = entry.key_id;
            let Some(key) = ModelMetadataKey::from_id(entry.key_id) else {
                if entry.required { return Err(MlError::new(MlErrorCode::UnsupportedMetadataKey, "unknown required model metadata key")); }
                continue;
            };
            if entry.required != key.is_required() {
                return Err(MlError::new(MlErrorCode::MalformedModelMetadata, "metadata requiredness does not match profile contract"));
            }
            let slot = entry.key_id as usize;
            if seen[slot] { return Err(MlError::new(MlErrorCode::DuplicateMetadataKey, "model metadata key occurs more than once")); }
            seen[slot] = true;
            let (block_index, range) = resolve_value(validated, entry.generic_key_id, entry.occurrence)?;
            let value = decode_value(key, &validated.blocks()[block_index], range.bytes())?;
            fields.push(MetadataField { key, value, generic_key_id: entry.generic_key_id, occurrence: entry.occurrence, block_index, range });
        }
        for required in [ModelMetadataKey::Architecture, ModelMetadataKey::ContextLength, ModelMetadataKey::EmbeddingLength, ModelMetadataKey::LayerCount, ModelMetadataKey::HeadCount] {
            if !fields.iter().any(|field| field.key == required) { return Err(MlError::new(MlErrorCode::MissingRequiredMetadata, "required model metadata key is absent")); }
        }
        Ok(Self { fields })
    }

    pub fn fields(&self) -> &[MetadataField<'a>] { &self.fields }
    pub fn get(&self, key: ModelMetadataKey) -> Option<&MetadataField<'a>> { self.fields.iter().find(|field| field.key == key) }
    pub fn architecture(&self) -> Option<&str> { match &self.get(ModelMetadataKey::Architecture)?.value { MetadataValue::Text(value) => Some(value), _ => None } }
    pub fn unsigned(&self, key: ModelMetadataKey) -> Option<u64> { match self.get(key)?.value { MetadataValue::Unsigned(value) => Some(value), _ => None } }
    pub fn float(&self, key: ModelMetadataKey) -> Option<f64> { match self.get(key)?.value { MetadataValue::Float(value) => Some(value), _ => None } }
    pub fn kv_head_count(&self) -> Option<u64> { self.unsigned(ModelMetadataKey::KVHeadCount) }
    pub fn key_head_dimension(&self) -> Option<u64> { self.unsigned(ModelMetadataKey::KeyHeadDimension) }
    pub fn value_head_dimension(&self) -> Option<u64> { self.unsigned(ModelMetadataKey::ValueHeadDimension) }
}

pub fn encode_payload(entries: &[MetadataEntry]) -> Result<Vec<u8>, MlError> {
    if entries.len() > MAX_METADATA_ENTRIES { return Err(MlError::new(MlErrorCode::MalformedModelMetadata, "metadata entry count exceeds profile maximum")); }
    let mut ordered: Vec<&MetadataEntry> = entries.iter().collect();
    ordered.sort_by_key(|entry| entry.key_id);
    for pair in ordered.windows(2) { if pair[0].key_id == pair[1].key_id { return Err(MlError::new(MlErrorCode::DuplicateMetadataKey, "model metadata key occurs more than once")); } }
    let mut output = Vec::with_capacity(HEADER_BYTES + ordered.len() * ENTRY_BYTES);
    output.extend_from_slice(&METADATA_MAGIC);
    output.extend_from_slice(&METADATA_VERSION.to_le_bytes());
    output.extend_from_slice(&0u16.to_le_bytes());
    output.extend_from_slice(&(ordered.len() as u16).to_le_bytes());
    output.extend_from_slice(&0u16.to_le_bytes());
    for entry in ordered {
        output.extend_from_slice(&entry.key_id.to_le_bytes());
        output.extend_from_slice(&(if entry.required { 1u16 } else { 0 }).to_le_bytes());
        output.extend_from_slice(&entry.generic_key_id.to_le_bytes());
        output.extend_from_slice(&entry.occurrence.to_le_bytes());
    }
    Ok(output)
}

fn parse_payload(bytes: &[u8]) -> Result<Vec<MetadataEntry>, MlError> {
    if bytes.len() < HEADER_BYTES || bytes.len() > MAX_METADATA_BYTES || bytes[..8] != METADATA_MAGIC { return Err(MlError::new(MlErrorCode::MalformedModelMetadata, "model metadata header is invalid")); }
    if u16::from_le_bytes(bytes[8..10].try_into().unwrap()) != METADATA_VERSION { return Err(MlError::new(MlErrorCode::MalformedModelMetadata, "unsupported model metadata version")); }
    if u16::from_le_bytes(bytes[10..12].try_into().unwrap()) != 0 || u16::from_le_bytes(bytes[14..16].try_into().unwrap()) != 0 { return Err(MlError::new(MlErrorCode::MalformedModelMetadata, "model metadata reserved fields are non-zero")); }
    let count = usize::from(u16::from_le_bytes(bytes[12..14].try_into().unwrap()));
    let expected = HEADER_BYTES.checked_add(count.checked_mul(ENTRY_BYTES).ok_or_else(|| MlError::new(MlErrorCode::MalformedModelMetadata, "metadata entry count overflows"))?).ok_or_else(|| MlError::new(MlErrorCode::MalformedModelMetadata, "metadata length overflows"))?;
    if count > MAX_METADATA_ENTRIES || expected != bytes.len() { return Err(MlError::new(MlErrorCode::MalformedModelMetadata, "metadata length does not match entry count")); }
    let mut entries = Vec::with_capacity(count);
    for index in 0..count {
        let offset = HEADER_BYTES + index * ENTRY_BYTES;
        let flags = u16::from_le_bytes(bytes[offset + 2..offset + 4].try_into().unwrap());
        if flags & !1 != 0 { return Err(MlError::new(MlErrorCode::MalformedModelMetadata, "metadata flags are invalid")); }
        entries.push(MetadataEntry::new(u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap()), flags != 0, u16::from_le_bytes(bytes[offset + 4..offset + 6].try_into().unwrap()), u16::from_le_bytes(bytes[offset + 6..offset + 8].try_into().unwrap())));
    }
    Ok(entries)
}

fn resolve_value<'a>(validated: &ValidatedV06<'a>, key_id: u16, occurrence: u16) -> Result<(usize, CheckedRange<'a>), MlError> {
    let index = validated.blocks().iter().enumerate().filter(|(_, block)| block.key_id == key_id).nth(occurrence as usize).map(|(index, _)| index).ok_or_else(|| MlError::new(MlErrorCode::MissingRequiredMetadata, "metadata canonical reference is absent"))?;
    if validated.blocks()[index].continuation { return Err(MlError::new(MlErrorCode::MetadataTypeMismatch, "metadata values may not be continuing in profile 0.1")); }
    Ok((index, validated.payload_range(index)?))
}

fn decode_value(key: ModelMetadataKey, block: &vbuf_core::v06::V06Block, bytes: &[u8]) -> Result<MetadataValue, MlError> {
    match key {
        ModelMetadataKey::Architecture => {
            if block.semantic != V06Semantic::Opaque || block.physical != V06Physical::Array || block.bit_width != 8 || bytes.is_empty() || bytes.contains(&0) { return Err(MlError::new(MlErrorCode::MetadataTypeMismatch, "architecture must be non-empty UTF-8 bytes")); }
            let text = String::from_utf8(bytes.to_vec()).map_err(|_| MlError::new(MlErrorCode::MetadataTypeMismatch, "architecture is not UTF-8"))?;
            Ok(MetadataValue::Text(text))
        }
        ModelMetadataKey::ContextLength | ModelMetadataKey::EmbeddingLength | ModelMetadataKey::LayerCount | ModelMetadataKey::HeadCount | ModelMetadataKey::FeedForwardLength | ModelMetadataKey::KVHeadCount | ModelMetadataKey::KeyHeadDimension | ModelMetadataKey::ValueHeadDimension => {
            let value = decode_unsigned(block, bytes)?;
            if value == 0 { return Err(MlError::new(MlErrorCode::MetadataValueOutOfRange, "model metadata integer must be non-zero")); }
            Ok(MetadataValue::Unsigned(value))
        }
        ModelMetadataKey::NormalizationEpsilon | ModelMetadataKey::RopeTheta => {
            let value = decode_float(block, bytes)?;
            if !value.is_finite() || value <= 0.0 { return Err(MlError::new(MlErrorCode::MetadataValueOutOfRange, "model metadata float must be finite and positive")); }
            Ok(MetadataValue::Float(value))
        }
    }
}

fn decode_unsigned(block: &vbuf_core::v06::V06Block, bytes: &[u8]) -> Result<u64, MlError> {
    if block.semantic != V06Semantic::Unsigned || block.physical != V06Physical::Scalar || block.count != 1 || !matches!(block.bit_width, 8 | 16 | 32 | 64) { return Err(MlError::new(MlErrorCode::MetadataTypeMismatch, "metadata value must be an unsigned scalar")); }
    Ok(match block.bit_width { 8 => u64::from(bytes[0]), 16 => u64::from(u16::from_le_bytes(bytes.try_into().unwrap())), 32 => u64::from(u32::from_le_bytes(bytes.try_into().unwrap())), 64 => u64::from_le_bytes(bytes.try_into().unwrap()), _ => unreachable!() })
}

fn decode_float(block: &vbuf_core::v06::V06Block, bytes: &[u8]) -> Result<f64, MlError> {
    if block.semantic != V06Semantic::Float || block.physical != V06Physical::Scalar || block.count != 1 { return Err(MlError::new(MlErrorCode::MetadataTypeMismatch, "metadata value must be a floating scalar")); }
    Ok(match block.bit_width { 32 => f32::from_le_bytes(bytes.try_into().unwrap()) as f64, 64 => f64::from_le_bytes(bytes.try_into().unwrap()), _ => return Err(MlError::new(MlErrorCode::MetadataTypeMismatch, "metadata float width is unsupported")) })
}
