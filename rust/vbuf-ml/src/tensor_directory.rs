use crate::bootstrap::Bootstrap;
use crate::error::{MlError, MlErrorCode};
use crate::region_roles::RegionRole;
use crate::representations::{representation_from_id, validate_external_tensor_representation, validate_tensor_representation, TensorRepresentation};
use crate::source::{SourceId, SourceRegistry, TensorRef};
use vbuf_core::v06::{V06Physical, V06Semantic, ValidatedV06};
use vbuf_layout::CheckedRange;

pub const DIRECTORY_MAGIC: [u8; 8] = *b"VBTDIR\0\0";
pub const DIRECTORY_VERSION: u16 = 1;
pub const MAX_DIRECTORY_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_TENSORS: u32 = 1_000_000;
pub const MAX_NAME_BYTES: usize = 4096;
pub const MAX_RANK: usize = 16;
const HEADER_BYTES: usize = 20;
const ENTRY_FIXED_BYTES: usize = 10;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorEntry {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub representation: TensorRepresentation,
    pub key_id: u16,
    pub occurrence: u16,
}

impl TensorEntry {
    pub fn new(name: impl Into<String>, dimensions: Vec<u64>, key_id: u16, occurrence: u16) -> Self {
        Self { name: name.into(), dimensions, representation: TensorRepresentation::CanonicalPrimitive, key_id, occurrence }
    }
}

#[derive(Debug)]
pub struct TensorDescriptor<'a> {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub representation: TensorRepresentation,
    pub key_id: u16,
    pub occurrence: u16,
    pub block_index: usize,
    pub range: Option<CheckedRange<'a>>,
    pub payload: TensorRef,
}

#[derive(Debug)]
pub struct TensorDirectory<'a> {
    tensors: Vec<TensorDescriptor<'a>>,
}

impl<'a> TensorDirectory<'a> {
    pub fn parse(validated: &ValidatedV06<'a>, bootstrap: &Bootstrap<'a>) -> Result<Self, MlError> {
        let registry = SourceRegistry::new(vec![SourceRegistry::self_descriptor(validated.bytes().len() as u64)]).map_err(|_| MlError::new(MlErrorCode::MalformedTensorDirectory, "self source registry is invalid"))?;
        Self::parse_with_sources(validated, bootstrap, &registry, &[])
    }

    pub fn parse_with_sources(validated: &ValidatedV06<'a>, bootstrap: &Bootstrap<'a>, sources: &SourceRegistry, external: &[(u16, u16, TensorRef)]) -> Result<Self, MlError> {
        let region = bootstrap.region(RegionRole::TensorDirectory).ok_or_else(|| MlError::new(MlErrorCode::TensorDirectoryMissing, "bootstrap has no tensor directory role"))?;
        let block = validated.blocks().get(region.block_index).ok_or_else(|| MlError::new(MlErrorCode::TensorReferenceMissing, "tensor directory block index is absent"))?;
        if block.semantic != V06Semantic::Opaque || block.physical != V06Physical::Array || block.bit_width != 8 || block.continuation {
            return Err(MlError::new(MlErrorCode::RegionTypeMismatch, "tensor directory must be a non-continuing opaque byte array"));
        }
        let raw_entries = parse_payload(region.range.bytes())?;
        let mut tensors = Vec::with_capacity(raw_entries.len());
        let mut previous_name: Option<Vec<u8>> = None;
        for raw in raw_entries {
            if let Some(previous) = previous_name.as_ref()
                && previous.as_slice() >= raw.name.as_slice()
            {
                return Err(MlError::new(if previous.as_slice() == raw.name.as_slice() { MlErrorCode::DuplicateTensorName } else { MlErrorCode::InvalidDirectoryOrder }, "tensor names must be unique and strictly sorted"));
            }
            previous_name = Some(raw.name.clone());
            let name = String::from_utf8(raw.name).map_err(|_| MlError::new(MlErrorCode::InvalidTensorName, "tensor name is not valid UTF-8"))?;
            let target_index = validated.blocks().iter().enumerate().filter(|(_, candidate)| candidate.key_id == raw.key_id).nth(raw.occurrence as usize).map(|(index, _)| index).ok_or_else(|| MlError::new(MlErrorCode::TensorReferenceMissing, "tensor canonical reference is absent"))?;
            let target = &validated.blocks()[target_index];
            if target.continuation {
                return Err(MlError::new(MlErrorCode::TensorRepresentationMismatch, "continuation tensor values are not supported by profile 0.1"));
            }
            let payload = if let Some((_, _, payload)) = external.iter().find(|(key_id, occurrence, _)| *key_id == raw.key_id && *occurrence == raw.occurrence) {
                *payload
            } else {
                let self_source = sources.get(SourceId::SELF).ok_or_else(|| MlError::new(MlErrorCode::TensorReferenceMissing, "self source descriptor is absent"))?;
                TensorRef::new(self_source, target.payload_start, target.payload_len).map_err(|_| MlError::new(MlErrorCode::TensorReferenceMissing, "canonical self range is not source-bounded"))?
            };
            if payload.source_id() == SourceId::SELF && target.payload_len != 0 {
                validate_tensor_representation(raw.representation, &raw.dimensions, target)?;
            } else {
                validate_external_tensor_representation(raw.representation, &raw.dimensions, target, payload.length())?;
            }
            if payload.source_id() == SourceId::SELF && payload.length() != target.payload_len {
                return Err(MlError::new(MlErrorCode::TensorPayloadSizeMismatch, "external tensor range length does not match canonical tensor geometry"));
            }
            if sources.get(payload.source_id()).is_none() {
                return Err(MlError::new(MlErrorCode::TensorReferenceMissing, "tensor source ID is not registered"));
            }
            tensors.push(TensorDescriptor {
                name,
                dimensions: raw.dimensions,
                representation: raw.representation,
                key_id: raw.key_id,
                occurrence: raw.occurrence,
                block_index: target_index,
                range: if payload.source_id() == SourceId::SELF { Some(validated.payload_range(target_index)?) } else { None },
                payload,
            });
        }
        Ok(Self { tensors })
    }

    pub fn tensors(&self) -> &[TensorDescriptor<'a>] { &self.tensors }

    pub fn get(&self, name: &str) -> Option<&TensorDescriptor<'a>> {
        self.tensors.binary_search_by(|tensor| tensor.name.as_str().cmp(name)).ok().map(|index| &self.tensors[index])
    }
}

pub fn encode_payload(entries: &[TensorEntry]) -> Result<Vec<u8>, MlError> {
    if entries.len() > MAX_TENSORS as usize {
        return Err(MlError::new(MlErrorCode::MalformedTensorDirectory, "tensor count exceeds profile maximum"));
    }
    let mut ordered: Vec<&TensorEntry> = entries.iter().collect();
    ordered.sort_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));
    for pair in ordered.windows(2) {
        if pair[0].name == pair[1].name {
            return Err(MlError::new(MlErrorCode::DuplicateTensorName, "tensor names must be unique"));
        }
    }
    let mut output = Vec::with_capacity(HEADER_BYTES);
    output.extend_from_slice(&DIRECTORY_MAGIC);
    output.extend_from_slice(&DIRECTORY_VERSION.to_le_bytes());
    output.extend_from_slice(&0u16.to_le_bytes());
    output.extend_from_slice(&(ordered.len() as u32).to_le_bytes());
    output.extend_from_slice(&0u32.to_le_bytes());
    for entry in ordered {
        validate_entry(entry)?;
        output.extend_from_slice(&(entry.name.len() as u16).to_le_bytes());
        output.push(u8::try_from(entry.dimensions.len()).map_err(|_| MlError::new(MlErrorCode::InvalidRank, "tensor rank exceeds profile maximum"))?);
        output.push(entry.representation as u8);
        output.extend_from_slice(&entry.key_id.to_le_bytes());
        output.extend_from_slice(&entry.occurrence.to_le_bytes());
        output.extend_from_slice(&0u16.to_le_bytes());
        output.extend_from_slice(entry.name.as_bytes());
        for dimension in &entry.dimensions {
            output.extend_from_slice(&dimension.to_le_bytes());
        }
    }
    if output.len() > MAX_DIRECTORY_BYTES {
        return Err(MlError::new(MlErrorCode::MalformedTensorDirectory, "tensor directory exceeds bounded size"));
    }
    Ok(output)
}

#[derive(Debug)]
struct RawEntry {
    name: Vec<u8>,
    dimensions: Vec<u64>,
    representation: TensorRepresentation,
    key_id: u16,
    occurrence: u16,
}

fn parse_payload(bytes: &[u8]) -> Result<Vec<RawEntry>, MlError> {
    if bytes.len() < HEADER_BYTES || bytes.len() > MAX_DIRECTORY_BYTES || bytes[..8] != DIRECTORY_MAGIC {
        return Err(MlError::new(MlErrorCode::MalformedTensorDirectory, "tensor directory header is invalid"));
    }
    let version = u16::from_le_bytes(bytes[8..10].try_into().unwrap());
    if version != DIRECTORY_VERSION {
        return Err(MlError::new(MlErrorCode::UnsupportedTensorDirectoryVersion, "unsupported tensor directory version"));
    }
    if u16::from_le_bytes(bytes[10..12].try_into().unwrap()) != 0 || bytes[16..20].iter().any(|byte| *byte != 0) || u32::from_le_bytes(bytes[12..16].try_into().unwrap()) > MAX_TENSORS {
        return Err(MlError::new(MlErrorCode::MalformedTensorDirectory, "tensor directory flags or count are invalid"));
    }
    let count = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let mut cursor = HEADER_BYTES;
    let mut output = Vec::with_capacity(count);
    for _ in 0..count {
        let fixed = bytes.get(cursor..cursor + ENTRY_FIXED_BYTES).ok_or_else(|| MlError::new(MlErrorCode::MalformedTensorDirectory, "truncated tensor directory entry"))?;
        let name_len = usize::from(u16::from_le_bytes(fixed[0..2].try_into().unwrap()));
        let rank = usize::from(fixed[2]);
        if name_len == 0 || name_len > MAX_NAME_BYTES {
            return Err(MlError::new(MlErrorCode::InvalidTensorName, "tensor name length is invalid"));
        }
        if rank > MAX_RANK {
            return Err(MlError::new(MlErrorCode::InvalidRank, "tensor rank exceeds profile maximum"));
        }
        if u16::from_le_bytes(fixed[8..10].try_into().unwrap()) != 0 {
            return Err(MlError::new(MlErrorCode::MalformedTensorDirectory, "tensor entry reserved field is non-zero"));
        }
        let representation = representation_from_id(fixed[3])?;
        cursor += ENTRY_FIXED_BYTES;
        let name_end = cursor.checked_add(name_len).ok_or_else(|| MlError::new(MlErrorCode::MalformedTensorDirectory, "tensor name length overflows"))?;
        let name = bytes.get(cursor..name_end).ok_or_else(|| MlError::new(MlErrorCode::MalformedTensorDirectory, "truncated tensor name"))?.to_vec();
        if std::str::from_utf8(&name).is_err() || name.contains(&0) {
            return Err(MlError::new(MlErrorCode::InvalidTensorName, "tensor name is invalid UTF-8 or contains NUL"));
        }
        cursor = name_end;
        let dimensions_bytes = rank.checked_mul(8).ok_or_else(|| MlError::new(MlErrorCode::MalformedTensorDirectory, "dimension table size overflows"))?;
        let dimensions_end = cursor.checked_add(dimensions_bytes).ok_or_else(|| MlError::new(MlErrorCode::MalformedTensorDirectory, "dimension table end overflows"))?;
        let dimension_data = bytes.get(cursor..dimensions_end).ok_or_else(|| MlError::new(MlErrorCode::MalformedTensorDirectory, "truncated tensor dimensions"))?;
        let mut dimensions = Vec::with_capacity(rank);
        for chunk in dimension_data.chunks_exact(8) {
            let dimension = u64::from_le_bytes(chunk.try_into().unwrap());
            if dimension == 0 {
                return Err(MlError::new(MlErrorCode::InvalidDimension, "zero tensor dimensions are not supported"));
            }
            dimensions.push(dimension);
        }
        cursor = dimensions_end;
        output.push(RawEntry { name, dimensions, representation, key_id: u16::from_le_bytes(fixed[4..6].try_into().unwrap()), occurrence: u16::from_le_bytes(fixed[6..8].try_into().unwrap()) });
    }
    if cursor != bytes.len() {
        return Err(MlError::new(MlErrorCode::MalformedTensorDirectory, "tensor directory has trailing bytes"));
    }
    Ok(output)
}

fn validate_entry(entry: &TensorEntry) -> Result<(), MlError> {
    if entry.name.is_empty() || entry.name.len() > MAX_NAME_BYTES || entry.name.as_bytes().contains(&0) || std::str::from_utf8(entry.name.as_bytes()).is_err() {
        return Err(MlError::new(MlErrorCode::InvalidTensorName, "tensor name is invalid"));
    }
    if entry.dimensions.len() > MAX_RANK {
        return Err(MlError::new(MlErrorCode::InvalidRank, "tensor rank exceeds profile maximum"));
    }
    for dimension in &entry.dimensions {
        if *dimension == 0 {
            return Err(MlError::new(MlErrorCode::InvalidDimension, "zero tensor dimensions are not supported"));
        }
    }
    entry.dimensions.iter().try_fold(1u64, |product, dimension| product.checked_mul(*dimension)).ok_or_else(|| MlError::new(MlErrorCode::ShapeOverflow, "tensor shape product overflows u64"))?;
    Ok(())
}
