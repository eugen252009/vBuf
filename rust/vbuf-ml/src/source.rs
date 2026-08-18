use std::fmt;
use crate::bootstrap::Bootstrap;
use crate::error::{MlError, MlErrorCode};
use crate::region_roles::RegionRole;
use vbuf_core::v06::ValidatedV06;
use vbuf_layout::{ByteRange, RangeError};

pub const SOURCE_METADATA_MAGIC: [u8; 8] = *b"VBVSRC\0\0";
pub const SOURCE_METADATA_VERSION: u16 = 1;
const SOURCE_HEADER_BYTES: usize = 24;
const SOURCE_ENTRY_BYTES: usize = 32;
const BINDING_BYTES: usize = 32;
const MAX_SOURCE_METADATA_BYTES: usize = 16 * 1024 * 1024;

/// Stable identity for a byte source. Location is deliberately not part of it.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SourceId(u64);

impl SourceId {
    pub const SELF: Self = Self(0);
    pub const fn new(value: u64) -> Self {
        Self(value)
    }
    pub const fn value(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SourceLocator {
    SelfArtifact,
    File(String),
    Http(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceHash {
    pub algorithm: u16,
    pub value: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceDescriptor {
    pub id: SourceId,
    pub declared_size: Option<u64>,
    pub locator: SourceLocator,
    pub hashes: Vec<SourceHash>,
}

impl SourceDescriptor {
    pub fn self_artifact(declared_size: u64) -> Self {
        Self {
            id: SourceId::SELF,
            declared_size: Some(declared_size),
            locator: SourceLocator::SelfArtifact,
            hashes: Vec::new(),
        }
    }

    pub fn checked_range(
        &self,
        offset: u64,
        length: u64,
    ) -> Result<CheckedSourceRange, SourceRangeError> {
        let range = ByteRange::new(offset, length).map_err(SourceRangeError::Range)?;
        let size = self.declared_size.ok_or(SourceRangeError::UnknownSize)?;
        if range.end() > size {
            return Err(SourceRangeError::OutsideSource);
        }
        Ok(CheckedSourceRange {
            source_id: self.id,
            declared_size: size,
            range,
        })
    }

    fn validate(&self) -> Result<(), SourceRegistryError> {
        if matches!(&self.locator, SourceLocator::File(value) | SourceLocator::Http(value) if value.is_empty())
        {
            return Err(SourceRegistryError::MalformedDescriptor);
        }
        if self
            .hashes
            .iter()
            .any(|hash| hash.algorithm == 0 || hash.value.is_empty())
        {
            return Err(SourceRegistryError::MalformedDescriptor);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceRegistry {
    descriptors: Vec<SourceDescriptor>,
}

#[derive(Debug)]
pub struct PersistentSourceMetadata {
    pub registry: SourceRegistry,
    pub bindings: Vec<(u16, u16, TensorRef)>,
}

pub fn encode_profile(registry: &SourceRegistry, bindings: &[(u16, u16, TensorRef)]) -> Result<Vec<u8>, MlError> {
    let mut output = vec![0u8; SOURCE_HEADER_BYTES];
    output[..8].copy_from_slice(&SOURCE_METADATA_MAGIC);
    output[8..10].copy_from_slice(&SOURCE_METADATA_VERSION.to_le_bytes());
    output[12..16].copy_from_slice(&(registry.descriptors.len() as u32).to_le_bytes());
    output[16..20].copy_from_slice(&(bindings.len() as u32).to_le_bytes());
    for descriptor in &registry.descriptors {
        descriptor.validate().map_err(|_| MlError::new(MlErrorCode::InvalidSourceLocator, "source descriptor is malformed"))?;
        let (kind, locator) = match &descriptor.locator {
            SourceLocator::SelfArtifact => (0u8, &[][..]),
            SourceLocator::File(value) => (1u8, value.as_bytes()),
            SourceLocator::Http(value) => (2u8, value.as_bytes()),
        };
        let size = descriptor.declared_size.ok_or_else(|| MlError::new(MlErrorCode::MalformedSourceMetadata, "persistent source size is required"))?;
        let locator_len = u32::try_from(locator.len()).map_err(|_| MlError::new(MlErrorCode::MalformedSourceMetadata, "source locator is too large"))?;
        let hash_count = u16::try_from(descriptor.hashes.len()).map_err(|_| MlError::new(MlErrorCode::MalformedSourceMetadata, "source hash count is too large"))?;
        output.extend_from_slice(&descriptor.id.value().to_le_bytes());
        output.extend_from_slice(&size.to_le_bytes());
        output.push(kind);
        output.extend_from_slice(&[0; 3]);
        output.extend_from_slice(&locator_len.to_le_bytes());
        output.extend_from_slice(&hash_count.to_le_bytes());
        output.extend_from_slice(&[0; 6]);
        output.extend_from_slice(locator);
        for hash in &descriptor.hashes {
            let length = u16::try_from(hash.value.len()).map_err(|_| MlError::new(MlErrorCode::InvalidSourceHash, "source hash is too large"))?;
            output.extend_from_slice(&hash.algorithm.to_le_bytes());
            output.extend_from_slice(&length.to_le_bytes());
            output.extend_from_slice(&hash.value);
        }
    }
    let mut seen = Vec::with_capacity(bindings.len());
    for (key_id, occurrence, reference) in bindings {
        if seen.contains(&(*key_id, *occurrence)) { return Err(MlError::new(MlErrorCode::DuplicateSourceBinding, "source binding occurs more than once")); }
        seen.push((*key_id, *occurrence));
        if registry.get(reference.source_id()).is_none() { return Err(MlError::new(MlErrorCode::UnknownRequiredRole, "source binding references an unknown source")); }
        output.extend_from_slice(&key_id.to_le_bytes());
        output.extend_from_slice(&occurrence.to_le_bytes());
        output.extend_from_slice(&[0; 4]);
        output.extend_from_slice(&reference.source_id().value().to_le_bytes());
        output.extend_from_slice(&reference.offset().to_le_bytes());
        output.extend_from_slice(&reference.length().to_le_bytes());
    }
    if output.len() > MAX_SOURCE_METADATA_BYTES { return Err(MlError::new(MlErrorCode::MalformedSourceMetadata, "source metadata exceeds bounded size")); }
    Ok(output)
}

pub fn parse_profile(_validated: &ValidatedV06<'_>, bootstrap: &Bootstrap<'_>) -> Result<Option<PersistentSourceMetadata>, MlError> {
    let Some(region) = bootstrap.region(RegionRole::SourceMetadata) else { return Ok(None); };
    let bytes = region.range.bytes();
    if bytes.len() < SOURCE_HEADER_BYTES || bytes.len() > MAX_SOURCE_METADATA_BYTES || bytes[..8] != SOURCE_METADATA_MAGIC { return Err(MlError::new(MlErrorCode::MalformedSourceMetadata, "source metadata header is invalid")); }
    if u16::from_le_bytes(bytes[8..10].try_into().unwrap()) != SOURCE_METADATA_VERSION { return Err(MlError::new(MlErrorCode::UnsupportedSourceMetadataVersion, "unsupported source metadata version")); }
    if bytes[10..12].iter().any(|byte| *byte != 0) || bytes[20..24].iter().any(|byte| *byte != 0) { return Err(MlError::new(MlErrorCode::MalformedSourceMetadata, "source metadata reserved fields are non-zero")); }
    let source_count = usize::try_from(u32::from_le_bytes(bytes[12..16].try_into().unwrap())).map_err(|_| MlError::new(MlErrorCode::MalformedSourceMetadata, "source count is invalid"))?;
    let binding_count = usize::try_from(u32::from_le_bytes(bytes[16..20].try_into().unwrap())).map_err(|_| MlError::new(MlErrorCode::MalformedSourceMetadata, "binding count is invalid"))?;
    let mut cursor = SOURCE_HEADER_BYTES;
    let mut descriptors = Vec::with_capacity(source_count);
    for _ in 0..source_count {
        let header = bytes.get(cursor..cursor + SOURCE_ENTRY_BYTES).ok_or_else(|| MlError::new(MlErrorCode::MalformedSourceMetadata, "source entry is truncated"))?;
        let id = SourceId::new(u64::from_le_bytes(header[..8].try_into().unwrap()));
        let size = u64::from_le_bytes(header[8..16].try_into().unwrap());
        let kind = header[16];
        if header[17..20].iter().any(|byte| *byte != 0) || header[26..32].iter().any(|byte| *byte != 0) { return Err(MlError::new(MlErrorCode::MalformedSourceMetadata, "source entry reserved fields are non-zero")); }
        let locator_len = usize::try_from(u32::from_le_bytes(header[20..24].try_into().unwrap())).map_err(|_| MlError::new(MlErrorCode::MalformedSourceMetadata, "source locator length is invalid"))?;
        let hash_count = usize::from(u16::from_le_bytes(header[24..26].try_into().unwrap()));
        cursor += SOURCE_ENTRY_BYTES;
        let locator_bytes = bytes.get(cursor..cursor + locator_len).ok_or_else(|| MlError::new(MlErrorCode::MalformedSourceMetadata, "source locator is truncated"))?;
        cursor += locator_len;
        let locator = match kind {
            0 if locator_bytes.is_empty() => SourceLocator::SelfArtifact,
            1 => SourceLocator::File(String::from_utf8(locator_bytes.to_vec()).map_err(|_| MlError::new(MlErrorCode::InvalidSourceLocator, "file locator is not UTF-8"))?),
            2 => SourceLocator::Http(String::from_utf8(locator_bytes.to_vec()).map_err(|_| MlError::new(MlErrorCode::InvalidSourceLocator, "HTTP locator is not UTF-8"))?),
            _ => return Err(MlError::new(MlErrorCode::InvalidSourceLocator, "source locator kind is unsupported")),
        };
        let mut hashes = Vec::with_capacity(hash_count);
        for _ in 0..hash_count {
            let header = bytes.get(cursor..cursor + 4).ok_or_else(|| MlError::new(MlErrorCode::MalformedSourceMetadata, "source hash header is truncated"))?;
            let algorithm = u16::from_le_bytes(header[..2].try_into().unwrap());
            let length = usize::from(u16::from_le_bytes(header[2..4].try_into().unwrap()));
            cursor += 4;
            let value = bytes.get(cursor..cursor + length).ok_or_else(|| MlError::new(MlErrorCode::InvalidSourceHash, "source hash is truncated"))?.to_vec();
            cursor += length;
            hashes.push(SourceHash { algorithm, value });
        }
        descriptors.push(SourceDescriptor { id, declared_size: Some(size), locator, hashes });
    }
    let registry = SourceRegistry::new(descriptors).map_err(|_| MlError::new(MlErrorCode::MalformedSourceMetadata, "source registry is invalid"))?;
    let mut bindings = Vec::with_capacity(binding_count);
    let mut seen = Vec::with_capacity(binding_count);
    for _ in 0..binding_count {
        let entry = bytes.get(cursor..cursor + BINDING_BYTES).ok_or_else(|| MlError::new(MlErrorCode::MalformedSourceMetadata, "source binding is truncated"))?;
        if entry[4..8].iter().any(|byte| *byte != 0) { return Err(MlError::new(MlErrorCode::MalformedSourceMetadata, "source binding reserved fields are non-zero")); }
        let key_id = u16::from_le_bytes(entry[..2].try_into().unwrap());
        let occurrence = u16::from_le_bytes(entry[2..4].try_into().unwrap());
        if seen.contains(&(key_id, occurrence)) { return Err(MlError::new(MlErrorCode::DuplicateSourceBinding, "source binding occurs more than once")); }
        seen.push((key_id, occurrence));
        let source_id = SourceId::new(u64::from_le_bytes(entry[8..16].try_into().unwrap()));
        let offset = u64::from_le_bytes(entry[16..24].try_into().unwrap());
        let length = u64::from_le_bytes(entry[24..32].try_into().unwrap());
        let reference = registry.checked_range(source_id, offset, length).map_err(|_| MlError::new(MlErrorCode::MalformedSourceMetadata, "source binding range is invalid"))?.tensor_ref();
        bindings.push((key_id, occurrence, reference));
        cursor += BINDING_BYTES;
    }
    if cursor != bytes.len() { return Err(MlError::new(MlErrorCode::MalformedSourceMetadata, "source metadata has trailing bytes")); }
    Ok(Some(PersistentSourceMetadata { registry, bindings }))
}

impl SourceRegistry {
    pub fn self_descriptor(declared_size: u64) -> SourceDescriptor {
        SourceDescriptor::self_artifact(declared_size)
    }

    pub fn new(descriptors: Vec<SourceDescriptor>) -> Result<Self, SourceRegistryError> {
        let mut ids = Vec::with_capacity(descriptors.len());
        for descriptor in &descriptors {
            descriptor.validate()?;
            if descriptor.locator == SourceLocator::SelfArtifact && descriptor.id != SourceId::SELF
            {
                return Err(SourceRegistryError::InvalidSelfSource);
            }
            if ids.contains(&descriptor.id) {
                return Err(SourceRegistryError::DuplicateSource);
            }
            ids.push(descriptor.id);
        }
        if !ids.contains(&SourceId::SELF) {
            return Err(SourceRegistryError::SelfSourceMissing);
        }
        Ok(Self { descriptors })
    }

    pub fn descriptors(&self) -> &[SourceDescriptor] {
        &self.descriptors
    }
    pub fn get(&self, id: SourceId) -> Option<&SourceDescriptor> {
        self.descriptors.iter().find(|source| source.id == id)
    }
    pub fn checked_range(
        &self,
        id: SourceId,
        offset: u64,
        length: u64,
    ) -> Result<CheckedSourceRange, SourceRangeError> {
        self.get(id)
            .ok_or(SourceRangeError::UnknownSource)?
            .checked_range(offset, length)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CheckedSourceRange {
    source_id: SourceId,
    declared_size: u64,
    range: ByteRange,
}

impl CheckedSourceRange {
    pub const fn source_id(self) -> SourceId {
        self.source_id
    }
    pub const fn declared_size(self) -> u64 {
        self.declared_size
    }
    pub const fn range(self) -> ByteRange {
        self.range
    }
    pub const fn offset(self) -> u64 {
        self.range.offset()
    }
    pub const fn length(self) -> u64 {
        self.range.length()
    }
    pub const fn end(self) -> u64 {
        self.range.end()
    }
    pub fn tensor_ref(self) -> TensorRef {
        TensorRef {
            source_id: self.source_id,
            range: self.range,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorRef {
    source_id: SourceId,
    range: ByteRange,
}

impl TensorRef {
    pub fn new(
        source: &SourceDescriptor,
        offset: u64,
        length: u64,
    ) -> Result<Self, SourceRangeError> {
        Ok(source.checked_range(offset, length)?.tensor_ref())
    }
    pub const fn source_id(self) -> SourceId {
        self.source_id
    }
    pub const fn range(self) -> ByteRange {
        self.range
    }
    pub const fn offset(self) -> u64 {
        self.range.offset()
    }
    pub const fn length(self) -> u64 {
        self.range.length()
    }
    pub const fn end(self) -> u64 {
        self.range.end()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceRangeError {
    Range(RangeError),
    UnknownSource,
    UnknownSize,
    OutsideSource,
}

impl fmt::Display for SourceRangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Range(error) => return error.fmt(f),
            Self::UnknownSource => "source ID is not registered",
            Self::UnknownSize => "source size is required for checked ranges",
            Self::OutsideSource => "range lies outside declared source size",
        })
    }
}

impl std::error::Error for SourceRangeError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceRegistryError {
    DuplicateSource,
    InvalidSelfSource,
    SelfSourceMissing,
    MalformedDescriptor,
}
