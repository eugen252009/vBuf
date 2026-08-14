use crate::error::{MlError, MlErrorCode};
use crate::region_roles::RegionRole;
use vbuf_core::v06::{ValidatedV06, V06Semantic};
use vbuf_layout::CheckedRange;

pub const BOOTSTRAP_KEY_ID: u16 = 0xF000;
pub const PROFILE_VERSION: u16 = 1;
pub const BOOTSTRAP_MAGIC: [u8; 8] = *b"VBUFML\0\0";
pub const MAX_BOOTSTRAP_BYTES: usize = 4096;
const HEADER_BYTES: usize = 16;
const ENTRY_BYTES: usize = 16;
const REQUIRED_FLAG: u16 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BootstrapEntry {
    pub role_id: u16,
    pub required: bool,
    pub key_id: u16,
    pub occurrence: u16,
}

impl BootstrapEntry {
    pub const fn new(role_id: u16, required: bool, key_id: u16, occurrence: u16) -> Self {
        Self { role_id, required, key_id, occurrence }
    }
}

#[derive(Debug)]
pub struct SemanticRegion<'a> {
    pub role: RegionRole,
    pub key_id: u16,
    pub occurrence: u16,
    /// Index into the authoritative validated canonical block sequence.
    pub block_index: usize,
    pub range: CheckedRange<'a>,
}

#[derive(Debug)]
pub struct Bootstrap<'a> {
    profile_version: u16,
    regions: Vec<SemanticRegion<'a>>,
}

impl<'a> Bootstrap<'a> {
    pub fn discover(validated: &'a ValidatedV06<'a>) -> Result<Self, MlError> {
        let mut bootstrap_index = None;
        for (index, block) in validated.blocks().iter().enumerate() {
            if block.key_id == BOOTSTRAP_KEY_ID && bootstrap_index.replace(index).is_some() {
                return Err(MlError::new(MlErrorCode::BootstrapDuplicate, "bootstrap Key-ID occurs more than once"));
            }
        }
        let index = bootstrap_index.ok_or_else(|| MlError::new(MlErrorCode::BootstrapNotFound, "bootstrap Key-ID is absent"))?;
        let block = &validated.blocks()[index];
        if block.semantic != V06Semantic::Opaque || block.bit_width != 8 {
            return Err(MlError::new(MlErrorCode::RegionTypeMismatch, "bootstrap must be an opaque byte region"));
        }
        let payload = validated.payload_range(index)?;
        let (profile_version, entries) = parse_payload(payload.bytes())?;
        let mut regions = Vec::new();
        let mut seen_roles = Vec::new();
        for entry in entries {
            let Some(role) = RegionRole::from_id(entry.role_id) else {
                if entry.required {
                    return Err(MlError::new(MlErrorCode::UnknownRequiredRole, "unknown required profile role"));
                }
                continue;
            };
            if seen_roles.contains(&role) {
                return Err(MlError::new(MlErrorCode::DuplicateRole, "profile role occurs more than once"));
            }
            if entry.required != role.is_required() {
                return Err(MlError::new(MlErrorCode::MalformedBootstrap, "role requiredness does not match profile contract"));
            }
            seen_roles.push(role);
            let target_index = validated
                .blocks()
                .iter()
                .enumerate()
                .filter(|(_, candidate)| candidate.key_id == entry.key_id)
                .nth(entry.occurrence as usize)
                .map(|(candidate_index, _)| candidate_index)
                .ok_or_else(|| MlError::new(MlErrorCode::ReferencedRegionMissing, "bootstrap region reference is absent"))?;
            regions.push(SemanticRegion {
                role,
                key_id: entry.key_id,
                occurrence: entry.occurrence,
                block_index: target_index,
                range: validated.payload_range(target_index)?,
            });
        }
        for required in [RegionRole::TensorDirectory, RegionRole::ModelMetadata] {
            if !seen_roles.contains(&required) {
                return Err(MlError::new(MlErrorCode::MissingRequiredRole, "required profile role is absent"));
            }
        }
        Ok(Self { profile_version, regions })
    }

    pub const fn profile_version(&self) -> u16 { self.profile_version }
    pub fn regions(&self) -> &[SemanticRegion<'a>] { &self.regions }
    pub fn region(&self, role: RegionRole) -> Option<&SemanticRegion<'a>> { self.regions.iter().find(|region| region.role == role) }
}

pub fn encode_payload(entries: &[BootstrapEntry]) -> Result<Vec<u8>, MlError> {
    let total = HEADER_BYTES.checked_add(entries.len().checked_mul(ENTRY_BYTES).ok_or_else(|| MlError::new(MlErrorCode::MalformedBootstrap, "bootstrap entry count overflows"))?).ok_or_else(|| MlError::new(MlErrorCode::MalformedBootstrap, "bootstrap size overflows"))?;
    if total > MAX_BOOTSTRAP_BYTES || entries.len() > u16::MAX as usize {
        return Err(MlError::new(MlErrorCode::MalformedBootstrap, "bootstrap exceeds bounded size"));
    }
    let mut output = vec![0u8; total];
    output[..8].copy_from_slice(&BOOTSTRAP_MAGIC);
    output[8..10].copy_from_slice(&PROFILE_VERSION.to_le_bytes());
    output[12..14].copy_from_slice(&(entries.len() as u16).to_le_bytes());
    for (index, entry) in entries.iter().enumerate() {
        let offset = HEADER_BYTES + index * ENTRY_BYTES;
        output[offset..offset + 2].copy_from_slice(&entry.role_id.to_le_bytes());
        output[offset + 2..offset + 4].copy_from_slice(&(if entry.required { REQUIRED_FLAG } else { 0 }).to_le_bytes());
        output[offset + 4..offset + 6].copy_from_slice(&entry.key_id.to_le_bytes());
        output[offset + 6..offset + 8].copy_from_slice(&entry.occurrence.to_le_bytes());
    }
    Ok(output)
}

fn parse_payload(bytes: &[u8]) -> Result<(u16, Vec<BootstrapEntry>), MlError> {
    if bytes.len() < HEADER_BYTES || bytes.len() > MAX_BOOTSTRAP_BYTES || bytes[..8] != BOOTSTRAP_MAGIC {
        return Err(MlError::new(MlErrorCode::MalformedBootstrap, "bootstrap header is invalid"));
    }
    let version = u16::from_le_bytes(bytes[8..10].try_into().unwrap());
    if version != PROFILE_VERSION {
        return Err(MlError::new(MlErrorCode::UnsupportedProfileVersion, "unsupported vBuf-ML profile version"));
    }
    if u16::from_le_bytes(bytes[10..12].try_into().unwrap()) != 0 || u16::from_le_bytes(bytes[14..16].try_into().unwrap()) != 0 {
        return Err(MlError::new(MlErrorCode::InvalidBootstrapFlags, "bootstrap reserved fields are non-zero"));
    }
    let count = usize::from(u16::from_le_bytes(bytes[12..14].try_into().unwrap()));
    let expected = HEADER_BYTES.checked_add(count.checked_mul(ENTRY_BYTES).ok_or_else(|| MlError::new(MlErrorCode::MalformedBootstrap, "bootstrap entry count overflows"))?).ok_or_else(|| MlError::new(MlErrorCode::MalformedBootstrap, "bootstrap size overflows"))?;
    if expected != bytes.len() || expected > MAX_BOOTSTRAP_BYTES {
        return Err(MlError::new(MlErrorCode::MalformedBootstrap, "bootstrap length does not match entry count"));
    }
    let mut entries = Vec::with_capacity(count);
    for index in 0..count {
        let offset = HEADER_BYTES + index * ENTRY_BYTES;
        let flags = u16::from_le_bytes(bytes[offset + 2..offset + 4].try_into().unwrap());
        if flags & !REQUIRED_FLAG != 0 || bytes[offset + 8..offset + ENTRY_BYTES].iter().any(|byte| *byte != 0) {
            return Err(MlError::new(MlErrorCode::InvalidBootstrapFlags, "bootstrap entry flags or reserved bytes are non-zero"));
        }
        entries.push(BootstrapEntry::new(
            u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap()),
            flags & REQUIRED_FLAG != 0,
            u16::from_le_bytes(bytes[offset + 4..offset + 6].try_into().unwrap()),
            u16::from_le_bytes(bytes[offset + 6..offset + 8].try_into().unwrap()),
        ));
    }
    Ok((version, entries))
}
