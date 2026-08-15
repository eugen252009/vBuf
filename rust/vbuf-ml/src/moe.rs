//! Architecture-neutral MoE expert catalog.
//!
//! This describes storage and dispatch facts only. It does not infer router
//! tensors or construct an execution graph; architecture loaders own those
//! semantics.

use crate::bootstrap::Bootstrap;
use crate::error::{MlError, MlErrorCode};
use crate::nested::NestedDirectory;
use crate::region_roles::RegionRole;
use vbuf_core::v06::{V06Physical, V06Semantic, ValidatedV06};

pub const MOE_DIRECTORY_MAGIC: [u8; 8] = *b"VBTMOE\0\0";
pub const MOE_DIRECTORY_VERSION: u16 = 1;
pub const MOE_FLAG_SHARED_EXPERTS: u32 = 1;
const HEADER_BYTES: usize = 36;
const ENTRY_FIXED_BYTES: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoeParameters {
    pub expert_count: u32,
    pub active_expert_count: u32,
    pub layer_count: u32,
    pub shared_experts: bool,
    pub shared_expert_count: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MoeEntry {
    pub layer_index: u32,
    pub expert_index: u32,
    pub role: u16,
    pub child_name: String,
}

#[derive(Debug)]
pub struct MoeDirectory {
    parameters: MoeParameters,
    entries: Vec<MoeEntry>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MoeLoaderKind {
    QwenMoE,
    DeepSeekMoE,
    Unsupported,
}

impl MoeLoaderKind {
    pub fn for_architecture(architecture: &str) -> Self {
        let architecture = architecture.to_ascii_lowercase();
        if architecture.contains("qwen") && architecture.contains("moe") {
            Self::QwenMoE
        } else if architecture.contains("deepseek") {
            Self::DeepSeekMoE
        } else {
            Self::Unsupported
        }
    }
}

impl MoeDirectory {
    pub fn parse<'a>(parent: &ValidatedV06<'a>, bootstrap: &Bootstrap<'a>, nested: &NestedDirectory<'a>) -> Result<Self, MlError> {
        let region = bootstrap.region(RegionRole::MoeDirectory).ok_or_else(|| MlError::new(MlErrorCode::MoeDirectoryMissing, "bootstrap has no MoE directory role"))?;
        let block = parent.blocks().get(region.block_index).ok_or_else(|| MlError::new(MlErrorCode::MalformedMoeDirectory, "MoE directory block is absent"))?;
        if block.semantic != V06Semantic::Opaque || block.physical != V06Physical::Array || block.bit_width != 8 || block.continuation {
            return Err(MlError::new(MlErrorCode::MalformedMoeDirectory, "MoE directory must be a non-continuing opaque byte array"));
        }
        let (parameters, entries) = parse_payload(region.range.bytes())?;
        let mut previous: Option<(u32, u32, u16)> = None;
        for entry in &entries {
            if entry.layer_index >= parameters.layer_count || entry.expert_index >= parameters.expert_count {
                return Err(MlError::new(MlErrorCode::MoeExpertIndexInvalid, "MoE expert entry index is outside declared bounds"));
            }
            if nested.get(&entry.child_name).is_none() {
                return Err(MlError::new(MlErrorCode::MoeExpertReferenceMissing, "MoE expert entry does not reference a nested child"));
            }
            let key = (entry.layer_index, entry.expert_index, entry.role);
            if previous.is_some_and(|previous| previous >= key) {
                return Err(MlError::new(MlErrorCode::MalformedMoeDirectory, "MoE entries must be strictly sorted and unique"));
            }
            previous = Some(key);
        }
        Ok(Self { parameters, entries })
    }

    pub fn parameters(&self) -> MoeParameters { self.parameters }
    pub fn entries(&self) -> &[MoeEntry] { &self.entries }
    pub fn entries_for_layer(&self, layer_index: u32) -> impl Iterator<Item = &MoeEntry> { self.entries.iter().filter(move |entry| entry.layer_index == layer_index) }
}

pub fn encode_payload(parameters: MoeParameters, entries: &[MoeEntry]) -> Result<Vec<u8>, MlError> {
    if parameters.expert_count == 0 || parameters.active_expert_count == 0 || parameters.active_expert_count > parameters.expert_count || parameters.layer_count == 0 || (parameters.shared_experts && parameters.shared_expert_count == 0) || (!parameters.shared_experts && parameters.shared_expert_count != 0) {
        return Err(MlError::new(MlErrorCode::MoeExpertIndexInvalid, "MoE parameters are invalid"));
    }
    let mut ordered = entries.to_vec();
    ordered.sort_by_key(|entry| (entry.layer_index, entry.expert_index, entry.role));
    for pair in ordered.windows(2) {
        if (pair[0].layer_index, pair[0].expert_index, pair[0].role) == (pair[1].layer_index, pair[1].expert_index, pair[1].role) {
            return Err(MlError::new(MlErrorCode::MalformedMoeDirectory, "duplicate MoE entry"));
        }
    }
    let mut output = Vec::with_capacity(HEADER_BYTES + ordered.len() * ENTRY_FIXED_BYTES);
    output.extend_from_slice(&MOE_DIRECTORY_MAGIC);
    output.extend_from_slice(&MOE_DIRECTORY_VERSION.to_le_bytes());
    output.extend_from_slice(&0u16.to_le_bytes());
    output.extend_from_slice(&parameters.expert_count.to_le_bytes());
    output.extend_from_slice(&parameters.active_expert_count.to_le_bytes());
    output.extend_from_slice(&parameters.layer_count.to_le_bytes());
    output.extend_from_slice(&(if parameters.shared_experts { MOE_FLAG_SHARED_EXPERTS } else { 0 }).to_le_bytes());
    output.extend_from_slice(&parameters.shared_expert_count.to_le_bytes());
    output.extend_from_slice(&(ordered.len() as u32).to_le_bytes());
    for entry in ordered {
        if entry.layer_index >= parameters.layer_count || entry.expert_index >= parameters.expert_count || entry.child_name.is_empty() || entry.child_name.len() > u32::MAX as usize {
            return Err(MlError::new(MlErrorCode::MoeExpertIndexInvalid, "MoE entry is invalid"));
        }
        output.extend_from_slice(&entry.layer_index.to_le_bytes());
        output.extend_from_slice(&entry.expert_index.to_le_bytes());
        output.extend_from_slice(&entry.role.to_le_bytes());
        output.extend_from_slice(&0u16.to_le_bytes());
        output.extend_from_slice(&(entry.child_name.len() as u32).to_le_bytes());
        output.extend_from_slice(entry.child_name.as_bytes());
    }
    Ok(output)
}

fn parse_payload(bytes: &[u8]) -> Result<(MoeParameters, Vec<MoeEntry>), MlError> {
    if bytes.len() < HEADER_BYTES || bytes[..8] != MOE_DIRECTORY_MAGIC || u16::from_le_bytes(bytes[8..10].try_into().unwrap()) != MOE_DIRECTORY_VERSION || u16::from_le_bytes(bytes[10..12].try_into().unwrap()) != 0 {
        return Err(MlError::new(MlErrorCode::UnsupportedMoeDirectoryVersion, "MoE directory header is invalid"));
    }
    let flags = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
    if flags & !MOE_FLAG_SHARED_EXPERTS != 0 {
        return Err(MlError::new(MlErrorCode::MalformedMoeDirectory, "MoE directory flags are invalid"));
    }
    let parameters = MoeParameters { expert_count: u32::from_le_bytes(bytes[12..16].try_into().unwrap()), active_expert_count: u32::from_le_bytes(bytes[16..20].try_into().unwrap()), layer_count: u32::from_le_bytes(bytes[20..24].try_into().unwrap()), shared_experts: flags & MOE_FLAG_SHARED_EXPERTS != 0, shared_expert_count: u32::from_le_bytes(bytes[28..32].try_into().unwrap()) };
    if parameters.expert_count == 0 || parameters.active_expert_count == 0 || parameters.active_expert_count > parameters.expert_count || parameters.layer_count == 0 || (parameters.shared_experts && parameters.shared_expert_count == 0) || (!parameters.shared_experts && parameters.shared_expert_count != 0) {
        return Err(MlError::new(MlErrorCode::MoeExpertIndexInvalid, "MoE parameters are invalid"));
    }
    let count = usize::try_from(u32::from_le_bytes(bytes[32..36].try_into().unwrap())).map_err(|_| MlError::new(MlErrorCode::MalformedMoeDirectory, "MoE entry count is too large"))?;
    let mut cursor = HEADER_BYTES;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let fixed = bytes.get(cursor..cursor + ENTRY_FIXED_BYTES).ok_or_else(|| MlError::new(MlErrorCode::MalformedMoeDirectory, "truncated MoE entry"))?;
        let name_len = usize::try_from(u32::from_le_bytes(fixed[12..16].try_into().unwrap())).map_err(|_| MlError::new(MlErrorCode::MalformedMoeDirectory, "MoE child name is too large"))?;
        let name_start = cursor + ENTRY_FIXED_BYTES;
        let name_end = name_start.checked_add(name_len).ok_or_else(|| MlError::new(MlErrorCode::MalformedMoeDirectory, "MoE child name length overflows"))?;
        let name = String::from_utf8(bytes.get(name_start..name_end).ok_or_else(|| MlError::new(MlErrorCode::MalformedMoeDirectory, "truncated MoE child name"))?.to_vec()).map_err(|_| MlError::new(MlErrorCode::MalformedMoeDirectory, "MoE child name is not UTF-8"))?;
        entries.push(MoeEntry { layer_index: u32::from_le_bytes(fixed[0..4].try_into().unwrap()), expert_index: u32::from_le_bytes(fixed[4..8].try_into().unwrap()), role: u16::from_le_bytes(fixed[8..10].try_into().unwrap()), child_name: name });
        cursor = name_end;
    }
    if cursor != bytes.len() { return Err(MlError::new(MlErrorCode::MalformedMoeDirectory, "MoE directory has trailing bytes")); }
    Ok((parameters, entries))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loader_selection_is_architecture_only() {
        assert_eq!(MoeLoaderKind::for_architecture("qwen3moe"), MoeLoaderKind::QwenMoE);
        assert_eq!(MoeLoaderKind::for_architecture("deepseek2"), MoeLoaderKind::DeepSeekMoE);
        assert_eq!(MoeLoaderKind::for_architecture("dense"), MoeLoaderKind::Unsupported);
    }

    #[test]
    fn generic_catalog_round_trips_and_sorts_entries() {
        let parameters = MoeParameters { expert_count: 8, active_expert_count: 2, layer_count: 2, shared_experts: true, shared_expert_count: 1 };
        let bytes = encode_payload(parameters, &[
            MoeEntry { layer_index: 1, expert_index: 2, role: 1, child_name: "blk.1.expert.2".into() },
            MoeEntry { layer_index: 0, expert_index: 0, role: 0, child_name: "blk.0.expert.0".into() },
        ]).unwrap();
        let (decoded, entries) = parse_payload(&bytes).unwrap();
        assert_eq!(decoded, parameters);
        assert_eq!(entries[0].child_name, "blk.0.expert.0");
        assert_eq!(entries[1].child_name, "blk.1.expert.2");
    }
}
