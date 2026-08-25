//! Architecture-neutral MoE expert catalog.
//!
//! This describes storage and dispatch facts only. It does not infer router
//! tensors or construct an execution graph; architecture loaders own those
//! semantics.

use crate::bootstrap::Bootstrap;
use crate::error::{MlError, MlErrorCode};
use crate::nested::NestedDirectory;
use crate::region_roles::RegionRole;
use crate::representations::TensorRepresentation;
use crate::tensor_directory::TensorDirectory;
use std::collections::HashSet;
use vbuf_core::v06::{V06Physical, V06Semantic, ValidatedV06};

pub const MOE_DIRECTORY_MAGIC: [u8; 8] = *b"VBTMOE\0\0";
pub const MOE_DIRECTORY_VERSION: u16 = 2;
const MOE_DIRECTORY_VERSION_V1: u16 = 1;
pub const MOE_FLAG_SHARED_EXPERTS: u32 = 1;
const HEADER_BYTES: usize = 36;
const ENTRY_FIXED_BYTES: usize = 16;
const V2_EXTENSION_BYTES: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoeParameters {
    pub expert_count: u32,
    pub active_expert_count: u32,
    pub layer_count: u32,
    pub shared_experts: bool,
    pub shared_expert_count: u32,
    pub normalize_topk_prob: bool,
    pub routing_group_count: u32,
    pub routing_topk_group_count: u32,
    pub routed_scaling_factor_bits: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MoeEntry {
    pub layer_index: u32,
    pub expert_index: u32,
    pub role: u16,
    pub child_name: String,
    /// Canonical tensor-directory identity for source-independent dispatch.
    /// None is retained for legacy v1 catalog entries.
    pub tensor_ordinal: Option<u16>,
    pub scale_ordinal: Option<u16>,
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
    pub fn parse<'a>(
        parent: &ValidatedV06<'a>,
        bootstrap: &Bootstrap<'a>,
        nested: &NestedDirectory<'a>,
    ) -> Result<Self, MlError> {
        let region = bootstrap.region(RegionRole::MoeDirectory).ok_or_else(|| {
            MlError::new(
                MlErrorCode::MoeDirectoryMissing,
                "bootstrap has no MoE directory role",
            )
        })?;
        let block = parent.blocks().get(region.block_index).ok_or_else(|| {
            MlError::new(
                MlErrorCode::MalformedMoeDirectory,
                "MoE directory block is absent",
            )
        })?;
        if block.semantic != V06Semantic::Opaque
            || block.physical != V06Physical::Array
            || block.bit_width != 8
            || block.continuation
        {
            return Err(MlError::new(
                MlErrorCode::MalformedMoeDirectory,
                "MoE directory must be a non-continuing opaque byte array",
            ));
        }
        let (parameters, entries) = parse_payload(region.range.bytes())?;
        let mut previous: Option<(u32, u32, u16)> = None;
        for entry in &entries {
            if entry.layer_index >= parameters.layer_count
                || entry.expert_index >= parameters.expert_count
            {
                return Err(MlError::new(
                    MlErrorCode::MoeExpertIndexInvalid,
                    "MoE expert entry index is outside declared bounds",
                ));
            }
            if nested.get(&entry.child_name).is_none() {
                return Err(MlError::new(
                    MlErrorCode::MoeExpertReferenceMissing,
                    "MoE expert entry does not reference a nested child",
                ));
            }
            let key = (entry.layer_index, entry.expert_index, entry.role);
            if previous.is_some_and(|previous| previous >= key) {
                return Err(MlError::new(
                    MlErrorCode::MalformedMoeDirectory,
                    "MoE entries must be strictly sorted and unique",
                ));
            }
            previous = Some(key);
        }
        Ok(Self {
            parameters,
            entries,
        })
    }

    pub fn parameters(&self) -> MoeParameters {
        self.parameters
    }
    pub fn entries(&self) -> &[MoeEntry] {
        &self.entries
    }
    pub fn entries_for_layer(&self, layer_index: u32) -> impl Iterator<Item = &MoeEntry> {
        self.entries
            .iter()
            .filter(move |entry| entry.layer_index == layer_index)
    }

    pub fn entries_with_tensor_bindings(&self) -> impl Iterator<Item = &MoeEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.tensor_ordinal.is_some())
    }

    pub fn validate_tensor_bindings(
        &self,
        directory: &TensorDirectory<'_>,
        tensor_key_id: u16,
    ) -> Result<(), MlError> {
        let mut seen_tensors = HashSet::new();
        for entry in self.entries_with_tensor_bindings() {
            let ordinal = entry.tensor_ordinal.ok_or_else(|| {
                MlError::new(
                    MlErrorCode::MoeExpertReferenceMissing,
                    "MoE tensor identity is absent",
                )
            })?;
            let weight = directory
                .get_by_identity(tensor_key_id, ordinal)
                .ok_or_else(|| {
                    MlError::new(
                        MlErrorCode::MoeExpertReferenceMissing,
                        "MoE tensor identity is not in the tensor directory",
                    )
                })?;
            if !seen_tensors.insert(ordinal)
                || (entry.scale_ordinal.is_some()
                    && weight.representation != TensorRepresentation::F8_E4M3)
            {
                return Err(MlError::new(
                    MlErrorCode::MoeExpertReferenceMissing,
                    "MoE tensor binding is duplicated or has an invalid representation",
                ));
            }
            if let Some(scale_ordinal) = entry.scale_ordinal {
                let scale = directory
                    .get_by_identity(tensor_key_id, scale_ordinal)
                    .ok_or_else(|| {
                        MlError::new(
                            MlErrorCode::MoeExpertReferenceMissing,
                            "MoE scale identity is not in the tensor directory",
                        )
                    })?;
                if scale.representation != TensorRepresentation::CanonicalPrimitive {
                    return Err(MlError::new(
                        MlErrorCode::MoeExpertReferenceMissing,
                        "MoE scale binding is not canonical float storage",
                    ));
                }
            }
        }
        Ok(())
    }
}

pub fn encode_payload(parameters: MoeParameters, entries: &[MoeEntry]) -> Result<Vec<u8>, MlError> {
    if parameters.expert_count == 0
        || parameters.active_expert_count == 0
        || parameters.active_expert_count > parameters.expert_count
        || parameters.layer_count == 0
        || (parameters.shared_experts && parameters.shared_expert_count == 0)
        || (!parameters.shared_experts && parameters.shared_expert_count != 0)
    {
        return Err(MlError::new(
            MlErrorCode::MoeExpertIndexInvalid,
            "MoE parameters are invalid",
        ));
    }
    let mut ordered = entries.to_vec();
    ordered.sort_by_key(|entry| (entry.layer_index, entry.expert_index, entry.role));
    for pair in ordered.windows(2) {
        if (pair[0].layer_index, pair[0].expert_index, pair[0].role)
            == (pair[1].layer_index, pair[1].expert_index, pair[1].role)
        {
            return Err(MlError::new(
                MlErrorCode::MalformedMoeDirectory,
                "duplicate MoE entry",
            ));
        }
    }
    let version = if ordered
        .iter()
        .any(|entry| entry.tensor_ordinal.is_some() || entry.scale_ordinal.is_some())
    {
        MOE_DIRECTORY_VERSION
    } else {
        MOE_DIRECTORY_VERSION_V1
    };
    let entry_bytes = if version == MOE_DIRECTORY_VERSION {
        ENTRY_FIXED_BYTES + 4
    } else {
        ENTRY_FIXED_BYTES
    };
    let mut output = Vec::with_capacity(HEADER_BYTES + ordered.len() * entry_bytes);
    output.extend_from_slice(&MOE_DIRECTORY_MAGIC);
    output.extend_from_slice(&version.to_le_bytes());
    output.extend_from_slice(&0u16.to_le_bytes());
    output.extend_from_slice(&parameters.expert_count.to_le_bytes());
    output.extend_from_slice(&parameters.active_expert_count.to_le_bytes());
    output.extend_from_slice(&parameters.layer_count.to_le_bytes());
    output.extend_from_slice(
        &(if parameters.shared_experts {
            MOE_FLAG_SHARED_EXPERTS
        } else {
            0
        })
        .to_le_bytes(),
    );
    output.extend_from_slice(&parameters.shared_expert_count.to_le_bytes());
    output.extend_from_slice(&(ordered.len() as u32).to_le_bytes());
    if version == MOE_DIRECTORY_VERSION {
        output.extend_from_slice(&u32::from(parameters.normalize_topk_prob).to_le_bytes());
        output.extend_from_slice(&parameters.routing_group_count.to_le_bytes());
        output.extend_from_slice(&parameters.routing_topk_group_count.to_le_bytes());
        output.extend_from_slice(&parameters.routed_scaling_factor_bits.to_le_bytes());
    }
    for entry in ordered {
        if entry.layer_index >= parameters.layer_count
            || entry.expert_index >= parameters.expert_count
            || entry.child_name.is_empty()
            || entry.child_name.len() > u32::MAX as usize
        {
            return Err(MlError::new(
                MlErrorCode::MoeExpertIndexInvalid,
                "MoE entry is invalid",
            ));
        }
        output.extend_from_slice(&entry.layer_index.to_le_bytes());
        output.extend_from_slice(&entry.expert_index.to_le_bytes());
        output.extend_from_slice(&entry.role.to_le_bytes());
        output.extend_from_slice(&0u16.to_le_bytes());
        if version == MOE_DIRECTORY_VERSION {
            output.extend_from_slice(
                &entry
                    .tensor_ordinal
                    .ok_or_else(|| {
                        MlError::new(
                            MlErrorCode::MoeExpertReferenceMissing,
                            "version 2 MoE entry has no tensor identity",
                        )
                    })?
                    .to_le_bytes(),
            );
            output.extend_from_slice(&entry.scale_ordinal.unwrap_or(u16::MAX).to_le_bytes());
        }
        output.extend_from_slice(&(entry.child_name.len() as u32).to_le_bytes());
        output.extend_from_slice(entry.child_name.as_bytes());
    }
    Ok(output)
}

fn parse_payload(bytes: &[u8]) -> Result<(MoeParameters, Vec<MoeEntry>), MlError> {
    if bytes.len() < HEADER_BYTES
        || bytes[..8] != MOE_DIRECTORY_MAGIC
        || !matches!(
            u16::from_le_bytes(bytes[8..10].try_into().unwrap()),
            MOE_DIRECTORY_VERSION_V1 | MOE_DIRECTORY_VERSION
        )
        || u16::from_le_bytes(bytes[10..12].try_into().unwrap()) != 0
    {
        return Err(MlError::new(
            MlErrorCode::UnsupportedMoeDirectoryVersion,
            "MoE directory header is invalid",
        ));
    }
    let flags = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
    if flags & !MOE_FLAG_SHARED_EXPERTS != 0 {
        return Err(MlError::new(
            MlErrorCode::MalformedMoeDirectory,
            "MoE directory flags are invalid",
        ));
    }
    let version = u16::from_le_bytes(bytes[8..10].try_into().unwrap());
    let extension = if version == MOE_DIRECTORY_VERSION {
        bytes
            .get(HEADER_BYTES..HEADER_BYTES + V2_EXTENSION_BYTES)
            .ok_or_else(|| {
                MlError::new(
                    MlErrorCode::MalformedMoeDirectory,
                    "MoE v2 routing extension is truncated",
                )
            })?
    } else {
        &[]
    };
    let parameters = MoeParameters {
        expert_count: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
        active_expert_count: u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
        layer_count: u32::from_le_bytes(bytes[20..24].try_into().unwrap()),
        shared_experts: flags & MOE_FLAG_SHARED_EXPERTS != 0,
        shared_expert_count: u32::from_le_bytes(bytes[28..32].try_into().unwrap()),
        normalize_topk_prob: extension.first().is_some_and(|value| *value != 0),
        routing_group_count: if extension.is_empty() {
            0
        } else {
            u32::from_le_bytes(extension[4..8].try_into().unwrap())
        },
        routing_topk_group_count: if extension.is_empty() {
            0
        } else {
            u32::from_le_bytes(extension[8..12].try_into().unwrap())
        },
        routed_scaling_factor_bits: if extension.is_empty() {
            1.0f32.to_bits()
        } else {
            u32::from_le_bytes(extension[12..16].try_into().unwrap())
        },
    };
    if version == MOE_DIRECTORY_VERSION
        && (extension[0] > 1
            || extension[1..4].iter().any(|byte| *byte != 0)
            || parameters.routing_group_count == 0
            || parameters.routing_topk_group_count == 0
            || parameters.routing_topk_group_count > parameters.routing_group_count
            || !f32::from_bits(parameters.routed_scaling_factor_bits).is_finite()
            || f32::from_bits(parameters.routed_scaling_factor_bits) <= 0.0)
    {
        return Err(MlError::new(
            MlErrorCode::MalformedMoeDirectory,
            "MoE v2 routing extension is invalid",
        ));
    }
    if parameters.expert_count == 0
        || parameters.active_expert_count == 0
        || parameters.active_expert_count > parameters.expert_count
        || parameters.layer_count == 0
        || (parameters.shared_experts && parameters.shared_expert_count == 0)
        || (!parameters.shared_experts && parameters.shared_expert_count != 0)
    {
        return Err(MlError::new(
            MlErrorCode::MoeExpertIndexInvalid,
            "MoE parameters are invalid",
        ));
    }
    let entry_fixed_bytes = if version == MOE_DIRECTORY_VERSION {
        ENTRY_FIXED_BYTES + 4
    } else {
        ENTRY_FIXED_BYTES
    };
    let count =
        usize::try_from(u32::from_le_bytes(bytes[32..36].try_into().unwrap())).map_err(|_| {
            MlError::new(
                MlErrorCode::MalformedMoeDirectory,
                "MoE entry count is too large",
            )
        })?;
    let mut cursor = HEADER_BYTES + extension.len();
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let fixed = bytes
            .get(cursor..cursor + entry_fixed_bytes)
            .ok_or_else(|| {
                MlError::new(MlErrorCode::MalformedMoeDirectory, "truncated MoE entry")
            })?;
        let name_offset = if version == MOE_DIRECTORY_VERSION {
            16
        } else {
            12
        };
        let name_len = usize::try_from(u32::from_le_bytes(
            fixed[name_offset..name_offset + 4].try_into().unwrap(),
        ))
        .map_err(|_| {
            MlError::new(
                MlErrorCode::MalformedMoeDirectory,
                "MoE child name is too large",
            )
        })?;
        let name_start = cursor + entry_fixed_bytes;
        let name_end = name_start.checked_add(name_len).ok_or_else(|| {
            MlError::new(
                MlErrorCode::MalformedMoeDirectory,
                "MoE child name length overflows",
            )
        })?;
        let name = String::from_utf8(
            bytes
                .get(name_start..name_end)
                .ok_or_else(|| {
                    MlError::new(
                        MlErrorCode::MalformedMoeDirectory,
                        "truncated MoE child name",
                    )
                })?
                .to_vec(),
        )
        .map_err(|_| {
            MlError::new(
                MlErrorCode::MalformedMoeDirectory,
                "MoE child name is not UTF-8",
            )
        })?;
        let (tensor_ordinal, scale_ordinal) = if version == MOE_DIRECTORY_VERSION {
            let tensor = u16::from_le_bytes(fixed[12..14].try_into().unwrap());
            let scale = u16::from_le_bytes(fixed[14..16].try_into().unwrap());
            (Some(tensor), (scale != u16::MAX).then_some(scale))
        } else {
            (None, None)
        };
        entries.push(MoeEntry {
            layer_index: u32::from_le_bytes(fixed[0..4].try_into().unwrap()),
            expert_index: u32::from_le_bytes(fixed[4..8].try_into().unwrap()),
            role: u16::from_le_bytes(fixed[8..10].try_into().unwrap()),
            child_name: name,
            tensor_ordinal,
            scale_ordinal,
        });
        cursor = name_end;
    }
    if cursor != bytes.len() {
        return Err(MlError::new(
            MlErrorCode::MalformedMoeDirectory,
            "MoE directory has trailing bytes",
        ));
    }
    Ok((parameters, entries))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loader_selection_is_architecture_only() {
        assert_eq!(
            MoeLoaderKind::for_architecture("qwen3moe"),
            MoeLoaderKind::QwenMoE
        );
        assert_eq!(
            MoeLoaderKind::for_architecture("deepseek2"),
            MoeLoaderKind::DeepSeekMoE
        );
        assert_eq!(
            MoeLoaderKind::for_architecture("dense"),
            MoeLoaderKind::Unsupported
        );
    }

    #[test]
    fn generic_catalog_round_trips_and_sorts_entries() {
        let parameters = MoeParameters {
            expert_count: 8,
            active_expert_count: 2,
            layer_count: 2,
            shared_experts: true,
            shared_expert_count: 1,
            normalize_topk_prob: false,
            routing_group_count: 0,
            routing_topk_group_count: 0,
            routed_scaling_factor_bits: 1.0f32.to_bits(),
        };
        let bytes = encode_payload(
            parameters,
            &[
                MoeEntry {
                    layer_index: 1,
                    expert_index: 2,
                    role: 1,
                    child_name: "blk.1.expert.2".into(),
                    tensor_ordinal: None,
                    scale_ordinal: None,
                },
                MoeEntry {
                    layer_index: 0,
                    expert_index: 0,
                    role: 0,
                    child_name: "blk.0.expert.0".into(),
                    tensor_ordinal: None,
                    scale_ordinal: None,
                },
            ],
        )
        .unwrap();
        let (decoded, entries) = parse_payload(&bytes).unwrap();
        assert_eq!(decoded, parameters);
        assert_eq!(entries[0].child_name, "blk.0.expert.0");
        assert_eq!(entries[1].child_name, "blk.1.expert.2");
    }

    #[test]
    fn tensor_bound_catalog_round_trips_without_source_names() {
        let parameters = MoeParameters {
            expert_count: 128,
            active_expert_count: 8,
            layer_count: 47,
            shared_experts: true,
            shared_expert_count: 1,
            normalize_topk_prob: true,
            routing_group_count: 1,
            routing_topk_group_count: 1,
            routed_scaling_factor_bits: 1.0f32.to_bits(),
        };
        let bytes = encode_payload(
            parameters,
            &[MoeEntry {
                layer_index: 1,
                expert_index: 7,
                role: 1,
                child_name: "glm.l001.e007.r01".into(),
                tensor_ordinal: Some(321),
                scale_ordinal: Some(322),
            }],
        )
        .unwrap();
        let (decoded, entries) = parse_payload(&bytes).unwrap();
        assert_eq!(decoded, parameters);
        assert_eq!(entries[0].tensor_ordinal, Some(321));
        assert_eq!(entries[0].scale_ordinal, Some(322));
    }
}
