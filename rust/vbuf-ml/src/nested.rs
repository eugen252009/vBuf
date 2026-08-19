//! Inline nested-vBuf directory for the vbuf-ml-0.1 profile.
//!
//! A parent vBuf stores this directory as a normal profile region. Each entry
//! points to a byte range inside one canonical parent block. The referenced
//! range must itself be a complete canonical v0.6 stream. Child streams are
//! validated before their bytes are exposed to callers.

use crate::bootstrap::Bootstrap;
use crate::error::{MlError, MlErrorCode};
use crate::region_roles::RegionRole;
use vbuf_core::v06::{V06Physical, V06Semantic, ValidatedV06, parse_v06};
use vbuf_layout::{ByteRange, CheckedRange};

pub const NESTED_DIRECTORY_MAGIC: [u8; 8] = *b"VBTNEST\0";
pub const NESTED_DIRECTORY_VERSION: u16 = 1;
pub const MAX_NESTED_DIRECTORY_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_NESTED_CHILDREN: u32 = 1_000_000;
pub const MAX_NESTED_NAME_BYTES: usize = 4096;
const HEADER_BYTES: usize = 20;
const ENTRY_FIXED_BYTES: usize = 24;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NestedEntry {
    pub name: String,
    pub key_id: u16,
    pub occurrence: u16,
    pub child_offset: u64,
    pub child_length: u64,
}

#[derive(Debug)]
pub struct NestedChild<'a> {
    pub name: String,
    pub key_id: u16,
    pub occurrence: u16,
    pub child_length: u64,
    pub range: CheckedRange<'a>,
    pub validated: ValidatedV06<'a>,
}

#[derive(Debug)]
pub struct NestedDirectory<'a> {
    children: Vec<NestedChild<'a>>,
}

impl<'a> NestedDirectory<'a> {
    pub fn parse(parent: &ValidatedV06<'a>, bootstrap: &Bootstrap<'a>) -> Result<Self, MlError> {
        let region = bootstrap
            .region(RegionRole::NestedDirectory)
            .ok_or_else(|| {
                MlError::new(
                    MlErrorCode::NestedDirectoryMissing,
                    "bootstrap has no nested directory role",
                )
            })?;
        let block = parent.blocks().get(region.block_index).ok_or_else(|| {
            MlError::new(
                MlErrorCode::NestedChildReferenceMissing,
                "nested directory block is absent",
            )
        })?;
        if block.semantic != V06Semantic::Opaque
            || block.physical != V06Physical::Array
            || block.bit_width != 8
            || block.continuation
        {
            return Err(MlError::new(
                MlErrorCode::RegionTypeMismatch,
                "nested directory must be a non-continuing opaque byte array",
            ));
        }
        let entries = parse_payload(region.range.bytes())?;
        let mut children = Vec::with_capacity(entries.len());
        let mut previous_name: Option<Vec<u8>> = None;
        let mut occupied: Vec<(u64, u64)> = Vec::new();
        for entry in entries {
            if let Some(previous) = previous_name.as_ref()
                && previous.as_slice() >= entry.name.as_bytes()
            {
                return Err(MlError::new(
                    if previous.as_slice() == entry.name.as_bytes() {
                        MlErrorCode::DuplicateTensorName
                    } else {
                        MlErrorCode::InvalidDirectoryOrder
                    },
                    "nested names must be unique and strictly sorted",
                ));
            }
            previous_name = Some(entry.name.as_bytes().to_vec());
            let target_index = parent
                .blocks()
                .iter()
                .enumerate()
                .filter(|(_, candidate)| candidate.key_id == entry.key_id)
                .nth(entry.occurrence as usize)
                .map(|(index, _)| index)
                .ok_or_else(|| {
                    MlError::new(
                        MlErrorCode::NestedChildReferenceMissing,
                        "nested child parent block reference is absent",
                    )
                })?;
            let target = &parent.blocks()[target_index];
            let child_end = entry
                .child_offset
                .checked_add(entry.child_length)
                .ok_or_else(|| {
                    MlError::new(
                        MlErrorCode::NestedChildRangeInvalid,
                        "nested child range overflows",
                    )
                })?;
            if entry.child_length == 0 || child_end > target.payload_len {
                return Err(MlError::new(
                    MlErrorCode::NestedChildRangeInvalid,
                    "nested child range is outside its parent payload",
                ));
            }
            if occupied
                .iter()
                .any(|(start, end)| entry.child_offset < *end && *start < child_end)
            {
                return Err(MlError::new(
                    MlErrorCode::NestedChildOverlap,
                    "nested child ranges overlap",
                ));
            }
            occupied.push((entry.child_offset, child_end));
            let absolute_start = target
                .payload_start
                .checked_add(entry.child_offset)
                .ok_or_else(|| {
                    MlError::new(
                        MlErrorCode::NestedChildRangeInvalid,
                        "nested child start overflows",
                    )
                })?;
            let absolute_end = absolute_start
                .checked_add(entry.child_length)
                .ok_or_else(|| {
                    MlError::new(
                        MlErrorCode::NestedChildRangeInvalid,
                        "nested child end overflows",
                    )
                })?;
            let range = CheckedRange::from_mapping(
                parent.bytes(),
                ByteRange::from_end(absolute_start, absolute_end).map_err(|_| {
                    MlError::new(
                        MlErrorCode::NestedChildRangeInvalid,
                        "nested child range cannot be represented",
                    )
                })?,
            )
            .map_err(|_| {
                MlError::new(
                    MlErrorCode::NestedChildRangeInvalid,
                    "nested child range cannot be exposed",
                )
            })?;
            let validated = parse_v06(range.bytes()).map_err(|_| {
                MlError::new(
                    MlErrorCode::NestedChildNotCanonical,
                    "nested child is not a valid canonical v0.6 stream",
                )
            })?;
            children.push(NestedChild {
                name: entry.name,
                key_id: entry.key_id,
                occurrence: entry.occurrence,
                child_length: entry.child_length,
                range,
                validated,
            });
        }
        Ok(Self { children })
    }

    pub fn children(&self) -> &[NestedChild<'a>] {
        &self.children
    }

    pub fn get(&self, name: &str) -> Option<&NestedChild<'a>> {
        self.children
            .binary_search_by(|child| child.name.as_str().cmp(name))
            .ok()
            .map(|index| &self.children[index])
    }
}

pub fn encode_payload(entries: &[NestedEntry]) -> Result<Vec<u8>, MlError> {
    if entries.len() > MAX_NESTED_CHILDREN as usize {
        return Err(MlError::new(
            MlErrorCode::MalformedNestedDirectory,
            "nested child count exceeds profile maximum",
        ));
    }
    let mut ordered: Vec<&NestedEntry> = entries.iter().collect();
    ordered.sort_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));
    for pair in ordered.windows(2) {
        if pair[0].name == pair[1].name {
            return Err(MlError::new(
                MlErrorCode::DuplicateTensorName,
                "nested child names must be unique",
            ));
        }
    }
    let mut output = Vec::with_capacity(HEADER_BYTES + ordered.len() * ENTRY_FIXED_BYTES);
    output.extend_from_slice(&NESTED_DIRECTORY_MAGIC);
    output.extend_from_slice(&NESTED_DIRECTORY_VERSION.to_le_bytes());
    output.extend_from_slice(&0u16.to_le_bytes());
    output.extend_from_slice(&(ordered.len() as u32).to_le_bytes());
    output.extend_from_slice(&0u32.to_le_bytes());
    for entry in ordered {
        validate_entry(entry)?;
        output.extend_from_slice(&(entry.name.len() as u16).to_le_bytes());
        output.extend_from_slice(&0u16.to_le_bytes());
        output.extend_from_slice(&entry.key_id.to_le_bytes());
        output.extend_from_slice(&entry.occurrence.to_le_bytes());
        output.extend_from_slice(&entry.child_offset.to_le_bytes());
        output.extend_from_slice(&entry.child_length.to_le_bytes());
        output.extend_from_slice(entry.name.as_bytes());
    }
    if output.len() > MAX_NESTED_DIRECTORY_BYTES {
        return Err(MlError::new(
            MlErrorCode::MalformedNestedDirectory,
            "nested directory exceeds bounded size",
        ));
    }
    Ok(output)
}

fn validate_entry(entry: &NestedEntry) -> Result<(), MlError> {
    if entry.name.is_empty()
        || entry.name.len() > MAX_NESTED_NAME_BYTES
        || entry.name.as_bytes().contains(&0)
    {
        return Err(MlError::new(
            MlErrorCode::InvalidTensorName,
            "nested child name is invalid",
        ));
    }
    if entry.child_length == 0 || entry.child_offset.checked_add(entry.child_length).is_none() {
        return Err(MlError::new(
            MlErrorCode::NestedChildRangeInvalid,
            "nested child range is invalid",
        ));
    }
    Ok(())
}

fn parse_payload(bytes: &[u8]) -> Result<Vec<NestedEntry>, MlError> {
    if bytes.len() < HEADER_BYTES
        || bytes.len() > MAX_NESTED_DIRECTORY_BYTES
        || bytes[..8] != NESTED_DIRECTORY_MAGIC
    {
        return Err(MlError::new(
            MlErrorCode::MalformedNestedDirectory,
            "nested directory header is invalid",
        ));
    }
    let version = u16::from_le_bytes(bytes[8..10].try_into().unwrap());
    if version != NESTED_DIRECTORY_VERSION {
        return Err(MlError::new(
            MlErrorCode::UnsupportedNestedDirectoryVersion,
            "unsupported nested directory version",
        ));
    }
    if u16::from_le_bytes(bytes[10..12].try_into().unwrap()) != 0
        || bytes[16..20].iter().any(|byte| *byte != 0)
    {
        return Err(MlError::new(
            MlErrorCode::MalformedNestedDirectory,
            "nested directory reserved fields are non-zero",
        ));
    }
    let count = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    if count > MAX_NESTED_CHILDREN {
        return Err(MlError::new(
            MlErrorCode::MalformedNestedDirectory,
            "nested child count exceeds profile maximum",
        ));
    }
    let mut cursor = HEADER_BYTES;
    let mut result = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let fixed = bytes
            .get(cursor..cursor + ENTRY_FIXED_BYTES)
            .ok_or_else(|| {
                MlError::new(
                    MlErrorCode::MalformedNestedDirectory,
                    "truncated nested directory entry",
                )
            })?;
        let name_len = usize::from(u16::from_le_bytes(fixed[0..2].try_into().unwrap()));
        if name_len == 0
            || name_len > MAX_NESTED_NAME_BYTES
            || fixed[2..4].iter().any(|byte| *byte != 0)
        {
            return Err(MlError::new(
                MlErrorCode::MalformedNestedDirectory,
                "nested entry header is invalid",
            ));
        }
        let name_start = cursor + ENTRY_FIXED_BYTES;
        let name_end = name_start.checked_add(name_len).ok_or_else(|| {
            MlError::new(
                MlErrorCode::MalformedNestedDirectory,
                "nested name length overflows",
            )
        })?;
        let name_bytes = bytes.get(name_start..name_end).ok_or_else(|| {
            MlError::new(
                MlErrorCode::MalformedNestedDirectory,
                "truncated nested child name",
            )
        })?;
        let name = String::from_utf8(name_bytes.to_vec()).map_err(|_| {
            MlError::new(
                MlErrorCode::InvalidTensorName,
                "nested child name is not UTF-8",
            )
        })?;
        result.push(NestedEntry {
            name,
            key_id: u16::from_le_bytes(fixed[4..6].try_into().unwrap()),
            occurrence: u16::from_le_bytes(fixed[6..8].try_into().unwrap()),
            child_offset: u64::from_le_bytes(fixed[8..16].try_into().unwrap()),
            child_length: u64::from_le_bytes(fixed[16..24].try_into().unwrap()),
        });
        cursor = name_end;
    }
    if cursor != bytes.len() {
        return Err(MlError::new(
            MlErrorCode::MalformedNestedDirectory,
            "nested directory has trailing bytes",
        ));
    }
    Ok(result)
}
