//! Optional vBuf-ML payload integrity metadata.
//!
//! Integrity is downstream semantic information. Canonical v0.6 validation
//! remains authoritative and verification is always an explicit operation.

use crate::bootstrap::Bootstrap;
use crate::error::{MlError, MlErrorCode};
use crate::region_roles::RegionRole;
use sha2::{Digest, Sha256};
use vbuf_core::v06::{V06Physical, V06Semantic, ValidatedV06};
use vbuf_layout::CheckedRange;

pub const INTEGRITY_MAGIC: [u8; 8] = *b"VBINTG\0\0";
pub const INTEGRITY_VERSION: u16 = 1;
pub const MAX_INTEGRITY_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_INTEGRITY_ENTRIES: usize = 1_000_000;
const HEADER_BYTES: usize = 20;
const ENTRY_BYTES: usize = 40;
const DIGEST_BYTES: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum IntegrityAlgorithm {
    Sha256 = 1,
}

impl IntegrityAlgorithm {
    fn from_id(id: u8) -> Result<Self, MlError> {
        match id {
            1 => Ok(Self::Sha256),
            _ => Err(MlError::new(MlErrorCode::UnsupportedIntegrityAlgorithm, "unsupported integrity algorithm")),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntegrityEntry {
    pub key_id: u16,
    pub occurrence: u16,
    pub digest: [u8; DIGEST_BYTES],
}

impl IntegrityEntry {
    pub const fn new(key_id: u16, occurrence: u16, digest: [u8; DIGEST_BYTES]) -> Self { Self { key_id, occurrence, digest } }
}

#[derive(Debug)]
pub struct IntegrityRecord<'a> {
    pub algorithm: IntegrityAlgorithm,
    pub key_id: u16,
    pub occurrence: u16,
    pub block_index: usize,
    /// Payload-only coverage; canonical headers and padding are excluded.
    pub range: CheckedRange<'a>,
    expected_digest: [u8; DIGEST_BYTES],
}

#[derive(Debug)]
pub struct IntegrityMetadata<'a> {
    records: Vec<IntegrityRecord<'a>>,
}

impl<'a> IntegrityMetadata<'a> {
    /// Returns `Ok(None)` when the optional bootstrap role is absent.
    pub fn discover(validated: &ValidatedV06<'a>, bootstrap: &Bootstrap<'a>) -> Result<Option<Self>, MlError> {
        let Some(region) = bootstrap.region(RegionRole::IntegrityMetadata) else { return Ok(None); };
        let block = validated.blocks().get(region.block_index).ok_or_else(|| MlError::new(MlErrorCode::IntegrityTargetMissing, "integrity metadata block index is absent"))?;
        if block.semantic != V06Semantic::Opaque || block.physical != V06Physical::Array || block.bit_width != 8 || block.continuation {
            return Err(MlError::new(MlErrorCode::MalformedIntegrityMetadata, "integrity metadata must be a non-continuing opaque byte array"));
        }
        let entries = parse_payload(region.range.bytes())?;
        let mut records = Vec::with_capacity(entries.len());
        let mut previous = None;
        for entry in entries {
            if let Some((key_id, occurrence)) = previous
                && (entry.key_id, entry.occurrence) <= (key_id, occurrence)
            {
                return Err(MlError::new(if (entry.key_id, entry.occurrence) == (key_id, occurrence) { MlErrorCode::DuplicateIntegrityTarget } else { MlErrorCode::MalformedIntegrityMetadata }, "integrity targets must be strictly sorted"));
            }
            previous = Some((entry.key_id, entry.occurrence));
            let target_index = validated.blocks().iter().enumerate().filter(|(_, candidate)| candidate.key_id == entry.key_id).nth(entry.occurrence as usize).map(|(index, _)| index).ok_or_else(|| MlError::new(MlErrorCode::IntegrityTargetMissing, "integrity target is absent"))?;
            let target = &validated.blocks()[target_index];
            if target.continuation { return Err(MlError::new(MlErrorCode::IntegrityTargetUnsupported, "continuation integrity targets are not supported by profile 0.1")); }
            records.push(IntegrityRecord { algorithm: IntegrityAlgorithm::Sha256, key_id: entry.key_id, occurrence: entry.occurrence, block_index: target_index, range: validated.payload_range(target_index)?, expected_digest: entry.digest });
        }
        Ok(Some(Self { records }))
    }

    pub fn records(&self) -> &[IntegrityRecord<'a>] { &self.records }

    pub fn record(&self, key_id: u16, occurrence: u16) -> Option<&IntegrityRecord<'a>> { self.records.iter().find(|record| record.key_id == key_id && record.occurrence == occurrence) }

    /// Verifies only the selected canonical payload range. This never performs
    /// a global scan and does not mutate portable or runtime state.
    pub fn verify_target(&self, key_id: u16, occurrence: u16) -> Result<(), MlError> {
        let record = self.record(key_id, occurrence).ok_or_else(|| MlError::new(MlErrorCode::NoIntegrityAvailable, "no integrity record covers target"))?;
        let actual = digest_payload(record.range.bytes());
        if actual != record.expected_digest { return Err(MlError::new(MlErrorCode::IntegrityMismatch, "integrity digest does not match canonical payload")); }
        Ok(())
    }
}

pub fn digest_payload(bytes: &[u8]) -> [u8; DIGEST_BYTES] {
    let digest = Sha256::digest(bytes);
    let mut output = [0u8; DIGEST_BYTES];
    output.copy_from_slice(&digest);
    output
}

pub fn encode_payload(entries: &[IntegrityEntry]) -> Result<Vec<u8>, MlError> {
    if entries.len() > MAX_INTEGRITY_ENTRIES { return Err(MlError::new(MlErrorCode::MalformedIntegrityMetadata, "integrity entry count exceeds profile maximum")); }
    let mut ordered: Vec<&IntegrityEntry> = entries.iter().collect();
    ordered.sort_by_key(|entry| (entry.key_id, entry.occurrence));
    for pair in ordered.windows(2) { if (pair[0].key_id, pair[0].occurrence) == (pair[1].key_id, pair[1].occurrence) { return Err(MlError::new(MlErrorCode::DuplicateIntegrityTarget, "integrity target occurs more than once")); } }
    let total = HEADER_BYTES.checked_add(ordered.len().checked_mul(ENTRY_BYTES).ok_or_else(|| MlError::new(MlErrorCode::MalformedIntegrityMetadata, "integrity size overflows"))?).ok_or_else(|| MlError::new(MlErrorCode::MalformedIntegrityMetadata, "integrity size overflows"))?;
    if total > MAX_INTEGRITY_BYTES { return Err(MlError::new(MlErrorCode::MalformedIntegrityMetadata, "integrity metadata exceeds profile maximum")); }
    let mut output = Vec::with_capacity(total);
    output.extend_from_slice(&INTEGRITY_MAGIC); output.extend_from_slice(&INTEGRITY_VERSION.to_le_bytes()); output.extend_from_slice(&0u16.to_le_bytes()); output.extend_from_slice(&(ordered.len() as u32).to_le_bytes()); output.extend_from_slice(&0u32.to_le_bytes());
    for entry in ordered { output.extend_from_slice(&entry.key_id.to_le_bytes()); output.extend_from_slice(&entry.occurrence.to_le_bytes()); output.push(IntegrityAlgorithm::Sha256 as u8); output.push(0); output.extend_from_slice(&0u16.to_le_bytes()); output.extend_from_slice(&entry.digest); }
    Ok(output)
}

fn parse_payload(bytes: &[u8]) -> Result<Vec<IntegrityEntry>, MlError> {
    if bytes.len() < HEADER_BYTES || bytes.len() > MAX_INTEGRITY_BYTES || bytes[..8] != INTEGRITY_MAGIC { return Err(MlError::new(MlErrorCode::MalformedIntegrityMetadata, "integrity metadata header is invalid")); }
    if u16::from_le_bytes(bytes[8..10].try_into().unwrap()) != INTEGRITY_VERSION || u16::from_le_bytes(bytes[10..12].try_into().unwrap()) != 0 || bytes[16..20].iter().any(|byte| *byte != 0) { return Err(MlError::new(MlErrorCode::MalformedIntegrityMetadata, "integrity metadata header is invalid")); }
    let count = usize::try_from(u32::from_le_bytes(bytes[12..16].try_into().unwrap())).map_err(|_| MlError::new(MlErrorCode::MalformedIntegrityMetadata, "integrity count exceeds host range"))?;
    let expected = HEADER_BYTES.checked_add(count.checked_mul(ENTRY_BYTES).ok_or_else(|| MlError::new(MlErrorCode::MalformedIntegrityMetadata, "integrity size overflows"))?).ok_or_else(|| MlError::new(MlErrorCode::MalformedIntegrityMetadata, "integrity size overflows"))?;
    if count > MAX_INTEGRITY_ENTRIES || expected != bytes.len() { return Err(MlError::new(MlErrorCode::MalformedIntegrityMetadata, "integrity length does not match entry count")); }
    let mut entries = Vec::with_capacity(count);
    for index in 0..count {
        let offset = HEADER_BYTES + index * ENTRY_BYTES;
        if IntegrityAlgorithm::from_id(bytes[offset + 4])? != IntegrityAlgorithm::Sha256 || bytes[offset + 5] != 0 || bytes[offset + 6..offset + 8].iter().any(|byte| *byte != 0) { return Err(MlError::new(MlErrorCode::InvalidIntegrityDigest, "integrity entry algorithm, flags, or reserved fields are invalid")); }
        let mut digest = [0u8; DIGEST_BYTES]; digest.copy_from_slice(&bytes[offset + 8..offset + 8 + DIGEST_BYTES]);
        entries.push(IntegrityEntry::new(u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap()), u16::from_le_bytes(bytes[offset + 2..offset + 4].try_into().unwrap()), digest));
    }
    Ok(entries)
}
