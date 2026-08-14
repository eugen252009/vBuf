//! Deterministic downstream placement policy for profile-owned writers.
//!
//! This module chooses block order and payload alignment requests. Canonical
//! v0.6 remains responsible for block geometry, padding, and range validity.
//! It does not allocate a general memory arena or encode physical offsets into
//! semantic references.

use std::collections::BTreeSet;
use std::fmt;
use std::io::{Seek, Write};
use vbuf_core::v06::{V06Physical, V06Semantic};
use vbuf_core::writer::{BlockOptions, VBufV06Writer, WriterError};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum LayoutClass {
    Bootstrap = 0,
    ModelMetadata = 1,
    TensorDirectory = 2,
    TokenizerControl = 3,
    TokenizerPayload = 4,
    TensorPayload = 5,
    Auxiliary = 6,
}

#[derive(Clone, Copy, Debug)]
pub struct PlacementRequest<'a> {
    pub class: LayoutClass,
    /// A deterministic source-order key. It must be unique in one plan.
    pub order: u64,
    pub key_id: u16,
    pub semantic: V06Semantic,
    pub physical: V06Physical,
    pub bit_width: u16,
    pub count: u64,
    /// Minimum requested payload alignment. `BaseStep` is always the minimum.
    pub payload_alignment: u64,
    pub payload: &'a [u8],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlannedBlock {
    pub request_index: usize,
    pub key_id: u16,
    pub occurrence: u16,
    pub payload_shift: u8,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LayoutError {
    InvalidBaseShift,
    InvalidPayloadAlignment,
    AlignmentExceedsEncoding,
    DuplicateOrder,
    KeyOccurrenceOverflow,
    Writer(String),
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBaseShift => f.write_str("BaseShift is outside canonical v0.6 range"),
            Self::InvalidPayloadAlignment => f.write_str("payload alignment is not a BaseStep power-of-two refinement"),
            Self::AlignmentExceedsEncoding => f.write_str("payload alignment cannot be encoded by v0.6 payload shift"),
            Self::DuplicateOrder => f.write_str("layout request order is not unique"),
            Self::KeyOccurrenceOverflow => f.write_str("Key-ID occurrence exceeds u16"),
            Self::Writer(error) => f.write_str(error),
        }
    }
}

impl std::error::Error for LayoutError {}

impl From<WriterError> for LayoutError {
    fn from(error: WriterError) -> Self { Self::Writer(error.to_string()) }
}

#[derive(Debug)]
pub struct LayoutPlan {
    entries: Vec<PlannedBlock>,
}

impl LayoutPlan {
    pub fn build(requests: &[PlacementRequest<'_>], base_shift: u8) -> Result<Self, LayoutError> {
        if !(3..=8).contains(&base_shift) { return Err(LayoutError::InvalidBaseShift); }
        let mut indices: Vec<usize> = (0..requests.len()).collect();
        indices.sort_by_key(|index| (requests[*index].class, requests[*index].order));
        let mut orders = BTreeSet::new();
        for request in requests {
            if !orders.insert(request.order) { return Err(LayoutError::DuplicateOrder); }
        }
        let mut entries = Vec::with_capacity(indices.len());
        let mut occurrences = [0u32; 65536];
        for request_index in indices {
            let request = requests[request_index];
            let payload_shift = payload_shift(base_shift, request.payload_alignment)?;
            let occurrence = occurrences[usize::from(request.key_id)];
            if occurrence > u32::from(u16::MAX) { return Err(LayoutError::KeyOccurrenceOverflow); }
            occurrences[usize::from(request.key_id)] = occurrence + 1;
            entries.push(PlannedBlock { request_index, key_id: request.key_id, occurrence: occurrence as u16, payload_shift });
        }
        Ok(Self { entries })
    }

    pub fn entries(&self) -> &[PlannedBlock] { &self.entries }
}

pub fn payload_shift(base_shift: u8, requested_alignment: u64) -> Result<u8, LayoutError> {
    if !(3..=8).contains(&base_shift) { return Err(LayoutError::InvalidBaseShift); }
    let base_step = 1u64 << base_shift;
    if requested_alignment < base_step || !requested_alignment.is_power_of_two() || !requested_alignment.is_multiple_of(base_step) {
        return Err(LayoutError::InvalidPayloadAlignment);
    }
    let shift = requested_alignment.trailing_zeros().saturating_sub(u32::from(base_shift));
    if u32::from(base_shift) + shift > 63 { return Err(LayoutError::AlignmentExceedsEncoding); }
    u8::try_from(shift).map_err(|_| LayoutError::AlignmentExceedsEncoding)
}

pub fn canonical_payload_alignment(base_shift: u8) -> Result<u64, LayoutError> {
    if !(3..=8).contains(&base_shift) { return Err(LayoutError::InvalidBaseShift); }
    Ok(1u64 << base_shift)
}

pub fn write_known_size<W: Write + Seek>(sink: W, base_shift: u8, requests: &[PlacementRequest<'_>]) -> Result<W, LayoutError> {
    write(sink, base_shift, requests, false)
}

pub fn write_indefinite<W: Write + Seek>(sink: W, base_shift: u8, requests: &[PlacementRequest<'_>]) -> Result<W, LayoutError> {
    write(sink, base_shift, requests, true)
}

fn write<W: Write + Seek>(sink: W, base_shift: u8, requests: &[PlacementRequest<'_>], indefinite: bool) -> Result<W, LayoutError> {
    let plan = LayoutPlan::build(requests, base_shift)?;
    let mut writer = if indefinite { VBufV06Writer::new_indefinite(sink, base_shift)? } else { VBufV06Writer::new_known_size(sink, base_shift)? };
    for entry in plan.entries {
        let request = requests[entry.request_index];
        writer.write_block(BlockOptions { key_id: request.key_id, physical: request.physical, continuation: false, payload_shift: entry.payload_shift }, request.semantic, request.bit_width, request.count, request.payload)?;
    }
    Ok(writer.finish()?)
}
