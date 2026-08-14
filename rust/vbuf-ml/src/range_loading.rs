//! Runtime-local partial loading over canonical validated ranges.
//!
//! Semantic selection, physical planning, source execution, and integrity are
//! deliberately separate. This module contains no portable wire state.

use std::fmt;
use std::fs::File;
use std::marker::PhantomData;
use std::io;
use std::path::Path;
use vbuf_layout::{ByteRange, RangeError};
use crate::tensor_directory::{TensorDescriptor, TensorDirectory};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Coalescing {
    None,
    ExactAdjacent,
    Gap(u64),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhysicalRange {
    pub offset: u64,
    pub length: u64,
}

impl PhysicalRange {
    pub fn end(self) -> Result<u64, RangeLoadError> { self.offset.checked_add(self.length).ok_or(RangeLoadError::ArithmeticOverflow) }
    fn from_byte_range(range: ByteRange) -> Self { Self { offset: range.offset(), length: range.length() } }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelectedTensor<'source> {
    ordinal: usize,
    key_id: u16,
    occurrence: u16,
    payload: ByteRange,
    _provenance: PhantomData<&'source ()>,
}

impl SelectedTensor<'_> {
    pub fn ordinal(&self) -> usize { self.ordinal }
    pub fn key_id(&self) -> u16 { self.key_id }
    pub fn occurrence(&self) -> u16 { self.occurrence }
    pub fn payload_range(&self) -> ByteRange { self.payload }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlannedTarget<'source> {
    pub selected: SelectedTensor<'source>,
    pub physical_read_index: usize,
    pub relative_offset: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadPlan<'source> {
    targets: Vec<PlannedTarget<'source>>,
    reads: Vec<PhysicalRange>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RangeLoadError {
    ArithmeticOverflow,
    InvalidRange,
    OutsideSource,
    HostIndexOverflow,
    SelectionOutOfBounds,
    Io,
}

impl fmt::Display for RangeLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ArithmeticOverflow => "partial-loading arithmetic overflow",
            Self::InvalidRange => "invalid partial-loading range",
            Self::OutsideSource => "partial-loading range lies outside source",
            Self::HostIndexOverflow => "partial-loading range exceeds host indexing",
            Self::SelectionOutOfBounds => "tensor selection is out of bounds",
            Self::Io => "positioned range read failed",
        })
    }
}

impl std::error::Error for RangeLoadError {}

impl From<RangeError> for RangeLoadError {
    fn from(error: RangeError) -> Self {
        match error {
            RangeError::Overflow | RangeError::EndBeforeStart => Self::ArithmeticOverflow,
            RangeError::OutsideMapping | RangeError::OutsideParent => Self::OutsideSource,
            RangeError::HostIndexOverflow => Self::HostIndexOverflow,
            RangeError::InvalidAlignment | RangeError::Misaligned => Self::InvalidRange,
        }
    }
}

pub fn select_tensor_names<'source>(directory: &'source TensorDirectory<'_>, names: &[&str]) -> Result<Vec<SelectedTensor<'source>>, RangeLoadError> {
    let mut selected = Vec::with_capacity(names.len());
    for name in names {
        let tensor = directory.get(name).ok_or(RangeLoadError::SelectionOutOfBounds)?;
        selected.push(selected_from_descriptor(directory, tensor)?);
    }
    Ok(selected)
}

pub fn select_tensor_ordinals<'source>(directory: &'source TensorDirectory<'_>, ordinals: &[usize]) -> Result<Vec<SelectedTensor<'source>>, RangeLoadError> {
    let mut selected = Vec::with_capacity(ordinals.len());
    for &ordinal in ordinals {
        let tensor = directory.tensors().get(ordinal).ok_or(RangeLoadError::SelectionOutOfBounds)?;
        selected.push(SelectedTensor { ordinal, key_id: tensor.key_id, occurrence: tensor.occurrence, payload: tensor.range.range(), _provenance: PhantomData });
    }
    Ok(selected)
}

fn selected_from_descriptor<'source>(directory: &'source TensorDirectory<'_>, tensor: &TensorDescriptor<'_>) -> Result<SelectedTensor<'source>, RangeLoadError> {
    let ordinal = directory.tensors().iter().position(|candidate| std::ptr::eq(candidate, tensor)).ok_or(RangeLoadError::SelectionOutOfBounds)?;
    Ok(SelectedTensor { ordinal, key_id: tensor.key_id, occurrence: tensor.occurrence, payload: tensor.range.range(), _provenance: PhantomData })
}

impl<'source> ReadPlan<'source> {
    pub fn build(selected: &[SelectedTensor<'source>], coalescing: Coalescing) -> Result<Self, RangeLoadError> {
        if selected.is_empty() { return Ok(Self { targets: Vec::new(), reads: Vec::new() }); }
        let mut order: Vec<(usize, ByteRange)> = selected.iter().enumerate().map(|(index, target)| (index, target.payload)).collect();
        order.sort_by_key(|(_, range)| (range.offset(), range.end()));
        let mut reads: Vec<PhysicalRange> = Vec::new();
        let mut owners: Vec<(usize, usize, u64)> = Vec::new();
        for (selection_index, range) in order {
            let candidate = PhysicalRange::from_byte_range(range);
            let mut merged = false;
            if let Some((read_index, current)) = reads.iter_mut().enumerate().next_back() {
                let current_end = current.end()?;
                let gap = candidate.offset.saturating_sub(current_end);
                let overlap = candidate.offset < current_end;
                let allowed = overlap || match coalescing { Coalescing::None => false, Coalescing::ExactAdjacent => candidate.offset <= current_end, Coalescing::Gap(limit) => gap <= limit };
                if allowed {
                    let new_end = current_end.max(candidate.end()?);
                    current.length = new_end.checked_sub(current.offset).ok_or(RangeLoadError::ArithmeticOverflow)?;
                    owners.push((selection_index, read_index, range.offset().checked_sub(current.offset).ok_or(RangeLoadError::ArithmeticOverflow)?));
                    merged = true;
                }
            }
            if !merged {
                let read_index = reads.len();
                reads.push(candidate);
                owners.push((selection_index, read_index, 0));
            }
        }
        let mut targets = vec![None; selected.len()];
        for (selection_index, read_index, relative_offset) in owners { targets[selection_index] = Some(PlannedTarget { selected: selected[selection_index], physical_read_index: read_index, relative_offset }); }
        Ok(Self { targets: targets.into_iter().map(|target| target.expect("every selected target has a read range")).collect(), reads })
    }

    pub fn targets(&self) -> &[PlannedTarget<'_>] { &self.targets }
    pub fn reads(&self) -> &[PhysicalRange] { &self.reads }
    pub fn semantic_bytes(&self) -> u64 { self.targets.iter().map(|target| target.selected.payload.length()).sum() }
    pub fn physical_bytes(&self) -> u64 { self.reads.iter().map(|read| read.length).sum() }
}

pub trait RangeSource {
    fn len(&self) -> u64;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> Result<(), RangeLoadError>;
}

#[derive(Debug)]
pub struct LoadedRead {
    pub range: PhysicalRange,
    pub bytes: Vec<u8>,
}

#[derive(Debug)]
pub struct LoadedPlan {
    reads: Vec<LoadedRead>,
}

impl LoadedPlan {
    pub fn reads(&self) -> &[LoadedRead] { &self.reads }
    pub fn target_bytes(&self, target: &PlannedTarget) -> Result<&[u8], RangeLoadError> {
        let read = self.reads.get(target.physical_read_index).ok_or(RangeLoadError::InvalidRange)?;
        let start = usize::try_from(target.relative_offset).map_err(|_| RangeLoadError::HostIndexOverflow)?;
        let length = usize::try_from(target.selected.payload.length()).map_err(|_| RangeLoadError::HostIndexOverflow)?;
        let end = start.checked_add(length).ok_or(RangeLoadError::ArithmeticOverflow)?;
        read.bytes.get(start..end).ok_or(RangeLoadError::OutsideSource)
    }
}

pub fn execute_plan<S: RangeSource>(source: &S, plan: &ReadPlan<'_>) -> Result<LoadedPlan, RangeLoadError> {
    let mut reads = Vec::with_capacity(plan.reads.len());
    for &range in &plan.reads {
        let end = range.end()?;
        if end > source.len() { return Err(RangeLoadError::OutsideSource); }
        let length = usize::try_from(range.length).map_err(|_| RangeLoadError::HostIndexOverflow)?;
        let mut bytes = vec![0u8; length];
        source.read_exact_at(range.offset, &mut bytes)?;
        reads.push(LoadedRead { range, bytes });
    }
    Ok(LoadedPlan { reads })
}

#[derive(Debug)]
pub struct MmapSource {
    mmap: memmap2::Mmap,
}

impl MmapSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, RangeLoadError> {
        let file = File::open(path).map_err(|_| RangeLoadError::Io)?;
        let mmap = unsafe { memmap2::Mmap::map(&file).map_err(|_| RangeLoadError::Io)? };
        Ok(Self { mmap })
    }

    pub fn checked_slice(&self, range: ByteRange) -> Result<&[u8], RangeLoadError> { Ok(&self.mmap[range.host_range(self.mmap.len())?]) }
}

impl RangeSource for MmapSource {
    fn len(&self) -> u64 { self.mmap.len() as u64 }
    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> Result<(), RangeLoadError> {
        let range = ByteRange::new(offset, u64::try_from(destination.len()).map_err(|_| RangeLoadError::HostIndexOverflow)?)?;
        destination.copy_from_slice(self.checked_slice(range)?);
        Ok(())
    }
}

#[derive(Debug)]
pub struct PositionedFileSource {
    file: File,
    length: u64,
}

impl PositionedFileSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, RangeLoadError> {
        let file = File::open(path).map_err(|_| RangeLoadError::Io)?;
        let length = file.metadata().map_err(|_| RangeLoadError::Io)?.len();
        Ok(Self { file, length })
    }
}

impl RangeSource for PositionedFileSource {
    fn len(&self) -> u64 { self.length }
    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> Result<(), RangeLoadError> {
        let mut done = 0usize;
        while done < destination.len() {
            let position = offset.checked_add(done as u64).ok_or(RangeLoadError::ArithmeticOverflow)?;
            let read = positioned_read(&self.file, &mut destination[done..], position).map_err(|_| RangeLoadError::Io)?;
            if read == 0 { return Err(RangeLoadError::Io); }
            done += read;
        }
        Ok(())
    }
}

#[cfg(unix)]
fn positioned_read(file: &File, destination: &mut [u8], offset: u64) -> io::Result<usize> { std::os::unix::fs::FileExt::read_at(file, destination, offset) }

#[cfg(windows)]
fn positioned_read(file: &File, destination: &mut [u8], offset: u64) -> io::Result<usize> { std::os::windows::fs::FileExt::seek_read(file, destination, offset) }

#[cfg(not(any(unix, windows)))]
fn positioned_read(_file: &File, _destination: &mut [u8], _offset: u64) -> io::Result<usize> { Err(io::Error::other("positioned reads are unsupported on this platform")) }

#[cfg(test)]
mod tests {
    use super::*;

    fn target(ordinal: usize, offset: u64, length: u64) -> SelectedTensor<'static> {
        SelectedTensor { ordinal, key_id: ordinal as u16, occurrence: 0, payload: ByteRange::new(offset, length).unwrap(), _provenance: PhantomData }
    }

    #[test]
    fn coalescing_merges_only_the_configured_physical_ranges() {
        let a = target(0, 10, 4);
        let b = target(1, 14, 3);
        let c = target(2, 18, 2);
        let exact = ReadPlan::build(&[a, b, c], Coalescing::ExactAdjacent).unwrap();
        assert_eq!(exact.reads(), &[PhysicalRange { offset: 10, length: 7 }, PhysicalRange { offset: 18, length: 2 }]);
        let gap = ReadPlan::build(&[a, b, c], Coalescing::Gap(1)).unwrap();
        assert_eq!(gap.reads(), &[PhysicalRange { offset: 10, length: 10 }]);
    }
}
