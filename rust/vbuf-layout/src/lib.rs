//! Wire-neutral checked byte ranges.
//!
//! This crate knows nothing about vBuf wire fields or downstream domains. A
//! `CheckedRange` only proves that a range is contained by the borrowed byte
//! mapping supplied by its creator; canonical provenance is supplied by the
//! validated container that creates it.

use std::fmt;
use std::ops::Range;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RangeError {
    Overflow,
    EndBeforeStart,
    OutsideMapping,
    OutsideParent,
    HostIndexOverflow,
    InvalidAlignment,
    Misaligned,
}

impl fmt::Display for RangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Overflow => "byte-range arithmetic overflow",
            Self::EndBeforeStart => "byte-range end precedes start",
            Self::OutsideMapping => "byte range lies outside mapping",
            Self::OutsideParent => "child range lies outside parent range",
            Self::HostIndexOverflow => "byte range cannot be represented by host indexing",
            Self::InvalidAlignment => "alignment is not a non-zero power of two",
            Self::Misaligned => "byte range is not aligned",
        })
    }
}

impl std::error::Error for RangeError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ByteRange {
    offset: u64,
    length: u64,
}

impl ByteRange {
    pub const fn new(offset: u64, length: u64) -> Result<Self, RangeError> {
        if offset.checked_add(length).is_none() {
            return Err(RangeError::Overflow);
        }
        Ok(Self { offset, length })
    }

    pub const fn from_end(offset: u64, end: u64) -> Result<Self, RangeError> {
        if end < offset {
            return Err(RangeError::EndBeforeStart);
        }
        Self::new(offset, end - offset)
    }

    pub const fn offset(self) -> u64 { self.offset }
    pub const fn length(self) -> u64 { self.length }
    pub const fn end(self) -> u64 { self.offset + self.length }

    pub fn contains(self, child: Self) -> bool {
        child.offset >= self.offset && child.end() <= self.end()
    }

    pub fn intersection(self, other: Self) -> Option<Self> {
        let start = self.offset.max(other.offset);
        let end = self.end().min(other.end());
        (start <= end).then(|| Self { offset: start, length: end - start })
    }

    pub fn refine(self, relative_offset: u64, length: u64) -> Result<Self, RangeError> {
        let offset = self.offset.checked_add(relative_offset).ok_or(RangeError::Overflow)?;
        let child = Self::new(offset, length)?;
        if self.contains(child) { Ok(child) } else { Err(RangeError::OutsideParent) }
    }

    pub fn is_offset_aligned(self, alignment: u64) -> Result<bool, RangeError> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(RangeError::InvalidAlignment);
        }
        Ok(self.offset.is_multiple_of(alignment))
    }

    pub fn host_range(self, mapping_len: usize) -> Result<Range<usize>, RangeError> {
        if self.end() > mapping_len as u64 {
            return Err(RangeError::OutsideMapping);
        }
        let start = usize::try_from(self.offset).map_err(|_| RangeError::HostIndexOverflow)?;
        let end = usize::try_from(self.end()).map_err(|_| RangeError::HostIndexOverflow)?;
        Ok(start..end)
    }
}

/// A range borrowed from one byte mapping. The lifetime prevents the mapping
/// from being dropped while this view exists.
#[derive(Debug, Eq, PartialEq)]
pub struct CheckedRange<'a> {
    bytes: &'a [u8],
    range: ByteRange,
}

impl<'a> CheckedRange<'a> {
    /// Constructs a bounded range from an already selected mapping. Canonical
    /// callers should keep this constructor behind their validated-container
    /// API so arbitrary offsets cannot acquire canonical provenance.
    pub fn from_mapping(bytes: &'a [u8], range: ByteRange) -> Result<Self, RangeError> {
        range.host_range(bytes.len())?;
        Ok(Self { bytes, range })
    }

    pub fn range(&self) -> ByteRange { self.range }
    pub fn offset(&self) -> u64 { self.range.offset() }
    pub fn length(&self) -> u64 { self.range.length() }
    pub fn end(&self) -> u64 { self.range.end() }
    pub fn bytes(&self) -> &'a [u8] {
        let host = self.range.host_range(self.bytes.len()).expect("checked range remains bounded");
        &self.bytes[host]
    }

    pub fn refine(&self, relative_offset: u64, length: u64) -> Result<CheckedRange<'a>, RangeError> {
        Self::from_mapping(self.bytes, self.range.refine(relative_offset, length)?)
    }

    pub fn require_alignment(&self, alignment: u64) -> Result<(), RangeError> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(RangeError::InvalidAlignment);
        }
        let address = (self.bytes.as_ptr() as usize)
            .checked_add(usize::try_from(self.offset()).map_err(|_| RangeError::HostIndexOverflow)?)
            .ok_or(RangeError::HostIndexOverflow)?;
        if !(address as u64).is_multiple_of(alignment) {
            return Err(RangeError::Misaligned);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_arithmetic_and_refinement_fail_closed() {
        assert_eq!(ByteRange::new(u64::MAX, 1), Err(RangeError::Overflow));
        let parent = ByteRange::new(10, 20).unwrap();
        assert_eq!(parent.refine(20, 1), Err(RangeError::OutsideParent));
        assert_eq!(parent.refine(u64::MAX, 1), Err(RangeError::Overflow));
        assert_eq!(parent.intersection(ByteRange::new(30, 2).unwrap()).unwrap().length(), 0);
    }

    #[test]
    fn mapping_and_host_conversion_are_checked() {
        let near_four_gib = ByteRange::new(1u64 << 32, 8).unwrap();
        assert_eq!(near_four_gib.end(), (1u64 << 32) + 8);
        assert_eq!(near_four_gib.host_range(0), Err(RangeError::OutsideMapping));
        let bytes = [1, 2, 3, 4];
        let range = CheckedRange::from_mapping(&bytes, ByteRange::new(1, 2).unwrap()).unwrap();
        assert_eq!(range.bytes(), &[2, 3]);
        assert_eq!(range.refine(1, 1).unwrap().bytes(), &[3]);
        assert_eq!(CheckedRange::from_mapping(&bytes, ByteRange::new(3, 2).unwrap()), Err(RangeError::OutsideMapping));
    }
}
