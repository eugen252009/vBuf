//! Generic recursive vBuf support.
//!
//! A nested root is a complete canonical v0.6 stream carried inside a checked
//! parent payload range. This module deliberately contains no profile names,
//! directories, or application semantics.

use crate::v06::{V06Error, V06ErrorCode, ValidatedV06, parse_v06};
use vbuf_layout::CheckedRange;

/// A canonically validated child vBuf and the checked range containing it.
#[derive(Debug)]
pub struct NestedV06<'a> {
    range: CheckedRange<'a>,
    validated: ValidatedV06<'a>,
}

impl<'a> NestedV06<'a> {
    /// Validate a complete child stream located at a relative offset in a
    /// parent's payload. The child uses its own local geometry and namespace.
    pub fn from_parent_payload(
        parent: &ValidatedV06<'a>,
        parent_block: usize,
        relative_offset: u64,
        length: u64,
    ) -> Result<Self, V06Error> {
        if length == 0 {
            return Err(V06Error::global(
                V06ErrorCode::LengthMismatch,
                "nested vBuf range must be non-empty",
            ));
        }
        let range = parent.payload_subrange(parent_block, relative_offset, length)?;
        let validated = parse_v06(range.bytes()).map_err(|_| {
            V06Error::global(
                V06ErrorCode::InvalidNestedStream,
                "nested range is not a canonical v0.6 stream",
            )
        })?;
        Ok(Self { range, validated })
    }

    pub fn range(&self) -> &CheckedRange<'a> {
        &self.range
    }

    pub fn validated(&self) -> &ValidatedV06<'a> {
        &self.validated
    }

    pub fn into_parts(self) -> (CheckedRange<'a>, ValidatedV06<'a>) {
        (self.range, self.validated)
    }
}
