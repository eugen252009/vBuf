//! Checked generic vBuf v0.6 reader.
//!
//! Parsing produces validated block descriptors before any payload reference is
//! exposed. Canonical headers and ranges remain authoritative.

use memmap2::Mmap;
use std::fmt;
use std::fs::File;
use std::mem::{align_of, size_of};
use std::path::Path;
use vbuf_layout::{ByteRange, CheckedRange, RangeError};

pub const V06_MAGIC: [u8; 4] = *b"VBUF";
pub const V06_VERSION: u32 = 0x0006_0000;
const MIN_HEADER_SIZE: u64 = 24;
const FLAG_INDEFINITE: u8 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum V06ErrorCode {
    TooShort = 1,
    BadMagic = 2,
    UnsupportedVersion = 3,
    InvalidBaseShift = 4,
    InvalidFlags = 5,
    InvalidHeaderSize = 6,
    InvalidReserved = 7,
    InvalidExtension = 8,
    UnknownRequiredExtension = 9,
    LengthMismatch = 10,
    ArithmeticOverflow = 11,
    TruncatedBlock = 12,
    InvalidAnchor = 13,
    InvalidRepresentation = 14,
    NonCanonicalCount = 15,
    InvalidContinuation = 16,
    NonZeroPadding = 17,
    Misaligned = 18,
    NotFound = 19,
    TypeMismatch = 20,
    HostUnsupported = 21,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V06Error {
    pub code: V06ErrorCode,
    pub offset: Option<u64>,
    pub detail: &'static str,
}

impl V06Error {
    fn at(code: V06ErrorCode, offset: u64, detail: &'static str) -> Self {
        Self {
            code,
            offset: Some(offset),
            detail,
        }
    }

    fn global(code: V06ErrorCode, detail: &'static str) -> Self {
        Self {
            code,
            offset: None,
            detail,
        }
    }
}

impl fmt::Display for V06Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.offset {
            Some(offset) => write!(f, "{} at byte {offset}", self.detail),
            None => f.write_str(self.detail),
        }
    }
}

impl std::error::Error for V06Error {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum V06Semantic {
    Unsigned = 0,
    Float = 1,
    Signed = 2,
    Opaque = 3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum V06Physical {
    Scalar = 0,
    Array = 1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V06Header {
    pub base_shift: u8,
    pub base_step: u64,
    pub indefinite: bool,
    pub header_size: u16,
    pub data_region_start: u64,
    pub data_region_size: u64,
    pub data_region_end: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V06Block {
    pub block_start: u64,
    pub payload_start: u64,
    pub payload_len: u64,
    pub payload_end: u64,
    pub next_block_start: u64,
    pub key_id: u16,
    pub semantic: V06Semantic,
    pub physical: V06Physical,
    pub continuation: bool,
    pub bit_width: u16,
    pub count: u64,
    pub payload_alignment: u64,
}

#[derive(Debug)]
pub struct ValidatedV06<'a> {
    bytes: &'a [u8],
    header: V06Header,
    blocks: Vec<V06Block>,
}

fn range_error(error: RangeError) -> V06Error {
    let code = match error {
        RangeError::InvalidAlignment | RangeError::Misaligned => V06ErrorCode::Misaligned,
        RangeError::OutsideMapping | RangeError::OutsideParent | RangeError::HostIndexOverflow => {
            V06ErrorCode::LengthMismatch
        }
        RangeError::Overflow | RangeError::EndBeforeStart => V06ErrorCode::ArithmeticOverflow,
    };
    V06Error::global(code, "checked physical range is invalid")
}

fn make_checked_range<'a>(
    bytes: &'a [u8],
    start: u64,
    end: u64,
) -> Result<CheckedRange<'a>, V06Error> {
    let range = ByteRange::from_end(start, end).map_err(range_error)?;
    CheckedRange::from_mapping(bytes, range).map_err(range_error)
}

#[derive(Debug)]
pub struct VBufV06 {
    mmap: Mmap,
    header: V06Header,
    blocks: Vec<V06Block>,
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, V06Error> {
    let raw = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| V06Error::at(V06ErrorCode::TooShort, offset as u64, "truncated u16"))?;
    Ok(u16::from_le_bytes(raw.try_into().expect("length checked")))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, V06Error> {
    let raw = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| V06Error::at(V06ErrorCode::TooShort, offset as u64, "truncated u32"))?;
    Ok(u32::from_le_bytes(raw.try_into().expect("length checked")))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, V06Error> {
    let raw = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| V06Error::at(V06ErrorCode::TooShort, offset as u64, "truncated u64"))?;
    Ok(u64::from_le_bytes(raw.try_into().expect("length checked")))
}

fn checked_add(a: u64, b: u64) -> Result<u64, V06Error> {
    a.checked_add(b)
        .ok_or_else(|| V06Error::global(V06ErrorCode::ArithmeticOverflow, "u64 addition overflow"))
}

fn checked_mul(a: u64, b: u64) -> Result<u64, V06Error> {
    a.checked_mul(b).ok_or_else(|| {
        V06Error::global(
            V06ErrorCode::ArithmeticOverflow,
            "u64 multiplication overflow",
        )
    })
}

fn align_up(value: u64, alignment: u64) -> Result<u64, V06Error> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(V06Error::global(
            V06ErrorCode::Misaligned,
            "invalid alignment",
        ));
    }
    Ok(checked_add(value, alignment - 1)? & !(alignment - 1))
}

fn to_usize(value: u64) -> Result<usize, V06Error> {
    usize::try_from(value).map_err(|_| {
        V06Error::global(
            V06ErrorCode::ArithmeticOverflow,
            "offset exceeds host usize",
        )
    })
}

fn require_zero(bytes: &[u8], start: u64, end: u64) -> Result<(), V06Error> {
    let start_usize = to_usize(start)?;
    let end_usize = to_usize(end)?;
    let range = bytes.get(start_usize..end_usize).ok_or_else(|| {
        V06Error::at(
            V06ErrorCode::LengthMismatch,
            start,
            "padding lies outside input",
        )
    })?;
    if let Some(relative) = range.iter().position(|byte| *byte != 0) {
        return Err(V06Error::at(
            V06ErrorCode::NonZeroPadding,
            checked_add(
                start,
                u64::try_from(relative).map_err(|_| {
                    V06Error::global(
                        V06ErrorCode::ArithmeticOverflow,
                        "padding offset exceeds u64",
                    )
                })?,
            )?,
            "non-zero canonical padding",
        ));
    }
    Ok(())
}

fn validate_representation(
    semantic: u8,
    physical: u8,
    bit_width: u16,
    count: u64,
    offset: u64,
) -> Result<(V06Semantic, V06Physical), V06Error> {
    let semantic = match semantic {
        0 if matches!(bit_width, 8 | 16 | 32 | 64) => V06Semantic::Unsigned,
        1 if matches!(bit_width, 32 | 64) => V06Semantic::Float,
        2 if matches!(bit_width, 8 | 16 | 32 | 64) => V06Semantic::Signed,
        3 if bit_width == 8 => V06Semantic::Opaque,
        _ => {
            return Err(V06Error::at(
                V06ErrorCode::InvalidRepresentation,
                offset,
                "invalid semantic/bit-width combination",
            ));
        }
    };
    let physical = match physical {
        0 if count == 1 => V06Physical::Scalar,
        1 => V06Physical::Array,
        _ => {
            return Err(V06Error::at(
                V06ErrorCode::InvalidRepresentation,
                offset,
                "invalid physical/count combination",
            ));
        }
    };
    Ok((semantic, physical))
}

pub fn parse_v06(bytes: &[u8]) -> Result<ValidatedV06<'_>, V06Error> {
    if bytes.len() < MIN_HEADER_SIZE as usize {
        return Err(V06Error::global(
            V06ErrorCode::TooShort,
            "v0.6 header is shorter than 24 bytes",
        ));
    }
    if bytes[0..4] != V06_MAGIC {
        return Err(V06Error::global(
            V06ErrorCode::BadMagic,
            "invalid vBuf magic",
        ));
    }
    if read_u32(bytes, 4)? != V06_VERSION {
        return Err(V06Error::global(
            V06ErrorCode::UnsupportedVersion,
            "unsupported vBuf version",
        ));
    }

    let base_shift = bytes[8];
    if !(3..=8).contains(&base_shift) {
        return Err(V06Error::global(
            V06ErrorCode::InvalidBaseShift,
            "BaseShift is outside 3..=8",
        ));
    }
    let base_step = 1u64.checked_shl(u32::from(base_shift)).ok_or_else(|| {
        V06Error::global(V06ErrorCode::InvalidBaseShift, "BaseStep shift overflow")
    })?;

    let flags = bytes[9];
    if flags & !FLAG_INDEFINITE != 0 {
        return Err(V06Error::global(
            V06ErrorCode::InvalidFlags,
            "unknown v0.6 global flags",
        ));
    }
    let indefinite = flags & FLAG_INDEFINITE != 0;
    let header_size = read_u16(bytes, 10)?;
    if header_size < MIN_HEADER_SIZE as u16 || header_size % 8 != 0 {
        return Err(V06Error::global(
            V06ErrorCode::InvalidHeaderSize,
            "invalid v0.6 HeaderSize",
        ));
    }
    let header_size_u64 = u64::from(header_size);
    if header_size as usize > bytes.len() {
        return Err(V06Error::global(
            V06ErrorCode::TooShort,
            "truncated extended global header",
        ));
    }
    if read_u32(bytes, 12)? != 0 {
        return Err(V06Error::global(
            V06ErrorCode::InvalidReserved,
            "global reserved field is non-zero",
        ));
    }
    let declared_size = read_u64(bytes, 16)?;
    if indefinite && declared_size != 0 {
        return Err(V06Error::global(
            V06ErrorCode::LengthMismatch,
            "indefinite stream has non-zero DataRegionSize",
        ));
    }

    let mut extension_offset = MIN_HEADER_SIZE;
    while extension_offset < header_size_u64 {
        let offset = to_usize(extension_offset)?;
        let _extension_type = read_u16(bytes, offset)?;
        let extension_flags = read_u16(bytes, offset + 2)?;
        let extension_size = u64::from(read_u32(bytes, offset + 4)?);
        if extension_flags & !1 != 0 || extension_size < 8 || extension_size % 8 != 0 {
            return Err(V06Error::at(
                V06ErrorCode::InvalidExtension,
                extension_offset,
                "invalid extension framing",
            ));
        }
        let extension_end = checked_add(extension_offset, extension_size)?;
        if extension_end > header_size_u64 {
            return Err(V06Error::at(
                V06ErrorCode::InvalidExtension,
                extension_offset,
                "extension exceeds HeaderSize",
            ));
        }
        if extension_flags & 1 != 0 {
            return Err(V06Error::at(
                V06ErrorCode::UnknownRequiredExtension,
                extension_offset,
                "unknown required extension",
            ));
        }
        extension_offset = extension_end;
    }
    if extension_offset != header_size_u64 {
        return Err(V06Error::global(
            V06ErrorCode::InvalidExtension,
            "extensions do not fill HeaderSize",
        ));
    }

    let data_region_start = align_up(header_size_u64, base_step)?;
    if data_region_start % base_step != 0 {
        return Err(V06Error::global(
            V06ErrorCode::Misaligned,
            "unaligned data region",
        ));
    }
    let physical_len = u64::try_from(bytes.len()).map_err(|_| {
        V06Error::global(V06ErrorCode::ArithmeticOverflow, "input length exceeds u64")
    })?;
    if data_region_start > physical_len {
        return Err(V06Error::global(
            V06ErrorCode::TooShort,
            "data-region start is outside input",
        ));
    }
    require_zero(bytes, header_size_u64, data_region_start)?;

    let data_region_end = if indefinite {
        physical_len
    } else {
        let end = checked_add(data_region_start, declared_size)?;
        if end != physical_len {
            return Err(V06Error::global(
                V06ErrorCode::LengthMismatch,
                "known DataRegionSize does not equal physical input",
            ));
        }
        end
    };
    let data_region_size = data_region_end - data_region_start;

    let mut blocks = Vec::new();
    let mut block_start = data_region_start;
    let mut required_continuation_key = None;
    while block_start < data_region_end {
        if block_start % base_step != 0 || (block_start - data_region_start) % base_step != 0 {
            return Err(V06Error::at(
                V06ErrorCode::Misaligned,
                block_start,
                "unaligned canonical block start",
            ));
        }
        let anchor_end = checked_add(block_start, 8)?;
        if anchor_end > data_region_end {
            return Err(V06Error::at(
                V06ErrorCode::TruncatedBlock,
                block_start,
                "truncated block anchor",
            ));
        }
        let anchor = read_u64(bytes, to_usize(block_start)?)?;
        if anchor == 0 {
            return Err(V06Error::at(
                V06ErrorCode::InvalidAnchor,
                block_start,
                "zero block anchor",
            ));
        }
        let semantic_code = (anchor & 0x0f) as u8;
        let physical_code = ((anchor >> 4) & 0x0f) as u8;
        let continuation = anchor & (1 << 8) != 0;
        let count64 = anchor & (1 << 9) != 0;
        let payload_shift = ((anchor >> 10) & 0x3f) as u8;
        let key_id = ((anchor >> 16) & 0xffff) as u16;
        let bit_width = ((anchor >> 32) & 0xffff) as u16;
        let inline_count = (anchor >> 48) & 0xffff;

        if let Some(required_key) = required_continuation_key.take()
            && required_key != key_id
        {
            return Err(V06Error::at(
                V06ErrorCode::InvalidContinuation,
                block_start,
                "continued block does not match the preceding KeyID",
            ));
        }

        let (count, block_header_size) = if count64 {
            let extended_end = checked_add(block_start, 16)?;
            if extended_end > data_region_end {
                return Err(V06Error::at(
                    V06ErrorCode::TruncatedBlock,
                    block_start,
                    "truncated extended count",
                ));
            }
            if inline_count != 0 {
                return Err(V06Error::at(
                    V06ErrorCode::NonCanonicalCount,
                    block_start,
                    "extended count has non-zero inline count",
                ));
            }
            let count_offset = checked_add(block_start, 8)?;
            let count = read_u64(bytes, to_usize(count_offset)?)?;
            if count <= 65535 {
                return Err(V06Error::at(
                    V06ErrorCode::NonCanonicalCount,
                    block_start,
                    "small count uses extended form",
                ));
            }
            (count, 16)
        } else {
            (inline_count, 8)
        };

        let (semantic, physical) =
            validate_representation(semantic_code, physical_code, bit_width, count, block_start)?;

        let combined_shift = u16::from(base_shift) + u16::from(payload_shift);
        if combined_shift > 63 {
            return Err(V06Error::at(
                V06ErrorCode::Misaligned,
                block_start,
                "payload alignment shift exceeds 63",
            ));
        }
        let payload_alignment = 1u64 << combined_shift;
        if payload_alignment < base_step || !payload_alignment.is_multiple_of(base_step) {
            return Err(V06Error::at(
                V06ErrorCode::Misaligned,
                block_start,
                "payload alignment contradicts BaseStep",
            ));
        }
        let header_end = checked_add(block_start, block_header_size)?;
        let payload_start = align_up(header_end, payload_alignment)?;
        require_zero(bytes, header_end, payload_start)?;

        let payload_bits = checked_mul(count, u64::from(bit_width))?;
        let payload_len = payload_bits / 8 + u64::from(payload_bits % 8 != 0);
        let payload_end = checked_add(payload_start, payload_len)?;
        if payload_end > data_region_end {
            return Err(V06Error::at(
                V06ErrorCode::TruncatedBlock,
                block_start,
                "declared payload exceeds data region",
            ));
        }
        if payload_start % base_step != 0 || payload_start % payload_alignment != 0 {
            return Err(V06Error::at(
                V06ErrorCode::Misaligned,
                payload_start,
                "payload is misaligned",
            ));
        }

        let next_block_start = align_up(payload_end, base_step)?;
        blocks.push(V06Block {
            block_start,
            payload_start,
            payload_len,
            payload_end,
            next_block_start,
            key_id,
            semantic,
            physical,
            continuation,
            bit_width,
            count,
            payload_alignment,
        });
        if continuation && (payload_end == data_region_end || next_block_start >= data_region_end) {
            return Err(V06Error::at(
                V06ErrorCode::InvalidContinuation,
                block_start,
                "final physical block cannot continue",
            ));
        }
        required_continuation_key = continuation.then_some(key_id);

        if payload_end == data_region_end {
            break;
        }
        if next_block_start >= data_region_end {
            return Err(V06Error::at(
                V06ErrorCode::LengthMismatch,
                payload_end,
                "final tail padding is not canonical",
            ));
        }
        require_zero(bytes, payload_end, next_block_start)?;
        block_start = next_block_start;
    }

    Ok(ValidatedV06 {
        bytes,
        header: V06Header {
            base_shift,
            base_step,
            indefinite,
            header_size,
            data_region_start,
            data_region_size,
            data_region_end,
        },
        blocks,
    })
}

impl<'a> ValidatedV06<'a> {
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }
    pub fn header(&self) -> &V06Header {
        &self.header
    }
    pub fn blocks(&self) -> &[V06Block] {
        &self.blocks
    }

    /// Returns the validated physical bytes of one canonical block. For a
    /// non-final block this includes canonical padding up to the next block;
    /// the final block ends at its payload end because no tail padding is
    /// required by v0.6.
    pub fn block_range(&self, index: usize) -> Result<CheckedRange<'a>, V06Error> {
        let block = self
            .blocks
            .get(index)
            .ok_or_else(|| V06Error::global(V06ErrorCode::NotFound, "block index not found"))?;
        let end = if block.payload_end == self.header.data_region_end {
            block.payload_end
        } else {
            block.next_block_start
        };
        make_checked_range(self.bytes, block.block_start, end)
    }

    /// Returns only the validated payload bytes of one canonical block.
    pub fn payload_range(&self, index: usize) -> Result<CheckedRange<'a>, V06Error> {
        let block = self
            .blocks
            .get(index)
            .ok_or_else(|| V06Error::global(V06ErrorCode::NotFound, "block index not found"))?;
        make_checked_range(self.bytes, block.payload_start, block.payload_end)
    }

    /// Refines a validated payload range without permitting escape from it.
    pub fn payload_subrange(
        &self,
        index: usize,
        relative_offset: u64,
        length: u64,
    ) -> Result<CheckedRange<'a>, V06Error> {
        self.payload_range(index)?
            .refine(relative_offset, length)
            .map_err(range_error)
    }

    pub fn block(&self, key_id: u16, occurrence: usize) -> Option<&V06Block> {
        self.blocks
            .iter()
            .filter(|block| block.key_id == key_id)
            .nth(occurrence)
    }

    pub fn opaque_bytes(&self, key_id: u16, occurrence: usize) -> Result<&'a [u8], V06Error> {
        let block = self.block(key_id, occurrence).ok_or_else(|| {
            V06Error::global(V06ErrorCode::NotFound, "KeyID occurrence not found")
        })?;
        if block.semantic != V06Semantic::Opaque || block.bit_width != 8 {
            return Err(V06Error::at(
                V06ErrorCode::TypeMismatch,
                block.block_start,
                "block is not opaque bytes",
            ));
        }
        payload_range(self.bytes, block)
    }
}

fn payload_range<'a>(bytes: &'a [u8], block: &V06Block) -> Result<&'a [u8], V06Error> {
    Ok(make_checked_range(bytes, block.payload_start, block.payload_end)?.bytes())
}

fn typed_view<'a, T>(
    bytes: &'a [u8],
    block: &V06Block,
    semantic: V06Semantic,
    bit_width: u16,
) -> Result<&'a [T], V06Error> {
    if cfg!(target_endian = "big") && size_of::<T>() > 1 {
        return Err(V06Error::global(
            V06ErrorCode::HostUnsupported,
            "zero-copy typed views require a little-endian host",
        ));
    }
    if block.semantic != semantic || block.bit_width != bit_width {
        return Err(V06Error::at(
            V06ErrorCode::TypeMismatch,
            block.block_start,
            "wire representation does not match requested native type",
        ));
    }
    let payload = payload_range(bytes, block)?;
    let count = usize::try_from(block.count).map_err(|_| {
        V06Error::global(V06ErrorCode::ArithmeticOverflow, "count exceeds host usize")
    })?;
    let expected = count.checked_mul(size_of::<T>()).ok_or_else(|| {
        V06Error::global(
            V06ErrorCode::ArithmeticOverflow,
            "native view length overflow",
        )
    })?;
    if expected != payload.len() {
        return Err(V06Error::at(
            V06ErrorCode::LengthMismatch,
            block.payload_start,
            "typed view length differs from validated payload",
        ));
    }
    let pointer = payload.as_ptr();
    if !(pointer as usize).is_multiple_of(align_of::<T>()) {
        return Err(V06Error::at(
            V06ErrorCode::Misaligned,
            block.payload_start,
            "native typed-view alignment is not satisfied",
        ));
    }

    // SAFETY: `payload_range` proves the complete range is inside `bytes`;
    // `expected == payload.len()` proves `count` elements fit; the pointer
    // alignment was checked immediately above; callers reach this helper only
    // through a concrete primitive method whose wire semantic/width is checked;
    // all bit patterns of the selected integer/IEEE float primitive are valid;
    // the returned lifetime is tied to `bytes`; and only a shared slice is made.
    Ok(unsafe { std::slice::from_raw_parts(pointer.cast::<T>(), count) })
}

macro_rules! primitive_view {
    ($name:ident, $ty:ty, $semantic:expr, $width:expr) => {
        pub fn $name(&self, key_id: u16, occurrence: usize) -> Result<&[$ty], V06Error> {
            let block = self.block(key_id, occurrence).ok_or_else(|| {
                V06Error::global(V06ErrorCode::NotFound, "KeyID occurrence not found")
            })?;
            typed_view(self.bytes(), block, $semantic, $width)
        }
    };
}

impl ValidatedV06<'_> {
    primitive_view!(u8_view, u8, V06Semantic::Unsigned, 8);
    primitive_view!(u16_view, u16, V06Semantic::Unsigned, 16);
    primitive_view!(u32_view, u32, V06Semantic::Unsigned, 32);
    primitive_view!(u64_view, u64, V06Semantic::Unsigned, 64);
    primitive_view!(i8_view, i8, V06Semantic::Signed, 8);
    primitive_view!(i16_view, i16, V06Semantic::Signed, 16);
    primitive_view!(i32_view, i32, V06Semantic::Signed, 32);
    primitive_view!(i64_view, i64, V06Semantic::Signed, 64);
    primitive_view!(f32_view, f32, V06Semantic::Float, 32);
    primitive_view!(f64_view, f64, V06Semantic::Float, 64);
}

impl VBufV06 {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, V06Error> {
        let file = File::open(path)
            .map_err(|_| V06Error::global(V06ErrorCode::TooShort, "cannot open v0.6 file"))?;
        // SAFETY: the read-only mapping is owned by the returned `VBufV06` and
        // no mutable alias is created by this reader.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|_| V06Error::global(V06ErrorCode::TooShort, "cannot map v0.6 file"))?;
        let validated = parse_v06(&mmap)?;
        let header = validated.header;
        let blocks = validated.blocks;
        Ok(Self {
            mmap,
            header,
            blocks,
        })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.mmap
    }
    pub fn header(&self) -> &V06Header {
        &self.header
    }
    pub fn blocks(&self) -> &[V06Block] {
        &self.blocks
    }

    pub fn block_range(&self, index: usize) -> Result<CheckedRange<'_>, V06Error> {
        let block = self
            .blocks
            .get(index)
            .ok_or_else(|| V06Error::global(V06ErrorCode::NotFound, "block index not found"))?;
        let end = if block.payload_end == self.header.data_region_end {
            block.payload_end
        } else {
            block.next_block_start
        };
        make_checked_range(self.bytes(), block.block_start, end)
    }

    pub fn payload_range(&self, index: usize) -> Result<CheckedRange<'_>, V06Error> {
        let block = self
            .blocks
            .get(index)
            .ok_or_else(|| V06Error::global(V06ErrorCode::NotFound, "block index not found"))?;
        make_checked_range(self.bytes(), block.payload_start, block.payload_end)
    }

    pub fn payload_subrange(
        &self,
        index: usize,
        relative_offset: u64,
        length: u64,
    ) -> Result<CheckedRange<'_>, V06Error> {
        self.payload_range(index)?
            .refine(relative_offset, length)
            .map_err(range_error)
    }

    pub fn block(&self, key_id: u16, occurrence: usize) -> Option<&V06Block> {
        self.blocks
            .iter()
            .filter(|block| block.key_id == key_id)
            .nth(occurrence)
    }

    pub fn opaque_bytes(&self, key_id: u16, occurrence: usize) -> Result<&[u8], V06Error> {
        let block = self.block(key_id, occurrence).ok_or_else(|| {
            V06Error::global(V06ErrorCode::NotFound, "KeyID occurrence not found")
        })?;
        if block.semantic != V06Semantic::Opaque || block.bit_width != 8 {
            return Err(V06Error::at(
                V06ErrorCode::TypeMismatch,
                block.block_start,
                "block is not opaque bytes",
            ));
        }
        payload_range(self.bytes(), block)
    }
}

macro_rules! owned_primitive_view {
    ($name:ident, $ty:ty, $semantic:expr, $width:expr) => {
        pub fn $name(&self, key_id: u16, occurrence: usize) -> Result<&[$ty], V06Error> {
            let block = self.block(key_id, occurrence).ok_or_else(|| {
                V06Error::global(V06ErrorCode::NotFound, "KeyID occurrence not found")
            })?;
            typed_view(self.bytes(), block, $semantic, $width)
        }
    };
}

impl VBufV06 {
    owned_primitive_view!(u8_view, u8, V06Semantic::Unsigned, 8);
    owned_primitive_view!(u16_view, u16, V06Semantic::Unsigned, 16);
    owned_primitive_view!(u32_view, u32, V06Semantic::Unsigned, 32);
    owned_primitive_view!(u64_view, u64, V06Semantic::Unsigned, 64);
    owned_primitive_view!(i8_view, i8, V06Semantic::Signed, 8);
    owned_primitive_view!(i16_view, i16, V06Semantic::Signed, 16);
    owned_primitive_view!(i32_view, i32, V06Semantic::Signed, 32);
    owned_primitive_view!(i64_view, i64, V06Semantic::Signed, 64);
    owned_primitive_view!(f32_view, f32, V06Semantic::Float, 32);
    owned_primitive_view!(f64_view, f64, V06Semantic::Float, 64);
}

#[cfg(test)]
mod arithmetic_tests {
    use super::*;

    #[test]
    fn near_u64_max_payload_and_alignment_operations_fail_closed() {
        assert_eq!(
            checked_add(u64::MAX - 4, 8).unwrap_err().code,
            V06ErrorCode::ArithmeticOverflow
        );
        assert_eq!(
            checked_mul(u64::MAX, 64).unwrap_err().code,
            V06ErrorCode::ArithmeticOverflow
        );
        assert_eq!(
            align_up(u64::MAX - 3, 8).unwrap_err().code,
            V06ErrorCode::ArithmeticOverflow
        );
    }
}
