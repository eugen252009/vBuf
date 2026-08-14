//! Canonical portable vBuf v0.6 writer.

use crate::v06::{V06_MAGIC, V06_VERSION, V06Physical, V06Semantic};
use std::fmt;
use std::io::{Seek, SeekFrom, Write};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockOptions {
    pub key_id: u16,
    pub physical: V06Physical,
    pub continuation: bool,
    pub payload_shift: u8,
}

impl BlockOptions {
    pub fn array(key_id: u16) -> Self {
        Self {
            key_id,
            physical: V06Physical::Array,
            continuation: false,
            payload_shift: 0,
        }
    }

    pub fn scalar(key_id: u16) -> Self {
        Self {
            key_id,
            physical: V06Physical::Scalar,
            continuation: false,
            payload_shift: 0,
        }
    }
}

#[derive(Debug)]
pub enum WriterError {
    Io(std::io::Error),
    InvalidBaseShift,
    InvalidPayloadShift,
    InvalidRepresentation,
    ArithmeticOverflow,
    PayloadLengthMismatch,
    ContinuationMismatch,
    UnterminatedContinuation,
    NonEmptySink,
}

impl fmt::Display for WriterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::InvalidBaseShift => f.write_str("BaseShift is outside 3..=8"),
            Self::InvalidPayloadShift => f.write_str("payload alignment shift exceeds 63"),
            Self::InvalidRepresentation => {
                f.write_str("invalid semantic/physical/count/width combination")
            }
            Self::ArithmeticOverflow => f.write_str("v0.6 writer arithmetic overflow"),
            Self::PayloadLengthMismatch => {
                f.write_str("payload length does not match count and width")
            }
            Self::ContinuationMismatch => f.write_str("continued block has a different KeyID"),
            Self::UnterminatedContinuation => {
                f.write_str("final physical block has Continuation set")
            }
            Self::NonEmptySink => f.write_str("canonical writer requires an empty sink"),
        }
    }
}

impl std::error::Error for WriterError {}

impl From<std::io::Error> for WriterError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StreamMode {
    KnownSize,
    Indefinite,
}

pub struct VBufV06Writer<W: Write + Seek> {
    inner: W,
    base_shift: u8,
    base_step: u64,
    data_region_start: u64,
    mode: StreamMode,
    blocks_written: u64,
    required_continuation_key: Option<u16>,
}

fn checked_add(a: u64, b: u64) -> Result<u64, WriterError> {
    a.checked_add(b).ok_or(WriterError::ArithmeticOverflow)
}

fn checked_mul(a: u64, b: u64) -> Result<u64, WriterError> {
    a.checked_mul(b).ok_or(WriterError::ArithmeticOverflow)
}

fn align_up(value: u64, alignment: u64) -> Result<u64, WriterError> {
    Ok(checked_add(value, alignment - 1)? & !(alignment - 1))
}

fn validate_representation(
    semantic: V06Semantic,
    physical: V06Physical,
    width: u16,
    count: u64,
) -> Result<(), WriterError> {
    let semantic_valid = match semantic {
        V06Semantic::Unsigned | V06Semantic::Signed => matches!(width, 8 | 16 | 32 | 64),
        V06Semantic::Float => matches!(width, 32 | 64),
        V06Semantic::Opaque => width == 8,
    };
    let physical_valid = match physical {
        V06Physical::Scalar => count == 1,
        V06Physical::Array => true,
    };
    if semantic_valid && physical_valid {
        Ok(())
    } else {
        Err(WriterError::InvalidRepresentation)
    }
}

impl<W: Write + Seek> VBufV06Writer<W> {
    pub fn new_known_size(inner: W, base_shift: u8) -> Result<Self, WriterError> {
        Self::new(inner, base_shift, StreamMode::KnownSize)
    }

    pub fn new_indefinite(inner: W, base_shift: u8) -> Result<Self, WriterError> {
        Self::new(inner, base_shift, StreamMode::Indefinite)
    }

    fn new(mut inner: W, base_shift: u8, mode: StreamMode) -> Result<Self, WriterError> {
        if !(3..=8).contains(&base_shift) {
            return Err(WriterError::InvalidBaseShift);
        }
        let base_step = 1u64 << base_shift;
        let data_region_start = align_up(24, base_step)?;
        let initial_position = inner.stream_position()?;
        let existing_length = inner.seek(SeekFrom::End(0))?;
        if initial_position != 0 || existing_length != 0 {
            return Err(WriterError::NonEmptySink);
        }
        inner.seek(SeekFrom::Start(0))?;
        let mut header = [0u8; 24];
        header[0..4].copy_from_slice(&V06_MAGIC);
        header[4..8].copy_from_slice(&V06_VERSION.to_le_bytes());
        header[8] = base_shift;
        header[9] = u8::from(mode == StreamMode::Indefinite);
        header[10..12].copy_from_slice(&24u16.to_le_bytes());
        inner.write_all(&header)?;
        write_zeros(&mut inner, data_region_start - 24)?;
        Ok(Self {
            inner,
            base_shift,
            base_step,
            data_region_start,
            mode,
            blocks_written: 0,
            required_continuation_key: None,
        })
    }

    pub fn write_block(
        &mut self,
        options: BlockOptions,
        semantic: V06Semantic,
        bit_width: u16,
        count: u64,
        payload: &[u8],
    ) -> Result<(), WriterError> {
        validate_representation(semantic, options.physical, bit_width, count)?;
        if let Some(required) = self.required_continuation_key
            && required != options.key_id
        {
            return Err(WriterError::ContinuationMismatch);
        }
        let combined_shift = u16::from(self.base_shift) + u16::from(options.payload_shift);
        if combined_shift > 63 {
            return Err(WriterError::InvalidPayloadShift);
        }
        let payload_bits = checked_mul(count, u64::from(bit_width))?;
        let expected_payload = payload_bits / 8 + u64::from(payload_bits % 8 != 0);
        if usize::try_from(expected_payload).ok() != Some(payload.len()) {
            return Err(WriterError::PayloadLengthMismatch);
        }

        let mut position = self.inner.stream_position()?;
        if self.blocks_written == 0 {
            if position != self.data_region_start {
                return Err(WriterError::Io(std::io::Error::other(
                    "unexpected first block offset",
                )));
            }
        } else {
            let block_start = align_up(position, self.base_step)?;
            write_zeros(&mut self.inner, block_start - position)?;
            position = block_start;
        }
        if !position.is_multiple_of(self.base_step) {
            return Err(WriterError::Io(std::io::Error::other(
                "unaligned block start",
            )));
        }

        let extended = count > 65535;
        let inline_count = if extended { 0 } else { count };
        let mut anchor = semantic as u64;
        anchor |= (options.physical as u64) << 4;
        anchor |= u64::from(options.continuation) << 8;
        anchor |= u64::from(extended) << 9;
        anchor |= u64::from(options.payload_shift) << 10;
        anchor |= u64::from(options.key_id) << 16;
        anchor |= u64::from(bit_width) << 32;
        anchor |= inline_count << 48;
        self.inner.write_all(&anchor.to_le_bytes())?;
        if extended {
            self.inner.write_all(&count.to_le_bytes())?;
        }

        let header_end = self.inner.stream_position()?;
        let payload_alignment = 1u64 << combined_shift;
        let payload_start = align_up(header_end, payload_alignment)?;
        write_zeros(&mut self.inner, payload_start - header_end)?;
        self.inner.write_all(payload)?;
        self.blocks_written = checked_add(self.blocks_written, 1)?;
        self.required_continuation_key = options.continuation.then_some(options.key_id);
        Ok(())
    }

    pub fn finish(mut self) -> Result<W, WriterError> {
        if self.required_continuation_key.is_some() {
            return Err(WriterError::UnterminatedContinuation);
        }
        let end = self.inner.stream_position()?;
        if self.mode == StreamMode::KnownSize {
            let data_size = end
                .checked_sub(self.data_region_start)
                .ok_or(WriterError::ArithmeticOverflow)?;
            self.inner.seek(SeekFrom::Start(16))?;
            self.inner.write_all(&data_size.to_le_bytes())?;
            self.inner.seek(SeekFrom::Start(end))?;
        }
        self.inner.flush()?;
        Ok(self.inner)
    }

    pub fn write_u8(&mut self, options: BlockOptions, values: &[u8]) -> Result<(), WriterError> {
        self.write_block(
            options,
            V06Semantic::Unsigned,
            8,
            slice_count(values)?,
            values,
        )
    }

    pub fn write_i8(&mut self, options: BlockOptions, values: &[i8]) -> Result<(), WriterError> {
        let mut payload = Vec::new();
        payload
            .try_reserve_exact(values.len())
            .map_err(|_| WriterError::ArithmeticOverflow)?;
        payload.extend(values.iter().map(|value| value.to_le_bytes()[0]));
        self.write_block(
            options,
            V06Semantic::Signed,
            8,
            slice_count(values)?,
            &payload,
        )
    }

    pub fn write_opaque(
        &mut self,
        options: BlockOptions,
        values: &[u8],
    ) -> Result<(), WriterError> {
        self.write_block(
            options,
            V06Semantic::Opaque,
            8,
            slice_count(values)?,
            values,
        )
    }
}

macro_rules! portable_method {
    ($name:ident, $ty:ty, $semantic:expr, $width:expr) => {
        impl<W: Write + Seek> VBufV06Writer<W> {
            pub fn $name(
                &mut self,
                options: BlockOptions,
                values: &[$ty],
            ) -> Result<(), WriterError> {
                let byte_len = values
                    .len()
                    .checked_mul(std::mem::size_of::<$ty>())
                    .ok_or(WriterError::ArithmeticOverflow)?;
                let mut payload = Vec::new();
                payload
                    .try_reserve_exact(byte_len)
                    .map_err(|_| WriterError::ArithmeticOverflow)?;
                for value in values {
                    payload.extend_from_slice(&value.to_le_bytes());
                }
                self.write_block(options, $semantic, $width, slice_count(values)?, &payload)
            }
        }
    };
}

portable_method!(write_u16, u16, V06Semantic::Unsigned, 16);
portable_method!(write_u32, u32, V06Semantic::Unsigned, 32);
portable_method!(write_u64, u64, V06Semantic::Unsigned, 64);
portable_method!(write_i16, i16, V06Semantic::Signed, 16);
portable_method!(write_i32, i32, V06Semantic::Signed, 32);
portable_method!(write_i64, i64, V06Semantic::Signed, 64);
portable_method!(write_f32, f32, V06Semantic::Float, 32);
portable_method!(write_f64, f64, V06Semantic::Float, 64);

fn slice_count<T>(values: &[T]) -> Result<u64, WriterError> {
    u64::try_from(values.len()).map_err(|_| WriterError::ArithmeticOverflow)
}

fn write_zeros<W: Write>(writer: &mut W, count: u64) -> Result<(), WriterError> {
    const ZEROES: [u8; 4096] = [0; 4096];
    let mut remaining = count;
    while remaining > 0 {
        let chunk = remaining.min(ZEROES.len() as u64) as usize;
        writer.write_all(&ZEROES[..chunk])?;
        remaining -= chunk as u64;
    }
    Ok(())
}
