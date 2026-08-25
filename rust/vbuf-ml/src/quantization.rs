//! Persistent vBuf-ML provenance for representations needing auxiliary scales.

use crate::bootstrap::Bootstrap;
use crate::error::{MlError, MlErrorCode};
use crate::region_roles::RegionRole;
use crate::representations::TensorRepresentation;
use crate::tensor_directory::TensorDirectory;
use vbuf_core::v06::{V06Physical, V06Semantic, ValidatedV06};

pub const QUANTIZATION_MAGIC: [u8; 8] = *b"VBTQNT\0\0";
pub const QUANTIZATION_VERSION: u16 = 1;
const HEADER_BYTES: usize = 20;
const MAX_ENTRIES: usize = 1_000_000;

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct F8QuantizationEntry {
    pub weight_name: String,
    pub scale_name: String,
    pub block_rows: u64,
    pub block_columns: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct F8QuantizationDirectory {
    entries: Vec<F8QuantizationEntry>,
}

impl F8QuantizationDirectory {
    pub fn new(mut entries: Vec<F8QuantizationEntry>) -> Result<Self, MlError> {
        entries.sort_by(|a, b| a.weight_name.as_bytes().cmp(b.weight_name.as_bytes()));
        for entry in &entries {
            validate_entry(entry)?;
        }
        for pair in entries.windows(2) {
            if pair[0].weight_name == pair[1].weight_name {
                return Err(MlError::new(
                    MlErrorCode::InvalidQuantizationMetadata,
                    "duplicate FP8 quantization weight",
                ));
            }
        }
        Ok(Self { entries })
    }

    pub fn entries(&self) -> &[F8QuantizationEntry] {
        &self.entries
    }

    pub fn get(&self, weight_name: &str) -> Option<&F8QuantizationEntry> {
        self.entries
            .binary_search_by(|entry| entry.weight_name.as_str().cmp(weight_name))
            .ok()
            .map(|index| &self.entries[index])
    }

    pub fn validate_against(&self, directory: &TensorDirectory<'_>) -> Result<(), MlError> {
        for entry in &self.entries {
            let weight = directory.get(&entry.weight_name).ok_or_else(|| {
                MlError::new(
                    MlErrorCode::InvalidQuantizationMetadata,
                    "FP8 weight named by quantization metadata is absent",
                )
            })?;
            let scale = directory.get(&entry.scale_name).ok_or_else(|| {
                MlError::new(
                    MlErrorCode::InvalidQuantizationMetadata,
                    "FP8 scale named by quantization metadata is absent",
                )
            })?;
            if weight.representation != TensorRepresentation::F8_E4M3
                || weight.dimensions.len() != 2
                || scale.representation != TensorRepresentation::CanonicalPrimitive
                || scale.dimensions.len() != 2
            {
                return Err(MlError::new(
                    MlErrorCode::InvalidQuantizationMetadata,
                    "FP8 quantization metadata has incompatible tensor representations",
                ));
            }
            let expected = [
                weight.dimensions[0].div_ceil(entry.block_rows),
                weight.dimensions[1].div_ceil(entry.block_columns),
            ];
            if scale.dimensions.as_slice() != expected
                || scale.payload.length()
                    != expected[0].saturating_mul(expected[1]).saturating_mul(4)
            {
                return Err(MlError::new(
                    MlErrorCode::InvalidQuantizationMetadata,
                    "FP8 scale shape does not match quantization metadata",
                ));
            }
        }
        Ok(())
    }

    pub fn parse(
        validated: &ValidatedV06<'_>,
        bootstrap: &Bootstrap<'_>,
    ) -> Result<Option<Self>, MlError> {
        let Some(region) = bootstrap.region(RegionRole::QuantizationMetadata) else {
            return Ok(None);
        };
        let block = validated.blocks().get(region.block_index).ok_or_else(|| {
            MlError::new(
                MlErrorCode::QuantizationMetadataMissing,
                "quantization metadata block is absent",
            )
        })?;
        if block.semantic != V06Semantic::Opaque
            || block.physical != V06Physical::Array
            || block.bit_width != 8
            || block.continuation
        {
            return Err(MlError::new(
                MlErrorCode::RegionTypeMismatch,
                "quantization metadata must be a non-continuing opaque byte array",
            ));
        }
        Ok(Some(parse_payload(region.range.bytes())?))
    }
}

fn validate_entry(entry: &F8QuantizationEntry) -> Result<(), MlError> {
    if entry.weight_name.is_empty()
        || entry.scale_name.is_empty()
        || entry.weight_name.len() > u16::MAX as usize
        || entry.scale_name.len() > u16::MAX as usize
        || entry.block_rows == 0
        || entry.block_columns == 0
    {
        return Err(MlError::new(
            MlErrorCode::InvalidQuantizationMetadata,
            "FP8 quantization metadata entry is invalid",
        ));
    }
    Ok(())
}

pub fn encode_payload(entries: &[F8QuantizationEntry]) -> Result<Vec<u8>, MlError> {
    let directory = F8QuantizationDirectory::new(entries.to_vec())?;
    if directory.entries.len() > MAX_ENTRIES {
        return Err(MlError::new(
            MlErrorCode::MalformedQuantizationMetadata,
            "quantization metadata entry count exceeds profile maximum",
        ));
    }
    let mut output = Vec::with_capacity(HEADER_BYTES);
    output.extend_from_slice(&QUANTIZATION_MAGIC);
    output.extend_from_slice(&QUANTIZATION_VERSION.to_le_bytes());
    output.extend_from_slice(&0u16.to_le_bytes());
    output.extend_from_slice(&(directory.entries.len() as u32).to_le_bytes());
    output.extend_from_slice(&0u32.to_le_bytes());
    for entry in directory.entries {
        output.extend_from_slice(&(entry.weight_name.len() as u16).to_le_bytes());
        output.extend_from_slice(&(entry.scale_name.len() as u16).to_le_bytes());
        output.extend_from_slice(&entry.block_rows.to_le_bytes());
        output.extend_from_slice(&entry.block_columns.to_le_bytes());
        output.extend_from_slice(entry.weight_name.as_bytes());
        output.extend_from_slice(entry.scale_name.as_bytes());
    }
    Ok(output)
}

fn parse_payload(bytes: &[u8]) -> Result<F8QuantizationDirectory, MlError> {
    if bytes.len() < HEADER_BYTES || bytes[..8] != QUANTIZATION_MAGIC {
        return Err(MlError::new(
            MlErrorCode::MalformedQuantizationMetadata,
            "quantization metadata header is invalid",
        ));
    }
    let version = u16::from_le_bytes([bytes[8], bytes[9]]);
    if version != QUANTIZATION_VERSION {
        return Err(MlError::new(
            MlErrorCode::MalformedQuantizationMetadata,
            "unsupported quantization metadata version",
        ));
    }
    let count = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    if count > MAX_ENTRIES {
        return Err(MlError::new(
            MlErrorCode::MalformedQuantizationMetadata,
            "quantization metadata entry count is too large",
        ));
    }
    let mut offset = HEADER_BYTES;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        if bytes.len().saturating_sub(offset) < 20 {
            return Err(MlError::new(
                MlErrorCode::MalformedQuantizationMetadata,
                "quantization metadata entry is truncated",
            ));
        }
        let weight_len = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap()) as usize;
        let scale_len =
            u16::from_le_bytes(bytes[offset + 2..offset + 4].try_into().unwrap()) as usize;
        let block_rows = u64::from_le_bytes(bytes[offset + 4..offset + 12].try_into().unwrap());
        let block_columns = u64::from_le_bytes(bytes[offset + 12..offset + 20].try_into().unwrap());
        offset += 20;
        let total = weight_len.checked_add(scale_len).ok_or_else(|| {
            MlError::new(
                MlErrorCode::MalformedQuantizationMetadata,
                "quantization name length overflows",
            )
        })?;
        if bytes.len().saturating_sub(offset) < total {
            return Err(MlError::new(
                MlErrorCode::MalformedQuantizationMetadata,
                "quantization metadata names are truncated",
            ));
        }
        let weight_name =
            String::from_utf8(bytes[offset..offset + weight_len].to_vec()).map_err(|_| {
                MlError::new(
                    MlErrorCode::InvalidQuantizationMetadata,
                    "FP8 weight name is not UTF-8",
                )
            })?;
        offset += weight_len;
        let scale_name =
            String::from_utf8(bytes[offset..offset + scale_len].to_vec()).map_err(|_| {
                MlError::new(
                    MlErrorCode::InvalidQuantizationMetadata,
                    "FP8 scale name is not UTF-8",
                )
            })?;
        offset += scale_len;
        entries.push(F8QuantizationEntry {
            weight_name,
            scale_name,
            block_rows,
            block_columns,
        });
    }
    if offset != bytes.len() {
        return Err(MlError::new(
            MlErrorCode::MalformedQuantizationMetadata,
            "quantization metadata has trailing bytes",
        ));
    }
    F8QuantizationDirectory::new(entries)
}

/// Decode one finite-number E4M3FN byte without using host floating-point ABI details.
pub fn e4m3fn_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;
    if exponent == 0 {
        return sign * (mantissa as f32) * 2.0f32.powi(-9);
    }
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::NAN;
    }
    let mantissa_value = 1.0 + (mantissa as f32) / 8.0;
    sign * mantissa_value * 2.0f32.powi(i32::from(exponent) - 7)
}

/// Materialize one 2-D block-scaled FP8 tensor into bounded F32 output storage.
pub fn dequantize_f8_e4m3(
    payload: &[u8],
    shape: [u64; 2],
    scales: &[u8],
    block_size: [u64; 2],
) -> Result<Vec<f32>, MlError> {
    let rows = usize::try_from(shape[0]).map_err(|_| {
        MlError::new(
            MlErrorCode::ShapeOverflow,
            "FP8 row count exceeds host limits",
        )
    })?;
    let columns = usize::try_from(shape[1]).map_err(|_| {
        MlError::new(
            MlErrorCode::ShapeOverflow,
            "FP8 column count exceeds host limits",
        )
    })?;
    let block_rows = usize::try_from(block_size[0]).map_err(|_| {
        MlError::new(
            MlErrorCode::InvalidQuantizationMetadata,
            "FP8 block row size exceeds host limits",
        )
    })?;
    let block_columns = usize::try_from(block_size[1]).map_err(|_| {
        MlError::new(
            MlErrorCode::InvalidQuantizationMetadata,
            "FP8 block column size exceeds host limits",
        )
    })?;
    if block_rows == 0 || block_columns == 0 {
        return Err(MlError::new(
            MlErrorCode::InvalidQuantizationMetadata,
            "FP8 block size must be non-zero",
        ));
    }
    let elements = rows.checked_mul(columns).ok_or_else(|| {
        MlError::new(
            MlErrorCode::ShapeOverflow,
            "FP8 element count overflows host limits",
        )
    })?;
    if payload.len() != elements {
        return Err(MlError::new(
            MlErrorCode::TensorPayloadSizeMismatch,
            "FP8 payload length does not match shape",
        ));
    }
    let scale_rows = rows.div_ceil(block_rows);
    let scale_columns = columns.div_ceil(block_columns);
    let scale_count = scale_rows.checked_mul(scale_columns).ok_or_else(|| {
        MlError::new(
            MlErrorCode::ShapeOverflow,
            "FP8 scale count overflows host limits",
        )
    })?;
    if scales.len() != scale_count.checked_mul(4).unwrap_or(usize::MAX) {
        return Err(MlError::new(
            MlErrorCode::TensorPayloadSizeMismatch,
            "FP8 scale payload length does not match block geometry",
        ));
    }
    let mut output = vec![0.0f32; elements];
    for row in 0..rows {
        for column in 0..columns {
            let scale_index = (row / block_rows) * scale_columns + column / block_columns;
            let start = scale_index * 4;
            let scale = f32::from_le_bytes(scales[start..start + 4].try_into().unwrap());
            output[row * columns + column] = e4m3fn_to_f32(payload[row * columns + column]) * scale;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3fn_exhaustive_is_finite_except_defined_nan_encodings() {
        fn reference(bits: u8) -> f32 {
            let sign = if bits & 0x80 != 0 { -1.0 } else { 1.0 };
            let exponent = (bits >> 3) & 0x0f;
            let mantissa = bits & 0x07;
            if exponent == 0 {
                return sign * (mantissa as f32) / 512.0;
            }
            if exponent == 15 && mantissa == 7 {
                return f32::NAN;
            }
            sign * (1.0 + (mantissa as f32) / 8.0) * 2.0f32.powi(i32::from(exponent) - 7)
        }
        for bits in 0..=u8::MAX {
            let value = e4m3fn_to_f32(bits);
            let expected = reference(bits);
            if expected.is_nan() {
                assert!(value.is_nan());
            } else {
                assert_eq!(value, expected);
            }
        }
        assert_eq!(e4m3fn_to_f32(0x38), 1.0);
        assert_eq!(e4m3fn_to_f32(0xb8), -1.0);
        assert_eq!(e4m3fn_to_f32(0x7e), 448.0);
    }

    #[test]
    fn malformed_quantization_entries_fail_closed() {
        assert!(
            F8QuantizationDirectory::new(vec![F8QuantizationEntry {
                weight_name: "weight".into(),
                scale_name: "scale".into(),
                block_rows: 0,
                block_columns: 128,
            }])
            .is_err()
        );
    }

    #[test]
    fn block_scaled_materialization_is_row_local_and_bounded() {
        let weights = [0x38, 0x40, 0xb8, 0x00];
        let scales = [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat();
        let values = dequantize_f8_e4m3(&weights, [2, 2], &scales, [1, 2]).unwrap();
        assert_eq!(values, vec![1.0, 2.0, -2.0, 0.0]);
        let input = [3.0f32, 4.0];
        let output = [
            values[0] * input[0] + values[1] * input[1],
            values[2] * input[0] + values[3] * input[1],
        ];
        assert_eq!(output, [11.0, -6.0]);
    }
}
