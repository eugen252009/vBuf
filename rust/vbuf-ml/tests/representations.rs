use vbuf_core::v06::{V06Block, V06Physical, V06Semantic};
use vbuf_ml::{
    BF16_BYTES_PER_ELEMENT, F8_E4M3_BYTES_PER_ELEMENT, IQ1_S_BLOCK_BYTES, IQ1_S_BLOCK_ELEMENTS,
    IQ2_S_BLOCK_BYTES, IQ2_S_BLOCK_ELEMENTS, IQ2_XS_BLOCK_BYTES, IQ2_XS_BLOCK_ELEMENTS,
    IQ2_XXS_BLOCK_BYTES, IQ2_XXS_BLOCK_ELEMENTS, IQ4_NL_BLOCK_BYTES, IQ4_NL_BLOCK_ELEMENTS,
    IQ4_XS_BLOCK_BYTES, IQ4_XS_BLOCK_ELEMENTS, MlErrorCode, Q2_K_BLOCK_BYTES, Q2_K_BLOCK_ELEMENTS,
    Q3_K_BLOCK_BYTES, Q3_K_BLOCK_ELEMENTS, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS, Q4_K_BLOCK_BYTES,
    Q4_K_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS, TensorRepresentation,
    bf16_bits_to_f32, expected_payload_bytes, validate_tensor_representation,
};

fn packed_block(payload_len: u64) -> V06Block {
    V06Block {
        block_start: 0,
        payload_start: 0,
        payload_len,
        payload_end: payload_len,
        next_block_start: payload_len,
        key_id: 1,
        semantic: V06Semantic::Opaque,
        physical: V06Physical::Array,
        continuation: false,
        bit_width: 8,
        count: payload_len,
        payload_alignment: 1,
    }
}

#[test]
fn canonical_f32_remains_the_generic_primitive_contract() {
    let block = V06Block {
        semantic: V06Semantic::Float,
        physical: V06Physical::Array,
        bit_width: 32,
        count: 6,
        payload_len: 24,
        ..packed_block(24)
    };
    validate_tensor_representation(TensorRepresentation::CanonicalPrimitive, &[2, 3], &block)
        .unwrap();
}

#[test]
fn bf16_has_exact_two_byte_storage_and_known_bits() {
    assert_eq!(BF16_BYTES_PER_ELEMENT, 2);
    assert_eq!(
        expected_payload_bytes(TensorRepresentation::Bf16, &[2, 3]).unwrap(),
        12
    );
    assert_eq!(bf16_bits_to_f32(0x0000), 0.0);
    assert_eq!(bf16_bits_to_f32(0x3f80), 1.0);
    assert_eq!(bf16_bits_to_f32(0xbf80), -1.0);
    let block = packed_block(12);
    validate_tensor_representation(TensorRepresentation::Bf16, &[2, 3], &block).unwrap();
    assert_eq!(
        validate_tensor_representation(TensorRepresentation::Bf16, &[2, 3], &packed_block(11))
            .unwrap_err()
            .code,
        MlErrorCode::TensorRepresentationMismatch
    );
}

#[test]
fn f8_e4m3_is_one_byte_opaque_storage() {
    assert_eq!(F8_E4M3_BYTES_PER_ELEMENT, 1);
    assert_eq!(
        expected_payload_bytes(TensorRepresentation::F8_E4M3, &[2, 3]).unwrap(),
        6
    );
    validate_tensor_representation(TensorRepresentation::F8_E4M3, &[2, 3], &packed_block(6))
        .unwrap();
    assert_eq!(
        validate_tensor_representation(TensorRepresentation::F8_E4M3, &[2, 3], &packed_block(5))
            .unwrap_err()
            .code,
        MlErrorCode::TensorRepresentationMismatch
    );
}

#[test]
fn q8_0_zero_block_fixture_matches_pinned_geometry() {
    // ggml quantize_row_q8_0_ref over 32 zero floats produces d = 0 and 32 zero q values.
    let mut block = vec![0u8; Q8_0_BLOCK_BYTES as usize];
    assert_eq!(block, vec![0; 34]);
    assert_eq!(
        expected_payload_bytes(TensorRepresentation::GgmlQ8_0, &[Q8_0_BLOCK_ELEMENTS]).unwrap(),
        34
    );
    validate_tensor_representation(
        TensorRepresentation::GgmlQ8_0,
        &[Q8_0_BLOCK_ELEMENTS],
        &packed_block(34),
    )
    .unwrap();
    block.extend_from_slice(&[0; 34]);
    assert_eq!(
        expected_payload_bytes(TensorRepresentation::GgmlQ8_0, &[64, 2]).unwrap(),
        136
    );
    validate_tensor_representation(TensorRepresentation::GgmlQ8_0, &[64, 2], &packed_block(136))
        .unwrap();
}

#[test]
fn q8_0_requires_innermost_row_divisibility() {
    assert_eq!(
        expected_payload_bytes(TensorRepresentation::GgmlQ8_0, &[31])
            .unwrap_err()
            .code,
        MlErrorCode::InvalidQuantizedShape
    );
}

#[test]
fn selected_low_bit_contracts_have_pinned_block_geometry() {
    for (representation, elements, bytes) in [
        (
            TensorRepresentation::GgmlQ4_0,
            Q4_0_BLOCK_ELEMENTS,
            Q4_0_BLOCK_BYTES,
        ),
        (
            TensorRepresentation::GgmlQ2_K,
            Q2_K_BLOCK_ELEMENTS,
            Q2_K_BLOCK_BYTES,
        ),
        (
            TensorRepresentation::GgmlIQ1_S,
            IQ1_S_BLOCK_ELEMENTS,
            IQ1_S_BLOCK_BYTES,
        ),
        (
            TensorRepresentation::GgmlQ4_K,
            Q4_K_BLOCK_ELEMENTS,
            Q4_K_BLOCK_BYTES,
        ),
        (
            TensorRepresentation::GgmlIQ4_NL,
            IQ4_NL_BLOCK_ELEMENTS,
            IQ4_NL_BLOCK_BYTES,
        ),
        (
            TensorRepresentation::GgmlIQ4_XS,
            IQ4_XS_BLOCK_ELEMENTS,
            IQ4_XS_BLOCK_BYTES,
        ),
        (
            TensorRepresentation::GgmlQ3_K,
            Q3_K_BLOCK_ELEMENTS,
            Q3_K_BLOCK_BYTES,
        ),
        (
            TensorRepresentation::GgmlIQ2_XXS,
            IQ2_XXS_BLOCK_ELEMENTS,
            IQ2_XXS_BLOCK_BYTES,
        ),
        (
            TensorRepresentation::GgmlIQ2_XS,
            IQ2_XS_BLOCK_ELEMENTS,
            IQ2_XS_BLOCK_BYTES,
        ),
        (
            TensorRepresentation::GgmlIQ2_S,
            IQ2_S_BLOCK_ELEMENTS,
            IQ2_S_BLOCK_BYTES,
        ),
    ] {
        assert_eq!(
            expected_payload_bytes(representation, &[elements]).unwrap(),
            bytes
        );
        validate_tensor_representation(representation, &[elements], &packed_block(bytes)).unwrap();
        assert_eq!(
            expected_payload_bytes(representation, &[elements - 1])
                .unwrap_err()
                .code,
            MlErrorCode::InvalidQuantizedShape
        );
    }
}

#[test]
fn packed_payload_length_and_storage_are_checked() {
    let mut block = packed_block(34);
    block.count = 33;
    assert_eq!(
        validate_tensor_representation(TensorRepresentation::GgmlQ8_0, &[32], &block)
            .unwrap_err()
            .code,
        MlErrorCode::TensorRepresentationMismatch
    );
    block = packed_block(34);
    block.semantic = V06Semantic::Unsigned;
    assert_eq!(
        validate_tensor_representation(TensorRepresentation::GgmlQ8_0, &[32], &block)
            .unwrap_err()
            .code,
        MlErrorCode::TensorRepresentationMismatch
    );
}
