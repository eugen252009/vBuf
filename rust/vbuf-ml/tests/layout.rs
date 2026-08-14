use std::io::Cursor;
use vbuf_core::v06::{parse_v06, V06Physical, V06Semantic};
use vbuf_ml::layout::{canonical_payload_alignment, payload_shift, write_indefinite, write_known_size, LayoutClass, LayoutError, LayoutPlan, PlacementRequest};

fn requests(alignment: u64) -> Vec<PlacementRequest<'static>> {
    vec![
        PlacementRequest { class: LayoutClass::TensorPayload, order: 2, key_id: 7, semantic: V06Semantic::Opaque, physical: V06Physical::Array, bit_width: 8, count: 3, payload_alignment: alignment, payload: b"abc" },
        PlacementRequest { class: LayoutClass::Bootstrap, order: 0, key_id: 0xF000, semantic: V06Semantic::Opaque, physical: V06Physical::Array, bit_width: 8, count: 3, payload_alignment: alignment, payload: b"ctl" },
        PlacementRequest { class: LayoutClass::ModelMetadata, order: 1, key_id: 7, semantic: V06Semantic::Opaque, physical: V06Physical::Array, bit_width: 8, count: 3, payload_alignment: alignment, payload: b"mdl" },
    ]
}

#[test]
fn control_first_order_and_occurrences_are_deterministic() {
    let requests = requests(8);
    let plan = LayoutPlan::build(&requests, 3).unwrap();
    assert_eq!(plan.entries().iter().map(|entry| entry.key_id).collect::<Vec<_>>(), vec![0xF000, 7, 7]);
    assert_eq!(plan.entries()[1].occurrence, 0);
    assert_eq!(plan.entries()[2].occurrence, 1);
    let bytes = write_known_size(Cursor::new(Vec::new()), 3, &requests).unwrap().into_inner();
    assert_eq!(plan.final_size(), bytes.len() as u64);
    let validated = parse_v06(&bytes).unwrap();
    for (planned, actual) in plan.entries().iter().zip(validated.blocks()) {
        assert_eq!(planned.block_start, actual.block_start);
        assert_eq!(planned.payload_start, actual.payload_start);
        assert_eq!(planned.payload_end, actual.payload_end);
    }
    assert_eq!(validated.blocks().iter().map(|block| block.key_id).collect::<Vec<_>>(), vec![0xF000, 7, 7]);
}

#[test]
fn alignment_is_a_payload_refinement_not_a_new_basestep() {
    assert_eq!(canonical_payload_alignment(4).unwrap(), 16);
    assert_eq!(payload_shift(4, 64).unwrap(), 2);
    let bytes = write_known_size(Cursor::new(Vec::new()), 4, &requests(64)).unwrap().into_inner();
    let validated = parse_v06(&bytes).unwrap();
    let tensor = &validated.blocks()[2];
    assert_eq!(tensor.payload_start % 64, 0);
    assert_eq!(validated.header().base_step, 16);
}

#[test]
fn invalid_alignment_and_ambiguous_order_fail_before_writing() {
    assert_eq!(payload_shift(4, 24).unwrap_err(), LayoutError::InvalidPayloadAlignment);
    assert_eq!(payload_shift(4, 8).unwrap_err(), LayoutError::InvalidPayloadAlignment);
    let mut duplicate = requests(8);
    duplicate[1].order = 2;
    assert_eq!(LayoutPlan::build(&duplicate, 3).unwrap_err(), LayoutError::DuplicateOrder);
}

#[test]
fn known_size_and_indefinite_writers_have_equivalent_blocks() {
    let requests = requests(16);
    let known = write_known_size(Cursor::new(Vec::new()), 3, &requests).unwrap().into_inner();
    let indefinite = write_indefinite(Cursor::new(Vec::new()), 3, &requests).unwrap().into_inner();
    let known_blocks = parse_v06(&known).unwrap().blocks().to_vec();
    let indefinite_blocks = parse_v06(&indefinite).unwrap().blocks().to_vec();
    assert_eq!(known_blocks, indefinite_blocks);
}
