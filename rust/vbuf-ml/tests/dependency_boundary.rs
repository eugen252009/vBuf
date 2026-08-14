use std::path::Path;
use vbuf_core::v06::parse_v06;

#[test]
fn descendant_consumes_validated_generic_range_without_domain_semantics() {
    let bytes = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/v06/valid-basic.vbuf"),
    )
    .unwrap();
    let validated = parse_v06(&bytes).unwrap();
    let payload = validated.payload_range(0).unwrap();
    assert_eq!(vbuf_ml::validated_range_len(&payload), 12);
}
