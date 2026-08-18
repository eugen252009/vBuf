use std::io::Cursor;
use vbuf_core::v06::{parse_v06, V06Semantic};
use vbuf_core::writer::{BlockOptions, VBufV06Writer};
use vbuf_ml::bootstrap::{encode_payload as encode_bootstrap, Bootstrap, BootstrapEntry, BOOTSTRAP_KEY_ID};
use vbuf_ml::tensor_directory::{encode_payload as encode_directory, TensorEntry};
use vbuf_ml::{encode_source_profile, parse_source_profile, SourceDescriptor, SourceId, SourceLocator, SourceRegistry, TensorDirectory, TensorRef};

fn artifact(self_size: u64) -> Vec<u8> {
    let directory = encode_directory(&[TensorEntry::new("weight", vec![1], 20, 0)]).unwrap();
    let source = SourceDescriptor { id: SourceId::new(7), declared_size: Some(4), locator: SourceLocator::File("authoritative.vbuf".into()), hashes: vec![] };
    let registry = SourceRegistry::new(vec![SourceDescriptor::self_artifact(self_size), source.clone()]).unwrap();
    let reference = TensorRef::new(&source, 0, 4).unwrap();
    let source_payload = encode_source_profile(&registry, &[(20, 0, reference)]).unwrap();
    let bootstrap_payload = encode_bootstrap(&[
        BootstrapEntry::new(1, true, 10, 0),
        BootstrapEntry::new(2, true, 11, 0),
        BootstrapEntry::new(7, false, 50, 0),
    ]).unwrap();
    let mut writer = VBufV06Writer::new_known_size(Cursor::new(Vec::new()), 3).unwrap();
    writer.write_opaque(BlockOptions::array(BOOTSTRAP_KEY_ID), &bootstrap_payload).unwrap();
    writer.write_opaque(BlockOptions::array(10), &directory).unwrap();
    writer.write_opaque(BlockOptions::array(11), b"model").unwrap();
    writer.write_block(BlockOptions::array(20), V06Semantic::Float, 32, 0, &[]).unwrap();
    writer.write_opaque(BlockOptions::array(50), &source_payload).unwrap();
    writer.finish().unwrap().into_inner()
}

#[test]
fn persistent_source_profile_round_trips_and_drives_normal_directory_parse() {
    let first = artifact(0);
    let bytes = artifact(first.len() as u64);
    let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let validated = parse_v06(bytes).unwrap();
    let bootstrap = Bootstrap::discover(&validated).unwrap();
    let profile = parse_source_profile(&validated, &bootstrap).unwrap().unwrap();
    assert_eq!(profile.registry.get(SourceId::new(7)).unwrap().declared_size, Some(4));
    assert_eq!(profile.bindings[0].2.offset(), 0);
    let directory = TensorDirectory::parse_with_sources(&validated, &bootstrap, &profile.registry, &profile.bindings).unwrap();
    assert_eq!(directory.get("weight").unwrap().payload.source_id(), SourceId::new(7));
    assert!(directory.get("weight").unwrap().range.is_none());
}

#[test]
fn old_profile_without_source_role_remains_self_compatible() {
    let bytes = include_bytes!("fixtures/minimal-tensor-directory.vbuf");
    let validated = parse_v06(bytes).unwrap();
    let bootstrap = Bootstrap::discover(&validated).unwrap();
    assert!(parse_source_profile(&validated, &bootstrap).unwrap().is_none());
    let directory = TensorDirectory::parse(&validated, &bootstrap).unwrap();
    assert_eq!(directory.get("layer.0").unwrap().payload.source_id(), SourceId::SELF);
}
