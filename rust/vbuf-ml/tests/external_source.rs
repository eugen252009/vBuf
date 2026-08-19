use std::io::Cursor;
use vbuf_core::v06::parse_v06;
use vbuf_core::writer::{BlockOptions, VBufV06Writer};
use vbuf_ml::bootstrap::{
    BOOTSTRAP_KEY_ID, Bootstrap, BootstrapEntry, encode_payload as encode_bootstrap,
};
use vbuf_ml::range_loading::{
    RangeSource, ReadPlan, SourceSet, execute_plan_with_sources, select_tensor_names,
};
use vbuf_ml::tensor_directory::{TensorEntry, encode_payload as encode_directory};
use vbuf_ml::{
    RangeLoadError, SourceDescriptor, SourceId, SourceLocator, SourceRangeError, SourceRegistry,
    TensorRef,
};

fn metadata_model() -> Vec<u8> {
    let directory = encode_directory(&[TensorEntry::new("weight", vec![1], 20, 0)]).unwrap();
    let mut writer = VBufV06Writer::new_known_size(Cursor::new(Vec::new()), 3).unwrap();
    writer
        .write_opaque(
            BlockOptions::array(BOOTSTRAP_KEY_ID),
            &encode_bootstrap(&[
                BootstrapEntry::new(1, true, 10, 0),
                BootstrapEntry::new(2, true, 11, 0),
            ])
            .unwrap(),
        )
        .unwrap();
    writer
        .write_opaque(BlockOptions::array(10), &directory)
        .unwrap();
    writer
        .write_opaque(BlockOptions::array(11), b"model")
        .unwrap();
    writer.write_f32(BlockOptions::array(20), &[0.0]).unwrap();
    writer.finish().unwrap().into_inner()
}

struct BytesSource {
    bytes: Vec<u8>,
}

impl RangeSource for BytesSource {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }
    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> Result<(), RangeLoadError> {
        let start = usize::try_from(offset).map_err(|_| RangeLoadError::HostIndexOverflow)?;
        let end = start
            .checked_add(destination.len())
            .ok_or(RangeLoadError::ArithmeticOverflow)?;
        destination.copy_from_slice(
            self.bytes
                .get(start..end)
                .ok_or(RangeLoadError::OutsideSource)?,
        );
        Ok(())
    }
}

#[test]
fn external_tensor_ref_uses_the_same_directory_validation_and_range_plan() {
    let metadata = metadata_model();
    let bytes: &'static [u8] = Box::leak(metadata.into_boxed_slice());
    let validated = parse_v06(bytes).unwrap();
    let bootstrap = Bootstrap::discover(&validated).unwrap();
    let source = SourceDescriptor {
        id: SourceId::new(7),
        declared_size: Some(4),
        locator: SourceLocator::File("/nas/model.vbuf".into()),
        hashes: Vec::new(),
    };
    let registry = SourceRegistry::new(vec![
        SourceDescriptor::self_artifact(bytes.len() as u64),
        source.clone(),
    ])
    .unwrap();
    let reference = TensorRef::new(&source, 0, 4).unwrap();
    let directory = vbuf_ml::TensorDirectory::parse_with_sources(
        &validated,
        &bootstrap,
        &registry,
        &[(20, 0, reference)],
    )
    .unwrap();
    assert_eq!(
        directory.get("weight").unwrap().payload.source_id(),
        SourceId::new(7)
    );
    assert!(directory.get("weight").unwrap().range.is_none());
    let selected = select_tensor_names(&directory, &["weight"]).unwrap();
    let plan = ReadPlan::build(&selected, vbuf_ml::Coalescing::None).unwrap();
    let source = BytesSource {
        bytes: vec![1, 2, 3, 4],
    };
    let loaded =
        execute_plan_with_sources(&SourceSet::new(vec![(SourceId::new(7), &source)]), &plan)
            .unwrap();
    assert_eq!(
        loaded.target_bytes(&plan.targets()[0]).unwrap(),
        &[1, 2, 3, 4]
    );
}

#[test]
fn checked_source_ranges_reject_overflow_bounds_and_unknown_size() {
    let source = SourceDescriptor {
        id: SourceId::new(8),
        declared_size: Some(1 << 40),
        locator: SourceLocator::Http("https://nas/model.vbuf".into()),
        hashes: vec![],
    };
    let reference = TensorRef::new(&source, (1 << 32) + 7, 9).unwrap();
    assert_eq!(reference.offset(), (1 << 32) + 7);
    assert_eq!(
        source.checked_range(u64::MAX, 1),
        Err(SourceRangeError::Range(vbuf_layout::RangeError::Overflow))
    );
    assert_eq!(
        source.checked_range(1 << 40, 1),
        Err(SourceRangeError::OutsideSource)
    );
    let unknown = SourceDescriptor {
        declared_size: None,
        ..source
    };
    assert_eq!(
        unknown.checked_range(0, 1),
        Err(SourceRangeError::UnknownSize)
    );
}

#[test]
fn source_identity_is_not_derived_from_location_and_hashes_are_optional_binary_values() {
    let source = SourceDescriptor {
        id: SourceId::new(9),
        declared_size: Some(10),
        locator: SourceLocator::Http("https://one/model.vbuf".into()),
        hashes: vec![
            vbuf_ml::SourceHash {
                algorithm: 1,
                value: vec![0, 1, 255],
            },
            vbuf_ml::SourceHash {
                algorithm: 2,
                value: vec![4, 5],
            },
        ],
    };
    let same_identity_elsewhere = SourceDescriptor {
        locator: SourceLocator::File("/mnt/model.vbuf".into()),
        ..source.clone()
    };
    assert_eq!(source.id, same_identity_elsewhere.id);
    assert_eq!(source.hashes.len(), 2);
    assert_eq!(source.hashes[0].value, vec![0, 1, 255]);
}

#[test]
fn malformed_source_descriptors_fail_closed() {
    let empty_locator = SourceDescriptor {
        id: SourceId::new(10),
        declared_size: Some(1),
        locator: SourceLocator::Http(String::new()),
        hashes: vec![],
    };
    assert_eq!(
        SourceRegistry::new(vec![SourceDescriptor::self_artifact(1), empty_locator]).unwrap_err(),
        vbuf_ml::SourceRegistryError::MalformedDescriptor
    );
    let empty_hash = SourceDescriptor {
        id: SourceId::new(11),
        declared_size: Some(1),
        locator: SourceLocator::File("model.vbuf".into()),
        hashes: vec![vbuf_ml::SourceHash {
            algorithm: 1,
            value: vec![],
        }],
    };
    assert_eq!(
        SourceRegistry::new(vec![SourceDescriptor::self_artifact(1), empty_hash]).unwrap_err(),
        vbuf_ml::SourceRegistryError::MalformedDescriptor
    );
}

#[test]
fn unknown_tensor_source_id_is_rejected_before_materialization() {
    let metadata = metadata_model();
    let bytes: &'static [u8] = Box::leak(metadata.into_boxed_slice());
    let validated = parse_v06(bytes).unwrap();
    let bootstrap = Bootstrap::discover(&validated).unwrap();
    let source = SourceDescriptor {
        id: SourceId::new(99),
        declared_size: Some(4),
        locator: SourceLocator::File("missing.vbuf".into()),
        hashes: vec![],
    };
    let reference = TensorRef::new(&source, 0, 4).unwrap();
    let registry =
        SourceRegistry::new(vec![SourceDescriptor::self_artifact(bytes.len() as u64)]).unwrap();
    assert_eq!(
        vbuf_ml::TensorDirectory::parse_with_sources(
            &validated,
            &bootstrap,
            &registry,
            &[(20, 0, reference)]
        )
        .unwrap_err()
        .code,
        vbuf_ml::MlErrorCode::TensorReferenceMissing
    );
}
