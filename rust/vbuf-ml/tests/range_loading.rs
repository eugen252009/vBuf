use std::cell::RefCell;
use std::io::Cursor;
use vbuf_core::v06::parse_v06;
use vbuf_core::writer::{BlockOptions, VBufV06Writer};
use vbuf_ml::bootstrap::{encode_payload as encode_bootstrap, Bootstrap, BootstrapEntry, BOOTSTRAP_KEY_ID};
use vbuf_ml::range_loading::{execute_plan, select_tensor_names, Coalescing, PhysicalRange, RangeSource, ReadPlan};
use vbuf_ml::tensor_directory::{encode_payload as encode_directory, TensorEntry};
use vbuf_ml::RangeLoadError;

fn model_bytes() -> Vec<u8> {
    let directory = encode_directory(&[
        TensorEntry::new("a", vec![1], 20, 0),
        TensorEntry::new("b", vec![1], 21, 0),
        TensorEntry::new("c", vec![1], 22, 0),
    ]).unwrap();
    let mut writer = VBufV06Writer::new_known_size(Cursor::new(Vec::new()), 3).unwrap();
    writer.write_opaque(BlockOptions::array(BOOTSTRAP_KEY_ID), &encode_bootstrap(&[
        BootstrapEntry::new(1, true, 12, 0), BootstrapEntry::new(2, true, 10, 0),
    ]).unwrap()).unwrap();
    writer.write_opaque(BlockOptions::array(10), b"model").unwrap();
    writer.write_opaque(BlockOptions::array(12), &directory).unwrap();
    writer.write_f32(BlockOptions::array(20), &[1.0]).unwrap();
    writer.write_opaque(BlockOptions::array(90), &[0xaa; 64]).unwrap();
    writer.write_f32(BlockOptions::array(21), &[2.0]).unwrap();
    writer.write_opaque(BlockOptions::array(91), &[0xbb; 64]).unwrap();
    writer.write_f32(BlockOptions::array(22), &[3.0]).unwrap();
    writer.finish().unwrap().into_inner()
}

struct RecordingSource<'a> {
    bytes: &'a [u8],
    calls: RefCell<Vec<PhysicalRange>>,
}

impl<'a> RangeSource for RecordingSource<'a> {
    fn len(&self) -> u64 { self.bytes.len() as u64 }
    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> Result<(), RangeLoadError> {
        let end = offset.checked_add(destination.len() as u64).ok_or(RangeLoadError::ArithmeticOverflow)?;
        let start = usize::try_from(offset).map_err(|_| RangeLoadError::HostIndexOverflow)?;
        let end = usize::try_from(end).map_err(|_| RangeLoadError::HostIndexOverflow)?;
        let source = self.bytes.get(start..end).ok_or(RangeLoadError::OutsideSource)?;
        destination.copy_from_slice(source);
        self.calls.borrow_mut().push(PhysicalRange { offset, length: destination.len() as u64 });
        Ok(())
    }
}

fn directory(bytes: &'static [u8]) -> vbuf_ml::TensorDirectory<'static> {
    let validated = parse_v06(bytes).unwrap();
    let bootstrap = Bootstrap::discover(&validated).unwrap();
    // Keep the canonical mapping alive for this test result.
    let bootstrap = Box::leak(Box::new(bootstrap));
    vbuf_ml::TensorDirectory::parse(Box::leak(Box::new(validated)), bootstrap).unwrap()
}

#[test]
fn selected_tensor_set_reads_only_selected_payload_ranges() {
    let bytes: &'static [u8] = Box::leak(model_bytes().into_boxed_slice());
    let tensor_directory = directory(bytes);
    let selected = select_tensor_names(&tensor_directory, &["c", "a"]).unwrap();
    let plan = ReadPlan::build(&selected, Coalescing::None).unwrap();
    assert_eq!(plan.targets().iter().map(|target| target.selected.ordinal()).collect::<Vec<_>>(), vec![2, 0]);
    assert_eq!(plan.reads().len(), 2);
    let source = RecordingSource { bytes, calls: RefCell::new(Vec::new()) };
    let loaded = execute_plan(&source, &plan).unwrap();
    assert_eq!(loaded.target_bytes(&plan.targets()[0]).unwrap().len(), 4);
    assert_eq!(loaded.target_bytes(&plan.targets()[1]).unwrap().len(), 4);
    assert_eq!(source.calls.borrow().len(), 2);
    assert!(source.calls.borrow().iter().all(|range| range.length == 4));
}

#[test]
fn empty_duplicate_and_order_independent_selection_are_safe() {
    let empty = ReadPlan::build(&[], Coalescing::ExactAdjacent).unwrap();
    assert!(empty.reads().is_empty());
    let bytes: &'static [u8] = Box::leak(model_bytes().into_boxed_slice());
    let tensor_directory = directory(bytes);
    let selected = select_tensor_names(&tensor_directory, &["a", "a"]).unwrap();
    let duplicate = ReadPlan::build(&selected, Coalescing::None).unwrap();
    assert_eq!(duplicate.reads().len(), 1);
    let forward = select_tensor_names(&tensor_directory, &["a", "c"]).unwrap();
    let reverse = select_tensor_names(&tensor_directory, &["c", "a"]).unwrap();
    assert_eq!(ReadPlan::build(&forward, Coalescing::None).unwrap().reads(), ReadPlan::build(&reverse, Coalescing::None).unwrap().reads());
}

#[test]
fn mmap_and_positioned_file_paths_return_identical_selected_bytes() {
    let bytes = model_bytes();
    let path = std::env::temp_dir().join(format!("vbuf-ml-range-{}.vbuf", std::process::id()));
    std::fs::write(&path, &bytes).unwrap();
    let mapped = vbuf_ml::MmapSource::open(&path).unwrap();
    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let tensor_directory = directory(leaked);
    let selected = select_tensor_names(&tensor_directory, &["a", "c"]).unwrap();
    let plan = ReadPlan::build(&selected, Coalescing::None).unwrap();
    let mapped_bytes = mapped.checked_slice(plan.targets()[0].selected.payload_range()).unwrap().to_vec();
    let positioned = vbuf_ml::PositionedFileSource::open(&path).unwrap();
    let loaded = execute_plan(&positioned, &plan).unwrap();
    assert_eq!(mapped_bytes, loaded.target_bytes(&plan.targets()[0]).unwrap());
    assert_eq!(loaded.target_bytes(&plan.targets()[1]).unwrap().len(), 4);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn bounds_and_overflow_are_checked() {
    let bytes: &'static [u8] = Box::leak(model_bytes().into_boxed_slice());
    let tensor_directory = directory(bytes);
    let target = select_tensor_names(&tensor_directory, &["a"]).unwrap();
    let source = RecordingSource { bytes: b"small", calls: RefCell::new(Vec::new()) };
    let plan = ReadPlan::build(&target, Coalescing::None).unwrap();
    assert_eq!(execute_plan(&source, &plan).unwrap_err(), RangeLoadError::OutsideSource);
}
