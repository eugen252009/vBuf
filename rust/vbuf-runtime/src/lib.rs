//! Small vBuf-owned execution control plane.
//!
//! This crate deliberately does not create a llama model. It validates the
//! artifact through vbuf-ml, reads tensor ranges only when a layer is acquired,
//! and keeps a bounded resident working set. The architecture-specific graph
//! executor is layered above this API and may use ggml through the optional
//! `ggml` feature.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use vbuf_ml::{BorrowedModel, MlError};

pub mod graph;
pub use graph::{ExecutionGraph, InputRef, Operation, OperationKind, PersistentTensor, TensorId, ValueId};
pub mod lowering;
pub use lowering::{lower_region, LoweringError, OperationAttributes, PortableInput, PortableOperation,
    PortableOperationKind, PortableProgram, PortableRegion, SemanticTensorKey, StateId, StateRef, TensorBinding};

#[cfg(feature = "ggml")]
pub mod ggml {
    #[repr(C)]
    pub struct Context {
        _private: [u8; 0],
    }

    unsafe extern "C" {
        pub fn ggml_time_init() -> bool;
    }
}

#[derive(Debug)]
pub enum RuntimeError {
    Model(MlError),
    Io(std::io::Error),
    BudgetTooSmall { requested: u64, budget: u64 },
    LayerNotFound(u32),
    InvalidTensorRange,
}

impl From<MlError> for RuntimeError {
    fn from(error: MlError) -> Self { Self::Model(error) }
}

impl From<std::io::Error> for RuntimeError {
    fn from(error: std::io::Error) -> Self { Self::Io(error) }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorRange {
    pub name: String,
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug)]
pub struct ResidentTensor {
    pub range: TensorRange,
    pub bytes: Vec<u8>,
}

#[derive(Debug)]
struct Cache {
    budget: u64,
    resident_bytes: u64,
    values: HashMap<String, ResidentTensor>,
    order: VecDeque<String>,
}

impl Cache {
    fn new(budget: u64) -> Self {
        Self { budget, resident_bytes: 0, values: HashMap::new(), order: VecDeque::new() }
    }

    fn touch(&mut self, name: &str) {
        self.order.retain(|entry| entry != name);
        self.order.push_back(name.to_owned());
    }

    fn evict_until(&mut self, required: u64) -> Result<(), RuntimeError> {
        if required > self.budget { return Err(RuntimeError::BudgetTooSmall { requested: required, budget: self.budget }); }
        while self.resident_bytes + required > self.budget {
            let Some(oldest) = self.order.pop_front() else { break };
            if let Some(value) = self.values.remove(&oldest) {
                self.resident_bytes -= value.bytes.len() as u64;
            }
        }
        Ok(())
    }

    fn insert(&mut self, value: ResidentTensor) -> Result<(), RuntimeError> {
        let bytes = value.bytes.len() as u64;
        self.evict_until(bytes)?;
        if let Some(previous) = self.values.remove(&value.range.name) {
            self.resident_bytes -= previous.bytes.len() as u64;
        }
        self.resident_bytes += bytes;
        self.touch(&value.range.name);
        self.values.insert(value.range.name.clone(), value);
        Ok(())
    }
}

#[derive(Debug)]
pub struct VBufRuntime {
    path: PathBuf,
    model: BorrowedModel,
    tensors: Vec<TensorRange>,
    layers: HashMap<u32, Vec<usize>>,
    cache: Cache,
}

impl VBufRuntime {
    pub fn open(path: impl AsRef<Path>, resident_budget: u64) -> Result<Self, RuntimeError> {
        let path = path.as_ref().to_owned();
        let model = BorrowedModel::open(&path)?;
        let mut tensors = Vec::new();
        let mut layers: HashMap<u32, Vec<usize>> = HashMap::new();
        for tensor in model.view().directory.tensors() {
            let offset = tensor.payload.offset();
            let length = tensor.payload.length();
            let index = tensors.len();
            tensors.push(TensorRange { name: tensor.name.clone(), offset, length });
            if let Some(layer) = parse_layer(&tensor.name) {
                layers.entry(layer).or_default().push(index);
            }
        }
        Ok(Self { path, model, tensors, layers, cache: Cache::new(resident_budget) })
    }

    pub fn model(&self) -> &BorrowedModel { &self.model }
    pub fn resident_bytes(&self) -> u64 { self.cache.resident_bytes }
    pub fn resident_budget(&self) -> u64 { self.cache.budget }
    pub fn tensor_ranges(&self) -> &[TensorRange] { &self.tensors }
    pub fn layer_ids(&self) -> impl Iterator<Item = u32> + '_ { self.layers.keys().copied() }

    /// Reads only the selected layer's tensor payloads and evicts old layers
    /// until the configured resident budget is respected.
    pub fn acquire_layer(&mut self, layer: u32) -> Result<Vec<&[u8]>, RuntimeError> {
        let indexes = self.layers.get(&layer).cloned().ok_or(RuntimeError::LayerNotFound(layer))?;
        let mut file = File::open(&self.path)?;
        for index in indexes.iter().copied() {
            let range = self.tensors[index].clone();
            if range.length > usize::MAX as u64 { return Err(RuntimeError::InvalidTensorRange); }
            let mut bytes = vec![0u8; range.length as usize];
            file.seek(SeekFrom::Start(range.offset))?;
            file.read_exact(&mut bytes)?;
            self.cache.insert(ResidentTensor { range, bytes })?;
        }
        let names: Vec<String> = indexes.iter().map(|index| self.tensors[*index].name.clone()).collect();
        Ok(names.iter().filter_map(|name| self.cache.values.get(name).map(|tensor| tensor.bytes.as_slice())).collect())
    }

    pub fn release_layer(&mut self, layer: u32) {
        if let Some(indexes) = self.layers.get(&layer) {
            for index in indexes {
                let name = &self.tensors[*index].name;
                if let Some(value) = self.cache.values.remove(name) {
                    self.cache.resident_bytes -= value.bytes.len() as u64;
                }
                self.cache.order.retain(|entry| entry != name);
            }
        }
    }
}

fn parse_layer(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blk.")?;
    let end = rest.find('.')?;
    rest[..end].parse().ok()
}

#[cfg(test)]
mod tests {
    use super::Cache;

    #[test]
    fn cache_evicts_oldest_values_before_inserting_new_layer() {
        let mut cache = Cache::new(6);
        cache.insert(super::ResidentTensor { range: super::TensorRange { name: "a".into(), offset: 0, length: 4 }, bytes: vec![0; 4] }).unwrap();
        cache.insert(super::ResidentTensor { range: super::TensorRange { name: "b".into(), offset: 4, length: 2 }, bytes: vec![0; 2] }).unwrap();
        cache.insert(super::ResidentTensor { range: super::TensorRange { name: "c".into(), offset: 6, length: 5 }, bytes: vec![0; 5] }).unwrap();
        assert_eq!(cache.resident_bytes, 5);
        assert!(!cache.values.contains_key("a"));
        assert!(cache.values.contains_key("c"));
    }
}
