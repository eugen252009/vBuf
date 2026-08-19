//! Architecture-neutral execution graph.
//!
//! Model adapters lower dense, MoE, attention, and SSM blocks into this view.
//! The scheduler only sees persistent tensor dependencies and value lifetimes;
//! it does not interpret model names or architecture metadata.

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TensorId(pub u32);

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ValueId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputRef {
    Tensor(TensorId),
    Value(ValueId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OperationKind {
    RmsNorm,
    MatMul,
    QuantizedMatMul,
    Activation,
    Attention,
    SsmScan,
    TopKRouter,
    ExpertDispatch,
    ResidualAdd,
    StateRead,
    StateWrite,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentTensor {
    pub id: TensorId,
    pub name: String,
    pub bytes: u64,
    pub source_offset: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Operation {
    pub id: String,
    pub kind: OperationKind,
    pub inputs: Vec<InputRef>,
    pub output: ValueId,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExecutionGraph {
    pub input: ValueId,
    pub output: ValueId,
    pub tensors: Vec<PersistentTensor>,
    pub operations: Vec<Operation>,
}

impl ExecutionGraph {
    pub fn persistent_bytes(&self) -> u64 {
        self.tensors.iter().map(|tensor| tensor.bytes).sum()
    }

    pub fn operation(&mut self, id: impl Into<String>, kind: OperationKind,
        inputs: impl IntoIterator<Item = InputRef>, output: ValueId) {
        self.operations.push(Operation { id: id.into(), kind, inputs: inputs.into_iter().collect(), output });
    }

    /// Returns tensor dependencies in first-use order. This is the common
    /// acquisition order consumed by the bounded residency runtime.
    pub fn acquisition_order(&self) -> Vec<TensorId> {
        let mut result = Vec::new();
        for operation in &self.operations {
            for input in &operation.inputs {
                if let InputRef::Tensor(id) = input && !result.contains(id) {
                    result.push(*id);
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_graph() -> ExecutionGraph {
        let mut graph = ExecutionGraph { input: ValueId(0), output: ValueId(2), ..Default::default() };
        graph.tensors.push(PersistentTensor { id: TensorId(0), name: "norm".into(), bytes: 128, source_offset: 10 });
        graph.tensors.push(PersistentTensor { id: TensorId(1), name: "weight".into(), bytes: 256, source_offset: 20 });
        graph.operation("norm", OperationKind::RmsNorm, [InputRef::Value(ValueId(0)), InputRef::Tensor(TensorId(0))], ValueId(1));
        graph.operation("matmul", OperationKind::QuantizedMatMul, [InputRef::Value(ValueId(1)), InputRef::Tensor(TensorId(1))], ValueId(2));
        graph
    }

    fn moe_graph() -> ExecutionGraph {
        let mut graph = ExecutionGraph { input: ValueId(0), output: ValueId(3), ..Default::default() };
        graph.tensors.push(PersistentTensor { id: TensorId(0), name: "router".into(), bytes: 64, source_offset: 10 });
        graph.tensors.push(PersistentTensor { id: TensorId(1), name: "expert.0".into(), bytes: 512, source_offset: 20 });
        graph.operation("route", OperationKind::TopKRouter, [InputRef::Value(ValueId(0)), InputRef::Tensor(TensorId(0))], ValueId(1));
        graph.operation("experts", OperationKind::ExpertDispatch, [InputRef::Value(ValueId(1)), InputRef::Tensor(TensorId(1))], ValueId(2));
        graph.operation("residual", OperationKind::ResidualAdd, [InputRef::Value(ValueId(0)), InputRef::Value(ValueId(2))], ValueId(3));
        graph
    }

    #[test]
    fn dense_and_moe_share_the_same_graph_and_residency_view() {
        let dense = dense_graph();
        let moe = moe_graph();
        assert_eq!(dense.acquisition_order(), [TensorId(0), TensorId(1)]);
        assert_eq!(moe.acquisition_order(), [TensorId(0), TensorId(1)]);
        assert_eq!(dense.persistent_bytes(), 384);
        assert_eq!(moe.persistent_bytes(), 576);
    }
}
