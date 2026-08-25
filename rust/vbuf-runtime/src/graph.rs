//! Architecture-neutral execution graph.
//!
//! Model adapters lower dense, MoE, attention, and SSM blocks into this view.
//! The scheduler only sees persistent tensor dependencies and value lifetimes;
//! it does not interpret model names or architecture metadata.

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TensorId(pub u32);

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ValueId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StateId(pub u32);

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
    BiasAdd,
    Rotary,
    ReshapeHeads,
    ElementwiseMul,
    ZeroLike,
    WeightedAdd,
    StateRead,
    StateWrite,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActivationKind {
    Silu,
    Sigmoid,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotaryAttributes {
    pub head_count: u64,
    pub head_dim: u64,
    pub rotary_dim: u64,
    pub theta: f32,
    pub position_start: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeadReshapeAttributes {
    pub head_count: u64,
    pub head_dim: u64,
    pub flatten: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertDispatchAttributes {
    pub expert_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionMaskKind {
    None,
    Causal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionPositionKind {
    StateLength,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionAttributes {
    pub batch_size: u64,
    pub query_head_count: u64,
    pub kv_head_count: u64,
    pub head_dim: u64,
    pub query_length: u64,
    /// Total visible K/V length after this operation commits its append.
    pub current_kv_length: u64,
    pub scale: f32,
    pub mask: AttentionMaskKind,
    pub position: AttentionPositionKind,
    pub state: StateId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatMulWeightOperand {
    Lhs,
    Rhs,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TopKOrder {
    Descending,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TopKTieBreak {
    LowerIndex,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OperationAttributes {
    pub epsilon: Option<f32>,
    pub activation: Option<ActivationKind>,
    pub attention: Option<AttentionAttributes>,
    pub rotary: Option<RotaryAttributes>,
    pub head_reshape: Option<HeadReshapeAttributes>,
    pub weighted_add: Option<f32>,
    pub expert_dispatch: Option<ExpertDispatchAttributes>,
    pub top_k: Option<u32>,
    pub matmul_weight_operand: Option<MatMulWeightOperand>,
    pub matmul_transpose_weight: Option<bool>,
    pub top_k_order: Option<TopKOrder>,
    pub top_k_tie_break: Option<TopKTieBreak>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentTensor {
    pub id: TensorId,
    pub name: String,
    pub bytes: u64,
    pub source_offset: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Operation {
    pub id: String,
    pub kind: OperationKind,
    pub inputs: Vec<InputRef>,
    pub output: ValueId,
    pub attributes: OperationAttributes,
}

#[derive(Clone, Debug, Default, PartialEq)]
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

    pub fn operation(
        &mut self,
        id: impl Into<String>,
        kind: OperationKind,
        inputs: impl IntoIterator<Item = InputRef>,
        output: ValueId,
        attributes: OperationAttributes,
    ) {
        self.operations.push(Operation {
            id: id.into(),
            kind,
            inputs: inputs.into_iter().collect(),
            output,
            attributes,
        });
    }

    /// Returns tensor dependencies in first-use order. This is the common
    /// acquisition order consumed by the bounded residency runtime.
    pub fn acquisition_order(&self) -> Vec<TensorId> {
        let mut result = Vec::new();
        for operation in &self.operations {
            for input in &operation.inputs {
                if let InputRef::Tensor(id) = input
                    && !result.contains(id)
                {
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
        let mut graph = ExecutionGraph {
            input: ValueId(0),
            output: ValueId(2),
            ..Default::default()
        };
        graph.tensors.push(PersistentTensor {
            id: TensorId(0),
            name: "norm".into(),
            bytes: 128,
            source_offset: 10,
        });
        graph.tensors.push(PersistentTensor {
            id: TensorId(1),
            name: "weight".into(),
            bytes: 256,
            source_offset: 20,
        });
        graph.operation(
            "norm",
            OperationKind::RmsNorm,
            [InputRef::Value(ValueId(0)), InputRef::Tensor(TensorId(0))],
            ValueId(1),
            OperationAttributes {
                epsilon: Some(1e-6),
                ..Default::default()
            },
        );
        graph.operation(
            "matmul",
            OperationKind::QuantizedMatMul,
            [InputRef::Value(ValueId(1)), InputRef::Tensor(TensorId(1))],
            ValueId(2),
            OperationAttributes {
                matmul_weight_operand: Some(MatMulWeightOperand::Rhs),
                matmul_transpose_weight: Some(false),
                ..Default::default()
            },
        );
        graph
    }

    fn moe_graph() -> ExecutionGraph {
        let mut graph = ExecutionGraph {
            input: ValueId(0),
            output: ValueId(3),
            ..Default::default()
        };
        graph.tensors.push(PersistentTensor {
            id: TensorId(0),
            name: "router".into(),
            bytes: 64,
            source_offset: 10,
        });
        graph.tensors.push(PersistentTensor {
            id: TensorId(1),
            name: "expert.0".into(),
            bytes: 512,
            source_offset: 20,
        });
        graph.operation(
            "route",
            OperationKind::TopKRouter,
            [InputRef::Value(ValueId(0)), InputRef::Tensor(TensorId(0))],
            ValueId(1),
            OperationAttributes {
                top_k: Some(2),
                top_k_order: Some(TopKOrder::Descending),
                top_k_tie_break: Some(TopKTieBreak::LowerIndex),
                ..Default::default()
            },
        );
        graph.operation(
            "experts",
            OperationKind::ExpertDispatch,
            [InputRef::Value(ValueId(1)), InputRef::Tensor(TensorId(1))],
            ValueId(2),
            OperationAttributes::default(),
        );
        graph.operation(
            "residual",
            OperationKind::ResidualAdd,
            [InputRef::Value(ValueId(0)), InputRef::Value(ValueId(2))],
            ValueId(3),
            OperationAttributes::default(),
        );
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
