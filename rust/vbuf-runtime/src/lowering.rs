//! Minimal model-agnostic lowering from imported semantic regions.
//!
//! This module deliberately lowers only semantic descriptors into the existing
//! architecture-neutral graph seed. It does not inspect model identity, source
//! tensor names, GGUF metadata, or GGML representation IDs. Backend execution
//! remains a separate adapter boundary.

use std::collections::HashMap;

use crate::graph::{ExecutionGraph, InputRef, OperationAttributes as GraphOperationAttributes,
    OperationKind, TensorId, ValueId};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SemanticTensorKey(pub String);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StateId(pub u32);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorBinding {
    pub semantic: SemanticTensorKey,
    pub tensor: TensorId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateRef {
    pub id: StateId,
    pub kind: String,
    pub lifetime: String,
    pub scope: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PortableInput {
    Value(ValueId),
    Tensor(SemanticTensorKey),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PortableOperationKind {
    RmsNorm,
    MatMul,
    Activation,
    TopK,
    IndexedMatMul,
    Attention,
    StateRead,
    StateWrite,
    ResidualAdd,
}

pub type OperationAttributes = GraphOperationAttributes;

#[derive(Clone, Debug, PartialEq)]
pub struct PortableOperation {
    pub id: String,
    pub kind: PortableOperationKind,
    pub inputs: Vec<PortableInput>,
    pub output: ValueId,
    pub attributes: OperationAttributes,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PortableRegion {
    pub id: String,
    pub input: ValueId,
    pub output: ValueId,
    pub operations: Vec<PortableOperation>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PortableProgram {
    pub tensor_bindings: Vec<TensorBinding>,
    pub state_refs: Vec<StateRef>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LoweringError {
    DuplicateBinding(SemanticTensorKey),
    MissingBinding(SemanticTensorKey),
    UnsupportedOperation(PortableOperationKind),
    InvalidAttribute(&'static str),
}

pub fn lower_region(
    program: &PortableProgram,
    region: &PortableRegion,
) -> Result<ExecutionGraph, LoweringError> {
    let bindings = program.tensor_bindings.iter().try_fold(
        HashMap::new(),
        |mut bindings: HashMap<SemanticTensorKey, TensorId>, binding| {
            if bindings
                .insert(binding.semantic.clone(), binding.tensor)
                .is_some()
            {
                return Err(LoweringError::DuplicateBinding(binding.semantic.clone()));
            }
            Ok(bindings)
        },
    )?;
    let mut graph = ExecutionGraph {
        input: region.input,
        output: region.output,
        ..Default::default()
    };
    for binding in &program.tensor_bindings {
        graph.tensors.push(crate::graph::PersistentTensor {
            id: binding.tensor,
            name: binding.semantic.0.clone(),
            bytes: 0,
            source_offset: 0,
        });
    }
    for operation in &region.operations {
        if operation.kind == PortableOperationKind::RmsNorm
            && operation.attributes.epsilon.is_none()
        {
            return Err(LoweringError::InvalidAttribute("RmsNorm requires epsilon"));
        }
        if operation.kind == PortableOperationKind::TopK && (operation.attributes.top_k.is_none()
            || operation.attributes.top_k_order.is_none()
            || operation.attributes.top_k_tie_break.is_none()) {
            return Err(LoweringError::InvalidAttribute("TopK requires top_k"));
        }
        if operation.kind == PortableOperationKind::MatMul
            && (operation.attributes.matmul_weight_operand.is_none()
                || operation.attributes.matmul_transpose_weight.is_none())
        {
            return Err(LoweringError::InvalidAttribute("MatMul requires weight operand and orientation"));
        }
        let kind = match operation.kind {
            PortableOperationKind::RmsNorm => OperationKind::RmsNorm,
            PortableOperationKind::MatMul => OperationKind::MatMul,
            PortableOperationKind::Activation => OperationKind::Activation,
            PortableOperationKind::TopK => OperationKind::TopKRouter,
            PortableOperationKind::IndexedMatMul => OperationKind::ExpertDispatch,
            PortableOperationKind::Attention => OperationKind::Attention,
            PortableOperationKind::StateRead => OperationKind::StateRead,
            PortableOperationKind::StateWrite => OperationKind::StateWrite,
            PortableOperationKind::ResidualAdd => OperationKind::ResidualAdd,
        };
        let inputs = operation
            .inputs
            .iter()
            .map(|input| match input {
                PortableInput::Value(value) => Ok(InputRef::Value(*value)),
                PortableInput::Tensor(semantic) => bindings
                    .get(semantic)
                    .copied()
                    .map(InputRef::Tensor)
                    .ok_or_else(|| LoweringError::MissingBinding(semantic.clone())),
            })
            .collect::<Result<Vec<_>, _>>()?;
        graph.operation(operation.id.clone(), kind, inputs, operation.output, GraphOperationAttributes {
            epsilon: operation.attributes.epsilon,
            top_k: operation.attributes.top_k,
            matmul_weight_operand: operation.attributes.matmul_weight_operand,
            matmul_transpose_weight: operation.attributes.matmul_transpose_weight,
            top_k_order: operation.attributes.top_k_order,
            top_k_tie_break: operation.attributes.top_k_tie_break,
        });
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{MatMulWeightOperand, TopKOrder, TopKTieBreak};

    fn deepseek_router_region() -> (PortableProgram, PortableRegion) {
        let norm = SemanticTensorKey("layer.1.mlp.input_norm".into());
        let router = SemanticTensorKey("layer.1.moe.router_weight".into());
        let program = PortableProgram {
            tensor_bindings: vec![
                TensorBinding {
                    semantic: norm.clone(),
                    tensor: TensorId(10),
                },
                TensorBinding {
                    semantic: router.clone(),
                    tensor: TensorId(11),
                },
            ],
            state_refs: vec![],
        };
        let region = PortableRegion {
            id: "layer.1.moe.router_prefix".into(),
            input: ValueId(0),
            output: ValueId(3),
            operations: vec![
                PortableOperation {
                    id: "normalize".into(),
                    kind: PortableOperationKind::RmsNorm,
                    inputs: vec![
                        PortableInput::Value(ValueId(0)),
                        PortableInput::Tensor(norm),
                    ],
                    output: ValueId(1),
                    attributes: OperationAttributes {
                        epsilon: Some(1e-6),
                        ..Default::default()
                    },
                },
                PortableOperation {
                    id: "router".into(),
                    kind: PortableOperationKind::MatMul,
                    inputs: vec![
                        PortableInput::Value(ValueId(1)),
                        PortableInput::Tensor(router),
                    ],
                    output: ValueId(2),
                    attributes: OperationAttributes {
                        matmul_weight_operand: Some(MatMulWeightOperand::Rhs),
                        matmul_transpose_weight: Some(false),
                        ..Default::default()
                    },
                },
                PortableOperation {
                    id: "select".into(),
                    kind: PortableOperationKind::TopK,
                    inputs: vec![PortableInput::Value(ValueId(2))],
                    output: ValueId(3),
                    attributes: OperationAttributes {
                        top_k: Some(6),
                        top_k_order: Some(TopKOrder::Descending),
                        top_k_tie_break: Some(TopKTieBreak::LowerIndex),
                        ..Default::default()
                    },
                },
            ],
        };
        (program, region)
    }

    #[test]
    fn lowers_real_deepseek_router_semantics_without_source_names() {
        let (program, region) = deepseek_router_region();
        let graph = lower_region(&program, &region).unwrap();
        assert_eq!(graph.acquisition_order(), [TensorId(10), TensorId(11)]);
        assert_eq!(
            graph
                .operations
                .iter()
                .map(|operation| operation.kind)
                .collect::<Vec<_>>(),
            [
                OperationKind::RmsNorm,
                OperationKind::MatMul,
                OperationKind::TopKRouter,
            ]
        );
        assert!(graph
            .tensors
            .iter()
            .all(|tensor| !tensor.name.starts_with("blk.")));
    }

    #[test]
    fn renamed_source_provenance_does_not_change_semantic_lowering() {
        let (program, mut region) = deepseek_router_region();
        let first = lower_region(&program, &region).unwrap();
        region.id = "unrelated.importer.region".into();
        let second = lower_region(&program, &region).unwrap();
        assert_eq!(first.acquisition_order(), second.acquisition_order());
        assert_eq!(first.operations, second.operations);
    }

    #[test]
    fn invalid_binding_and_operation_attributes_fail_closed() {
        let (mut program, mut region) = deepseek_router_region();
        program.tensor_bindings.pop();
        assert!(matches!(
            lower_region(&program, &region),
            Err(LoweringError::MissingBinding(_))
        ));
        let (program, _) = deepseek_router_region();
        region.operations[2].attributes.top_k = None;
        assert_eq!(
            lower_region(&program, &region),
            Err(LoweringError::InvalidAttribute("TopK requires top_k"))
        );
    }
}
