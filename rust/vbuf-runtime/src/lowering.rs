//! Minimal model-agnostic lowering from imported semantic regions.
//!
//! This module deliberately lowers only semantic descriptors into the existing
//! architecture-neutral graph seed. It does not inspect model identity, source
//! tensor names, source metadata, or backend representation IDs. Backend execution
//! remains a separate adapter boundary.

use std::collections::HashMap;

use crate::graph::{
    AttentionMaskKind, AttentionPositionKind, ExecutionGraph, InputRef,
    OperationAttributes as GraphOperationAttributes, OperationKind, StateId, TensorId, ValueId,
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SemanticTensorKey(pub String);

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
    BiasAdd,
    Rotary,
    ReshapeHeads,
    ElementwiseMul,
    ZeroLike,
    WeightedAdd,
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
    MissingState(StateId),
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
        if operation.kind == PortableOperationKind::Activation
            && operation.attributes.activation.is_none()
        {
            return Err(LoweringError::InvalidAttribute(
                "Activation requires an activation kind",
            ));
        }
        if operation.kind == PortableOperationKind::Rotary {
            let Some(rotary) = operation.attributes.rotary else {
                return Err(LoweringError::InvalidAttribute(
                    "Rotary requires geometry and theta",
                ));
            };
            if operation.inputs.len() != 1
                || rotary.head_count == 0
                || rotary.head_dim == 0
                || rotary.rotary_dim == 0
                || rotary.rotary_dim > rotary.head_dim
                || rotary.rotary_dim % 2 != 0
                || !rotary.theta.is_finite()
                || rotary.theta <= 0.0
            {
                return Err(LoweringError::InvalidAttribute(
                    "Rotary geometry or theta is invalid",
                ));
            }
        }
        if operation.kind == PortableOperationKind::ReshapeHeads {
            let Some(shape) = operation.attributes.head_reshape else {
                return Err(LoweringError::InvalidAttribute(
                    "Head reshape requires geometry",
                ));
            };
            if operation.inputs.len() != 1 || shape.head_count == 0 || shape.head_dim == 0 {
                return Err(LoweringError::InvalidAttribute(
                    "Head reshape geometry is invalid",
                ));
            }
        }
        if operation.kind == PortableOperationKind::WeightedAdd
            && (operation.inputs.len() != 2
                || operation.attributes.weighted_add.is_none()
                || !operation.attributes.weighted_add.unwrap().is_finite())
        {
            return Err(LoweringError::InvalidAttribute(
                "WeightedAdd requires two inputs and a finite weight",
            ));
        }
        if operation.kind == PortableOperationKind::IndexedMatMul
            && (operation.inputs.len() != 4 || operation.attributes.expert_dispatch.is_none())
        {
            return Err(LoweringError::InvalidAttribute(
                "IndexedMatMul requires activation, three weights, and expert identity",
            ));
        }
        if matches!(
            operation.kind,
            PortableOperationKind::BiasAdd
                | PortableOperationKind::ElementwiseMul
                | PortableOperationKind::ZeroLike
        ) {
            let expected = if operation.kind == PortableOperationKind::ZeroLike {
                1
            } else {
                2
            };
            if operation.inputs.len() != expected {
                return Err(LoweringError::InvalidAttribute(
                    "Elementwise operation has an invalid input count",
                ));
            }
        }
        if operation.kind == PortableOperationKind::Attention {
            let Some(attention) = operation.attributes.attention else {
                return Err(LoweringError::InvalidAttribute(
                    "Attention requires geometry and state attributes",
                ));
            };
            if operation.inputs.len() != 3 {
                return Err(LoweringError::InvalidAttribute(
                    "Attention requires Q, K, and V inputs",
                ));
            }
            if attention.batch_size == 0
                || attention.query_head_count == 0
                || attention.kv_head_count == 0
                || attention.head_dim == 0
                || attention.query_length == 0
                || attention.query_head_count % attention.kv_head_count != 0
                || attention.current_kv_length < attention.query_length
                || !attention.scale.is_finite()
                || attention.scale <= 0.0
            {
                return Err(LoweringError::InvalidAttribute(
                    "Attention geometry or scale is invalid",
                ));
            }
            if !program
                .state_refs
                .iter()
                .any(|state| state.id == attention.state)
            {
                return Err(LoweringError::MissingState(attention.state));
            }
            if !matches!(
                attention.mask,
                AttentionMaskKind::None | AttentionMaskKind::Causal
            ) || !matches!(attention.position, AttentionPositionKind::StateLength)
            {
                return Err(LoweringError::InvalidAttribute(
                    "Attention mask or position semantics are unsupported",
                ));
            }
        }
        if operation.kind == PortableOperationKind::TopK
            && (operation.attributes.top_k.is_none()
                || operation.attributes.top_k_order.is_none()
                || operation.attributes.top_k_tie_break.is_none())
        {
            return Err(LoweringError::InvalidAttribute("TopK requires top_k"));
        }
        if operation.kind == PortableOperationKind::MatMul
            && (operation.attributes.matmul_weight_operand.is_none()
                || operation.attributes.matmul_transpose_weight.is_none())
        {
            return Err(LoweringError::InvalidAttribute(
                "MatMul requires weight operand and orientation",
            ));
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
            PortableOperationKind::BiasAdd => OperationKind::BiasAdd,
            PortableOperationKind::Rotary => OperationKind::Rotary,
            PortableOperationKind::ReshapeHeads => OperationKind::ReshapeHeads,
            PortableOperationKind::ElementwiseMul => OperationKind::ElementwiseMul,
            PortableOperationKind::ZeroLike => OperationKind::ZeroLike,
            PortableOperationKind::WeightedAdd => OperationKind::WeightedAdd,
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
        graph.operation(
            operation.id.clone(),
            kind,
            inputs,
            operation.output,
            GraphOperationAttributes {
                epsilon: operation.attributes.epsilon,
                activation: operation.attributes.activation,
                attention: operation.attributes.attention,
                rotary: operation.attributes.rotary,
                head_reshape: operation.attributes.head_reshape,
                weighted_add: operation.attributes.weighted_add,
                expert_dispatch: operation.attributes.expert_dispatch,
                top_k: operation.attributes.top_k,
                matmul_weight_operand: operation.attributes.matmul_weight_operand,
                matmul_transpose_weight: operation.attributes.matmul_transpose_weight,
                top_k_order: operation.attributes.top_k_order,
                top_k_tie_break: operation.attributes.top_k_tie_break,
            },
        );
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{
        ActivationKind, AttentionAttributes, MatMulWeightOperand, TopKOrder, TopKTieBreak,
    };

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
        assert!(
            graph
                .tensors
                .iter()
                .all(|tensor| !tensor.name.starts_with("blk."))
        );
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

    #[test]
    fn activation_requires_and_preserves_generic_kind() {
        let activation = SemanticTensorKey("activation.input".into());
        let program = PortableProgram {
            tensor_bindings: vec![TensorBinding {
                semantic: activation.clone(),
                tensor: TensorId(12),
            }],
            state_refs: vec![],
        };
        let mut region = PortableRegion {
            id: "activation".into(),
            input: ValueId(0),
            output: ValueId(1),
            operations: vec![PortableOperation {
                id: "silu".into(),
                kind: PortableOperationKind::Activation,
                inputs: vec![PortableInput::Value(ValueId(0))],
                output: ValueId(1),
                attributes: OperationAttributes {
                    activation: Some(ActivationKind::Silu),
                    ..Default::default()
                },
            }],
        };
        let graph = lower_region(&program, &region).unwrap();
        assert_eq!(graph.operations[0].kind, OperationKind::Activation);
        assert_eq!(
            graph.operations[0].attributes.activation,
            Some(ActivationKind::Silu)
        );
        region.operations[0].attributes.activation = None;
        assert_eq!(
            lower_region(&program, &region),
            Err(LoweringError::InvalidAttribute(
                "Activation requires an activation kind"
            ))
        );
    }

    #[test]
    fn lowers_mha_and_gqa_attention_with_explicit_state_contract() {
        let state = StateId(7);
        let program = PortableProgram {
            tensor_bindings: vec![TensorBinding {
                semantic: SemanticTensorKey("value".into()),
                tensor: TensorId(12),
            }],
            state_refs: vec![StateRef {
                id: state,
                kind: "kv-cache".into(),
                lifetime: "request".into(),
                scope: "attention".into(),
            }],
            ..Default::default()
        };
        let attention = PortableOperation {
            id: "attention".into(),
            kind: PortableOperationKind::Attention,
            inputs: vec![
                PortableInput::Value(ValueId(1)),
                PortableInput::Value(ValueId(2)),
                PortableInput::Tensor(SemanticTensorKey("value".into())),
            ],
            output: ValueId(3),
            attributes: OperationAttributes {
                attention: Some(AttentionAttributes {
                    batch_size: 1,
                    query_head_count: 4,
                    kv_head_count: 2,
                    head_dim: 64,
                    query_length: 8,
                    current_kv_length: 8,
                    scale: 1.0 / 8.0,
                    mask: AttentionMaskKind::Causal,
                    position: AttentionPositionKind::StateLength,
                    state,
                }),
                ..Default::default()
            },
        };
        let region = PortableRegion {
            id: "attention".into(),
            input: ValueId(1),
            output: ValueId(3),
            operations: vec![attention],
        };
        let graph = lower_region(&program, &region).unwrap();
        assert_eq!(graph.operations[0].kind, OperationKind::Attention);
        assert_eq!(graph.operations[0].inputs.len(), 3);
        assert_eq!(
            graph.operations[0].attributes.attention.unwrap().state,
            state
        );
    }

    #[test]
    fn attention_rejects_missing_state_and_invalid_gqa_geometry() {
        let mut operation = PortableOperation {
            id: "attention".into(),
            kind: PortableOperationKind::Attention,
            inputs: vec![
                PortableInput::Value(ValueId(1)),
                PortableInput::Value(ValueId(2)),
                PortableInput::Value(ValueId(3)),
            ],
            output: ValueId(4),
            attributes: OperationAttributes {
                attention: Some(AttentionAttributes {
                    batch_size: 1,
                    query_head_count: 2,
                    kv_head_count: 2,
                    head_dim: 8,
                    query_length: 1,
                    current_kv_length: 1,
                    scale: 1.0,
                    mask: AttentionMaskKind::Causal,
                    position: AttentionPositionKind::StateLength,
                    state: StateId(99),
                }),
                ..Default::default()
            },
        };
        let region = PortableRegion {
            id: "attention".into(),
            input: ValueId(1),
            output: ValueId(4),
            operations: vec![operation.clone()],
        };
        assert_eq!(
            lower_region(&PortableProgram::default(), &region),
            Err(LoweringError::MissingState(StateId(99)))
        );
        operation
            .attributes
            .attention
            .as_mut()
            .unwrap()
            .kv_head_count = 4;
        let region = PortableRegion {
            operations: vec![operation],
            ..region
        };
        assert_eq!(
            lower_region(
                &PortableProgram {
                    state_refs: vec![StateRef {
                        id: StateId(99),
                        kind: "kv-cache".into(),
                        lifetime: "request".into(),
                        scope: "attention".into(),
                    }],
                    ..Default::default()
                },
                &region,
            ),
            Err(LoweringError::InvalidAttribute(
                "Attention geometry or scale is invalid"
            ))
        );
    }
}
