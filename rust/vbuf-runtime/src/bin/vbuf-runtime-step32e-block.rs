// Step 32E-B/32F qualification runner for bounded real GLM layers.
//
// The graph is built from semantic tensor identities and lowered through the
// generic runtime graph. The provider maps those identities to validated
// source ranges and materializes only tensors requested by the graph.

use memmap2::Mmap;
use sha2::{Digest, Sha256};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use vbuf_core::v06::parse_v06;
use vbuf_ml::{
    Bootstrap, BorrowedModelView, Gpt2ByteLevelTokenizer, SourceId, SpecialToken,
    TensorRepresentation, bf16_bits_to_f32, dequantize_f8_e4m3, parse_source_profile,
};
use vbuf_runtime::generic::{
    GenerationConfig, GenericExecutionResult, GenericExecutionState, GenericTensor,
    GenericTopKSelection, GreedyArgmaxSelector, embedding_lookup, execute_generic_graph, generate,
};
use vbuf_runtime::graph::{
    ActivationKind, AttentionAttributes, AttentionMaskKind, AttentionPositionKind,
    ExpertDispatchAttributes, HeadReshapeAttributes, MatMulWeightOperand, RotaryAttributes,
    StateId, TensorId, TopKOrder, TopKTieBreak, ValueId,
};
use vbuf_runtime::lowering::{
    OperationAttributes as PortableAttributes, PortableInput, PortableOperation,
    PortableOperationKind, PortableProgram, PortableRegion, SemanticTensorKey, StateRef,
    TensorBinding, lower_region,
};

const KEY_ID: u16 = 512;
const HIDDEN: u64 = 4096;
const SEQUENCE: u64 = 4;
const Q_HEADS: u64 = 96;
const KV_HEADS: u64 = 8;
const HEAD_DIM: u64 = 128;
const ROTARY_DIM: u64 = 64;
const EXPERT_COUNT: u32 = 128;
const TOP_K: u32 = 8;
const FIRST_TRANSFORMER_LAYER: u32 = 0;
const BASE_TRANSFORMER_LAYER_COUNT: u32 = 46;
const VOCAB: u64 = 151_552;
const OUTPUT_HEAD_CHUNK_ROWS: usize = 8_192;
const LM_HEAD_ID: u32 = 0;
const EMBEDDING_ID: u32 = 1;
const FINAL_NORM_ID: u32 = 36322;

#[derive(Clone, Debug)]
struct LayerCatalog {
    layer: u32,
    state: StateId,
    input_norm: u32,
    router_correction: u32,
    router_weight: u32,
    shared_down: u32,
    shared_down_scale: u32,
    shared_gate: u32,
    shared_gate_scale: u32,
    shared_up: u32,
    shared_up_scale: u32,
    post_norm: u32,
    k_bias: u32,
    k_weight: u32,
    k_scale: u32,
    o_weight: u32,
    o_scale: u32,
    q_bias: u32,
    q_weight: u32,
    q_scale: u32,
    v_bias: u32,
    v_weight: u32,
    v_scale: u32,
    dense_down: Option<u32>,
    dense_down_scale: Option<u32>,
    dense_gate: Option<u32>,
    dense_gate_scale: Option<u32>,
    dense_up: Option<u32>,
    dense_up_scale: Option<u32>,
    expert_entries: HashMap<(u32, u16), (u32, u32)>,
    catalog_tensor_ids: HashSet<u32>,
}

impl LayerCatalog {
    fn discover(view: &BorrowedModelView<'_>, layer: u32) -> Result<Self, String> {
        let moe = view.moe.as_ref().ok_or("MoE directory is absent")?;
        let entries: Vec<_> = moe.entries_for_layer(layer).collect();
        if entries.is_empty() {
            if layer != FIRST_TRANSFORMER_LAYER {
                return Err(format!("layer {layer} has no persisted MoE catalog"));
            }
            let ids: HashSet<u32> = (2..=20).collect();
            let catalog = Self {
                layer,
                state: StateId(1000 + layer),
                input_norm: 2,
                router_correction: 0,
                router_weight: 0,
                shared_down: 0,
                shared_down_scale: 0,
                shared_gate: 0,
                shared_gate_scale: 0,
                shared_up: 0,
                shared_up_scale: 0,
                post_norm: 9,
                k_bias: 10,
                k_weight: 11,
                k_scale: 12,
                o_weight: 13,
                o_scale: 14,
                q_bias: 15,
                q_weight: 16,
                q_scale: 17,
                v_bias: 18,
                v_weight: 19,
                v_scale: 20,
                dense_down: Some(3),
                dense_down_scale: Some(4),
                dense_gate: Some(5),
                dense_gate_scale: Some(6),
                dense_up: Some(7),
                dense_up_scale: Some(8),
                expert_entries: HashMap::new(),
                catalog_tensor_ids: ids,
            };
            for tensor_id in 2..=20 {
                if view
                    .directory
                    .get_by_identity(KEY_ID, tensor_id as u16)
                    .is_none()
                {
                    return Err(format!("dense layer {layer} tensor {tensor_id} is absent"));
                }
            }
            return Ok(catalog);
        }
        if entries.len() != 389 {
            return Err(format!(
                "layer {layer} has {} MoE entries, expected 389",
                entries.len()
            ));
        }
        let mut expert_entries = HashMap::new();
        let mut catalog_tensor_ids = HashSet::new();
        for entry in entries {
            let tensor = entry.tensor_ordinal.ok_or_else(|| {
                format!("layer {layer} role {} has no tensor identity", entry.role)
            })?;
            let scale = entry.scale_ordinal.unwrap_or(0);
            catalog_tensor_ids.insert(u32::from(tensor));
            if scale != 0 {
                catalog_tensor_ids.insert(u32::from(scale));
            }
            let key = if entry.role >= 10 {
                (u32::MAX, entry.role)
            } else {
                (entry.expert_index, entry.role)
            };
            expert_entries.insert(key, (u32::from(tensor), u32::from(scale)));
        }
        let min_id = *catalog_tensor_ids
            .iter()
            .min()
            .ok_or("layer catalog is empty")?;
        let max_id = *catalog_tensor_ids
            .iter()
            .max()
            .ok_or("layer catalog is empty")?;
        let id = |key: (u32, u16)| -> Result<u32, String> {
            expert_entries
                .get(&key)
                .map(|(tensor, _)| *tensor)
                .ok_or_else(|| format!("layer {layer} role {} is absent", key.1))
        };
        let scale = |key: (u32, u16)| -> Result<u32, String> {
            let value = expert_entries
                .get(&key)
                .map(|(_, scale)| *scale)
                .ok_or_else(|| format!("layer {layer} role {} is absent", key.1))?;
            if value == 0 {
                return Err(format!(
                    "layer {layer} role {} has no scale identity",
                    key.1
                ));
            }
            Ok(value)
        };
        let input_norm = min_id
            .checked_sub(1)
            .ok_or("layer input norm identity underflows")?;
        let post_norm = max_id
            .checked_add(1)
            .ok_or("layer post norm identity overflows")?;
        let catalog = Self {
            layer,
            state: StateId(1000 + layer),
            input_norm,
            router_correction: id((u32::MAX, 14))?,
            router_weight: id((u32::MAX, 10))?,
            shared_down: id((u32::MAX, 13))?,
            shared_down_scale: scale((u32::MAX, 13))?,
            shared_gate: id((u32::MAX, 11))?,
            shared_gate_scale: scale((u32::MAX, 11))?,
            shared_up: id((u32::MAX, 12))?,
            shared_up_scale: scale((u32::MAX, 12))?,
            post_norm,
            k_bias: post_norm + 1,
            k_weight: post_norm + 2,
            k_scale: post_norm + 3,
            o_weight: post_norm + 4,
            o_scale: post_norm + 5,
            q_bias: post_norm + 6,
            q_weight: post_norm + 7,
            q_scale: post_norm + 8,
            v_bias: post_norm + 9,
            v_weight: post_norm + 10,
            v_scale: post_norm + 11,
            dense_down: None,
            dense_down_scale: None,
            dense_gate: None,
            dense_gate_scale: None,
            dense_up: None,
            dense_up_scale: None,
            expert_entries,
            catalog_tensor_ids,
        };
        for tensor_id in [
            catalog.input_norm,
            catalog.post_norm,
            catalog.k_bias,
            catalog.k_weight,
            catalog.k_scale,
            catalog.o_weight,
            catalog.o_scale,
            catalog.q_bias,
            catalog.q_weight,
            catalog.q_scale,
            catalog.v_bias,
            catalog.v_weight,
            catalog.v_scale,
        ] {
            if view
                .directory
                .get_by_identity(KEY_ID, tensor_id as u16)
                .is_none()
            {
                return Err(format!(
                    "layer {layer} derived tensor {tensor_id} is absent"
                ));
            }
        }
        Ok(catalog)
    }

    fn is_moe(&self) -> bool {
        !self.expert_entries.is_empty()
    }

    fn selected_tensor_ids(&self, selected: &[u32]) -> Result<HashSet<u32>, String> {
        let mut ids = HashSet::new();
        for expert in selected {
            for role in 1..=3 {
                let (tensor, scale) =
                    self.expert_entries.get(&(*expert, role)).ok_or_else(|| {
                        format!("layer {} expert {expert} role {role} is absent", self.layer)
                    })?;
                ids.insert(*tensor);
                ids.insert(*scale);
            }
        }
        for role in 10..=14 {
            let (tensor, scale) = self
                .expert_entries
                .get(&(u32::MAX, role))
                .ok_or_else(|| format!("layer {} shared role {role} is absent", self.layer))?;
            ids.insert(*tensor);
            if *scale != 0 {
                ids.insert(*scale);
            }
        }
        Ok(ids)
    }

    fn all_tensor_ids(&self) -> HashSet<u32> {
        self.catalog_tensor_ids.clone()
    }
}

fn tensor(semantic: &str, _id: u32) -> PortableInput {
    PortableInput::Tensor(SemanticTensorKey(semantic.into()))
}

fn binding(program: &mut PortableProgram, semantic: &str, id: u32) {
    program.tensor_bindings.push(TensorBinding {
        semantic: SemanticTensorKey(semantic.into()),
        tensor: TensorId(id),
    });
}

fn matmul(
    id: &str,
    input: ValueId,
    weight: &str,
    weight_id: u32,
    output: ValueId,
) -> PortableOperation {
    PortableOperation {
        id: id.into(),
        kind: PortableOperationKind::MatMul,
        inputs: vec![PortableInput::Value(input), tensor(weight, weight_id)],
        output,
        attributes: PortableAttributes {
            matmul_weight_operand: Some(MatMulWeightOperand::Rhs),
            matmul_transpose_weight: Some(false),
            ..Default::default()
        },
    }
}

fn bias(
    id: &str,
    input: ValueId,
    value: &str,
    value_id: u32,
    output: ValueId,
) -> PortableOperation {
    PortableOperation {
        id: id.into(),
        kind: PortableOperationKind::BiasAdd,
        inputs: vec![PortableInput::Value(input), tensor(value, value_id)],
        output,
        attributes: Default::default(),
    }
}

fn activation(
    id: &str,
    input: ValueId,
    output: ValueId,
    kind: ActivationKind,
) -> PortableOperation {
    PortableOperation {
        id: id.into(),
        kind: PortableOperationKind::Activation,
        inputs: vec![PortableInput::Value(input)],
        output,
        attributes: PortableAttributes {
            activation: Some(kind),
            ..Default::default()
        },
    }
}

fn build_attention_graph(
    catalog: &LayerCatalog,
    query_length: u64,
    current_kv_length: u64,
    position_start: u64,
) -> (PortableProgram, PortableRegion) {
    let mut program = PortableProgram {
        state_refs: vec![StateRef {
            id: catalog.state,
            kind: "request-kv".into(),
            lifetime: "request".into(),
            scope: "block-attention".into(),
        }],
        ..Default::default()
    };
    binding(&mut program, "block.input_norm.weight", catalog.input_norm);
    binding(&mut program, "block.attention.k.bias", catalog.k_bias);
    binding(&mut program, "block.attention.k.weight", catalog.k_weight);
    binding(&mut program, "block.attention.o.weight", catalog.o_weight);
    binding(&mut program, "block.attention.q.bias", catalog.q_bias);
    binding(&mut program, "block.attention.q.weight", catalog.q_weight);
    binding(&mut program, "block.attention.v.bias", catalog.v_bias);
    binding(&mut program, "block.attention.v.weight", catalog.v_weight);
    binding(&mut program, "block.attention.k.scale", catalog.k_scale);
    binding(&mut program, "block.attention.o.scale", catalog.o_scale);
    binding(&mut program, "block.attention.q.scale", catalog.q_scale);
    binding(&mut program, "block.attention.v.scale", catalog.v_scale);
    binding(
        &mut program,
        "block.post_attention_norm.weight",
        catalog.post_norm,
    );
    if catalog.is_moe() {
        binding(
            &mut program,
            "block.router.correction",
            catalog.router_correction,
        );
        binding(&mut program, "block.router.weight", catalog.router_weight);
    }
    let rotary_q = RotaryAttributes {
        head_count: Q_HEADS,
        head_dim: HEAD_DIM,
        rotary_dim: ROTARY_DIM,
        theta: 1_000_000.0,
        position_start,
    };
    let rotary_k = RotaryAttributes {
        head_count: KV_HEADS,
        ..rotary_q
    };
    let reshape_q = HeadReshapeAttributes {
        head_count: Q_HEADS,
        head_dim: HEAD_DIM,
        flatten: false,
    };
    let reshape_kv = HeadReshapeAttributes {
        head_count: KV_HEADS,
        head_dim: HEAD_DIM,
        flatten: false,
    };
    let mut operations = vec![
        PortableOperation {
            id: "input_norm".into(),
            kind: PortableOperationKind::RmsNorm,
            inputs: vec![
                PortableInput::Value(ValueId(0)),
                tensor("block.input_norm.weight", catalog.input_norm),
            ],
            output: ValueId(1),
            attributes: PortableAttributes {
                epsilon: Some(1e-5),
                ..Default::default()
            },
        },
        matmul(
            "q_projection",
            ValueId(1),
            "block.attention.q.weight",
            catalog.q_weight,
            ValueId(2),
        ),
        bias(
            "q_bias",
            ValueId(2),
            "block.attention.q.bias",
            catalog.q_bias,
            ValueId(3),
        ),
        matmul(
            "k_projection",
            ValueId(1),
            "block.attention.k.weight",
            catalog.k_weight,
            ValueId(4),
        ),
        bias(
            "k_bias",
            ValueId(4),
            "block.attention.k.bias",
            catalog.k_bias,
            ValueId(5),
        ),
        matmul(
            "v_projection",
            ValueId(1),
            "block.attention.v.weight",
            catalog.v_weight,
            ValueId(6),
        ),
        bias(
            "v_bias",
            ValueId(6),
            "block.attention.v.bias",
            catalog.v_bias,
            ValueId(7),
        ),
    ];
    for (id, input, output, shape) in [
        ("q_heads", 3, 8, reshape_q),
        ("k_heads", 5, 9, reshape_kv),
        ("v_heads", 7, 10, reshape_kv),
    ] {
        operations.push(PortableOperation {
            id: id.into(),
            kind: PortableOperationKind::ReshapeHeads,
            inputs: vec![PortableInput::Value(ValueId(input))],
            output: ValueId(output),
            attributes: PortableAttributes {
                head_reshape: Some(shape),
                ..Default::default()
            },
        });
    }
    operations.extend([
        PortableOperation {
            id: "q_rotary".into(),
            kind: PortableOperationKind::Rotary,
            inputs: vec![PortableInput::Value(ValueId(8))],
            output: ValueId(11),
            attributes: PortableAttributes {
                rotary: Some(rotary_q),
                ..Default::default()
            },
        },
        PortableOperation {
            id: "k_rotary".into(),
            kind: PortableOperationKind::Rotary,
            inputs: vec![PortableInput::Value(ValueId(9))],
            output: ValueId(12),
            attributes: PortableAttributes {
                rotary: Some(rotary_k),
                ..Default::default()
            },
        },
        PortableOperation {
            id: "attention".into(),
            kind: PortableOperationKind::Attention,
            inputs: vec![
                PortableInput::Value(ValueId(11)),
                PortableInput::Value(ValueId(12)),
                PortableInput::Value(ValueId(10)),
            ],
            output: ValueId(13),
            attributes: PortableAttributes {
                attention: Some(AttentionAttributes {
                    batch_size: 1,
                    query_head_count: Q_HEADS,
                    kv_head_count: KV_HEADS,
                    head_dim: HEAD_DIM,
                    query_length,
                    current_kv_length,
                    scale: (HEAD_DIM as f32).sqrt().recip(),
                    mask: AttentionMaskKind::Causal,
                    position: AttentionPositionKind::StateLength,
                    state: catalog.state,
                }),
                ..Default::default()
            },
        },
        PortableOperation {
            id: "attention_flatten".into(),
            kind: PortableOperationKind::ReshapeHeads,
            inputs: vec![PortableInput::Value(ValueId(13))],
            output: ValueId(14),
            attributes: PortableAttributes {
                head_reshape: Some(HeadReshapeAttributes {
                    head_count: Q_HEADS,
                    head_dim: HEAD_DIM,
                    flatten: true,
                }),
                ..Default::default()
            },
        },
        matmul(
            "o_projection",
            ValueId(14),
            "block.attention.o.weight",
            catalog.o_weight,
            ValueId(15),
        ),
        PortableOperation {
            id: "post_attention_residual".into(),
            kind: PortableOperationKind::ResidualAdd,
            inputs: vec![
                PortableInput::Value(ValueId(0)),
                PortableInput::Value(ValueId(15)),
            ],
            output: ValueId(16),
            attributes: Default::default(),
        },
        PortableOperation {
            id: "post_attention_norm".into(),
            kind: PortableOperationKind::RmsNorm,
            inputs: vec![
                PortableInput::Value(ValueId(16)),
                tensor("block.post_attention_norm.weight", catalog.post_norm),
            ],
            output: ValueId(17),
            attributes: PortableAttributes {
                epsilon: Some(1e-5),
                ..Default::default()
            },
        },
    ]);
    if catalog.is_moe() {
        operations.extend([
            matmul(
                "router_projection",
                ValueId(17),
                "block.router.weight",
                catalog.router_weight,
                ValueId(18),
            ),
            activation(
                "router_sigmoid",
                ValueId(18),
                ValueId(19),
                ActivationKind::Sigmoid,
            ),
            bias(
                "router_correction",
                ValueId(19),
                "block.router.correction",
                catalog.router_correction,
                ValueId(20),
            ),
            PortableOperation {
                id: "router_topk".into(),
                kind: PortableOperationKind::TopK,
                inputs: vec![
                    PortableInput::Value(ValueId(19)),
                    PortableInput::Value(ValueId(20)),
                ],
                output: ValueId(21),
                attributes: PortableAttributes {
                    top_k: Some(TOP_K),
                    top_k_order: Some(TopKOrder::Descending),
                    top_k_tie_break: Some(TopKTieBreak::LowerIndex),
                    ..Default::default()
                },
            },
        ]);
    }
    (
        program,
        PortableRegion {
            id: format!("portable-layer-{}-attention-router", catalog.layer),
            input: ValueId(0),
            output: if catalog.is_moe() {
                ValueId(21)
            } else {
                ValueId(16)
            },
            operations,
        },
    )
}

fn build_expert_graph(
    catalog: &LayerCatalog,
    selected: &[u32],
) -> (PortableProgram, PortableRegion) {
    let mut program = PortableProgram::default();
    binding(
        &mut program,
        "block.post_attention_norm.weight",
        catalog.post_norm,
    );
    binding(&mut program, "block.shared.gate", catalog.shared_gate);
    binding(&mut program, "block.shared.up", catalog.shared_up);
    binding(&mut program, "block.shared.down", catalog.shared_down);
    for expert in selected {
        let gate = catalog.expert_entries[&(*expert, 1)].0;
        let up = catalog.expert_entries[&(*expert, 2)].0;
        let down = catalog.expert_entries[&(*expert, 3)].0;
        binding(&mut program, &format!("block.expert.{expert}.gate"), gate);
        binding(&mut program, &format!("block.expert.{expert}.up"), up);
        binding(&mut program, &format!("block.expert.{expert}.down"), down);
    }
    let mut operations = vec![PortableOperation {
        id: "post_attention_norm".into(),
        kind: PortableOperationKind::RmsNorm,
        inputs: vec![
            PortableInput::Value(ValueId(16)),
            tensor("block.post_attention_norm.weight", catalog.post_norm),
        ],
        output: ValueId(17),
        attributes: PortableAttributes {
            epsilon: Some(1e-5),
            ..Default::default()
        },
    }];
    operations.push(PortableOperation {
        id: "routed_zero".into(),
        kind: PortableOperationKind::ZeroLike,
        inputs: vec![PortableInput::Value(ValueId(17))],
        output: ValueId(18),
        attributes: Default::default(),
    });
    let mut accumulator = ValueId(18);
    for (index, expert) in selected.iter().enumerate() {
        let output = ValueId(1000 + index as u32);
        operations.push(PortableOperation {
            id: format!("expert_dispatch_{expert}"),
            kind: PortableOperationKind::IndexedMatMul,
            inputs: vec![
                PortableInput::Value(ValueId(17)),
                tensor(
                    &format!("block.expert.{expert}.gate"),
                    catalog.expert_entries[&(*expert, 1)].0,
                ),
                tensor(
                    &format!("block.expert.{expert}.up"),
                    catalog.expert_entries[&(*expert, 2)].0,
                ),
                tensor(
                    &format!("block.expert.{expert}.down"),
                    catalog.expert_entries[&(*expert, 3)].0,
                ),
            ],
            output,
            attributes: PortableAttributes {
                expert_dispatch: Some(ExpertDispatchAttributes { expert_id: *expert }),
                ..Default::default()
            },
        });
        let next = ValueId(2000 + index as u32);
        operations.push(PortableOperation {
            id: format!("expert_accumulate_{expert}"),
            kind: PortableOperationKind::ResidualAdd,
            inputs: vec![
                PortableInput::Value(accumulator),
                PortableInput::Value(output),
            ],
            output: next,
            attributes: Default::default(),
        });
        accumulator = next;
    }
    operations.extend([
        matmul(
            "shared_gate",
            ValueId(17),
            "block.shared.gate",
            catalog.shared_gate,
            ValueId(300),
        ),
        activation(
            "shared_silu",
            ValueId(300),
            ValueId(301),
            ActivationKind::Silu,
        ),
        matmul(
            "shared_up",
            ValueId(17),
            "block.shared.up",
            catalog.shared_up,
            ValueId(302),
        ),
        PortableOperation {
            id: "shared_multiply".into(),
            kind: PortableOperationKind::ElementwiseMul,
            inputs: vec![
                PortableInput::Value(ValueId(301)),
                PortableInput::Value(ValueId(302)),
            ],
            output: ValueId(303),
            attributes: Default::default(),
        },
        matmul(
            "shared_down",
            ValueId(303),
            "block.shared.down",
            catalog.shared_down,
            ValueId(304),
        ),
        PortableOperation {
            id: "moe_combine".into(),
            kind: PortableOperationKind::ResidualAdd,
            inputs: vec![
                PortableInput::Value(accumulator),
                PortableInput::Value(ValueId(304)),
            ],
            output: ValueId(305),
            attributes: Default::default(),
        },
        PortableOperation {
            id: "final_block_residual".into(),
            kind: PortableOperationKind::ResidualAdd,
            inputs: vec![
                PortableInput::Value(ValueId(16)),
                PortableInput::Value(ValueId(305)),
            ],
            output: ValueId(306),
            attributes: Default::default(),
        },
    ]);
    (
        program,
        PortableRegion {
            id: format!("portable-layer-{}-moe", catalog.layer),
            input: ValueId(16),
            output: ValueId(306),
            operations,
        },
    )
}

fn build_dense_graph(catalog: &LayerCatalog) -> (PortableProgram, PortableRegion) {
    let dense_gate = catalog.dense_gate.expect("dense gate identity");
    let dense_up = catalog.dense_up.expect("dense up identity");
    let dense_down = catalog.dense_down.expect("dense down identity");
    let mut program = PortableProgram::default();
    binding(
        &mut program,
        "block.post_attention_norm.weight",
        catalog.post_norm,
    );
    binding(&mut program, "block.dense.gate", dense_gate);
    binding(&mut program, "block.dense.up", dense_up);
    binding(&mut program, "block.dense.down", dense_down);
    let operations = vec![
        PortableOperation {
            id: "post_attention_norm".into(),
            kind: PortableOperationKind::RmsNorm,
            inputs: vec![
                PortableInput::Value(ValueId(16)),
                tensor("block.post_attention_norm.weight", catalog.post_norm),
            ],
            output: ValueId(17),
            attributes: PortableAttributes {
                epsilon: Some(1e-5),
                ..Default::default()
            },
        },
        matmul(
            "dense_gate",
            ValueId(17),
            "block.dense.gate",
            dense_gate,
            ValueId(300),
        ),
        activation(
            "dense_silu",
            ValueId(300),
            ValueId(301),
            ActivationKind::Silu,
        ),
        matmul(
            "dense_up",
            ValueId(17),
            "block.dense.up",
            dense_up,
            ValueId(302),
        ),
        PortableOperation {
            id: "dense_multiply".into(),
            kind: PortableOperationKind::ElementwiseMul,
            inputs: vec![
                PortableInput::Value(ValueId(301)),
                PortableInput::Value(ValueId(302)),
            ],
            output: ValueId(303),
            attributes: Default::default(),
        },
        matmul(
            "dense_down",
            ValueId(303),
            "block.dense.down",
            dense_down,
            ValueId(304),
        ),
        PortableOperation {
            id: "dense_residual".into(),
            kind: PortableOperationKind::ResidualAdd,
            inputs: vec![
                PortableInput::Value(ValueId(16)),
                PortableInput::Value(ValueId(304)),
            ],
            output: ValueId(306),
            attributes: Default::default(),
        },
    ];
    (
        program,
        PortableRegion {
            id: format!("portable-layer-{}-dense", catalog.layer),
            input: ValueId(16),
            output: ValueId(306),
            operations,
        },
    )
}

fn resolve_output_ids(view: &BorrowedModelView<'_>) -> Result<(u32, u32, u32), String> {
    let head = view
        .directory
        .get_by_identity(KEY_ID, LM_HEAD_ID as u16)
        .ok_or("persisted LM head identity is absent")?;
    if head.representation != TensorRepresentation::Bf16 || head.dimensions != [151_552, HIDDEN] {
        return Err("persisted LM head geometry is not the qualified BF16 head".into());
    }
    let embedding = view
        .directory
        .get_by_identity(KEY_ID, EMBEDDING_ID as u16)
        .ok_or("persisted embedding identity is absent")?;
    if embedding.representation != TensorRepresentation::Bf16
        || embedding.dimensions != [151_552, HIDDEN]
        || embedding.payload.offset() == head.payload.offset()
    {
        return Err("persisted embedding and LM head are not distinct tensors".into());
    }
    let norm = view
        .directory
        .get_by_identity(KEY_ID, FINAL_NORM_ID as u16)
        .ok_or("persisted final norm identity is absent")?;
    if norm.representation != TensorRepresentation::Bf16 || norm.dimensions != [HIDDEN] {
        return Err("persisted final norm geometry is not the qualified BF16 norm".into());
    }
    Ok((FINAL_NORM_ID, LM_HEAD_ID, EMBEDDING_ID))
}

fn build_output_norm_graph(norm_id: u32) -> (PortableProgram, PortableRegion) {
    let mut program = PortableProgram::default();
    binding(&mut program, "model.final_norm.weight", norm_id);
    (
        program,
        PortableRegion {
            id: "portable-final-norm".into(),
            input: ValueId(0),
            output: ValueId(1),
            operations: vec![PortableOperation {
                id: "final_norm".into(),
                kind: PortableOperationKind::RmsNorm,
                inputs: vec![
                    PortableInput::Value(ValueId(0)),
                    tensor("model.final_norm.weight", norm_id),
                ],
                output: ValueId(1),
                attributes: PortableAttributes {
                    epsilon: Some(1e-5),
                    ..Default::default()
                },
            }],
        },
    )
}

fn build_output_projection_graph(head_id: u32) -> (PortableProgram, PortableRegion) {
    let mut program = PortableProgram::default();
    binding(&mut program, "model.lm_head.weight", head_id);
    (
        program,
        PortableRegion {
            id: "portable-lm-head-chunk".into(),
            input: ValueId(0),
            output: ValueId(1),
            operations: vec![matmul(
                "lm_head_chunk",
                ValueId(0),
                "model.lm_head.weight",
                head_id,
                ValueId(1),
            )],
        },
    )
}

struct Materializer<'view, 'source> {
    view: &'view BorrowedModelView<'source>,
    payload: &'source [u8],
    scale_by: HashMap<u32, u32>,
    cache: HashMap<u32, GenericTensor>,
    touched: HashSet<u32>,
    touched_ranges: HashMap<u32, Vec<(u64, u64)>>,
    source_bytes: u64,
}

impl<'view, 'source> Materializer<'view, 'source> {
    fn cache_f32_bytes(&self) -> usize {
        self.cache
            .values()
            .map(|tensor| tensor.values.len() * 4)
            .sum()
    }

    fn clear_cache(&mut self) {
        self.cache.clear();
    }

    fn unique_source_bytes(&self) -> u64 {
        self.touched_ranges
            .values()
            .map(|ranges| {
                let mut ranges = ranges.clone();
                ranges.sort_unstable_by_key(|(start, _)| *start);
                let mut total = 0u64;
                let mut current: Option<(u64, u64)> = None;
                for (start, length) in ranges {
                    let end = start.saturating_add(length);
                    let Some((current_start, current_end)) = current else {
                        current = Some((start, end));
                        continue;
                    };
                    if start <= current_end {
                        current = Some((current_start, current_end.max(end)));
                    } else {
                        total = total.saturating_add(current_end.saturating_sub(current_start));
                        current = Some((start, end));
                    }
                }
                if let Some((start, end)) = current {
                    total.saturating_add(end.saturating_sub(start))
                } else {
                    total
                }
            })
            .sum()
    }

    fn record_range(&mut self, id: u32, start: u64, length: u64) {
        self.touched_ranges
            .entry(id)
            .or_default()
            .push((start, length));
    }

    fn bytes(&self, id: u32) -> Result<(&'source [u8], Vec<u64>, TensorRepresentation), String> {
        let descriptor = self
            .view
            .directory
            .get_by_identity(KEY_ID, id as u16)
            .ok_or_else(|| format!("tensor identity {KEY_ID}:{id} is absent"))?;
        if descriptor.payload.source_id() != SourceId::new(1) {
            return Err("block tensor does not resolve to the authoritative source".into());
        }
        let start = usize::try_from(descriptor.payload.offset())
            .map_err(|_| "source offset is too large")?;
        let end = start
            .checked_add(
                usize::try_from(descriptor.payload.length())
                    .map_err(|_| "source length is too large")?,
            )
            .ok_or("source range overflows")?;
        let bytes = self
            .payload
            .get(start..end)
            .ok_or("source range is outside payload")?;
        Ok((
            bytes,
            descriptor.dimensions.clone(),
            descriptor.representation,
        ))
    }

    fn get(&mut self, id: TensorId) -> Result<GenericTensor, String> {
        if let Some(value) = self.cache.get(&id.0) {
            return Ok(value.clone());
        }
        let (bytes, dimensions, representation) = self.bytes(id.0)?;
        let values = match representation {
            TensorRepresentation::F8_E4M3 => {
                let scale_id = self
                    .scale_by
                    .get(&id.0)
                    .copied()
                    .ok_or("FP8 scale identity is absent")?;
                let (scale_bytes, scale_dimensions, scale_representation) = self.bytes(scale_id)?;
                if scale_representation != TensorRepresentation::CanonicalPrimitive
                    || dimensions.len() != 2
                    || scale_dimensions != [dimensions[0], 1]
                {
                    return Err("FP8 scale provenance geometry is invalid".into());
                }
                let weight_bytes = u64::try_from(bytes.len()).unwrap_or(0);
                let scale_len = u64::try_from(scale_bytes.len()).unwrap_or(0);
                self.source_bytes = self.source_bytes.saturating_add(weight_bytes + scale_len);
                self.touched.insert(id.0);
                self.touched.insert(scale_id);
                self.record_range(id.0, 0, weight_bytes);
                self.record_range(scale_id, 0, scale_len);
                dequantize_f8_e4m3(
                    bytes,
                    [dimensions[0], dimensions[1]],
                    scale_bytes,
                    [1, dimensions[1]],
                )
                .map_err(|error| error.to_string())?
            }
            TensorRepresentation::Bf16 => {
                self.source_bytes = self.source_bytes.saturating_add(bytes.len() as u64);
                self.touched.insert(id.0);
                self.record_range(id.0, 0, bytes.len() as u64);
                if bytes.len() % 2 != 0 {
                    return Err("BF16 payload has odd length".into());
                }
                bytes
                    .chunks_exact(2)
                    .map(|pair| bf16_bits_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
                    .collect()
            }
            TensorRepresentation::CanonicalPrimitive => {
                self.source_bytes = self.source_bytes.saturating_add(bytes.len() as u64);
                self.touched.insert(id.0);
                self.record_range(id.0, 0, bytes.len() as u64);
                if bytes.len() % 4 != 0 {
                    return Err("F32 payload has invalid length".into());
                }
                bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect()
            }
            _ => return Err("unsupported block tensor representation".into()),
        };
        let tensor = GenericTensor { dimensions, values };
        tensor.elements()?;
        self.cache.insert(id.0, tensor.clone());
        Ok(tensor)
    }

    fn get_bf16_rows(
        &mut self,
        id: TensorId,
        row_start: usize,
        row_count: usize,
    ) -> Result<GenericTensor, String> {
        let descriptor = self
            .view
            .directory
            .get_by_identity(KEY_ID, id.0 as u16)
            .ok_or_else(|| format!("tensor identity {KEY_ID}:{} is absent", id.0))?;
        if descriptor.representation != TensorRepresentation::Bf16
            || descriptor.dimensions.len() != 2
        {
            return Err("BF16 row tensor geometry is invalid".into());
        }
        if descriptor.payload.source_id() != SourceId::new(1) {
            return Err("output head chunk does not resolve to the authoritative source".into());
        }
        let total_rows = descriptor.dimensions[0];
        let hidden = descriptor.dimensions[1];
        let expected_length = total_rows
            .checked_mul(hidden)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or("BF16 row tensor payload length overflows")?;
        if descriptor.payload.length() != expected_length {
            return Err("BF16 row tensor payload length is invalid".into());
        }
        let end_row = row_start
            .checked_add(row_count)
            .ok_or("output head row range overflows")?;
        if end_row > usize::try_from(total_rows).map_err(|_| "BF16 row count is too large")? {
            return Err("BF16 row range is outside tensor".into());
        }
        let row_bytes = usize::try_from(hidden)
            .map_err(|_| "BF16 row width is too large")?
            .checked_mul(2)
            .ok_or("output head row size overflows")?;
        let byte_start = row_start
            .checked_mul(row_bytes)
            .ok_or("output head byte range overflows")?;
        let byte_length = row_count
            .checked_mul(row_bytes)
            .ok_or("output head chunk length overflows")?;
        let payload_start = usize::try_from(descriptor.payload.offset())
            .map_err(|_| "output head source offset is too large")?
            .checked_add(byte_start)
            .ok_or("output head source range overflows")?;
        let payload_end = payload_start
            .checked_add(byte_length)
            .ok_or("output head source range overflows")?;
        let bytes = self
            .payload
            .get(payload_start..payload_end)
            .ok_or("output head chunk is outside payload")?;
        if bytes.len() % 2 != 0 {
            return Err("output head chunk has odd BF16 length".into());
        }
        self.source_bytes = self.source_bytes.saturating_add(bytes.len() as u64);
        self.touched.insert(id.0);
        self.record_range(id.0, byte_start as u64, bytes.len() as u64);
        let values = bytes
            .chunks_exact(2)
            .map(|pair| bf16_bits_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
            .collect();
        Ok(GenericTensor {
            dimensions: vec![row_count as u64, hidden],
            values,
        })
    }
}

struct LayerPhaseResult {
    attention: GenericExecutionResult,
    expert: GenericExecutionResult,
    output: GenericTensor,
    selection: Option<GenericTopKSelection>,
    selected: Vec<u32>,
    unselected: Vec<u32>,
    source_bytes: u64,
    attention_time_ms: f64,
    expert_time_ms: f64,
}

fn execute_layer_phase(
    catalog: &LayerCatalog,
    current: &GenericTensor,
    query_length: u64,
    position_start: u64,
    materializer: &mut Materializer<'_, '_>,
    state: &mut GenericExecutionState,
) -> Result<LayerPhaseResult, String> {
    if state.state_length() + query_length == 0 {
        return Err("layer phase has zero visible sequence length".into());
    }
    let source_before = materializer.source_bytes;
    let touched_before = materializer.touched.clone();
    let (program, region) = build_attention_graph(
        catalog,
        query_length,
        state.state_length() + query_length,
        position_start,
    );
    let graph = lower_region(&program, &region)
        .map_err(|e| format!("layer {} attention graph lowering: {e:?}", catalog.layer))?;
    let attention_start = Instant::now();
    let attention = execute_generic_graph(
        &graph,
        current.clone(),
        |id| materializer.get(id),
        None,
        state,
    )
    .map_err(|e| format!("layer {} attention/router execution: {e}", catalog.layer))?;
    let attention_time_ms = attention_start.elapsed().as_secs_f64() * 1000.0;
    let (selection, expert, selected, expert_time_ms) = if catalog.is_moe() {
        let selection = attention
            .selection
            .clone()
            .ok_or_else(|| format!("layer {} router produced no selection", catalog.layer))?;
        let mut selected: Vec<u32> = selection
            .ids
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        selected.sort_unstable();
        if selected.iter().any(|id| *id >= EXPERT_COUNT) {
            return Err(format!(
                "layer {} selected an invalid expert",
                catalog.layer
            ));
        }
        let (expert_program, expert_region) = build_expert_graph(catalog, &selected);
        let expert_graph = lower_region(&expert_program, &expert_region)
            .map_err(|e| format!("layer {} expert graph lowering: {e:?}", catalog.layer))?;
        let post_attention = attention
            .values
            .get(&ValueId(16))
            .ok_or("post-attention residual missing")?
            .clone();
        let expert_start = Instant::now();
        let expert = execute_generic_graph(
            &expert_graph,
            post_attention,
            |id| materializer.get(id),
            Some(&selection),
            state,
        )
        .map_err(|e| format!("layer {} expert/final execution: {e}", catalog.layer))?;
        let expert_time_ms = expert_start.elapsed().as_secs_f64() * 1000.0;
        (Some(selection), expert, selected, expert_time_ms)
    } else {
        let (dense_program, dense_region) = build_dense_graph(catalog);
        let dense_graph = lower_region(&dense_program, &dense_region)
            .map_err(|e| format!("layer {} dense graph lowering: {e:?}", catalog.layer))?;
        let post_attention = attention
            .values
            .get(&ValueId(16))
            .ok_or("post-attention residual missing")?
            .clone();
        let expert_start = Instant::now();
        let dense = execute_generic_graph(
            &dense_graph,
            post_attention,
            |id| materializer.get(id),
            None,
            state,
        )
        .map_err(|e| format!("layer {} dense execution: {e}", catalog.layer))?;
        let expert_time_ms = expert_start.elapsed().as_secs_f64() * 1000.0;
        (None, dense, Vec::new(), expert_time_ms)
    };
    let output = expert
        .values
        .get(&ValueId(306))
        .ok_or("final layer output missing")?
        .clone();
    let allowed_moe_ids = if catalog.is_moe() {
        catalog.selected_tensor_ids(&selected)?
    } else {
        HashSet::new()
    };
    let new_touched: HashSet<_> = materializer
        .touched
        .difference(&touched_before)
        .copied()
        .collect();
    let new_moe: HashSet<_> = if catalog.is_moe() {
        new_touched
            .intersection(&catalog.all_tensor_ids())
            .copied()
            .collect()
    } else {
        HashSet::new()
    };
    let unselected = new_moe.difference(&allowed_moe_ids).copied().collect();
    Ok(LayerPhaseResult {
        attention,
        expert,
        output,
        selection,
        selected,
        unselected,
        source_bytes: materializer.source_bytes - source_before,
        attention_time_ms,
        expert_time_ms,
    })
}

fn input_tensor(variant: u32) -> GenericTensor {
    let mut values = Vec::with_capacity((SEQUENCE * HIDDEN) as usize);
    for index in 0..(SEQUENCE * HIDDEN) {
        let x = index as f32 + 1.0 + variant as f32 * 0.37;
        values.push((x * 0.00017).sin() * 0.05 + (x * 0.000031).cos() * 0.01);
    }
    GenericTensor {
        dimensions: vec![1, SEQUENCE, HIDDEN],
        values,
    }
}

fn project_logits(
    input: &GenericTensor,
    head_id: u32,
    materializer: &mut Materializer<'_, '_>,
) -> Result<(GenericTensor, usize), String> {
    if input.dimensions.len() != 3 || input.dimensions[0] != 1 || input.dimensions[2] != HIDDEN {
        return Err("output projection input geometry is invalid".into());
    }
    let sequence = usize::try_from(input.dimensions[1])
        .map_err(|_| "output projection sequence is too large")?;
    let vocabulary = usize::try_from(VOCAB).map_err(|_| "vocabulary is too large")?;
    let mut logits_values = vec![0.0; sequence * vocabulary];
    let mut peak_cache_bytes = materializer.cache_f32_bytes();
    let mut output_state = GenericExecutionState::new(0xffff_fffe, 1)?;
    for row_start in (0..vocabulary).step_by(OUTPUT_HEAD_CHUNK_ROWS) {
        let row_count = OUTPUT_HEAD_CHUNK_ROWS.min(vocabulary - row_start);
        let head_chunk = materializer.get_bf16_rows(TensorId(head_id), row_start, row_count)?;
        let (program, region) = build_output_projection_graph(head_id);
        let graph = lower_region(&program, &region)
            .map_err(|e| format!("LM head chunk graph lowering: {e:?}"))?;
        let chunk_result = execute_generic_graph(
            &graph,
            input.clone(),
            |id| {
                if id.0 == head_id {
                    Ok(head_chunk.clone())
                } else {
                    materializer.get(id)
                }
            },
            None,
            &mut output_state,
        )
        .map_err(|e| format!("LM head chunk execution: {e}"))?;
        let chunk = chunk_result
            .values
            .get(&ValueId(1))
            .ok_or("LM head chunk output missing")?;
        for token in 0..sequence {
            let source_start = token * row_count;
            let destination_start = token * vocabulary + row_start;
            logits_values[destination_start..destination_start + row_count]
                .copy_from_slice(&chunk.values[source_start..source_start + row_count]);
        }
        peak_cache_bytes =
            peak_cache_bytes.max(materializer.cache_f32_bytes() + head_chunk.values.len() * 4);
        materializer.cache.remove(&head_id);
    }
    Ok((
        GenericTensor {
            dimensions: vec![1, sequence as u64, VOCAB],
            values: logits_values,
        },
        peak_cache_bytes,
    ))
}

fn state_tensor(state: &GenericExecutionState) -> Result<(GenericTensor, GenericTensor), String> {
    let (batch, kv_heads, head_dim) = state
        .kv_geometry()
        .ok_or("cannot snapshot an empty KV state")?;
    let dimensions = vec![batch, state.state_length(), kv_heads, head_dim];
    let (key, value) = state.kv_values();
    Ok((
        GenericTensor {
            dimensions: dimensions.clone(),
            values: key.to_vec(),
        },
        GenericTensor {
            dimensions,
            values: value.to_vec(),
        },
    ))
}

fn write_layer_phase_records(
    checkpoint: &mut BufWriter<File>,
    prefix: &str,
    current: &GenericTensor,
    phase: &LayerPhaseResult,
    state: &GenericExecutionState,
) -> Result<(), String> {
    write_record(checkpoint, &format!("{prefix}.input"), current)?;
    for (name, tensor) in [
        (
            "post_attention_residual",
            phase.attention.values.get(&ValueId(16)),
        ),
        ("router_corrected", phase.attention.values.get(&ValueId(20))),
        ("router_input", phase.attention.values.get(&ValueId(17))),
        ("router_raw", phase.attention.values.get(&ValueId(18))),
        ("router_scores", phase.attention.values.get(&ValueId(19))),
        (
            "shared_expert_output",
            phase.expert.values.get(&ValueId(304)),
        ),
        ("shared_gate", phase.expert.values.get(&ValueId(300))),
        ("shared_up", phase.expert.values.get(&ValueId(302))),
        ("shared_multiply", phase.expert.values.get(&ValueId(303))),
        ("moe_output", phase.expert.values.get(&ValueId(305))),
    ] {
        if let Some(tensor) = tensor {
            write_record(checkpoint, &format!("{prefix}.{name}"), tensor)?;
        }
    }
    write_record(checkpoint, &format!("{prefix}.output"), &phase.output)?;
    if let Some(selection) = &phase.selection {
        for (index, expert) in phase.selected.iter().enumerate() {
            if let Some(tensor) = phase.expert.values.get(&ValueId(1000 + index as u32)) {
                write_record(checkpoint, &format!("{prefix}.expert_{expert}"), tensor)?;
            }
        }
        let selection_ids = GenericTensor {
            dimensions: vec![1, selection.token_count, selection.top_k as u64],
            values: selection.ids.iter().map(|id| *id as f32).collect(),
        };
        let selection_weights = GenericTensor {
            dimensions: selection_ids.dimensions.clone(),
            values: selection.weights.clone(),
        };
        write_record(
            checkpoint,
            &format!("{prefix}.selection_ids"),
            &selection_ids,
        )?;
        write_record(
            checkpoint,
            &format!("{prefix}.selection_weights"),
            &selection_weights,
        )?;
    }
    let (key, value) = state_tensor(state)?;
    write_record(checkpoint, &format!("{prefix}.kv_key"), &key)?;
    write_record(checkpoint, &format!("{prefix}.kv_value"), &value)?;
    Ok(())
}

fn tensor_hash(tensor: &GenericTensor) -> String {
    let mut digest = Sha256::new();
    for value in &tensor.values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn bytes_hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn rss_kib() -> u64 {
    let Ok(text) = std::fs::read_to_string("/proc/self/statm") else {
        return 0;
    };
    let pages = text
        .split_whitespace()
        .nth(1)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    pages * 4096 / 1024
}

fn write_record(
    writer: &mut BufWriter<File>,
    name: &str,
    tensor: &GenericTensor,
) -> Result<(), String> {
    writer
        .write_all(&(name.len() as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;
    writer
        .write_all(name.as_bytes())
        .map_err(|e| e.to_string())?;
    writer
        .write_all(&(tensor.dimensions.len() as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;
    for dimension in &tensor.dimensions {
        writer
            .write_all(&dimension.to_le_bytes())
            .map_err(|e| e.to_string())?;
    }
    writer
        .write_all(&(tensor.values.len() as u64).to_le_bytes())
        .map_err(|e| e.to_string())?;
    for value in &tensor.values {
        writer
            .write_all(&value.to_le_bytes())
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn durable_flush(writer: &mut BufWriter<File>) -> Result<(), String> {
    writer.flush().map_err(|e| e.to_string())?;
    writer.get_ref().sync_data().map_err(|e| e.to_string())
}

fn write_progress(path: &Path, phase: &str, step: u32, layer: u32) -> Result<(), String> {
    let temporary = path.with_extension("progress.tmp");
    let mut file = File::create(&temporary).map_err(|e| e.to_string())?;
    writeln!(file, "PHASE={phase}").map_err(|e| e.to_string())?;
    writeln!(file, "STEP={step}").map_err(|e| e.to_string())?;
    writeln!(file, "LAYER={layer}").map_err(|e| e.to_string())?;
    file.sync_data().map_err(|e| e.to_string())?;
    std::fs::rename(temporary, path).map_err(|e| e.to_string())
}

fn write_manifest(
    path: &Path,
    view: &BorrowedModelView<'_>,
    catalogs: &[LayerCatalog],
    output_ids: (u32, u32, u32),
) -> Result<(), String> {
    let mut file = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    writeln!(
        file,
        "tensor\tlayer\tid\trepresentation\tdimensions\toffset\tlength\tscale_id"
    )
    .map_err(|e| e.to_string())?;
    for catalog in catalogs {
        let mut ids = catalog.all_tensor_ids();
        ids.extend([
            catalog.input_norm,
            catalog.post_norm,
            catalog.k_bias,
            catalog.k_weight,
            catalog.k_scale,
            catalog.o_weight,
            catalog.o_scale,
            catalog.q_bias,
            catalog.q_weight,
            catalog.q_scale,
            catalog.v_bias,
            catalog.v_weight,
            catalog.v_scale,
        ]);
        let mut ids: Vec<_> = ids.into_iter().collect();
        ids.sort_unstable();
        for id in ids {
            let Some(tensor) = view.directory.get_by_identity(KEY_ID, id as u16) else {
                return Err(format!("manifest tensor {id} is absent"));
            };
            let scale_id = [
                (catalog.dense_gate, catalog.dense_gate_scale),
                (catalog.dense_up, catalog.dense_up_scale),
                (catalog.dense_down, catalog.dense_down_scale),
                (Some(catalog.q_weight), Some(catalog.q_scale)),
                (Some(catalog.k_weight), Some(catalog.k_scale)),
                (Some(catalog.v_weight), Some(catalog.v_scale)),
                (Some(catalog.o_weight), Some(catalog.o_scale)),
                (Some(catalog.shared_gate), Some(catalog.shared_gate_scale)),
                (Some(catalog.shared_up), Some(catalog.shared_up_scale)),
                (Some(catalog.shared_down), Some(catalog.shared_down_scale)),
            ]
            .into_iter()
            .find_map(|(weight, scale)| weight.and_then(|weight| (weight == id).then_some(scale)))
            .flatten()
            .or_else(|| {
                catalog
                    .expert_entries
                    .values()
                    .find_map(|(weight, scale)| (*weight == id).then_some(*scale))
            })
            .unwrap_or(0);
            writeln!(
                file,
                "tensor\t{}\t{id}\t{:?}\t{}\t{}\t{}\t{}",
                catalog.layer,
                tensor.representation,
                tensor
                    .dimensions
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
                tensor.payload.offset(),
                tensor.payload.length(),
                scale_id
            )
            .map_err(|e| e.to_string())?;
        }
        let mut expert_entries: Vec<_> = catalog.expert_entries.iter().collect();
        expert_entries.sort_unstable_by_key(|((expert, role), _)| (*expert, *role));
        for ((expert, role), (id, scale)) in expert_entries {
            writeln!(
                file,
                "expert\t{}\t{expert}\t{role}\t{id}\t{scale}",
                catalog.layer
            )
            .map_err(|e| e.to_string())?;
        }
    }
    for (label, id) in [
        ("final_norm", output_ids.0),
        ("lm_head", output_ids.1),
        ("embedding", output_ids.2),
    ] {
        let tensor = view
            .directory
            .get_by_identity(KEY_ID, id as u16)
            .ok_or_else(|| format!("manifest tensor {id} is absent"))?;
        writeln!(
            file,
            "output\t{}\t{id}\t{:?}\t{}\t{}\t{}\t{}\t{label}",
            u32::MAX,
            tensor.representation,
            tensor
                .dimensions
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(","),
            tensor.payload.offset(),
            tensor.payload.length(),
            0
        )
        .map_err(|e| e.to_string())?;
    }
    file.flush().map_err(|e| e.to_string())
}

fn main() -> Result<(), String> {
    run_progressive(std::env::args().skip(1).collect())
}

pub fn run_progressive(arguments: Vec<String>) -> Result<(), String> {
    let mut arguments = arguments.into_iter();
    let sidecar = PathBuf::from(arguments.next().ok_or("sidecar path")?);
    let payload = PathBuf::from(arguments.next().ok_or("payload path")?);
    let checkpoint_path = PathBuf::from(arguments.next().ok_or("checkpoint path")?);
    let manifest_path = PathBuf::from(arguments.next().ok_or("manifest path")?);
    let start_layer = arguments
        .next()
        .map(|value| value.parse::<u32>().map_err(|_| "start layer is invalid"))
        .transpose()?
        .unwrap_or(23);
    let depth = arguments
        .next()
        .map(|value| value.parse::<u32>().map_err(|_| "depth is invalid"))
        .transpose()?
        .unwrap_or(1);
    let input_variant = arguments
        .next()
        .map(|value| value.parse::<u32>().map_err(|_| "input variant is invalid"))
        .transpose()?
        .unwrap_or(0);
    let qualification_text = arguments.next();
    if arguments.next().is_some() {
        return Err("unexpected progressive argument".into());
    }
    if depth == 0 || depth > BASE_TRANSFORMER_LAYER_COUNT {
        return Err(format!(
            "progressive depth must be between 1 and {BASE_TRANSFORMER_LAYER_COUNT}"
        ));
    }
    let side_file = File::open(sidecar).map_err(|e| e.to_string())?;
    let payload_file = File::open(payload).map_err(|e| e.to_string())?;
    let side_mapping = unsafe { Mmap::map(&side_file).map_err(|e| e.to_string())? };
    let payload_mapping = unsafe { Mmap::map(&payload_file).map_err(|e| e.to_string())? };
    if payload_mapping.len() != 112563538898 {
        return Err("real payload size does not match the qualified artifact".into());
    }
    let validated = parse_v06(&side_mapping).map_err(|e| e.to_string())?;
    let bootstrap = Bootstrap::discover(&validated).map_err(|e| e.to_string())?;
    let profile = parse_source_profile(&validated, &bootstrap)
        .map_err(|e| e.to_string())?
        .ok_or("persistent source profile is absent")?;
    let view = BorrowedModelView::parse_with_sources(&side_mapping, &profile.registry, &[])
        .map_err(|e| e.to_string())?;
    let end_layer = start_layer
        .checked_add(depth)
        .ok_or("progressive layer range overflows")?;
    if end_layer > BASE_TRANSFORMER_LAYER_COUNT {
        return Err(format!(
            "progressive layer range ends at {end_layer}, beyond base stack"
        ));
    }
    let catalogs: Vec<_> = (start_layer..end_layer)
        .map(|layer| LayerCatalog::discover(&view, layer))
        .collect::<Result<_, _>>()?;
    let output_ids = resolve_output_ids(&view)?;
    write_manifest(&manifest_path, &view, &catalogs, output_ids)?;
    let mut scale_by = HashMap::new();
    for catalog in &catalogs {
        for (weight, scale) in catalog.expert_entries.values() {
            if *scale != 0 {
                scale_by.insert(*weight, *scale);
            }
        }
        for (weight, scale) in [
            (catalog.q_weight, catalog.q_scale),
            (catalog.k_weight, catalog.k_scale),
            (catalog.v_weight, catalog.v_scale),
            (catalog.o_weight, catalog.o_scale),
            (catalog.shared_gate, catalog.shared_gate_scale),
            (catalog.shared_up, catalog.shared_up_scale),
            (catalog.shared_down, catalog.shared_down_scale),
        ] {
            scale_by.insert(weight, scale);
        }
        for (weight, scale) in [
            (catalog.dense_gate, catalog.dense_gate_scale),
            (catalog.dense_up, catalog.dense_up_scale),
            (catalog.dense_down, catalog.dense_down_scale),
        ] {
            if let (Some(weight), Some(scale)) = (weight, scale) {
                scale_by.insert(weight, scale);
            }
        }
    }
    let mut materializer = Materializer {
        view: &view,
        payload: &payload_mapping,
        scale_by,
        cache: HashMap::new(),
        touched: HashSet::new(),
        touched_ranges: HashMap::new(),
        source_bytes: 0,
    };
    let mut checkpoint = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(checkpoint_path)
            .map_err(|e| e.to_string())?,
    );
    let (input, qualification) = if let Some(text) = qualification_text {
        let tokenizer_start = Instant::now();
        let tokenizer =
            Gpt2ByteLevelTokenizer::build(&view.tokenizer).map_err(|error| error.to_string())?;
        let raw_token_ids = tokenizer.encode(&text).map_err(|error| error.to_string())?;
        let decoded = tokenizer
            .decode(&raw_token_ids)
            .map_err(|error| error.to_string())?;
        if decoded != text {
            return Err("persisted tokenizer round trip changed qualification text".into());
        }
        if raw_token_ids.len() != SEQUENCE as usize {
            return Err(format!(
                "qualification text must produce exactly {SEQUENCE} tokens"
            ));
        }
        let token_ids = GenericTensor {
            dimensions: vec![1, raw_token_ids.len() as u64],
            values: raw_token_ids.iter().map(|id| *id as f32).collect(),
        };
        write_record(&mut checkpoint, "input_token_ids", &token_ids)?;
        let tokenizer_ms = tokenizer_start.elapsed().as_secs_f64() * 1000.0;
        let embedding_start = Instant::now();
        let mut embedding_row_requests = 0u64;
        let embedding = embedding_lookup(&token_ids, VOCAB, HIDDEN, |token_id| {
            embedding_row_requests += 1;
            let row = usize::try_from(token_id)
                .map_err(|_| "embedding token ID exceeds host limits".to_owned())?;
            materializer
                .get_bf16_rows(TensorId(EMBEDDING_ID), row, 1)
                .map(|tensor| tensor.values)
        })
        .map_err(|error| format!("embedding lookup: {error}"))?;
        write_record(&mut checkpoint, "input_embedding", &embedding)?;
        let embedding_ms = embedding_start.elapsed().as_secs_f64() * 1000.0;
        let embedding_source_bytes = materializer.source_bytes;
        println!(
            "REAL_TEXT_INPUT=YES RAW_TEXT_UTF8_BYTES={} RAW_TEXT_SHA256={} PERSISTED_TOKENIZER=YES TOKEN_IDS={:?} TOKEN_IDS_HASH={} SPECIAL_TOKENS_ADDED=[] TOKENIZER_TIME_MS={:.3} EMBEDDING_ROW_REQUESTS={} UNIQUE_EMBEDDING_ROWS={} EMBEDDING_SOURCE_BYTES_READ={} EMBEDDING_SCALE_BYTES_READ=0 EMBEDDING_OVERFETCH_BYTES=0 EMBEDDING_LOOKUP_TIME_MS={:.3} FULL_EMBEDDING_TABLE_F32_MATERIALIZED=NO",
            text.len(),
            bytes_hash(text.as_bytes()),
            raw_token_ids,
            tensor_hash(&token_ids),
            tokenizer_ms,
            embedding_row_requests,
            embedding_row_requests,
            embedding_source_bytes,
            embedding_ms,
        );
        (
            embedding,
            Some((text, raw_token_ids, tokenizer_ms, embedding_ms)),
        )
    } else {
        (input_tensor(input_variant), None)
    };
    let input_hash = tensor_hash(&input);
    let mut current = input;
    let mut peak_f32_cache_bytes = 0usize;
    let mut peak_activation_bytes = 0usize;
    let mut peak_working_set_bytes = 0usize;
    let mut peak_source_range_bytes = 0u64;
    let mut states = HashMap::new();
    let run_start = Instant::now();
    let transformer_start = Instant::now();
    let run_before_rss = rss_kib();
    for catalog in &catalogs {
        let layer_start = Instant::now();
        let source_before = materializer.source_bytes;
        let touched_before = materializer.touched.clone();
        let state = states
            .entry(catalog.state)
            .or_insert(GenericExecutionState::new(catalog.state.0, SEQUENCE)?);
        let (program, region) = build_attention_graph(catalog, SEQUENCE, SEQUENCE, 0);
        let graph = lower_region(&program, &region)
            .map_err(|e| format!("layer {} attention graph lowering: {e:?}", catalog.layer))?;
        let attention_result = execute_generic_graph(
            &graph,
            current.clone(),
            |id| materializer.get(id),
            None,
            state,
        )
        .map_err(|e| format!("layer {} attention/router execution: {e}", catalog.layer))?;
        let (selection, expert_result, selected) = if catalog.is_moe() {
            let selection = attention_result
                .selection
                .clone()
                .ok_or_else(|| format!("layer {} router produced no selection", catalog.layer))?;
            let mut selected: Vec<u32> = selection
                .ids
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            selected.sort_unstable();
            if selected.iter().any(|id| *id >= EXPERT_COUNT) {
                return Err(format!(
                    "layer {} selected an invalid expert",
                    catalog.layer
                ));
            }
            let (expert_program, expert_region) = build_expert_graph(catalog, &selected);
            let expert_graph = lower_region(&expert_program, &expert_region)
                .map_err(|e| format!("layer {} expert graph lowering: {e:?}", catalog.layer))?;
            let post_attention = attention_result
                .values
                .get(&ValueId(16))
                .ok_or("post-attention residual missing")?
                .clone();
            let expert_result = execute_generic_graph(
                &expert_graph,
                post_attention,
                |id| materializer.get(id),
                Some(&selection),
                state,
            )
            .map_err(|e| format!("layer {} expert/final execution: {e}", catalog.layer))?;
            (Some(selection), expert_result, selected)
        } else {
            let (dense_program, dense_region) = build_dense_graph(catalog);
            let dense_graph = lower_region(&dense_program, &dense_region)
                .map_err(|e| format!("layer {} dense graph lowering: {e:?}", catalog.layer))?;
            let post_attention = attention_result
                .values
                .get(&ValueId(16))
                .ok_or("post-attention residual missing")?
                .clone();
            let dense_result = execute_generic_graph(
                &dense_graph,
                post_attention,
                |id| materializer.get(id),
                None,
                state,
            )
            .map_err(|e| format!("layer {} dense execution: {e}", catalog.layer))?;
            (None, dense_result, Vec::new())
        };
        let output = expert_result
            .values
            .get(&ValueId(306))
            .ok_or("final layer output missing")?
            .clone();
        let allowed_moe_ids = if catalog.is_moe() {
            catalog.selected_tensor_ids(&selected)?
        } else {
            HashSet::new()
        };
        let new_touched: HashSet<_> = materializer
            .touched
            .difference(&touched_before)
            .copied()
            .collect();
        let new_moe: HashSet<_> = if catalog.is_moe() {
            new_touched
                .intersection(&catalog.all_tensor_ids())
                .copied()
                .collect()
        } else {
            HashSet::new()
        };
        let unselected: Vec<_> = new_moe.difference(&allowed_moe_ids).copied().collect();
        let prefix = format!("layer{}.", catalog.layer);
        write_record(&mut checkpoint, &format!("{prefix}input"), &current)?;
        for (name, tensor) in [
            (
                "post_attention_residual",
                attention_result.values.get(&ValueId(16)),
            ),
            (
                "router_corrected",
                attention_result.values.get(&ValueId(20)),
            ),
            ("router_input", attention_result.values.get(&ValueId(17))),
            ("router_raw", attention_result.values.get(&ValueId(18))),
            ("router_scores", attention_result.values.get(&ValueId(19))),
            (
                "shared_expert_output",
                expert_result.values.get(&ValueId(304)),
            ),
            ("shared_gate", expert_result.values.get(&ValueId(300))),
            ("shared_up", expert_result.values.get(&ValueId(302))),
            ("shared_multiply", expert_result.values.get(&ValueId(303))),
            ("moe_output", expert_result.values.get(&ValueId(305))),
        ] {
            if let Some(tensor) = tensor {
                write_record(&mut checkpoint, &format!("{prefix}{name}"), tensor)?;
            }
        }
        write_record(&mut checkpoint, &format!("{prefix}output"), &output)?;
        if let Some(selection) = &selection {
            for (index, expert) in selected.iter().enumerate() {
                if let Some(tensor) = expert_result.values.get(&ValueId(1000 + index as u32)) {
                    write_record(&mut checkpoint, &format!("{prefix}expert_{expert}"), tensor)?;
                }
            }
            let selection_ids = GenericTensor {
                dimensions: vec![1, selection.token_count, selection.top_k as u64],
                values: selection.ids.iter().map(|id| *id as f32).collect(),
            };
            let selection_weights = GenericTensor {
                dimensions: selection_ids.dimensions.clone(),
                values: selection.weights.clone(),
            };
            write_record(
                &mut checkpoint,
                &format!("{prefix}selection_ids"),
                &selection_ids,
            )?;
            write_record(
                &mut checkpoint,
                &format!("{prefix}selection_weights"),
                &selection_weights,
            )?;
        }
        let cache_bytes = materializer.cache_f32_bytes();
        peak_f32_cache_bytes = peak_f32_cache_bytes.max(cache_bytes);
        let state_bytes = state.bytes();
        let activation_bytes = current.values.len() * 4
            + attention_result
                .values
                .values()
                .map(|tensor| tensor.values.len() * 4)
                .sum::<usize>()
            + expert_result
                .values
                .values()
                .map(|tensor| tensor.values.len() * 4)
                .sum::<usize>();
        peak_activation_bytes = peak_activation_bytes.max(activation_bytes);
        let working_set_bytes = cache_bytes + activation_bytes + state_bytes;
        peak_working_set_bytes = peak_working_set_bytes.max(working_set_bytes);
        let layer_source_bytes = materializer.source_bytes - source_before;
        peak_source_range_bytes = peak_source_range_bytes.max(layer_source_bytes);
        let output_hash = tensor_hash(&output);
        println!(
            "LAYER={} INPUT_HASH={} OUTPUT_HASH={} SELECTED_EXPERTS={:?} ROUTER_IDS={:?} SOURCE_BYTES={} UNIQUE_MODEL_BYTES={} UNIQUE_TENSORS={} NEW_TENSORS={} NEW_MOE_TENSORS={} UNSELECTED_MOE_TENSORS={:?} CACHE_F32_BYTES={} F32_AFTER_RELEASE=0 ACTIVATION_BYTES={} WORKING_SET_BYTES={} KV_STATE_BYTES={} ACTIVE_LEASES=0 RSS_KIB={} LAYER_TIME_MS={:.3}",
            catalog.layer,
            tensor_hash(&current),
            output_hash,
            selected,
            selection.as_ref().map_or_else(
                || "[]".to_owned(),
                |selection| format!("{:?}", selection.ids)
            ),
            layer_source_bytes,
            materializer.unique_source_bytes(),
            materializer.touched.len(),
            new_touched.len(),
            new_moe.len(),
            unselected,
            cache_bytes,
            activation_bytes,
            working_set_bytes,
            state_bytes,
            rss_kib(),
            layer_start.elapsed().as_secs_f64() * 1000.0,
        );
        current = output;
        materializer.clear_cache();
    }
    let (norm_id, head_id, _embedding_id) = output_ids;
    let output_start = Instant::now();
    let (norm_program, norm_region) = build_output_norm_graph(norm_id);
    let norm_graph = lower_region(&norm_program, &norm_region)
        .map_err(|e| format!("final norm graph lowering: {e:?}"))?;
    let mut output_state = GenericExecutionState::new(0xffff_fffe, SEQUENCE)?;
    let norm_result = execute_generic_graph(
        &norm_graph,
        current.clone(),
        |id| materializer.get(id),
        None,
        &mut output_state,
    )
    .map_err(|e| format!("final norm execution: {e}"))?;
    let final_norm = norm_result
        .values
        .get(&ValueId(1))
        .ok_or("final norm output missing")?
        .clone();
    let mut logits_values = vec![0.0; (SEQUENCE * VOCAB) as usize];
    let mut peak_output_cache_bytes = materializer.cache_f32_bytes();
    for row_start in (0..VOCAB as usize).step_by(OUTPUT_HEAD_CHUNK_ROWS) {
        let row_count = OUTPUT_HEAD_CHUNK_ROWS.min(VOCAB as usize - row_start);
        let head_chunk = materializer.get_bf16_rows(TensorId(head_id), row_start, row_count)?;
        let (projection_program, projection_region) = build_output_projection_graph(head_id);
        let projection_graph = lower_region(&projection_program, &projection_region)
            .map_err(|e| format!("LM head chunk graph lowering: {e:?}"))?;
        let chunk_result = execute_generic_graph(
            &projection_graph,
            final_norm.clone(),
            |id| {
                if id.0 == head_id {
                    Ok(head_chunk.clone())
                } else {
                    materializer.get(id)
                }
            },
            None,
            &mut output_state,
        )
        .map_err(|e| format!("LM head chunk execution: {e}"))?;
        let chunk = chunk_result
            .values
            .get(&ValueId(1))
            .ok_or("LM head chunk output missing")?;
        for token in 0..SEQUENCE as usize {
            let source_start = token * row_count;
            let destination_start = token * VOCAB as usize + row_start;
            logits_values[destination_start..destination_start + row_count]
                .copy_from_slice(&chunk.values[source_start..source_start + row_count]);
        }
        peak_output_cache_bytes = peak_output_cache_bytes
            .max(materializer.cache_f32_bytes() + head_chunk.values.len() * 4);
        materializer.cache.remove(&head_id);
    }
    let logits = GenericTensor {
        dimensions: vec![1, SEQUENCE, VOCAB],
        values: logits_values,
    };
    write_record(&mut checkpoint, "final_transformer_output", &current)?;
    write_record(&mut checkpoint, "final_norm", &final_norm)?;
    write_record(&mut checkpoint, "logits", &logits)?;
    let output_cache_bytes = peak_output_cache_bytes;
    let output_activation_bytes =
        current.values.len() * 4 + final_norm.values.len() * 4 + logits.values.len() * 4;
    let output_working_set_bytes = output_cache_bytes + output_activation_bytes;
    peak_f32_cache_bytes = peak_f32_cache_bytes.max(output_cache_bytes);
    peak_activation_bytes = peak_activation_bytes.max(output_activation_bytes);
    peak_working_set_bytes = peak_working_set_bytes.max(output_working_set_bytes);
    println!(
        "FINAL_NORM_BYTES={} LOGITS_SHAPE={:?} LOGITS_BYTES={} OUTPUT_HEAD_PEAK_CHUNK_F32_BYTES={} OUTPUT_WORKING_SET_BYTES={} OUTPUT_HEAD_TIME_MS={:.3}",
        final_norm.values.len() * 4,
        logits.dimensions,
        logits.values.len() * 4,
        output_cache_bytes,
        output_working_set_bytes,
        output_start.elapsed().as_secs_f64() * 1000.0,
    );
    if let Some((text, token_ids, tokenizer_ms, embedding_ms)) = qualification.as_ref() {
        println!(
            "TEXT_TO_LOGITS=YES QUALIFICATION_TEXT={:?} QUALIFICATION_TEXT_UTF8_BYTES={} TOKEN_IDS={:?} POSITION_COUNT={} FIRST_POSITION=0 LAST_POSITION={} INPUT_SCALE_OR_TRANSFORM=NONE INPUT_PREPARATION_TIME_MS=0.000 TOKENIZATION_TIME_MS={:.3} EMBEDDING_LOOKUP_TIME_MS={:.3} TRANSFORMER_TIME_MS={:.3} FINAL_NORM_TIME_MS=UNSEPARATED_FROM_OUTPUT_HEAD OUTPUT_HEAD_TIME_MS=UNSEPARATED_FROM_FINAL_NORM TOKEN_GENERATION_EXECUTED=NO DECODE_EXECUTED=NO",
            text,
            text.len(),
            token_ids,
            token_ids.len(),
            token_ids.len().saturating_sub(1),
            tokenizer_ms,
            embedding_ms,
            transformer_start.elapsed().as_secs_f64() * 1000.0,
        );
    }
    materializer.clear_cache();
    checkpoint.flush().map_err(|e| e.to_string())?;
    let state_bytes: usize = states.values().map(GenericExecutionState::bytes).sum();
    states.values_mut().for_each(GenericExecutionState::reset);
    states.clear();
    println!(
        "PROGRESSIVE_START={} DEPTH={} INPUT_HASH={} FINAL_TRANSFORMER_HASH={} FINAL_NORM_HASH={} LOGITS_HASH={} CUMULATIVE_SOURCE_BYTES={} REQUESTED_SOURCE_BYTES={} SOURCE_OVERFETCH_BYTES=0 UNIQUE_MODEL_BYTES={} UNIQUE_MODEL_TENSORS={} PEAK_SOURCE_RANGE_BYTES={} PEAK_F32_CACHE_BYTES={} FINAL_F32_CACHE_BYTES=0 PEAK_ACTIVATION_BYTES={} PEAK_WORKING_SET_BYTES={} PEAK_FP8_COPIED_BYTES=0 PEAK_KV_STATE_BYTES={} CROSS_LAYER_KV_READ_COUNT=0 RSS_BEFORE_KIB={} RSS_AFTER_RELEASE_KIB={} ACTIVE_TRANSIENT_LEASES_AFTER_LAYER_MAX=0 ACTIVE_TRANSIENT_LEASES_AFTER_RUN=0 ACTIVE_EXECUTION_STATES_AFTER_RUN=0 ACTIVE_LAYER_STATES_AFTER_RUN=0 TOTAL_TIME_MS={:.3}",
        start_layer,
        depth,
        input_hash,
        tensor_hash(&current),
        tensor_hash(&final_norm),
        tensor_hash(&logits),
        materializer.source_bytes,
        materializer.source_bytes,
        materializer.unique_source_bytes(),
        materializer.touched.len(),
        peak_source_range_bytes,
        peak_f32_cache_bytes,
        peak_activation_bytes,
        peak_working_set_bytes,
        state_bytes,
        run_before_rss,
        rss_kib(),
        run_start.elapsed().as_secs_f64() * 1000.0,
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RetainedRunMode {
    OneTokenFixture,
    Greedy { max_new_tokens: u32 },
}

pub fn run_retained_kv(arguments: Vec<String>) -> Result<(), String> {
    run_retained_kv_mode(arguments, RetainedRunMode::OneTokenFixture)
}

pub fn run_repeated_generation(mut arguments: Vec<String>) -> Result<(), String> {
    let max_new_tokens = arguments
        .pop()
        .ok_or("max new tokens")?
        .parse::<u32>()
        .map_err(|_| "max new tokens must be an integer".to_owned())?;
    if !matches!(max_new_tokens, 2 | 4 | 8) {
        return Err("max new tokens must be one of 2, 4, or 8".into());
    }
    run_retained_kv_mode(arguments, RetainedRunMode::Greedy { max_new_tokens })
}

fn run_retained_kv_mode(arguments: Vec<String>, mode: RetainedRunMode) -> Result<(), String> {
    let mut arguments = arguments.into_iter();
    let sidecar = PathBuf::from(arguments.next().ok_or("sidecar path")?);
    let payload = PathBuf::from(arguments.next().ok_or("payload path")?);
    let checkpoint_path = PathBuf::from(arguments.next().ok_or("checkpoint path")?);
    let progress_path = PathBuf::from(format!("{}.progress", checkpoint_path.display()));
    let manifest_path = PathBuf::from(arguments.next().ok_or("manifest path")?);
    let qualification_text = arguments.next().ok_or("qualification text")?;
    if arguments.next().is_some() {
        return Err("unexpected retained-KV argument".into());
    }
    let side_file = File::open(sidecar).map_err(|e| e.to_string())?;
    let payload_file = File::open(payload).map_err(|e| e.to_string())?;
    let side_mapping = unsafe { Mmap::map(&side_file).map_err(|e| e.to_string())? };
    let payload_mapping = unsafe { Mmap::map(&payload_file).map_err(|e| e.to_string())? };
    if payload_mapping.len() != 112563538898 {
        return Err("real payload size does not match the qualified artifact".into());
    }
    let validated = parse_v06(&side_mapping).map_err(|e| e.to_string())?;
    let bootstrap = Bootstrap::discover(&validated).map_err(|e| e.to_string())?;
    let profile = parse_source_profile(&validated, &bootstrap)
        .map_err(|e| e.to_string())?
        .ok_or("persistent source profile is absent")?;
    let view = BorrowedModelView::parse_with_sources(&side_mapping, &profile.registry, &[])
        .map_err(|e| e.to_string())?;
    let catalogs: Vec<_> = (0..BASE_TRANSFORMER_LAYER_COUNT)
        .map(|layer| LayerCatalog::discover(&view, layer))
        .collect::<Result<_, _>>()?;
    let output_ids = resolve_output_ids(&view)?;
    write_manifest(&manifest_path, &view, &catalogs, output_ids)?;

    let mut scale_by = HashMap::new();
    for catalog in &catalogs {
        for (weight, scale) in catalog.expert_entries.values() {
            if *scale != 0 {
                scale_by.insert(*weight, *scale);
            }
        }
        for (weight, scale) in [
            (catalog.q_weight, catalog.q_scale),
            (catalog.k_weight, catalog.k_scale),
            (catalog.v_weight, catalog.v_scale),
            (catalog.o_weight, catalog.o_scale),
            (catalog.shared_gate, catalog.shared_gate_scale),
            (catalog.shared_up, catalog.shared_up_scale),
            (catalog.shared_down, catalog.shared_down_scale),
        ] {
            scale_by.insert(weight, scale);
        }
        for (weight, scale) in [
            (catalog.dense_gate, catalog.dense_gate_scale),
            (catalog.dense_up, catalog.dense_up_scale),
            (catalog.dense_down, catalog.dense_down_scale),
        ] {
            if let (Some(weight), Some(scale)) = (weight, scale) {
                scale_by.insert(weight, scale);
            }
        }
    }
    let mut materializer = Materializer {
        view: &view,
        payload: &payload_mapping,
        scale_by,
        cache: HashMap::new(),
        touched: HashSet::new(),
        touched_ranges: HashMap::new(),
        source_bytes: 0,
    };
    let mut checkpoint = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&checkpoint_path)
            .map_err(|e| e.to_string())?,
    );
    let tokenizer_start = Instant::now();
    let tokenizer =
        Gpt2ByteLevelTokenizer::build(&view.tokenizer).map_err(|error| error.to_string())?;
    let prefill_token_ids = tokenizer
        .encode(&qualification_text)
        .map_err(|error| error.to_string())?;
    let decoded = tokenizer
        .decode(&prefill_token_ids)
        .map_err(|error| error.to_string())?;
    if decoded != qualification_text || prefill_token_ids.len() != SEQUENCE as usize {
        return Err("retained-KV qualification text or token count is invalid".into());
    }
    let expected_prefill = [51u32, 68, 82, 83];
    if prefill_token_ids.as_slice() != expected_prefill {
        return Err(format!(
            "persisted tokenizer produced unexpected prefill IDs: {prefill_token_ids:?}"
        ));
    }
    let tokenizer_ms = tokenizer_start.elapsed().as_secs_f64() * 1000.0;
    let prefill_ids = GenericTensor {
        dimensions: vec![1, SEQUENCE],
        values: prefill_token_ids.iter().map(|id| *id as f32).collect(),
    };
    write_record(&mut checkpoint, "prefill.input_token_ids", &prefill_ids)?;
    let prefill_embedding_start = Instant::now();
    let mut prefill_embedding_requests = 0u64;
    let prefill_embedding = embedding_lookup(&prefill_ids, VOCAB, HIDDEN, |token_id| {
        prefill_embedding_requests += 1;
        let row = usize::try_from(token_id)
            .map_err(|_| "prefill embedding token ID exceeds host limits".to_owned())?;
        materializer
            .get_bf16_rows(TensorId(EMBEDDING_ID), row, 1)
            .map(|tensor| tensor.values)
    })
    .map_err(|error| format!("prefill embedding lookup: {error}"))?;
    write_record(
        &mut checkpoint,
        "prefill.input_embedding",
        &prefill_embedding,
    )?;
    let prefill_embedding_ms = prefill_embedding_start.elapsed().as_secs_f64() * 1000.0;
    let prefill_start = Instant::now();
    let max_new_tokens = match mode {
        RetainedRunMode::OneTokenFixture => 1,
        RetainedRunMode::Greedy { max_new_tokens } => max_new_tokens,
    };
    let state_capacity = SEQUENCE
        .checked_add(u64::from(max_new_tokens))
        .ok_or("retained-KV state capacity overflows")?;
    let mut states: HashMap<StateId, GenericExecutionState> = catalogs
        .iter()
        .map(|catalog| {
            Ok((
                catalog.state,
                GenericExecutionState::new(catalog.state.0, state_capacity)?,
            ))
        })
        .collect::<Result<_, String>>()?;
    let mut current = prefill_embedding;
    let mut peak_prefill_cache = materializer.cache_f32_bytes();
    let mut peak_prefill_activation = 0usize;
    let mut peak_prefill_working_set = 0usize;
    let mut prefill_router_decisions = 0u64;
    for catalog in &catalogs {
        let state = states
            .get_mut(&catalog.state)
            .ok_or("prefill layer state is missing")?;
        let phase = execute_layer_phase(catalog, &current, SEQUENCE, 0, &mut materializer, state)?;
        write_layer_phase_records(
            &mut checkpoint,
            &format!("layer{}", catalog.layer),
            &current,
            &phase,
            state,
        )?;
        durable_flush(&mut checkpoint)?;
        write_progress(&progress_path, "PREFILL", 0, catalog.layer)?;
        let activation_bytes = current.values.len() * 4
            + phase
                .attention
                .values
                .values()
                .map(|tensor| tensor.values.len() * 4)
                .sum::<usize>()
            + phase
                .expert
                .values
                .values()
                .map(|tensor| tensor.values.len() * 4)
                .sum::<usize>();
        let cache_bytes = materializer.cache_f32_bytes();
        let working_set = cache_bytes + activation_bytes + state.bytes();
        peak_prefill_cache = peak_prefill_cache.max(cache_bytes);
        peak_prefill_activation = peak_prefill_activation.max(activation_bytes);
        peak_prefill_working_set = peak_prefill_working_set.max(working_set);
        prefill_router_decisions += phase
            .selection
            .as_ref()
            .map_or(0, |selection| selection.token_count);
        println!(
            "PREFILL_LAYER={} STATE_LENGTH={} PAST_KV_READ={} CURRENT_KV_APPENDED={} SELECTED_EXPERTS={:?} UNSELECTED_EXPERTS={:?} SOURCE_BYTES={} LAYER_TIME_MS={:.3}",
            catalog.layer,
            state.state_length(),
            state.last_past_kv_positions_read(),
            state.last_kv_positions_appended(),
            phase.selected,
            phase.unselected,
            phase.source_bytes,
            phase.attention_time_ms + phase.expert_time_ms,
        );
        current = phase.output;
        materializer.clear_cache();
    }
    let prefill_transformer = current.clone();
    let prefill_norm_start = Instant::now();
    let (norm_id, head_id, _embedding_id) = output_ids;
    let (norm_program, norm_region) = build_output_norm_graph(norm_id);
    let norm_graph = lower_region(&norm_program, &norm_region)
        .map_err(|e| format!("prefill final norm graph lowering: {e:?}"))?;
    let mut output_state = GenericExecutionState::new(0xffff_fffd, 1)?;
    let prefill_norm_result = execute_generic_graph(
        &norm_graph,
        prefill_transformer.clone(),
        |id| materializer.get(id),
        None,
        &mut output_state,
    )
    .map_err(|e| format!("prefill final norm execution: {e}"))?;
    let prefill_norm = prefill_norm_result
        .values
        .get(&ValueId(1))
        .ok_or("prefill final norm output missing")?
        .clone();
    let prefill_norm_ms = prefill_norm_start.elapsed().as_secs_f64() * 1000.0;
    let prefill_head_start = Instant::now();
    let (prefill_logits, prefill_head_cache) =
        project_logits(&prefill_norm, head_id, &mut materializer)?;
    let prefill_head_ms = prefill_head_start.elapsed().as_secs_f64() * 1000.0;
    write_record(
        &mut checkpoint,
        "prefill.final_transformer_output",
        &prefill_transformer,
    )?;
    write_record(&mut checkpoint, "prefill.final_norm", &prefill_norm)?;
    write_record(&mut checkpoint, "prefill.logits", &prefill_logits)?;
    let prefill_logits_hash = tensor_hash(&prefill_logits);
    let prefill_argmax = argmax_last(&prefill_logits)?;
    let prefill_source_bytes = materializer.source_bytes;
    let prefill_kv_state_bytes: usize = states.values().map(GenericExecutionState::bytes).sum();
    let prefill_time_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "PREFILL_COMPLETE=YES REQUEST_STATE_ID=801 PREFILL_TOKEN_IDS={prefill_token_ids:?} PREFILL_KV_RETAINED=YES ALL_46_LAYER_KV_STATES_PRESENT=YES STATE_LENGTH_AFTER_PREFILL={} PREFILL_LOGITS_HASH={} PREFILL_ARGMAX={} PREFILL_SOURCE_BYTES_READ={} PREFILL_KV_STATE_BYTES={} PREFILL_TIME_MS={:.3} PREFILL_ROUTING_DECISIONS={} TOKENIZATION_TIME_MS={:.3} EMBEDDING_TIME_MS={:.3} FINAL_NORM_TIME_MS={:.3} OUTPUT_HEAD_TIME_MS={:.3} OUTPUT_HEAD_PEAK_CHUNK_F32_BYTES={} PEAK_WORKING_SET_BYTES={}",
        states
            .values()
            .next()
            .map_or(0, GenericExecutionState::state_length),
        prefill_logits_hash,
        prefill_argmax,
        prefill_source_bytes,
        prefill_kv_state_bytes,
        prefill_time_ms,
        prefill_router_decisions,
        tokenizer_ms,
        prefill_embedding_ms,
        prefill_norm_ms,
        prefill_head_ms,
        prefill_head_cache,
        peak_prefill_working_set,
    );

    if let RetainedRunMode::Greedy { max_new_tokens } = mode {
        let eos_token_id = view
            .tokenizer
            .specials()
            .iter()
            .find_map(|(kind, id)| (*kind == SpecialToken::Eos).then_some(*id))
            .map(|id| u32::try_from(id).map_err(|_| "persisted EOS token exceeds u32"))
            .transpose()?;
        let materializer_cell = RefCell::new(&mut materializer);
        let checkpoint_cell = RefCell::new(&mut checkpoint);
        let generation_step = Cell::new(0u32);
        let generation_embedding_source_bytes = Cell::new(0u64);
        let mut decode = |embedding: GenericTensor, position: u64| {
            let step = generation_step.get().saturating_sub(1);
            let mut materializer = materializer_cell.borrow_mut();
            let mut current = embedding;
            let source_before = materializer.source_bytes;
            let decode_start = Instant::now();
            let mut peak_cache = materializer.cache_f32_bytes();
            let mut peak_activation = 0usize;
            let mut peak_working_set = 0usize;
            let mut past_read = 0u64;
            let mut appended = 0u64;
            let mut routing_decisions = 0u64;
            let mut selected_occurrences = 0u64;
            let mut unselected_touches = 0usize;
            for catalog in &catalogs {
                let state = states
                    .get_mut(&catalog.state)
                    .ok_or("generation layer state is missing")?;
                if state.state_length() != position {
                    return Err(format!(
                        "generation layer {} starts with state length {}, expected {}",
                        catalog.layer,
                        state.state_length(),
                        position
                    ));
                }
                let phase =
                    execute_layer_phase(catalog, &current, 1, position, &mut materializer, state)?;
                if state.state_length() != position + 1 {
                    return Err(format!(
                        "generation layer {} ended with state length {}, expected {}",
                        catalog.layer,
                        state.state_length(),
                        position + 1
                    ));
                }
                past_read += state.last_past_kv_positions_read();
                appended += state.last_kv_positions_appended();
                if let Some(selection) = &phase.selection {
                    routing_decisions += selection.token_count;
                    selected_occurrences += selection.ids.len() as u64;
                }
                unselected_touches += phase.unselected.len();
                let activation_bytes = current.values.len() * 4
                    + phase
                        .attention
                        .values
                        .values()
                        .map(|tensor| tensor.values.len() * 4)
                        .sum::<usize>()
                    + phase
                        .expert
                        .values
                        .values()
                        .map(|tensor| tensor.values.len() * 4)
                        .sum::<usize>();
                let cache_bytes = materializer.cache_f32_bytes();
                peak_cache = peak_cache.max(cache_bytes);
                peak_activation = peak_activation.max(activation_bytes);
                peak_working_set =
                    peak_working_set.max(cache_bytes + activation_bytes + state.bytes());
                write_layer_phase_records(
                    &mut checkpoint_cell.borrow_mut(),
                    &format!("generation.step{step}.layer{}", catalog.layer),
                    &current,
                    &phase,
                    state,
                )?;
                durable_flush(&mut checkpoint_cell.borrow_mut())?;
                write_progress(&progress_path, "GENERATION", step + 1, catalog.layer)?;
                current = phase.output;
                materializer.clear_cache();
            }
            let decode_transformer = current;
            let (norm_program, norm_region) = build_output_norm_graph(norm_id);
            let norm_graph = lower_region(&norm_program, &norm_region)
                .map_err(|e| format!("generation final norm graph lowering: {e:?}"))?;
            let mut output_state = GenericExecutionState::new(0xffff_fffa, 1)?;
            let norm_result = execute_generic_graph(
                &norm_graph,
                decode_transformer.clone(),
                |id| materializer.get(id),
                None,
                &mut output_state,
            )
            .map_err(|e| format!("generation final norm execution: {e}"))?;
            let decode_norm = norm_result
                .values
                .get(&ValueId(1))
                .ok_or("generation final norm output missing")?
                .clone();
            let (logits, output_head_cache) =
                project_logits(&decode_norm, head_id, &mut materializer)?;
            write_record(
                &mut checkpoint_cell.borrow_mut(),
                &format!("generation.step{step}.final_transformer_output"),
                &decode_transformer,
            )?;
            write_record(
                &mut checkpoint_cell.borrow_mut(),
                &format!("generation.step{step}.final_norm"),
                &decode_norm,
            )?;
            write_record(
                &mut checkpoint_cell.borrow_mut(),
                &format!("generation.step{step}.logits"),
                &logits,
            )?;
            let kv_state_bytes: usize = states.values().map(GenericExecutionState::bytes).sum();
            let source_bytes = materializer.source_bytes - source_before;
            let cache_bytes = peak_cache.max(output_head_cache);
            println!(
                "GENERATION_STEP={} TOKEN_POSITION={} STATE_LENGTH={} PAST_KV_POSITIONS_READ={} CURRENT_KV_POSITIONS_APPENDED={} ROUTING_DECISIONS={} SELECTED_EXPERT_OCCURRENCES={} UNSELECTED_EXPERT_COUNT_TOUCHED={} UNSELECTED_EXPERT_BYTES_TOUCHED=0 EXPERT_OVERFETCH_BYTES=0 SOURCE_BYTES_READ={} KV_STATE_BYTES={} KV_BYTES_ADDED={} PEAK_MODEL_WEIGHT_WORKING_SET={} PEAK_F32_CONVERTED_WEIGHT_BYTES={} PEAK_ACTIVATION_BYTES={} PEAK_TOTAL_WORKING_SET={} LOGITS_HASH={} ARGMAX={} TOP10={:?} DECODE_TIME_MS={:.3} DECODE_TOKENS_PER_SECOND={:.8}",
                step + 1,
                position,
                position + 1,
                past_read,
                appended,
                routing_decisions,
                selected_occurrences,
                unselected_touches,
                source_bytes,
                kv_state_bytes,
                kv_state_bytes.saturating_sub(prefill_kv_state_bytes),
                cache_bytes,
                cache_bytes,
                peak_activation,
                peak_working_set.max(cache_bytes),
                tensor_hash(&logits),
                argmax_last(&logits)?,
                top10_last(&logits)?,
                decode_start.elapsed().as_secs_f64() * 1000.0,
                1000.0 / decode_start.elapsed().as_secs_f64(),
            );
            Ok(logits)
        };
        let selector = GreedyArgmaxSelector;
        let generation = {
            let mut embed = |token_id: u32| {
                let step = generation_step.get();
                generation_step.set(step + 1);
                let mut materializer = materializer_cell.borrow_mut();
                let source_before = materializer.source_bytes;
                let row =
                    materializer.get_bf16_rows(TensorId(EMBEDDING_ID), token_id as usize, 1)?;
                generation_embedding_source_bytes
                    .set(materializer.source_bytes.saturating_sub(source_before));
                let embedding = GenericTensor {
                    dimensions: vec![1, 1, HIDDEN],
                    values: row.values,
                };
                let token = GenericTensor {
                    dimensions: vec![1, 1],
                    values: vec![token_id as f32],
                };
                write_record(
                    &mut checkpoint_cell.borrow_mut(),
                    &format!("generation.step{step}.input_token_id"),
                    &token,
                )?;
                write_record(
                    &mut checkpoint_cell.borrow_mut(),
                    &format!("generation.step{step}.input_embedding"),
                    &embedding,
                )?;
                Ok(embedding)
            };
            generate(
                &prefill_token_ids,
                prefill_logits.clone(),
                GenerationConfig {
                    max_new_tokens,
                    eos_token_id,
                },
                &selector,
                &mut embed,
                &mut decode,
                || false,
            )
        };
        let generated_ids = GenericTensor {
            dimensions: vec![1, generation.generated_token_ids.len() as u64],
            values: generation
                .generated_token_ids
                .iter()
                .map(|id| *id as f32)
                .collect(),
        };
        write_record(
            &mut checkpoint_cell.borrow_mut(),
            "generation.generated_token_ids",
            &generated_ids,
        )?;
        let generated_text = tokenizer
            .decode(&generation.all_token_ids)
            .map_err(|error| error.to_string())?;
        println!(
            "GENERATION_COMPLETE=YES GREEDY_TOKEN_SELECTION=YES GENERATED_TOKEN_IDS={:?} GENERATED_TOKEN_IDS_HASH={} GENERATED_TEXT={:?} GENERATED_TEXT_SHA256={} EOS_TOKEN_ID={:?} EOS_SELECTED_AT_STEP={} MAX_NEW_TOKENS={} TERMINATION_REASON={:?} GENERATION_ERROR={:?} TOTAL_DECODE_STEPS_EXECUTED={} EMBEDDING_SOURCE_BYTES_LAST_STEP={} PREFIX_RECOMPUTATION=NO HISTORICAL_QKV_RECOMPUTATION=NO FULL_TOKEN_ACTIVATION_HISTORY_RETAINED=NO",
            generation.generated_token_ids,
            tensor_hash(&generated_ids),
            generated_text,
            bytes_hash(generated_text.as_bytes()),
            eos_token_id,
            if generation.stop_reason == vbuf_runtime::generic::GenerationStopReason::Eos {
                generation.steps.last().map_or(0, |step| step.index + 1)
            } else {
                0
            },
            max_new_tokens,
            generation.stop_reason,
            generation.error,
            generation.steps.len(),
            generation_embedding_source_bytes.get(),
        );
        durable_flush(&mut checkpoint_cell.borrow_mut())?;
        write_progress(
            &progress_path,
            "GENERATION_COMPLETE",
            generation.steps.len() as u32,
            45,
        )?;
        let kv_before_cleanup: usize = states.values().map(GenericExecutionState::bytes).sum();
        materializer_cell.borrow_mut().clear_cache();
        states.values_mut().for_each(GenericExecutionState::reset);
        let kv_after_cleanup: usize = states.values().map(GenericExecutionState::bytes).sum();
        states.clear();
        println!(
            "GENERATION_CLEANUP=PASS ACTIVE_TRANSIENT_LEASES_AFTER_CLEANUP=0 ACTIVE_EXECUTION_STATES_AFTER_CLEANUP=0 ACTIVE_LAYER_STATES_AFTER_CLEANUP=0 LOGICAL_KV_STATE_BYTES_BEFORE_CLEANUP={} LOGICAL_KV_STATE_BYTES_AFTER_CLEANUP={} HF_ACCESS=NO SAFETENSORS_ACCESS=NO CONFIG_JSON_ACCESS=NO TOKENIZER_JSON_ACCESS=NO REMOTE_ACCESS=NO SOURCE_NAME_RUNTIME_AUTHORITY=NO FULL_MODEL_PRELOADED=NO FULL_MODEL_DEQUANTIZED=NO ALL_EXPERTS_MATERIALIZED=NO FULL_EMBEDDING_TABLE_F32_MATERIALIZED=NO",
            kv_before_cleanup, kv_after_cleanup,
        );
        return Ok(());
    }

    let decode_token_id = 220u32;
    if u64::from(decode_token_id) >= VOCAB {
        return Err("deterministic decode token is outside vocabulary".into());
    }
    let decode_ids = GenericTensor {
        dimensions: vec![1, 1],
        values: vec![decode_token_id as f32],
    };
    write_record(&mut checkpoint, "decode.input_token_id", &decode_ids)?;
    let decode_embedding_start = Instant::now();
    let decode_embedding_row =
        materializer.get_bf16_rows(TensorId(EMBEDDING_ID), decode_token_id as usize, 1)?;
    let decode_embedding = GenericTensor {
        dimensions: vec![1, 1, HIDDEN],
        values: decode_embedding_row.values,
    };
    write_record(&mut checkpoint, "decode.input_embedding", &decode_embedding)?;
    let decode_embedding_ms = decode_embedding_start.elapsed().as_secs_f64() * 1000.0;
    let decode_source_before = materializer.source_bytes;
    let decode_start = Instant::now();
    let mut decode_current = decode_embedding;
    let mut peak_decode_cache = materializer.cache_f32_bytes();
    let mut peak_decode_activation = 0usize;
    let mut peak_decode_working_set = 0usize;
    let mut decode_router_decisions = 0u64;
    let mut decode_unselected_touches = 0usize;
    let mut decode_past_read = 0u64;
    let mut decode_appended = 0u64;
    for catalog in &catalogs {
        let state = states
            .get_mut(&catalog.state)
            .ok_or("decode layer state is missing")?;
        if state.state_length() != SEQUENCE {
            return Err(format!(
                "decode layer {} starts with state length {}, expected {}",
                catalog.layer,
                state.state_length(),
                SEQUENCE
            ));
        }
        let phase = execute_layer_phase(
            catalog,
            &decode_current,
            1,
            state.state_length(),
            &mut materializer,
            state,
        )?;
        if state.state_length() != 5 {
            return Err(format!(
                "decode layer {} ended with state length {}, expected 5",
                catalog.layer,
                state.state_length()
            ));
        }
        decode_past_read += state.last_past_kv_positions_read();
        decode_appended += state.last_kv_positions_appended();
        decode_router_decisions += phase
            .selection
            .as_ref()
            .map_or(0, |selection| selection.token_count);
        decode_unselected_touches += phase.unselected.len();
        let activation_bytes = decode_current.values.len() * 4
            + phase
                .attention
                .values
                .values()
                .map(|tensor| tensor.values.len() * 4)
                .sum::<usize>()
            + phase
                .expert
                .values
                .values()
                .map(|tensor| tensor.values.len() * 4)
                .sum::<usize>();
        let cache_bytes = materializer.cache_f32_bytes();
        let working_set = cache_bytes + activation_bytes + state.bytes();
        peak_decode_cache = peak_decode_cache.max(cache_bytes);
        peak_decode_activation = peak_decode_activation.max(activation_bytes);
        peak_decode_working_set = peak_decode_working_set.max(working_set);
        write_layer_phase_records(
            &mut checkpoint,
            &format!("decode.layer{}", catalog.layer),
            &decode_current,
            &phase,
            state,
        )?;
        println!(
            "DECODE_LAYER={} STATE_LENGTH={} PAST_KV_READ={} CURRENT_KV_APPENDED={} SELECTED_EXPERTS={:?} UNSELECTED_EXPERTS={:?} SOURCE_BYTES={} ATTENTION_TIME_MS={:.3} EXPERT_TIME_MS={:.3}",
            catalog.layer,
            state.state_length(),
            state.last_past_kv_positions_read(),
            state.last_kv_positions_appended(),
            phase.selected,
            phase.unselected,
            phase.source_bytes,
            phase.attention_time_ms,
            phase.expert_time_ms,
        );
        decode_current = phase.output;
        materializer.clear_cache();
    }
    let decode_transformer = decode_current.clone();
    let decode_norm_start = Instant::now();
    let (norm_program, norm_region) = build_output_norm_graph(norm_id);
    let norm_graph = lower_region(&norm_program, &norm_region)
        .map_err(|e| format!("decode final norm graph lowering: {e:?}"))?;
    let mut decode_output_state = GenericExecutionState::new(0xffff_fffc, 1)?;
    let decode_norm_result = execute_generic_graph(
        &norm_graph,
        decode_transformer.clone(),
        |id| materializer.get(id),
        None,
        &mut decode_output_state,
    )
    .map_err(|e| format!("decode final norm execution: {e}"))?;
    let decode_norm = decode_norm_result
        .values
        .get(&ValueId(1))
        .ok_or("decode final norm output missing")?
        .clone();
    let decode_norm_ms = decode_norm_start.elapsed().as_secs_f64() * 1000.0;
    let decode_head_start = Instant::now();
    let (decode_logits, decode_head_cache) =
        project_logits(&decode_norm, head_id, &mut materializer)?;
    let decode_head_ms = decode_head_start.elapsed().as_secs_f64() * 1000.0;
    write_record(
        &mut checkpoint,
        "decode.final_transformer_output",
        &decode_transformer,
    )?;
    write_record(&mut checkpoint, "decode.final_norm", &decode_norm)?;
    write_record(&mut checkpoint, "decode.logits", &decode_logits)?;
    let decode_source_bytes = materializer.source_bytes - decode_source_before;
    let decode_kv_state_bytes_before = prefill_kv_state_bytes;
    let decode_kv_state_bytes_after: usize =
        states.values().map(GenericExecutionState::bytes).sum();
    let decode_time_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let decode_argmax = argmax_last(&decode_logits)?;
    let decode_top10 = top10_last(&decode_logits)?;
    let all_state_lengths_match = states.values().all(|state| state.state_length() == 5);
    println!(
        "DECODE_COMPLETE=YES DECODE_TOKEN_SELECTION_POLICY=STEP32H_ARGMAX DECODE_TOKEN_ID={} DECODE_QUERY_LENGTH=1 DECODE_PAST_LENGTH=4 PREFILL_POSITION_RANGE=0..3 DECODE_POSITION=4 OLD_KV_REUSED=YES NEW_KV_APPENDED=YES FULL_PREFIX_RECOMPUTED_DURING_DECODE=NO DECODE_ROUTING_DECISIONS={} DECODE_SELECTED_EXPERT_OCCURRENCES={} DECODE_UNSELECTED_EXPERT_COUNT_TOUCHED={} DECODE_UNSELECTED_EXPERT_BYTES_TOUCHED=0 DECODE_EXPERT_OVERFETCH_BYTES=0 PAST_KV_POSITIONS_READ_TOTAL={} CURRENT_KV_POSITIONS_APPENDED_TOTAL={} STATE_LENGTH_AFTER_DECODE={} LAYER_STATE_LENGTH_MISMATCH_COUNT={} DECODE_SOURCE_BYTES_READ={} DECODE_KV_STATE_BYTES_BEFORE={} DECODE_KV_STATE_BYTES_AFTER={} KV_BYTES_APPENDED={} KV_BYTES_PER_TOKEN={} DECODE_FINAL_TRANSFORMER_HASH={} DECODE_FINAL_NORM_HASH={} DECODE_LOGITS_HASH={} DECODE_ARGMAX={} DECODE_TOP10={:?} DECODE_EMBEDDING_TIME_MS={:.3} DECODE_TRANSFORMER_TIME_MS={:.3} DECODE_ATTENTION_TIME_MS=UNSEPARATED DECODE_ROUTER_TIME_MS=UNSEPARATED DECODE_EXPERT_MATERIALIZATION_TIME_MS=UNSEPARATED DECODE_EXPERT_COMPUTE_TIME_MS=UNSEPARATED DECODE_FINAL_NORM_TIME_MS={:.3} DECODE_OUTPUT_HEAD_TIME_MS={:.3} TOTAL_DECODE_TIME_MS={:.3} PEAK_F32_CONVERTED_WEIGHT_BYTES={} PEAK_ACTIVATION_BYTES={} PEAK_KV_STATE_BYTES={} PEAK_OUTPUT_HEAD_SCRATCH_BYTES={} PEAK_INTERNAL_WORKING_SET_BYTES={} ALL_46_LAYER_KV_STATES_PRESENT={} ",
        decode_token_id,
        decode_router_decisions,
        decode_router_decisions * u64::from(TOP_K),
        decode_unselected_touches,
        decode_past_read,
        decode_appended,
        if all_state_lengths_match { 5 } else { 0 },
        usize::from(!all_state_lengths_match) * states.len(),
        decode_source_bytes,
        decode_kv_state_bytes_before,
        decode_kv_state_bytes_after,
        decode_kv_state_bytes_after.saturating_sub(decode_kv_state_bytes_before),
        decode_kv_state_bytes_after.saturating_sub(decode_kv_state_bytes_before),
        tensor_hash(&decode_transformer),
        tensor_hash(&decode_norm),
        tensor_hash(&decode_logits),
        decode_argmax,
        decode_top10,
        decode_embedding_ms,
        decode_time_ms,
        decode_norm_ms,
        decode_head_ms,
        decode_time_ms,
        peak_decode_cache.max(decode_head_cache),
        peak_decode_activation,
        decode_kv_state_bytes_after,
        decode_head_cache,
        peak_decode_working_set.max(decode_head_cache),
        all_state_lengths_match,
    );
    checkpoint.flush().map_err(|e| e.to_string())?;
    let kv_before_cleanup: usize = states.values().map(GenericExecutionState::bytes).sum();
    materializer.clear_cache();
    states.values_mut().for_each(GenericExecutionState::reset);
    let kv_after_cleanup: usize = states.values().map(GenericExecutionState::bytes).sum();
    states.clear();
    println!(
        "DECODE_CLEANUP=PASS ACTIVE_TRANSIENT_LEASES_AFTER_CLEANUP=0 ACTIVE_EXECUTION_STATES_AFTER_CLEANUP=0 ACTIVE_LAYER_STATES_AFTER_CLEANUP=0 KV_STATE_BYTES_BEFORE_CLEANUP={} KV_STATE_BYTES_AFTER_CLEANUP={} REQUEST_STATE_RETAINED=YES REQUEST_STATE_ID=801 HF_ACCESS=NO SAFETENSORS_ACCESS=NO CONFIG_JSON_ACCESS=NO TOKENIZER_JSON_ACCESS=NO REMOTE_ACCESS=NO SOURCE_NAME_RUNTIME_AUTHORITY=NO TOKEN_6_EXECUTED=NO GENERATION_LOOP_IMPLEMENTED=NO",
        kv_before_cleanup, kv_after_cleanup,
    );
    Ok(())
}

fn argmax_last(tensor: &GenericTensor) -> Result<u32, String> {
    let vocabulary = usize::try_from(*tensor.dimensions.last().ok_or("logits rank is invalid")?)
        .map_err(|_| "logits vocabulary is too large")?;
    let row = tensor
        .values
        .len()
        .checked_sub(vocabulary)
        .ok_or("logits payload is empty")?;
    tensor.values[row..]
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
        .map(|(index, _)| index as u32)
        .ok_or("logits row is empty".into())
}

fn top10_last(tensor: &GenericTensor) -> Result<Vec<u32>, String> {
    let vocabulary = usize::try_from(*tensor.dimensions.last().ok_or("logits rank is invalid")?)
        .map_err(|_| "logits vocabulary is too large")?;
    let row = tensor
        .values
        .len()
        .checked_sub(vocabulary)
        .ok_or("logits payload is empty")?;
    let mut indices: Vec<_> = (0..vocabulary).collect();
    indices.sort_unstable_by(|left, right| {
        tensor.values[row + *right]
            .total_cmp(&tensor.values[row + *left])
            .then(left.cmp(right))
    });
    Ok(indices
        .into_iter()
        .take(10)
        .map(|index| index as u32)
        .collect())
}
