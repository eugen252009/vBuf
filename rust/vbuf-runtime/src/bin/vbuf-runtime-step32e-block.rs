// Step 32E-B/32F qualification runner for bounded real GLM layers.
//
// The graph is built from semantic tensor identities and lowered through the
// generic runtime graph. The provider maps those identities to validated
// source ranges and materializes only tensors requested by the graph.

use memmap2::Mmap;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use vbuf_core::v06::parse_v06;
use vbuf_ml::{
    Bootstrap, BorrowedModelView, SourceId, TensorRepresentation, bf16_bits_to_f32,
    dequantize_f8_e4m3, parse_source_profile,
};
use vbuf_runtime::generic::{GenericExecutionState, GenericTensor, execute_generic_graph};
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
    expert_entries: HashMap<(u32, u16), (u32, u32)>,
    catalog_tensor_ids: HashSet<u32>,
}

impl LayerCatalog {
    fn discover(view: &BorrowedModelView<'_>, layer: u32) -> Result<Self, String> {
        let moe = view.moe.as_ref().ok_or("MoE directory is absent")?;
        let entries: Vec<_> = moe.entries_for_layer(layer).collect();
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

fn build_attention_graph(catalog: &LayerCatalog) -> (PortableProgram, PortableRegion) {
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
    binding(
        &mut program,
        "block.router.correction",
        catalog.router_correction,
    );
    binding(&mut program, "block.router.weight", catalog.router_weight);
    let rotary_q = RotaryAttributes {
        head_count: Q_HEADS,
        head_dim: HEAD_DIM,
        rotary_dim: ROTARY_DIM,
        theta: 1_000_000.0,
        position_start: 0,
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
                    query_length: SEQUENCE,
                    current_kv_length: SEQUENCE,
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
    (
        program,
        PortableRegion {
            id: format!("portable-layer-{}-attention-router", catalog.layer),
            input: ValueId(0),
            output: ValueId(21),
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

struct Materializer<'view, 'source> {
    view: &'view BorrowedModelView<'source>,
    payload: &'source [u8],
    scale_by: HashMap<u32, u32>,
    cache: HashMap<u32, GenericTensor>,
    touched: HashSet<u32>,
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
        self.touched
            .iter()
            .filter_map(|id| {
                self.view
                    .directory
                    .get_by_identity(KEY_ID, *id as u16)
                    .map(|tensor| tensor.payload.length())
            })
            .sum()
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

fn tensor_hash(tensor: &GenericTensor) -> String {
    let mut digest = Sha256::new();
    for value in &tensor.values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
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

fn write_manifest(
    path: &Path,
    view: &BorrowedModelView<'_>,
    catalogs: &[LayerCatalog],
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
                (catalog.q_weight, catalog.q_scale),
                (catalog.k_weight, catalog.k_scale),
                (catalog.v_weight, catalog.v_scale),
                (catalog.o_weight, catalog.o_scale),
                (catalog.shared_gate, catalog.shared_gate_scale),
                (catalog.shared_up, catalog.shared_up_scale),
                (catalog.shared_down, catalog.shared_down_scale),
            ]
            .into_iter()
            .find_map(|(weight, scale)| (weight == id).then_some(scale))
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
        for ((expert, role), (id, scale)) in &catalog.expert_entries {
            writeln!(
                file,
                "expert\t{}\t{expert}\t{role}\t{id}\t{scale}",
                catalog.layer
            )
            .map_err(|e| e.to_string())?;
        }
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
    if depth == 0 || depth > 8 {
        return Err("progressive depth must be between 1 and 8".into());
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
    let catalogs: Vec<_> = (start_layer..start_layer + depth)
        .map(|layer| LayerCatalog::discover(&view, layer))
        .collect::<Result<_, _>>()?;
    write_manifest(&manifest_path, &view, &catalogs)?;
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
    }
    let mut materializer = Materializer {
        view: &view,
        payload: &payload_mapping,
        scale_by,
        cache: HashMap::new(),
        touched: HashSet::new(),
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
    let input = input_tensor(input_variant);
    let input_hash = tensor_hash(&input);
    let mut current = input;
    let mut peak_f32_cache_bytes = 0usize;
    let mut peak_activation_bytes = 0usize;
    let mut peak_working_set_bytes = 0usize;
    let mut peak_source_range_bytes = 0u64;
    let mut states = HashMap::new();
    let run_start = Instant::now();
    let run_before_rss = rss_kib();
    for catalog in &catalogs {
        let layer_start = Instant::now();
        let source_before = materializer.source_bytes;
        let touched_before = materializer.touched.clone();
        let state = states
            .entry(catalog.state)
            .or_insert(GenericExecutionState::new(catalog.state.0, SEQUENCE)?);
        let (program, region) = build_attention_graph(catalog);
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
        let output = expert_result
            .values
            .get(&ValueId(306))
            .ok_or("final layer output missing")?
            .clone();
        let selected_tensor_ids = catalog.selected_tensor_ids(&selected)?;
        let allowed_moe_ids = selected_tensor_ids;
        let new_touched: HashSet<_> = materializer
            .touched
            .difference(&touched_before)
            .copied()
            .collect();
        let new_moe: HashSet<_> = new_touched
            .intersection(&catalog.all_tensor_ids())
            .copied()
            .collect();
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
            ("moe_output", expert_result.values.get(&ValueId(305))),
        ] {
            if let Some(tensor) = tensor {
                write_record(&mut checkpoint, &format!("{prefix}{name}"), tensor)?;
            }
        }
        write_record(&mut checkpoint, &format!("{prefix}output"), &output)?;
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
            selection.ids,
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
    checkpoint.flush().map_err(|e| e.to_string())?;
    let state_bytes: usize = states.values().map(GenericExecutionState::bytes).sum();
    states.values_mut().for_each(GenericExecutionState::reset);
    states.clear();
    println!(
        "PROGRESSIVE_START={} DEPTH={} INPUT_HASH={} FINAL_HASH={} CUMULATIVE_SOURCE_BYTES={} UNIQUE_MODEL_BYTES={} UNIQUE_MODEL_TENSORS={} PEAK_SOURCE_RANGE_BYTES={} PEAK_F32_CACHE_BYTES={} FINAL_F32_CACHE_BYTES=0 PEAK_ACTIVATION_BYTES={} PEAK_WORKING_SET_BYTES={} PEAK_FP8_COPIED_BYTES=0 PEAK_KV_STATE_BYTES={} CROSS_LAYER_KV_READ_COUNT=0 RSS_BEFORE_KIB={} RSS_AFTER_RELEASE_KIB={} ACTIVE_TRANSIENT_LEASES_AFTER_LAYER_MAX=0 ACTIVE_TRANSIENT_LEASES_AFTER_RUN=0 ACTIVE_EXECUTION_STATES_AFTER_RUN=0 ACTIVE_LAYER_STATES_AFTER_RUN=0 TOTAL_TIME_MS={:.3}",
        start_layer,
        depth,
        input_hash,
        tensor_hash(&current),
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
