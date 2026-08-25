//! Step 32E-B qualification runner for one real GLM block.
//!
//! The graph is built from semantic tensor identities and lowered through the
//! generic runtime graph. The provider maps those identities to validated
//! source ranges and materializes only tensors requested by the graph.

use memmap2::Mmap;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
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

const LAYER: u32 = 23;
const KEY_ID: u16 = 512;
const STATE_ID: StateId = StateId(7);
const HIDDEN: u64 = 4096;
const SEQUENCE: u64 = 4;
const Q_HEADS: u64 = 96;
const KV_HEADS: u64 = 8;
const HEAD_DIM: u64 = 128;
const ROTARY_DIM: u64 = 64;
const EXPERT_COUNT: u32 = 128;
const TOP_K: u32 = 8;

const INPUT_NORM: u32 = 11856;
const ROUTER_CORRECTION: u32 = 12625;
const ROUTER_WEIGHT: u32 = 12626;
const SHARED_DOWN: u32 = 12627;
const SHARED_DOWN_SCALE: u32 = 12628;
const SHARED_GATE: u32 = 12629;
const SHARED_GATE_SCALE: u32 = 12630;
const SHARED_UP: u32 = 12631;
const SHARED_UP_SCALE: u32 = 12632;
const POST_NORM: u32 = 12633;
const K_BIAS: u32 = 12634;
const K_WEIGHT: u32 = 12635;
const K_SCALE: u32 = 12636;
const O_WEIGHT: u32 = 12637;
const O_SCALE: u32 = 12638;
const Q_BIAS: u32 = 12639;
const Q_WEIGHT: u32 = 12640;
const Q_SCALE: u32 = 12641;
const V_BIAS: u32 = 12642;
const V_WEIGHT: u32 = 12643;
const V_SCALE: u32 = 12644;

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

fn build_attention_graph() -> (PortableProgram, PortableRegion) {
    let mut program = PortableProgram {
        state_refs: vec![StateRef {
            id: STATE_ID,
            kind: "request-kv".into(),
            lifetime: "request".into(),
            scope: "block-attention".into(),
        }],
        ..Default::default()
    };
    binding(&mut program, "block.input_norm.weight", INPUT_NORM);
    binding(&mut program, "block.attention.k.bias", K_BIAS);
    binding(&mut program, "block.attention.k.weight", K_WEIGHT);
    binding(&mut program, "block.attention.o.weight", O_WEIGHT);
    binding(&mut program, "block.attention.q.bias", Q_BIAS);
    binding(&mut program, "block.attention.q.weight", Q_WEIGHT);
    binding(&mut program, "block.attention.v.bias", V_BIAS);
    binding(&mut program, "block.attention.v.weight", V_WEIGHT);
    binding(&mut program, "block.attention.k.scale", K_SCALE);
    binding(&mut program, "block.attention.o.scale", O_SCALE);
    binding(&mut program, "block.attention.q.scale", Q_SCALE);
    binding(&mut program, "block.attention.v.scale", V_SCALE);
    binding(&mut program, "block.post_attention_norm.weight", POST_NORM);
    binding(&mut program, "block.router.correction", ROUTER_CORRECTION);
    binding(&mut program, "block.router.weight", ROUTER_WEIGHT);
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
                tensor("block.input_norm.weight", INPUT_NORM),
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
            Q_WEIGHT,
            ValueId(2),
        ),
        bias(
            "q_bias",
            ValueId(2),
            "block.attention.q.bias",
            Q_BIAS,
            ValueId(3),
        ),
        matmul(
            "k_projection",
            ValueId(1),
            "block.attention.k.weight",
            K_WEIGHT,
            ValueId(4),
        ),
        bias(
            "k_bias",
            ValueId(4),
            "block.attention.k.bias",
            K_BIAS,
            ValueId(5),
        ),
        matmul(
            "v_projection",
            ValueId(1),
            "block.attention.v.weight",
            V_WEIGHT,
            ValueId(6),
        ),
        bias(
            "v_bias",
            ValueId(6),
            "block.attention.v.bias",
            V_BIAS,
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
                    state: STATE_ID,
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
            O_WEIGHT,
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
                tensor("block.post_attention_norm.weight", POST_NORM),
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
            ROUTER_WEIGHT,
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
            ROUTER_CORRECTION,
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
            id: "real-glm-block-23-attention-router".into(),
            input: ValueId(0),
            output: ValueId(21),
            operations,
        },
    )
}

fn build_expert_graph(
    selected: &[u32],
    expert_entries: &HashMap<(u32, u16), (u32, u32)>,
) -> (PortableProgram, PortableRegion) {
    let mut program = PortableProgram::default();
    binding(&mut program, "block.post_attention_norm.weight", POST_NORM);
    binding(&mut program, "block.shared.gate", SHARED_GATE);
    binding(&mut program, "block.shared.up", SHARED_UP);
    binding(&mut program, "block.shared.down", SHARED_DOWN);
    for expert in selected {
        let gate = expert_entries[&(*expert, 1)].0;
        let up = expert_entries[&(*expert, 2)].0;
        let down = expert_entries[&(*expert, 3)].0;
        binding(&mut program, &format!("block.expert.{expert}.gate"), gate);
        binding(&mut program, &format!("block.expert.{expert}.up"), up);
        binding(&mut program, &format!("block.expert.{expert}.down"), down);
    }
    let mut operations = vec![PortableOperation {
        id: "post_attention_norm".into(),
        kind: PortableOperationKind::RmsNorm,
        inputs: vec![
            PortableInput::Value(ValueId(16)),
            tensor("block.post_attention_norm.weight", POST_NORM),
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
                    expert_entries[&(*expert, 1)].0,
                ),
                tensor(
                    &format!("block.expert.{expert}.up"),
                    expert_entries[&(*expert, 2)].0,
                ),
                tensor(
                    &format!("block.expert.{expert}.down"),
                    expert_entries[&(*expert, 3)].0,
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
            SHARED_GATE,
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
            SHARED_UP,
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
            SHARED_DOWN,
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
            id: "real-glm-block-23-moe".into(),
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

fn input_tensor() -> GenericTensor {
    let mut values = Vec::with_capacity((SEQUENCE * HIDDEN) as usize);
    for index in 0..(SEQUENCE * HIDDEN) {
        let x = index as f32 + 1.0;
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
    expert_entries: &HashMap<(u32, u16), (u32, u32)>,
) -> Result<(), String> {
    let mut file = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    let scale_by: HashMap<u32, u32> = expert_entries.values().copied().collect();
    writeln!(
        file,
        "tensor\tid\trepresentation\tdimensions\toffset\tlength\tscale_id"
    )
    .map_err(|e| e.to_string())?;
    for id in INPUT_NORM..=V_SCALE {
        let Some(tensor) = view.directory.get_by_identity(KEY_ID, id as u16) else {
            continue;
        };
        let scale_id = scale_by
            .iter()
            .find_map(|(weight, scale)| (*weight == id).then_some(*scale))
            .unwrap_or(0);
        writeln!(
            file,
            "tensor\t{id}\t{:?}\t{}\t{}\t{}\t{}",
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
    for ((expert, role), (id, scale)) in expert_entries {
        writeln!(file, "expert\t{expert}\t{role}\t{id}\t{scale}").map_err(|e| e.to_string())?;
    }
    file.flush().map_err(|e| e.to_string())
}

fn main() -> Result<(), String> {
    let sidecar = PathBuf::from(std::env::args().nth(1).ok_or("sidecar path")?);
    let payload = PathBuf::from(std::env::args().nth(2).ok_or("payload path")?);
    let checkpoint_path = PathBuf::from(std::env::args().nth(3).ok_or("checkpoint path")?);
    let manifest_path = PathBuf::from(std::env::args().nth(4).ok_or("manifest path")?);
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
    let moe = view.moe.as_ref().ok_or("MoE directory is absent")?;
    let mut expert_entries = HashMap::new();
    for entry in moe.entries_for_layer(LAYER) {
        if let (Some(tensor), scale) = (entry.tensor_ordinal, entry.scale_ordinal) {
            expert_entries.insert(
                (entry.expert_index, entry.role),
                (u32::from(tensor), u32::from(scale.unwrap_or(0))),
            );
        }
    }
    for (id, scale) in [
        (Q_WEIGHT, Q_SCALE),
        (K_WEIGHT, K_SCALE),
        (V_WEIGHT, V_SCALE),
        (O_WEIGHT, O_SCALE),
        (SHARED_GATE, SHARED_GATE_SCALE),
        (SHARED_UP, SHARED_UP_SCALE),
        (SHARED_DOWN, SHARED_DOWN_SCALE),
    ] {
        // Normal block identities are fixed semantic identities in this qualified sidecar.
        expert_entries.insert((u32::MAX, id as u16), (id, scale));
    }
    write_manifest(&manifest_path, &view, &expert_entries)?;
    let scale_by: HashMap<u32, u32> = expert_entries
        .iter()
        .filter_map(|((expert, role), (id, scale))| {
            if *expert != u32::MAX && *role <= 3 {
                Some((*id, *scale))
            } else {
                None
            }
        })
        .chain([
            (Q_WEIGHT, Q_SCALE),
            (K_WEIGHT, K_SCALE),
            (V_WEIGHT, V_SCALE),
            (O_WEIGHT, O_SCALE),
            (SHARED_GATE, SHARED_GATE_SCALE),
            (SHARED_UP, SHARED_UP_SCALE),
            (SHARED_DOWN, SHARED_DOWN_SCALE),
        ])
        .collect();
    let mut materializer = Materializer {
        view: &view,
        payload: &payload_mapping,
        scale_by,
        cache: HashMap::new(),
        touched: HashSet::new(),
        source_bytes: 0,
    };
    let input = input_tensor();
    let input_hash = tensor_hash(&input);
    let (program, region) = build_attention_graph();
    let graph =
        lower_region(&program, &region).map_err(|e| format!("attention graph lowering: {e:?}"))?;
    let mut state = GenericExecutionState::new(STATE_ID.0, SEQUENCE).map_err(|e| e.to_string())?;
    let before_rss = rss_kib();
    let attention_result = execute_generic_graph(
        &graph,
        input.clone(),
        |id| materializer.get(id),
        None,
        &mut state,
    )
    .map_err(|e| format!("generic attention/router execution: {e}"))?;
    let selection = attention_result
        .selection
        .clone()
        .ok_or("router produced no selection")?;
    let selected: Vec<u32> = selection
        .ids
        .iter()
        .copied()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let mut selected = selected;
    selected.sort_unstable();
    if selected.iter().any(|id| *id >= EXPERT_COUNT) {
        return Err("router selected an invalid expert".into());
    }
    let (expert_program, expert_region) = build_expert_graph(&selected, &expert_entries);
    let expert_graph = lower_region(&expert_program, &expert_region)
        .map_err(|e| format!("expert graph lowering: {e:?}"))?;
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
        &mut state,
    )
    .map_err(|e| format!("generic expert/final execution: {e}"))?;
    let mut checkpoint = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(checkpoint_path)
            .map_err(|e| e.to_string())?,
    );
    for (name, values, source) in [
        ("block_input", ValueId(0), &attention_result.values),
        ("normalized_input", ValueId(1), &attention_result.values),
        ("q_projection", ValueId(2), &attention_result.values),
        ("k_projection", ValueId(4), &attention_result.values),
        ("v_projection", ValueId(6), &attention_result.values),
        ("q", ValueId(11), &attention_result.values),
        ("k", ValueId(12), &attention_result.values),
        ("v", ValueId(10), &attention_result.values),
        ("attention_core", ValueId(13), &attention_result.values),
        ("o_projection", ValueId(15), &attention_result.values),
        (
            "post_attention_residual",
            ValueId(16),
            &attention_result.values,
        ),
        ("router_raw", ValueId(18), &attention_result.values),
        ("router_scores", ValueId(19), &attention_result.values),
        ("router_corrected", ValueId(20), &attention_result.values),
    ] {
        if let Some(tensor) = source.get(&values) {
            write_record(&mut checkpoint, name, tensor)?;
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
    write_record(&mut checkpoint, "selection_ids", &selection_ids)?;
    write_record(&mut checkpoint, "selection_weights", &selection_weights)?;
    for (name, id) in [
        ("post_attention_norm", ValueId(17)),
        ("shared_expert_output", ValueId(304)),
        ("moe_combined_output", ValueId(305)),
        ("final_block_output", ValueId(306)),
    ] {
        if let Some(tensor) = expert_result.values.get(&id) {
            write_record(&mut checkpoint, name, tensor)?;
        }
    }
    for (index, expert) in selected.iter().enumerate() {
        if let Some(tensor) = expert_result.values.get(&ValueId(1000 + index as u32)) {
            write_record(
                &mut checkpoint,
                &format!("selected_expert_{expert}"),
                tensor,
            )?;
        }
    }
    checkpoint.flush().map_err(|e| e.to_string())?;
    println!(
        "REAL_BLOCK=23 INPUT_HASH={} SELECTED_EXPERTS={:?} ROUTER_IDS={:?} ROUTING_WEIGHTS={:?} SOURCE_BYTES={} TOUCHED_TENSORS={} CACHE_F32_BYTES={} RSS_BEFORE_KIB={} RSS_AFTER_KIB={} STATE_LENGTH={} FINAL_ELEMENTS={}",
        input_hash,
        selected,
        selection.ids,
        selection.weights,
        materializer.source_bytes,
        materializer.touched.len(),
        materializer
            .cache
            .values()
            .map(|tensor| tensor.values.len() * 4)
            .sum::<usize>(),
        before_rss,
        rss_kib(),
        state.length,
        expert_result
            .values
            .get(&ValueId(306))
            .map_or(0, |tensor| tensor.values.len()),
    );
    Ok(())
}
