//! Small portable F32 graph executor used for backend qualification.
//!
//! Persistent tensor acquisition remains outside this module. The provider
//! receives only semantic `TensorId` values and returns validated materialized
//! F32 tensors. No model or source naming is interpreted here.

use crate::graph::{
    ActivationKind, AttentionMaskKind, ExecutionGraph, InputRef, OperationKind, TensorId, ValueId,
};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
pub struct GenericTensor {
    pub dimensions: Vec<u64>,
    pub values: Vec<f32>,
}

impl GenericTensor {
    pub fn elements(&self) -> Result<usize, String> {
        let mut elements = 1u64;
        for dimension in &self.dimensions {
            if *dimension == 0 {
                return Err("generic tensor has a zero dimension".into());
            }
            elements = elements
                .checked_mul(*dimension)
                .ok_or("generic tensor shape product overflows")?;
        }
        let elements = usize::try_from(elements).map_err(|_| "generic tensor is too large")?;
        if elements != self.values.len() {
            return Err("generic tensor payload length does not match shape".into());
        }
        Ok(elements)
    }
}

/// Gather validated rows from an embedding matrix without requiring the matrix
/// itself to be materialized. The row provider owns persistence and residency;
/// this generic seam only validates token geometry and assembles the result.
pub fn embedding_lookup<T>(
    token_ids: &GenericTensor,
    vocabulary_size: u64,
    hidden_size: u64,
    mut row_provider: T,
) -> Result<GenericTensor, String>
where
    T: FnMut(u64) -> Result<Vec<f32>, String>,
{
    if token_ids.dimensions.len() != 2
        || vocabulary_size == 0
        || hidden_size == 0
        || token_ids.values.len()
            != usize::try_from(
                token_ids.dimensions[0]
                    .checked_mul(token_ids.dimensions[1])
                    .ok_or("embedding token geometry overflows")?,
            )
            .map_err(|_| "embedding token geometry exceeds host limits")?
    {
        return Err("embedding token geometry is invalid".into());
    }
    let hidden_size = usize::try_from(hidden_size).map_err(|_| "embedding width is too large")?;
    let mut rows = HashMap::<u64, Vec<f32>>::new();
    let mut output = Vec::with_capacity(
        token_ids
            .values
            .len()
            .checked_mul(hidden_size)
            .ok_or("embedding output geometry overflows")?,
    );
    for token in &token_ids.values {
        if !token.is_finite() || *token < 0.0 || token.fract() != 0.0 {
            return Err("embedding token ID is not a non-negative integer".into());
        }
        let token_id = *token as u64;
        if token_id >= vocabulary_size {
            return Err("embedding token ID is outside vocabulary".into());
        }
        if !rows.contains_key(&token_id) {
            let row = row_provider(token_id)?;
            if row.len() != hidden_size {
                return Err("embedding row width does not match hidden size".into());
            }
            rows.insert(token_id, row);
        }
        let row = rows.get(&token_id).expect("validated embedding row");
        output.extend_from_slice(row);
    }
    Ok(GenericTensor {
        dimensions: vec![
            token_ids.dimensions[0],
            token_ids.dimensions[1],
            hidden_size as u64,
        ],
        values: output,
    })
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenericExecutionState {
    pub binding_id: u32,
    pub capacity: u64,
    pub length: u64,
    batch_size: u64,
    kv_head_count: u64,
    head_dim: u64,
    key: Vec<f32>,
    value: Vec<f32>,
}

impl GenericExecutionState {
    pub fn new(binding_id: u32, capacity: u64) -> Result<Self, String> {
        if capacity == 0 {
            return Err("generic execution state capacity is zero".into());
        }
        Ok(Self {
            binding_id,
            capacity,
            ..Default::default()
        })
    }

    pub fn reset(&mut self) {
        self.length = 0;
        self.batch_size = 0;
        self.kv_head_count = 0;
        self.head_dim = 0;
        self.key.clear();
        self.value.clear();
    }

    pub fn bytes(&self) -> usize {
        (self.key.len() + self.value.len()) * std::mem::size_of::<f32>()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenericTopKSelection {
    pub ids: Vec<u32>,
    pub weights: Vec<f32>,
    pub token_count: u64,
    pub top_k: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenericExecutionResult {
    pub values: HashMap<ValueId, GenericTensor>,
    pub selection: Option<GenericTopKSelection>,
}

fn input_value<'a>(
    values: &'a HashMap<ValueId, GenericTensor>,
    input: &InputRef,
) -> Result<&'a GenericTensor, String> {
    match input {
        InputRef::Value(id) => values
            .get(id)
            .ok_or_else(|| format!("generic value {} is unavailable", id.0)),
        InputRef::Tensor(_) => Err("generic operation expected a value input".into()),
    }
}

fn tensor_input<T>(
    operation: &crate::graph::Operation,
    index: usize,
    provider: &mut T,
) -> Result<GenericTensor, String>
where
    T: FnMut(TensorId) -> Result<GenericTensor, String>,
{
    match operation.inputs.get(index) {
        Some(InputRef::Tensor(id)) => provider(*id),
        _ => Err("generic operation expected a tensor input".into()),
    }
}

fn checked_last_dimension(tensor: &GenericTensor) -> Result<usize, String> {
    tensor
        .dimensions
        .last()
        .copied()
        .ok_or_else(|| "generic tensor has no dimensions".to_owned())
        .and_then(|value| {
            usize::try_from(value).map_err(|_| "generic dimension is too large".into())
        })
}

fn matmul<T>(
    input: &GenericTensor,
    weight: &GenericTensor,
    provider: &mut T,
) -> Result<GenericTensor, String>
where
    T: FnMut(TensorId) -> Result<GenericTensor, String>,
{
    let input_width = checked_last_dimension(input)?;
    if weight.dimensions.len() != 2 || weight.dimensions[1] != input_width as u64 {
        return Err("generic matmul geometry is invalid".into());
    }
    input.elements()?;
    weight.elements()?;
    let output_width =
        usize::try_from(weight.dimensions[0]).map_err(|_| "matmul output is too large")?;
    let rows = input.values.len() / input_width;
    let mut dimensions = input.dimensions.clone();
    *dimensions.last_mut().unwrap() = output_width as u64;
    let mut output = vec![0.0; rows * output_width];
    for row in 0..rows {
        for out in 0..output_width {
            let mut sum = 0.0f32;
            for column in 0..input_width {
                sum += input.values[row * input_width + column]
                    * weight.values[out * input_width + column];
            }
            output[row * output_width + out] = sum;
        }
    }
    let _ = provider;
    Ok(GenericTensor {
        dimensions,
        values: output,
    })
}

fn rms_norm(
    input: &GenericTensor,
    weight: &GenericTensor,
    epsilon: f32,
) -> Result<GenericTensor, String> {
    let width = checked_last_dimension(input)?;
    if weight.dimensions != [width as u64] || weight.values.len() != width || !epsilon.is_finite() {
        return Err("generic RMSNorm geometry is invalid".into());
    }
    let rows = input.values.len() / width;
    let mut output = vec![0.0; input.values.len()];
    for row in 0..rows {
        let start = row * width;
        let mean = input.values[start..start + width]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / width as f32;
        let scale = (mean + epsilon).sqrt().recip();
        for column in 0..width {
            output[start + column] = input.values[start + column] * scale * weight.values[column];
        }
    }
    Ok(GenericTensor {
        dimensions: input.dimensions.clone(),
        values: output,
    })
}

fn bias_add(input: &GenericTensor, bias: &GenericTensor) -> Result<GenericTensor, String> {
    let width = checked_last_dimension(input)?;
    if bias.dimensions != [width as u64] || bias.values.len() != width {
        return Err("generic bias geometry is invalid".into());
    }
    let mut output = input.clone();
    for row in output.values.chunks_exact_mut(width) {
        for (column, value) in row.iter_mut().enumerate() {
            *value += bias.values[column];
        }
    }
    Ok(output)
}

fn reshape_heads(
    input: &GenericTensor,
    head_count: u64,
    head_dim: u64,
    flatten: bool,
) -> Result<GenericTensor, String> {
    if input.dimensions.len() != if flatten { 4 } else { 3 } {
        return Err("generic head reshape rank is invalid".into());
    }
    let expected = head_count
        .checked_mul(head_dim)
        .ok_or("head reshape overflows")?;
    let dimensions = if flatten {
        if input.dimensions[2] != head_count || input.dimensions[3] != head_dim {
            return Err("generic head flatten geometry is invalid".into());
        }
        vec![input.dimensions[0], input.dimensions[1], expected]
    } else {
        if input.dimensions[2] != expected {
            return Err("generic head reshape geometry is invalid".into());
        }
        vec![
            input.dimensions[0],
            input.dimensions[1],
            head_count,
            head_dim,
        ]
    };
    Ok(GenericTensor {
        dimensions,
        values: input.values.clone(),
    })
}

fn rotary(
    input: &GenericTensor,
    attrs: crate::graph::RotaryAttributes,
) -> Result<GenericTensor, String> {
    if input.dimensions.len() != 4
        || input.dimensions[2] != attrs.head_count
        || input.dimensions[3] != attrs.head_dim
        || attrs.rotary_dim % 2 != 0
    {
        return Err("generic rotary geometry is invalid".into());
    }
    let mut output = input.values.clone();
    let half = attrs.rotary_dim as usize / 2;
    let head_dim = attrs.head_dim as usize;
    let sequence = input.dimensions[1] as usize;
    let heads = attrs.head_count as usize;
    for batch in 0..input.dimensions[0] as usize {
        for position in 0..sequence {
            for head in 0..heads {
                let base = ((batch * sequence + position) * heads + head) * head_dim;
                for index in 0..half {
                    let inverse = attrs
                        .theta
                        .powf(-(2.0 * index as f32) / attrs.rotary_dim as f32);
                    let angle = (attrs.position_start + position as u64) as f32 * inverse;
                    let cosine = angle.cos();
                    let sine = angle.sin();
                    let left = input.values[base + index];
                    let right = input.values[base + half + index];
                    output[base + index] = left * cosine - right * sine;
                    output[base + half + index] = right * cosine + left * sine;
                }
            }
        }
    }
    Ok(GenericTensor {
        dimensions: input.dimensions.clone(),
        values: output,
    })
}

fn attention(
    query: &GenericTensor,
    key: &GenericTensor,
    value: &GenericTensor,
    attrs: crate::graph::AttentionAttributes,
    state: &mut GenericExecutionState,
) -> Result<GenericTensor, String> {
    let expected_query = [
        attrs.batch_size,
        attrs.query_length,
        attrs.query_head_count,
        attrs.head_dim,
    ];
    let expected_kv = [
        attrs.batch_size,
        attrs.query_length,
        attrs.kv_head_count,
        attrs.head_dim,
    ];
    if query.dimensions != expected_query
        || key.dimensions != expected_kv
        || value.dimensions != expected_kv
        || state.binding_id != attrs.state.0
        || attrs.current_kv_length != state.length + attrs.query_length
        || attrs.query_length > state.capacity.saturating_sub(state.length)
        || attrs.query_head_count % attrs.kv_head_count != 0
    {
        return Err("generic attention geometry or state is invalid".into());
    }
    query.elements()?;
    key.elements()?;
    value.elements()?;
    if state.length != 0 {
        if state.batch_size != attrs.batch_size
            || state.kv_head_count != attrs.kv_head_count
            || state.head_dim != attrs.head_dim
        {
            return Err("generic attention state geometry changed".into());
        }
    }
    let old_length = state.length;
    let visible = attrs.current_kv_length as usize;
    let q_heads = attrs.query_head_count as usize;
    let kv_heads = attrs.kv_head_count as usize;
    let dimension = attrs.head_dim as usize;
    let query_length = attrs.query_length as usize;
    let batch_size = attrs.batch_size as usize;
    let mut next_key = state.key.clone();
    let mut next_value = state.value.clone();
    next_key.extend_from_slice(&key.values);
    next_value.extend_from_slice(&value.values);
    let mut output = vec![0.0; query.values.len()];
    let group = q_heads / kv_heads;
    for batch in 0..batch_size {
        for query_position in 0..query_length {
            for query_head in 0..q_heads {
                let kv_head = query_head / group;
                let last = if attrs.mask == AttentionMaskKind::Causal {
                    old_length as usize + query_position
                } else {
                    visible - 1
                };
                let mut scores = vec![0.0f32; last + 1];
                for key_position in 0..=last {
                    let mut dot = 0.0;
                    for component in 0..dimension {
                        let q_index = (((batch * query_length + query_position) * q_heads
                            + query_head)
                            * dimension)
                            + component;
                        let k_index = (((batch * attrs.query_length as usize
                            + key_position.min(query_length - 1))
                            * kv_heads
                            + kv_head)
                            * dimension)
                            + component;
                        let state_index =
                            (((batch * old_length as usize + key_position) * kv_heads + kv_head)
                                * dimension)
                                + component;
                        let kval = if key_position < old_length as usize {
                            state.key[state_index]
                        } else {
                            key.values[k_index]
                        };
                        dot += query.values[q_index] * kval;
                    }
                    scores[key_position] = dot * attrs.scale;
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut total = 0.0;
                for score in &mut scores {
                    *score = (*score - maximum).exp();
                    total += *score;
                }
                for component in 0..dimension {
                    let mut sum = 0.0;
                    for key_position in 0..=last {
                        let v_index = (((batch * attrs.query_length as usize
                            + key_position.min(query_length - 1))
                            * kv_heads
                            + kv_head)
                            * dimension)
                            + component;
                        let state_index =
                            (((batch * old_length as usize + key_position) * kv_heads + kv_head)
                                * dimension)
                                + component;
                        let vval = if key_position < old_length as usize {
                            state.value[state_index]
                        } else {
                            value.values[v_index]
                        };
                        sum += scores[key_position] / total * vval;
                    }
                    let output_index = (((batch * query_length + query_position) * q_heads
                        + query_head)
                        * dimension)
                        + component;
                    output[output_index] = sum;
                }
            }
        }
    }
    state.batch_size = attrs.batch_size;
    state.kv_head_count = attrs.kv_head_count;
    state.head_dim = attrs.head_dim;
    state.length += attrs.query_length;
    state.key = next_key;
    state.value = next_value;
    Ok(GenericTensor {
        dimensions: expected_query.to_vec(),
        values: output,
    })
}

fn top_k<T>(
    scores: &GenericTensor,
    corrected: &GenericTensor,
    top_k: u32,
) -> Result<GenericTopKSelection, String> {
    let expert_count = *scores
        .dimensions
        .last()
        .ok_or("router score rank is invalid")? as usize;
    if scores.dimensions != corrected.dimensions
        || expert_count == 0
        || top_k as usize > expert_count
    {
        return Err("router score geometry is invalid".into());
    }
    let token_count = scores.values.len() / expert_count;
    let mut ids = Vec::with_capacity(token_count * top_k as usize);
    let mut weights = Vec::with_capacity(ids.capacity());
    for token in 0..token_count {
        let start = token * expert_count;
        let mut candidates: Vec<_> = (0..expert_count)
            .map(|index| (corrected.values[start + index], index))
            .collect();
        candidates.sort_by(|left, right| right.0.total_cmp(&left.0).then(left.1.cmp(&right.1)));
        let selected = &candidates[..top_k as usize];
        let denominator: f32 = selected
            .iter()
            .map(|(_, index)| scores.values[start + *index])
            .sum();
        for (_, index) in selected {
            ids.push(*index as u32);
            weights.push(scores.values[start + *index] / (denominator + 1e-20));
        }
    }
    Ok(GenericTopKSelection {
        ids,
        weights,
        token_count: token_count as u64,
        top_k,
    })
}

fn expert_dispatch(
    input: &GenericTensor,
    gate: &GenericTensor,
    up: &GenericTensor,
    down: &GenericTensor,
    expert_id: u32,
    selection: &GenericTopKSelection,
) -> Result<GenericTensor, String> {
    if input.dimensions.len() != 3
        || gate.dimensions.len() != 2
        || up.dimensions != gate.dimensions
        || down.dimensions.len() != 2
        || gate.dimensions[1] != input.dimensions[2]
        || down.dimensions[0] != input.dimensions[2]
        || down.dimensions[1] != gate.dimensions[0]
    {
        return Err("generic expert dispatch geometry is invalid".into());
    }
    let token_count = (input.dimensions[0] * input.dimensions[1]) as usize;
    if selection.token_count as usize != token_count {
        return Err("generic expert selection token count is invalid".into());
    }
    let hidden = input.dimensions[2] as usize;
    let intermediate = gate.dimensions[0] as usize;
    let top_k = selection.top_k as usize;
    let mut output = vec![0.0; input.values.len()];
    for token in 0..token_count {
        for rank in 0..top_k {
            let selection_index = token * top_k + rank;
            if selection.ids[selection_index] != expert_id {
                continue;
            }
            let input_offset = token * hidden;
            let mut activated = vec![0.0; intermediate];
            for channel in 0..intermediate {
                let mut gate_sum = 0.0;
                let mut up_sum = 0.0;
                for column in 0..hidden {
                    gate_sum += input.values[input_offset + column]
                        * gate.values[channel * hidden + column];
                    up_sum +=
                        input.values[input_offset + column] * up.values[channel * hidden + column];
                }
                let silu = gate_sum / (1.0 + (-gate_sum).exp());
                activated[channel] = silu * up_sum;
            }
            let weight = selection.weights[selection_index];
            for column in 0..hidden {
                let mut sum = 0.0;
                for channel in 0..intermediate {
                    sum += activated[channel] * down.values[column * intermediate + channel];
                }
                output[input_offset + column] += weight * sum;
            }
        }
    }
    Ok(GenericTensor {
        dimensions: input.dimensions.clone(),
        values: output,
    })
}

pub fn execute_generic_graph<T>(
    graph: &ExecutionGraph,
    input: GenericTensor,
    mut provider: T,
    selection: Option<&GenericTopKSelection>,
    state: &mut GenericExecutionState,
) -> Result<GenericExecutionResult, String>
where
    T: FnMut(TensorId) -> Result<GenericTensor, String>,
{
    let mut values = HashMap::from([(graph.input, input)]);
    let mut result = GenericExecutionResult {
        values: HashMap::new(),
        selection: None,
    };
    for operation in &graph.operations {
        let value = |index: usize| -> Result<GenericTensor, String> {
            Ok(input_value(
                &values,
                operation
                    .inputs
                    .get(index)
                    .ok_or("operation input is missing")?,
            )?
            .clone())
        };
        let output = match operation.kind {
            OperationKind::RmsNorm => rms_norm(
                &value(0)?,
                &tensor_input(operation, 1, &mut provider)?,
                operation
                    .attributes
                    .epsilon
                    .ok_or("RMSNorm epsilon missing")?,
            )?,
            OperationKind::MatMul | OperationKind::QuantizedMatMul => matmul(
                &value(0)?,
                &tensor_input(operation, 1, &mut provider)?,
                &mut provider,
            )?,
            OperationKind::BiasAdd => {
                bias_add(&value(0)?, &tensor_input(operation, 1, &mut provider)?)?
            }
            OperationKind::Activation => {
                let input = value(0)?;
                let activation = operation
                    .attributes
                    .activation
                    .ok_or("activation kind missing")?;
                let values = input
                    .values
                    .into_iter()
                    .map(|x| match activation {
                        ActivationKind::Silu => x / (1.0 + (-x).exp()),
                        ActivationKind::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                    })
                    .collect();
                GenericTensor {
                    dimensions: input.dimensions,
                    values,
                }
            }
            OperationKind::ReshapeHeads => {
                let attrs = operation
                    .attributes
                    .head_reshape
                    .ok_or("head reshape attributes missing")?;
                reshape_heads(&value(0)?, attrs.head_count, attrs.head_dim, attrs.flatten)?
            }
            OperationKind::Rotary => rotary(
                &value(0)?,
                operation
                    .attributes
                    .rotary
                    .ok_or("rotary attributes missing")?,
            )?,
            OperationKind::ResidualAdd | OperationKind::ElementwiseMul => {
                let left = value(0)?;
                let right = value(1)?;
                if left.dimensions != right.dimensions || left.values.len() != right.values.len() {
                    return Err("elementwise geometry is invalid".into());
                }
                let values = if operation.kind == OperationKind::ResidualAdd {
                    left.values
                        .iter()
                        .zip(right.values.iter())
                        .map(|(a, b)| a + b)
                        .collect()
                } else {
                    left.values
                        .iter()
                        .zip(right.values.iter())
                        .map(|(a, b)| a * b)
                        .collect()
                };
                GenericTensor {
                    dimensions: left.dimensions,
                    values,
                }
            }
            OperationKind::ZeroLike => {
                let input = value(0)?;
                GenericTensor {
                    dimensions: input.dimensions,
                    values: vec![0.0; input.values.len()],
                }
            }
            OperationKind::WeightedAdd => {
                let left = value(0)?;
                let right = value(1)?;
                if left.dimensions != right.dimensions || left.values.len() != right.values.len() {
                    return Err("weighted add geometry is invalid".into());
                }
                let weight = operation
                    .attributes
                    .weighted_add
                    .ok_or("weighted add weight missing")?;
                let values = left
                    .values
                    .iter()
                    .zip(right.values.iter())
                    .map(|(a, b)| a + weight * b)
                    .collect();
                GenericTensor {
                    dimensions: left.dimensions,
                    values,
                }
            }
            OperationKind::Attention => attention(
                &value(0)?,
                &value(1)?,
                &value(2)?,
                operation
                    .attributes
                    .attention
                    .ok_or("attention attributes missing")?,
                state,
            )?,
            OperationKind::TopKRouter => {
                let scores = value(0)?;
                let corrected = value(1)?;
                result.selection = Some(top_k::<T>(
                    &scores,
                    &corrected,
                    operation.attributes.top_k.ok_or("top-k missing")?,
                )?);
                scores
            }
            OperationKind::ExpertDispatch => {
                let selection = selection.ok_or("expert dispatch selection is missing")?;
                expert_dispatch(
                    &value(0)?,
                    &tensor_input(operation, 1, &mut provider)?,
                    &tensor_input(operation, 2, &mut provider)?,
                    &tensor_input(operation, 3, &mut provider)?,
                    operation
                        .attributes
                        .expert_dispatch
                        .ok_or("expert dispatch identity is missing")?
                        .expert_id,
                    selection,
                )?
            }
            _ => {
                return Err(format!(
                    "generic operation {:?} is unsupported",
                    operation.kind
                ));
            }
        };
        values.insert(operation.output, output);
    }
    result.values = values;
    Ok(result)
}

#[cfg(test)]
mod embedding_tests {
    use super::{GenericTensor, embedding_lookup};
    use std::cell::Cell;

    fn ids(values: Vec<f32>, dimensions: Vec<u64>) -> GenericTensor {
        GenericTensor { dimensions, values }
    }

    #[test]
    fn embedding_lookup_deduplicates_rows_and_preserves_token_order() {
        let calls = Cell::new(0);
        let result = embedding_lookup(&ids(vec![1.0, 2.0, 1.0], vec![1, 3]), 4, 2, |id| {
            calls.set(calls.get() + 1);
            Ok(vec![id as f32, id as f32 + 0.5])
        })
        .unwrap();
        assert_eq!(calls.get(), 2);
        assert_eq!(result.dimensions, [1, 3, 2]);
        assert_eq!(result.values, [1.0, 1.5, 2.0, 2.5, 1.0, 1.5]);
    }

    #[test]
    fn embedding_lookup_rejects_invalid_ids_and_geometry() {
        for value in [-1.0, 4.0, 1.5, f32::NAN, f32::INFINITY] {
            assert!(
                embedding_lookup(&ids(vec![value], vec![1, 1]), 4, 2, |_| Ok(vec![0.0; 2]))
                    .is_err()
            );
        }
        assert!(embedding_lookup(&ids(vec![0.0], vec![1]), 4, 2, |_| Ok(vec![0.0; 2])).is_err());
        assert!(embedding_lookup(&ids(vec![0.0], vec![1, 1]), 4, 2, |_| Ok(vec![0.0])).is_err());
    }
}
