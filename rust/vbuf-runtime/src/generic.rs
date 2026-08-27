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

pub trait TokenSelector {
    fn select(&self, logits: &GenericTensor) -> Result<u32, String>;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GreedyArgmaxSelector;

impl TokenSelector for GreedyArgmaxSelector {
    fn select(&self, logits: &GenericTensor) -> Result<u32, String> {
        logits.elements()?;
        let vocabulary = logits
            .dimensions
            .last()
            .copied()
            .ok_or("logits rank is invalid")?;
        let vocabulary =
            usize::try_from(vocabulary).map_err(|_| "logits vocabulary is too large")?;
        if vocabulary == 0 {
            return Err("logits vocabulary is empty".into());
        }
        let row_start = logits
            .values
            .len()
            .checked_sub(vocabulary)
            .ok_or("logits row is empty")?;
        let row = &logits.values[row_start..];
        if row.iter().any(|value| !value.is_finite()) {
            return Err("logits contain a non-finite value".into());
        }
        let best =
            row.iter().enumerate().fold(
                0usize,
                |best, (index, value)| {
                    if *value > row[best] { index } else { best }
                },
            );
        u32::try_from(best).map_err(|_| "selected token ID exceeds u32".into())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenerationStopReason {
    MaxNewTokens,
    Eos,
    Cancelled,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GenerationConfig {
    pub max_new_tokens: u32,
    pub eos_token_id: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationStep {
    pub index: u32,
    pub token_id: u32,
    pub position: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenerationResult {
    pub all_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub steps: Vec<GenerationStep>,
    pub stop_reason: GenerationStopReason,
    pub error: Option<String>,
}

pub fn generate<E, D, C, S>(
    prompt_token_ids: &[u32],
    initial_logits: GenericTensor,
    config: GenerationConfig,
    selector: &S,
    mut embed: E,
    mut decode: D,
    mut cancelled: C,
) -> GenerationResult
where
    E: FnMut(u32) -> Result<GenericTensor, String>,
    D: FnMut(GenericTensor, u64) -> Result<GenericTensor, String>,
    C: FnMut() -> bool,
    S: TokenSelector,
{
    let mut result = GenerationResult {
        all_token_ids: prompt_token_ids.to_vec(),
        generated_token_ids: Vec::new(),
        steps: Vec::new(),
        stop_reason: GenerationStopReason::MaxNewTokens,
        error: None,
    };
    let mut logits = initial_logits;
    for index in 0..config.max_new_tokens {
        if cancelled() {
            result.stop_reason = GenerationStopReason::Cancelled;
            return result;
        }
        let token_id = match selector.select(&logits) {
            Ok(token_id) => token_id,
            Err(error) => {
                result.stop_reason = GenerationStopReason::Failed;
                result.error = Some(error);
                return result;
            }
        };
        let position = match u64::try_from(result.all_token_ids.len()) {
            Ok(position) => position,
            Err(_) => {
                result.stop_reason = GenerationStopReason::Failed;
                result.error = Some("logical token sequence exceeds host limits".into());
                return result;
            }
        };
        result.all_token_ids.push(token_id);
        result.generated_token_ids.push(token_id);
        result.steps.push(GenerationStep {
            index,
            token_id,
            position,
        });
        if config.eos_token_id == Some(token_id) {
            result.stop_reason = GenerationStopReason::Eos;
            return result;
        }
        let embedding = match embed(token_id) {
            Ok(embedding) => embedding,
            Err(error) => {
                result.stop_reason = GenerationStopReason::Failed;
                result.error = Some(error);
                return result;
            }
        };
        logits = match decode(embedding, position) {
            Ok(logits) => logits,
            Err(error) => {
                result.stop_reason = GenerationStopReason::Failed;
                result.error = Some(error);
                return result;
            }
        };
    }
    result
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
    last_past_kv_positions_read: u64,
    last_kv_positions_appended: u64,
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
        self.last_past_kv_positions_read = 0;
        self.last_kv_positions_appended = 0;
    }

    pub fn bytes(&self) -> usize {
        (self.key.len() + self.value.len()) * std::mem::size_of::<f32>()
    }

    pub fn state_length(&self) -> u64 {
        self.length
    }

    pub fn kv_geometry(&self) -> Option<(u64, u64, u64)> {
        (self.length != 0).then_some((self.batch_size, self.kv_head_count, self.head_dim))
    }

    pub fn kv_values(&self) -> (&[f32], &[f32]) {
        (&self.key, &self.value)
    }

    pub fn last_past_kv_positions_read(&self) -> u64 {
        self.last_past_kv_positions_read
    }

    pub fn last_kv_positions_appended(&self) -> u64 {
        self.last_kv_positions_appended
    }

    fn validate_payload(&self) -> Result<(), String> {
        if self.length > self.capacity {
            return Err("generic attention state exceeds capacity".into());
        }
        if self.length == 0 {
            if self.batch_size != 0
                || self.kv_head_count != 0
                || self.head_dim != 0
                || !self.key.is_empty()
                || !self.value.is_empty()
            {
                return Err("empty generic attention state has a payload".into());
            }
            return Ok(());
        }
        if self.batch_size == 0 || self.kv_head_count == 0 || self.head_dim == 0 {
            return Err("generic attention state geometry is empty".into());
        }
        let elements = self
            .batch_size
            .checked_mul(self.length)
            .and_then(|value| value.checked_mul(self.kv_head_count))
            .and_then(|value| value.checked_mul(self.head_dim))
            .ok_or("generic attention state geometry overflows")?;
        let elements =
            usize::try_from(elements).map_err(|_| "generic attention state is too large")?;
        if self.key.len() != elements || self.value.len() != elements {
            return Err("generic attention state payload geometry is invalid".into());
        }
        Ok(())
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
    state.validate_payload()?;
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
                        let current_position = key_position.saturating_sub(old_length as usize);
                        let k_index = (((batch * query_length + current_position) * kv_heads
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
                        let current_position = key_position.saturating_sub(old_length as usize);
                        let v_index = (((batch * query_length + current_position) * kv_heads
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
    state.last_past_kv_positions_read = old_length;
    state.last_kv_positions_appended = attrs.query_length;
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
    let state_before = state.clone();
    let execution = (|| -> Result<GenericExecutionResult, String> {
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
                    if left.dimensions != right.dimensions
                        || left.values.len() != right.values.len()
                    {
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
                    if left.dimensions != right.dimensions
                        || left.values.len() != right.values.len()
                    {
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
    })();
    if execution.is_err() {
        *state = state_before;
    }
    execution
}

#[cfg(test)]
mod embedding_tests {
    use super::{GenericExecutionState, GenericTensor, attention, embedding_lookup};
    use crate::graph::{AttentionAttributes, AttentionMaskKind, AttentionPositionKind, StateId};
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

    fn sequence_tensor(
        sequence: usize,
        heads: usize,
        dimension: usize,
        offset: f32,
    ) -> GenericTensor {
        let mut values = Vec::with_capacity(sequence * heads * dimension);
        for position in 0..sequence {
            for head in 0..heads {
                for component in 0..dimension {
                    values.push(
                        offset
                            + (position * heads * dimension + head * dimension + component) as f32
                                * 0.03125,
                    );
                }
            }
        }
        GenericTensor {
            dimensions: vec![1, sequence as u64, heads as u64, dimension as u64],
            values,
        }
    }

    fn sequence_slice(tensor: &GenericTensor, start: usize, end: usize) -> GenericTensor {
        let width = tensor.dimensions[2] as usize * tensor.dimensions[3] as usize;
        GenericTensor {
            dimensions: vec![
                1,
                (end - start) as u64,
                tensor.dimensions[2],
                tensor.dimensions[3],
            ],
            values: tensor.values[start * width..end * width].to_vec(),
        }
    }

    fn attention_attributes(
        query_heads: u64,
        kv_heads: u64,
        dimension: u64,
        query_length: u64,
        current_kv_length: u64,
        state: StateId,
    ) -> AttentionAttributes {
        AttentionAttributes {
            batch_size: 1,
            query_head_count: query_heads,
            kv_head_count: kv_heads,
            head_dim: dimension,
            query_length,
            current_kv_length,
            scale: (dimension as f32).sqrt().recip(),
            mask: AttentionMaskKind::Causal,
            position: AttentionPositionKind::StateLength,
            state,
        }
    }

    fn assert_close(left: &GenericTensor, right: &GenericTensor) {
        assert_eq!(left.dimensions, right.dimensions);
        let maximum = left
            .values
            .iter()
            .zip(&right.values)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(maximum <= 1e-6, "maximum attention error was {maximum}");
    }

    fn assert_prefill_decode_equivalence(query_heads: usize, kv_heads: usize) {
        let sequence = 3;
        let dimension = 3;
        let query = sequence_tensor(sequence, query_heads, dimension, 0.25);
        let key = sequence_tensor(sequence, kv_heads, dimension, -0.5);
        let value = sequence_tensor(sequence, kv_heads, dimension, 0.75);
        let state_id = StateId(31 + query_heads as u32);
        let mut full_state = GenericExecutionState::new(state_id.0, sequence as u64).unwrap();
        let full = attention(
            &query,
            &key,
            &value,
            attention_attributes(
                query_heads as u64,
                kv_heads as u64,
                dimension as u64,
                sequence as u64,
                sequence as u64,
                state_id,
            ),
            &mut full_state,
        )
        .unwrap();

        let mut incremental_state =
            GenericExecutionState::new(state_id.0, sequence as u64).unwrap();
        let prefix = attention(
            &sequence_slice(&query, 0, 2),
            &sequence_slice(&key, 0, 2),
            &sequence_slice(&value, 0, 2),
            attention_attributes(
                query_heads as u64,
                kv_heads as u64,
                dimension as u64,
                2,
                2,
                state_id,
            ),
            &mut incremental_state,
        )
        .unwrap();
        assert_eq!(
            prefix.dimensions,
            [1, 2, query_heads as u64, dimension as u64]
        );
        let decode = attention(
            &sequence_slice(&query, 2, 3),
            &sequence_slice(&key, 2, 3),
            &sequence_slice(&value, 2, 3),
            attention_attributes(
                query_heads as u64,
                kv_heads as u64,
                dimension as u64,
                1,
                3,
                state_id,
            ),
            &mut incremental_state,
        )
        .unwrap();
        let final_offset = (sequence - 1) * query_heads * dimension;
        let full_final = GenericTensor {
            dimensions: decode.dimensions.clone(),
            values: full.values[final_offset..].to_vec(),
        };
        assert_close(&decode, &full_final);
        assert_eq!(incremental_state.state_length(), 3);
        assert_eq!(
            incremental_state.kv_geometry(),
            Some((1, kv_heads as u64, dimension as u64))
        );
        assert_eq!(
            incremental_state.kv_values().0.len(),
            sequence * kv_heads * dimension
        );
    }

    #[test]
    fn prefill_decode_equivalence_covers_mha_and_gqa() {
        assert_prefill_decode_equivalence(2, 2);
        assert_prefill_decode_equivalence(4, 2);
    }

    #[test]
    fn multi_token_continuation_uses_each_new_kv_position() {
        let query = sequence_tensor(3, 2, 2, 0.1);
        let key = sequence_tensor(3, 2, 2, 0.2);
        let value = sequence_tensor(3, 2, 2, 0.3);
        let state_id = StateId(41);
        let mut full_state = GenericExecutionState::new(state_id.0, 3).unwrap();
        let full = attention(
            &query,
            &key,
            &value,
            attention_attributes(2, 2, 2, 3, 3, state_id),
            &mut full_state,
        )
        .unwrap();
        let mut continuation_state = GenericExecutionState::new(state_id.0, 3).unwrap();
        attention(
            &sequence_slice(&query, 0, 1),
            &sequence_slice(&key, 0, 1),
            &sequence_slice(&value, 0, 1),
            attention_attributes(2, 2, 2, 1, 1, state_id),
            &mut continuation_state,
        )
        .unwrap();
        let continuation = attention(
            &sequence_slice(&query, 1, 3),
            &sequence_slice(&key, 1, 3),
            &sequence_slice(&value, 1, 3),
            attention_attributes(2, 2, 2, 2, 3, state_id),
            &mut continuation_state,
        )
        .unwrap();
        let final_offset = 2 * 2 * 2;
        assert_close(
            &GenericTensor {
                dimensions: vec![1, 1, 2, 2],
                values: full.values[final_offset..].to_vec(),
            },
            &GenericTensor {
                dimensions: vec![1, 1, 2, 2],
                values: continuation.values[4..].to_vec(),
            },
        );
        assert_eq!(continuation_state.state_length(), 3);
    }

    #[test]
    fn failed_decode_preserves_state_and_rejects_malformed_payload() {
        let query = sequence_tensor(2, 2, 2, 0.1);
        let key = sequence_tensor(2, 2, 2, 0.2);
        let value = sequence_tensor(2, 2, 2, 0.3);
        let state_id = StateId(51);
        let mut state = GenericExecutionState::new(state_id.0, 2).unwrap();
        attention(
            &sequence_slice(&query, 0, 1),
            &sequence_slice(&key, 0, 1),
            &sequence_slice(&value, 0, 1),
            attention_attributes(2, 2, 2, 1, 1, state_id),
            &mut state,
        )
        .unwrap();
        let before = state.clone();
        assert!(
            attention(
                &sequence_slice(&query, 1, 2),
                &sequence_slice(&key, 1, 2),
                &sequence_slice(&value, 1, 2),
                attention_attributes(2, 2, 2, 1, 3, state_id),
                &mut state,
            )
            .is_err()
        );
        assert_eq!(state, before);

        let mut malformed = state.clone();
        malformed.key.pop();
        assert!(
            attention(
                &sequence_slice(&query, 1, 2),
                &sequence_slice(&key, 1, 2),
                &sequence_slice(&value, 1, 2),
                attention_attributes(2, 2, 2, 1, 2, state_id),
                &mut malformed,
            )
            .is_err()
        );
        assert_eq!(malformed, {
            let mut expected = state.clone();
            expected.key.pop();
            expected
        });
    }
}

#[cfg(test)]
mod generation_tests {
    use super::{
        GenerationConfig, GenerationStopReason, GenericTensor, GreedyArgmaxSelector, TokenSelector,
        generate,
    };

    fn logits(values: Vec<f32>) -> GenericTensor {
        GenericTensor {
            dimensions: vec![1, 1, values.len() as u64],
            values,
        }
    }

    #[test]
    fn greedy_argmax_selects_unique_negative_and_lower_index_ties() {
        let selector = GreedyArgmaxSelector;
        assert_eq!(selector.select(&logits(vec![-3.0, -1.0, -2.0])), Ok(1));
        assert_eq!(selector.select(&logits(vec![4.0, 4.0, 1.0])), Ok(0));
    }

    #[test]
    fn greedy_argmax_rejects_invalid_logits() {
        let selector = GreedyArgmaxSelector;
        assert!(selector.select(&logits(vec![1.0, f32::NAN])).is_err());
        assert!(
            selector
                .select(&GenericTensor {
                    dimensions: vec![1, 1, 0],
                    values: Vec::new(),
                })
                .is_err()
        );
    }

    #[test]
    fn greedy_argmax_handles_a_large_vocabulary() {
        let selector = GreedyArgmaxSelector;
        let mut values = vec![-1.0; 4096];
        values[3071] = 2.0;
        assert_eq!(selector.select(&logits(values)), Ok(3071));
    }

    #[test]
    fn generation_tracks_positions_and_stops_at_the_limit() {
        let result = generate(
            &[51, 68, 82, 83],
            logits(vec![0.0, 2.0, 1.0]),
            GenerationConfig {
                max_new_tokens: 2,
                eos_token_id: None,
            },
            &GreedyArgmaxSelector,
            |token| Ok(logits(vec![token as f32, 0.0, 0.0])),
            |embedding, _position| Ok(embedding),
            || false,
        );
        assert_eq!(result.stop_reason, GenerationStopReason::MaxNewTokens);
        assert_eq!(result.generated_token_ids, vec![1, 0]);
        assert_eq!(
            result
                .steps
                .iter()
                .map(|step| step.position)
                .collect::<Vec<_>>(),
            vec![4, 5]
        );
    }

    #[test]
    fn generation_handles_eos_cancellation_and_failure() {
        let eos = generate(
            &[1],
            logits(vec![0.0, 3.0]),
            GenerationConfig {
                max_new_tokens: 4,
                eos_token_id: Some(1),
            },
            &GreedyArgmaxSelector,
            |_| panic!("EOS must not request an embedding"),
            |embedding, _| Ok(embedding),
            || false,
        );
        assert_eq!(eos.stop_reason, GenerationStopReason::Eos);
        assert_eq!(eos.generated_token_ids, vec![1]);

        let cancelled = generate(
            &[1],
            logits(vec![1.0]),
            GenerationConfig {
                max_new_tokens: 1,
                eos_token_id: None,
            },
            &GreedyArgmaxSelector,
            |_| panic!("cancelled generation must not request an embedding"),
            |embedding, _| Ok(embedding),
            || true,
        );
        assert_eq!(cancelled.stop_reason, GenerationStopReason::Cancelled);

        let failed = generate(
            &[1],
            logits(vec![1.0]),
            GenerationConfig {
                max_new_tokens: 1,
                eos_token_id: None,
            },
            &GreedyArgmaxSelector,
            |_| Err("embedding failed".into()),
            |embedding, _| Ok(embedding),
            || false,
        );
        assert_eq!(failed.stop_reason, GenerationStopReason::Failed);
        assert_eq!(failed.error.as_deref(), Some("embedding failed"));
    }
}
