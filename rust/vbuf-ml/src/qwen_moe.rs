//! Qwen MoE tensor contract for the existing llama.cpp graph path.

use crate::{ModelMetadata, ModelMetadataKey, MoeDirectory, TensorDirectory};
use crate::error::{MlError, MlErrorCode};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenMoELoader {
    pub architecture: String,
    pub embedding_length: u64,
    pub layer_count: u32,
    pub expert_count: u32,
    pub active_expert_count: u32,
    pub expert_feed_forward_length: u64,
    pub shared_expert_feed_forward_length: Option<u64>,
}

impl QwenMoELoader {
    pub fn from_model(architecture: &str, metadata: &ModelMetadata<'_>, tensors: &TensorDirectory<'_>, moe: &MoeDirectory) -> Result<Self, MlError> {
        let architecture = architecture.to_owned();
        let is_qwen2 = architecture == "qwen2moe";
        let is_qwen3 = matches!(architecture.as_str(), "qwen3moe" | "qwen3vlmoe");
        if !is_qwen2 && !is_qwen3 {
            return Err(MlError::new(MlErrorCode::UnsupportedMoeArchitecture, "architecture is not a supported Qwen MoE variant"));
        }
        let parameters = moe.parameters();
        if is_qwen2 && (!parameters.shared_experts || parameters.shared_expert_count == 0) {
            return Err(MlError::new(MlErrorCode::MoeExpertIndexInvalid, "Qwen2 MoE requires shared expert metadata"));
        }
        let embedding_length = metadata.unsigned(ModelMetadataKey::EmbeddingLength).ok_or_else(|| MlError::new(MlErrorCode::MissingRequiredMetadata, "Qwen MoE embedding length is absent"))?;
        let layer_count = metadata.unsigned(ModelMetadataKey::LayerCount).ok_or_else(|| MlError::new(MlErrorCode::MissingRequiredMetadata, "Qwen MoE layer count is absent"))?;
        if layer_count != u64::from(parameters.layer_count) {
            return Err(MlError::new(MlErrorCode::MoeExpertIndexInvalid, "Qwen MoE layer count disagrees with the expert catalog"));
        }
        let mut expert_feed_forward_length = None;
        for layer in 0..parameters.layer_count {
            let router = tensor(tensors, &format!("blk.{layer}.ffn_gate_inp.weight"))?;
            require_shape(router, &[embedding_length, u64::from(parameters.expert_count)])?;
            let gate = tensor(tensors, &format!("blk.{layer}.ffn_gate_exps.weight"))?;
            let up = tensor(tensors, &format!("blk.{layer}.ffn_up_exps.weight"))?;
            let down = tensor(tensors, &format!("blk.{layer}.ffn_down_exps.weight"))?;
            let gate_shape = &[embedding_length, gate_shape_ff(gate)?, u64::from(parameters.expert_count)];
            require_shape(gate, gate_shape)?;
            require_shape(up, gate_shape)?;
            require_shape(down, &[gate_shape[1], embedding_length, u64::from(parameters.expert_count)])?;
            if expert_feed_forward_length.replace(gate_shape[1]).is_some_and(|length| length != gate_shape[1]) {
                return Err(MlError::new(MlErrorCode::TensorRepresentationMismatch, "Qwen expert feed-forward dimensions differ between layers"));
            }
            if is_qwen2 {
                for suffix in ["ffn_gate_inp_shexp", "ffn_gate_shexp", "ffn_down_shexp", "ffn_up_shexp"] {
                    let _ = tensor(tensors, &format!("blk.{layer}.{suffix}.weight"))?;
                }
            }
        }
        let expert_feed_forward_length = expert_feed_forward_length.ok_or_else(|| MlError::new(MlErrorCode::MoeExpertReferenceMissing, "Qwen MoE has no expert layers"))?;
        Ok(Self { architecture, embedding_length, layer_count: parameters.layer_count, expert_count: parameters.expert_count, active_expert_count: parameters.active_expert_count, expert_feed_forward_length, shared_expert_feed_forward_length: None })
    }
}

fn tensor<'a>(tensors: &'a TensorDirectory<'a>, name: &str) -> Result<&'a crate::TensorDescriptor<'a>, MlError> {
    tensors.get(name).ok_or_else(|| MlError::new(MlErrorCode::MoeExpertReferenceMissing, "Qwen MoE tensor is missing"))
}

fn gate_shape_ff(tensor: &crate::TensorDescriptor<'_>) -> Result<u64, MlError> {
    if tensor.dimensions.len() != 3 || tensor.dimensions[1] == 0 { return Err(MlError::new(MlErrorCode::TensorRepresentationMismatch, "Qwen expert tensor rank or dimension is invalid")); }
    Ok(tensor.dimensions[1])
}

fn require_shape(tensor: &crate::TensorDescriptor<'_>, expected: &[u64]) -> Result<(), MlError> {
    if tensor.dimensions != expected { return Err(MlError::new(MlErrorCode::TensorRepresentationMismatch, "Qwen MoE tensor shape does not match the architecture contract")); }
    Ok(())
}
