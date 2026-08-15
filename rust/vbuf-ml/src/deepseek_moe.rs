//! DeepSeek2/DeepSeek3 MoE tensor contract.

use crate::{ModelMetadata, ModelMetadataKey, MoeDirectory, TensorDirectory};
use crate::error::{MlError, MlErrorCode};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekMoELoader {
    pub architecture: String,
    pub embedding_length: u64,
    pub layer_count: u32,
    pub dense_layer_count: u32,
    pub moe_layer_count: u32,
    pub expert_count: u32,
    pub active_expert_count: u32,
    pub expert_feed_forward_length: u64,
    pub shared_expert_count: u32,
}

impl DeepSeekMoELoader {
    pub fn from_model(architecture: &str, metadata: &ModelMetadata<'_>, tensors: &TensorDirectory<'_>, moe: &MoeDirectory) -> Result<Self, MlError> {
        if !matches!(architecture, "deepseek2" | "deepseek32" | "deepseek2-ocr") {
            return Err(MlError::new(MlErrorCode::UnsupportedMoeArchitecture, "architecture is not a supported DeepSeek MoE variant"));
        }
        let parameters = moe.parameters();
        if !parameters.shared_experts || parameters.shared_expert_count == 0 {
            return Err(MlError::new(MlErrorCode::MoeExpertIndexInvalid, "DeepSeek MoE requires shared expert metadata"));
        }
        let embedding_length = metadata.unsigned(ModelMetadataKey::EmbeddingLength).ok_or_else(|| MlError::new(MlErrorCode::MissingRequiredMetadata, "DeepSeek MoE embedding length is absent"))?;
        let layer_count = metadata.unsigned(ModelMetadataKey::LayerCount).ok_or_else(|| MlError::new(MlErrorCode::MissingRequiredMetadata, "DeepSeek MoE layer count is absent"))?;
        if layer_count != u64::from(parameters.layer_count) {
            return Err(MlError::new(MlErrorCode::MoeExpertIndexInvalid, "DeepSeek MoE layer count disagrees with the expert catalog"));
        }
        let mut dense_layer_count = 0;
        let mut moe_layer_count = 0;
        let mut expert_ff = None;
        for layer in 0..parameters.layer_count {
            let prefix = format!("blk.{layer}");
            if tensors.get(&format!("{prefix}.ffn_gate_inp.weight")).is_some() {
                moe_layer_count += 1;
                let router = tensor(tensors, &format!("{prefix}.ffn_gate_inp.weight"))?;
                require_shape(router, &[embedding_length, u64::from(parameters.expert_count)])?;
                if let Some(bias) = tensors.get(&format!("{prefix}.ffn_exp_probs_b.bias")).or_else(|| tensors.get(&format!("{prefix}.exp_probs_b.bias"))) {
                    require_shape(bias, &[u64::from(parameters.expert_count)])?;
                }
                let down = tensor(tensors, &format!("{prefix}.ffn_down_exps.weight"))?;
                let gate = tensor(tensors, &format!("{prefix}.ffn_gate_exps.weight"))?;
                let up = tensor(tensors, &format!("{prefix}.ffn_up_exps.weight"))?;
                let ff = expert_ff_from(down, embedding_length, parameters.expert_count)?;
                require_shape(gate, &[embedding_length, ff, u64::from(parameters.expert_count)])?;
                require_shape(up, &[embedding_length, ff, u64::from(parameters.expert_count)])?;
                require_shape(down, &[ff, embedding_length, u64::from(parameters.expert_count)])?;
                if expert_ff.replace(ff).is_some_and(|old| old != ff) {
                    return Err(MlError::new(MlErrorCode::TensorRepresentationMismatch, "DeepSeek expert feed-forward dimensions differ between layers"));
                }
                let shared_ff = ff.checked_mul(u64::from(parameters.shared_expert_count)).ok_or_else(|| MlError::new(MlErrorCode::RepresentationArithmeticOverflow, "DeepSeek shared expert dimension overflows"))?;
                require_shape(tensor(tensors, &format!("{prefix}.ffn_gate_shexp.weight"))?, &[embedding_length, shared_ff])?;
                require_shape(tensor(tensors, &format!("{prefix}.ffn_down_shexp.weight"))?, &[shared_ff, embedding_length])?;
                require_shape(tensor(tensors, &format!("{prefix}.ffn_up_shexp.weight"))?, &[embedding_length, shared_ff])?;
            } else {
                dense_layer_count += 1;
                for suffix in ["ffn_gate", "ffn_down", "ffn_up"] {
                    let _ = tensor(tensors, &format!("{prefix}.{suffix}.weight"))?;
                }
            }
        }
        let expert_feed_forward_length = expert_ff.ok_or_else(|| MlError::new(MlErrorCode::MoeExpertReferenceMissing, "DeepSeek model has no MoE layers"))?;
        Ok(Self { architecture: architecture.to_owned(), embedding_length, layer_count: parameters.layer_count, dense_layer_count, moe_layer_count, expert_count: parameters.expert_count, active_expert_count: parameters.active_expert_count, expert_feed_forward_length, shared_expert_count: parameters.shared_expert_count })
    }
}

fn tensor<'a>(tensors: &'a TensorDirectory<'a>, name: &str) -> Result<&'a crate::TensorDescriptor<'a>, MlError> {
    tensors.get(name).ok_or_else(|| MlError::new(MlErrorCode::MoeExpertReferenceMissing, "DeepSeek MoE tensor is missing"))
}

fn expert_ff_from(tensor: &crate::TensorDescriptor<'_>, embedding: u64, experts: u32) -> Result<u64, MlError> {
    if tensor.dimensions.len() != 3 || tensor.dimensions[1] != embedding || tensor.dimensions[2] != u64::from(experts) || tensor.dimensions[0] == 0 {
        return Err(MlError::new(MlErrorCode::TensorRepresentationMismatch, "DeepSeek expert down tensor shape is invalid"));
    }
    Ok(tensor.dimensions[0])
}

fn require_shape(tensor: &crate::TensorDescriptor<'_>, expected: &[u64]) -> Result<(), MlError> {
    if tensor.dimensions != expected { return Err(MlError::new(MlErrorCode::TensorRepresentationMismatch, "DeepSeek MoE tensor shape does not match the architecture contract")); }
    Ok(())
}
