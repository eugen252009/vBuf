use std::path::PathBuf;
use vbuf_ml::{ConsumerModel, ConsumerTensorType};

fn artifact(name: &str) -> Option<ConsumerModel> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../research-models").join(name);
    if !path.exists() { return None; }
    Some(ConsumerModel::open(path).unwrap())
}

#[test]
fn real_vbuf_models_expose_validated_consumer_descriptors() {
    for name in ["Qwen3-0.6B-BF16.vbuf", "Qwen3-0.6B-Q8_0.vbuf"] {
        let Some(model) = artifact(name) else { return };
        assert!(model.is_validated().unwrap());
        assert_eq!(model.tensor_count().unwrap(), if name.contains("BF16") { 311 } else { 310 });
        let metadata = model.model_metadata().unwrap();
        assert_eq!(metadata.architecture, "qwen3");
        assert_eq!(metadata.kv_head_count, 8);
        assert_eq!(metadata.key_head_dimension, 128);
        assert_eq!(metadata.value_head_dimension, 128);
        assert_eq!(model.tokenizer_count().unwrap(), 151_936);
        assert_eq!(model.merge_count().unwrap(), 151_387);
        assert_eq!(model.add_bos().unwrap(), Some(false));
        assert_eq!(model.chat_template().unwrap().unwrap().len(), if name.contains("BF16") { 4168 } else { 4100 });
        assert_eq!(model.tensor_type(0).unwrap(), Some(if name.contains("BF16") { ConsumerTensorType::Bf16 } else { ConsumerTensorType::Q8_0 }));
        assert!(model.tensor_payload(0).unwrap().is_some());
    }
}

#[test]
fn consumer_descriptor_rejects_missing_files() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../research-models/does-not-exist.vbuf");
    assert!(ConsumerModel::open(path).is_err());
}
