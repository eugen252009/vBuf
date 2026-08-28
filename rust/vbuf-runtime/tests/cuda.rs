#![cfg(feature = "cuda")]

use vbuf_runtime::cuda::CudaBackend;
use vbuf_runtime::device::DeviceBackend;
use vbuf_runtime::generic::GenericTopKSelection;
use vbuf_runtime::graph::{
    ActivationKind, AttentionAttributes, AttentionMaskKind, AttentionPositionKind, StateId,
};

fn backend() -> CudaBackend {
    assert!(CudaBackend::device_count() > 0);
    CudaBackend::new(0).expect("CUDA device 0")
}

#[test]
fn cuda_upload_download_and_cleanup() {
    let backend = backend();
    let tensor = backend.upload_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert_eq!(backend.download_f32(&tensor).unwrap(), [1.0, 2.0, 3.0, 4.0]);
    drop(tensor);
    backend.synchronize().unwrap();
    assert_eq!(backend.live_bytes(), 0);
    assert_eq!(backend.live_tensors(), 0);
}

#[test]
fn cuda_matmul_rmsnorm_activation_and_attention() {
    let backend = backend();
    let input = backend
        .upload_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2])
        .unwrap();
    let weight = backend.upload_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
    let product = backend.matmul(&input, &weight).unwrap();
    assert_eq!(product.dimensions(), &[1, 2, 2]);
    let norm = backend
        .rms_norm(
            &product,
            &backend.upload_f32(&[1.0, 1.0], &[2]).unwrap(),
            1e-5,
        )
        .unwrap();
    let activated = backend.activation(&norm, ActivationKind::Silu).unwrap();
    assert!(
        backend
            .download_f32(&activated)
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );

    let query = backend
        .upload_f32(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 1, 2])
        .unwrap();
    let key = query.clone();
    let value = backend
        .upload_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 1, 2])
        .unwrap();
    let attention = backend
        .attention(
            &query,
            &key,
            &value,
            AttentionAttributes {
                batch_size: 1,
                query_head_count: 1,
                kv_head_count: 1,
                head_dim: 2,
                query_length: 2,
                current_kv_length: 2,
                scale: 2.0f32.sqrt().recip(),
                mask: AttentionMaskKind::Causal,
                position: AttentionPositionKind::StateLength,
                state: StateId(1),
            },
        )
        .unwrap();
    assert_eq!(attention.dimensions(), &[1, 2, 1, 2]);
    assert!(
        backend
            .download_f32(&attention)
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
    drop((
        attention, value, key, query, activated, norm, product, weight, input,
    ));
    backend.synchronize().unwrap();
    assert_eq!(backend.live_bytes(), 0);
}

#[test]
fn cuda_rejects_foreign_device_tensors() {
    let first = backend();
    if CudaBackend::device_count() < 2 {
        return;
    }
    let second = CudaBackend::new(1).unwrap();
    let left = first.upload_f32(&[1.0, 2.0], &[1, 2]).unwrap();
    let right = second.upload_f32(&[3.0, 4.0], &[1, 2]).unwrap();
    assert!(first.add(&left, &right).is_err());
}

#[test]
fn cuda_selected_expert_dispatch_uses_only_the_requested_expert() {
    let backend = backend();
    let input = backend
        .upload_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2])
        .unwrap();
    let gate = backend
        .upload_f32(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2])
        .unwrap();
    let up = backend
        .upload_f32(&[1.0, 1.0, 1.0, 1.0, 2.0, 2.0], &[3, 2])
        .unwrap();
    let down = backend
        .upload_f32(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[2, 3])
        .unwrap();
    let selection = GenericTopKSelection {
        ids: vec![0, 1],
        weights: vec![1.0, 1.0],
        token_count: 2,
        top_k: 1,
    };
    let output = backend
        .expert_dispatch(&input, &gate, &up, &down, &selection, 0)
        .unwrap();
    let values = backend.download_f32(&output).unwrap();
    assert!(values[0].is_finite() && values[2] == 0.0);
    drop((output, down, up, gate, input));
    backend.synchronize().unwrap();
    assert_eq!(backend.live_bytes(), 0);
}
