use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;
use vbuf_core::v06::{V06Physical, V06Semantic};
use vbuf_ml::hf_import::{
    HfTransport, ImportOptions, RangeResponse, RepositoryMetadata, ShardPlan, SourceIdentity,
    build_plan, execute_plan, execute_plan_parallel, parse_safetensors_header, space_gate,
};
use vbuf_ml::layout::{LayoutClass, LayoutPlan, PlacementLengthRequest};
use vbuf_ml::{Bootstrap, F8QuantizationDirectory, ModelMetadata, TensorDirectory};

#[derive(Clone)]
struct FixtureTransport {
    shards: BTreeMap<String, Vec<u8>>,
    fail_once: bool,
    calls: usize,
    always_fail: bool,
}

impl vbuf_ml::hf_import::RepositoryTransport for FixtureTransport {
    fn metadata(&mut self, _name: &str) -> Result<Option<Vec<u8>>, String> {
        Ok(None)
    }

    fn range(&mut self, shard: &str, start: u64, end: u64) -> Result<RangeResponse, String> {
        self.calls += 1;
        if self.always_fail {
            return Err("controlled interruption".into());
        }
        if self.fail_once {
            self.fail_once = false;
            return Err("controlled connection reset".into());
        }
        let bytes = self
            .shards
            .get(shard)
            .ok_or_else(|| "unknown shard".to_string())?;
        let start = usize::try_from(start).map_err(|_| "start overflow")?;
        let end = usize::try_from(end).map_err(|_| "end overflow")?;
        Ok(RangeResponse {
            status: 206,
            content_range: Some((start as u64, end as u64, Some(bytes.len() as u64))),
            body: bytes
                .get(start..end)
                .ok_or_else(|| "range outside fixture".to_string())?
                .to_vec(),
        })
    }
}

fn fixture() -> (
    RepositoryMetadata,
    Vec<(ShardPlan, Vec<u8>, Vec<u8>)>,
    FixtureTransport,
) {
    let payload = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let header = serde_json::json!({
        "weight": {"dtype":"F32", "shape":[2], "data_offsets":[0,8]}
    })
    .to_string()
    .into_bytes();
    let mut source = (header.len() as u64).to_le_bytes().to_vec();
    source.extend_from_slice(&header);
    source.extend_from_slice(&payload);
    let first = source[..8].to_vec();
    let header_bytes = source[8..8 + header.len()].to_vec();
    let config = serde_json::json!({
        "architectures":["TinyForCausalLM"], "max_position_embeddings":128,
        "hidden_size":4, "num_hidden_layers":1, "num_attention_heads":1,
        "intermediate_size":8, "rms_norm_eps":1e-5, "rope_theta":10000.0
    })
    .to_string()
    .into_bytes();
    let metadata = RepositoryMetadata {
        source: SourceIdentity {
            repository: "fixture/model".into(),
            requested_revision: "main".into(),
            resolved_revision: "0123456789012345678901234567890123456789".into(),
        },
        config,
        index: None,
        metadata_bytes: 0,
    };
    let shard = ShardPlan {
        id: "model.safetensors".into(),
        file_name: "model.safetensors".into(),
        size: source.len() as u64,
        header_length: header.len() as u64,
    };
    let transport = FixtureTransport {
        shards: BTreeMap::from([("model.safetensors".into(), source)]),
        fail_once: false,
        calls: 0,
        always_fail: false,
    };
    (metadata, vec![(shard, first, header_bytes)], transport)
}

#[test]
fn fixture_plans_with_u64_offsets_and_direct_scatter() {
    let (metadata, shards, mut transport) = fixture();
    let plan = build_plan(metadata, None, shards, 64).unwrap();
    assert!(plan.can_execute);
    assert!(plan.all_destination_offsets_known);
    assert_eq!(plan.tensors.len(), 1);
    assert_eq!(plan.tensors[0].source_length, 8);
    assert_eq!(
        plan.tensors[0].destination_offset % plan.destination_alignment,
        0
    );
    let root = std::env::temp_dir().join(format!("vbuf-hf-import-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let output = root.join("fixture.vbuf");
    let stats = execute_plan(
        &mut transport,
        &plan,
        &output,
        &ImportOptions {
            staging_bytes: 64,
            max_retries: 2,
            parallel_requests: 1,
        },
    )
    .unwrap();
    assert_eq!(stats.destination_written_bytes, 8);
    let bytes = fs::read(&output).unwrap();
    let validated = vbuf_core::v06::parse_v06(&bytes).unwrap();
    let bootstrap = Bootstrap::discover(&validated).unwrap();
    ModelMetadata::parse(&validated, &bootstrap).unwrap();
    let start = usize::try_from(plan.tensors[0].destination_offset).unwrap();
    assert_eq!(&bytes[start..start + 8], &[1, 2, 3, 4, 5, 6, 7, 8]);
    assert!(!Path::new(&format!("{}.partial", output.display())).exists());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn parallel_executor_preserves_direct_scatter_and_state_contract() {
    let (metadata, shards, transport) = fixture();
    let plan = build_plan(metadata, None, shards, 64).unwrap();
    let root = std::env::temp_dir().join(format!("vbuf-hf-parallel-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let output = root.join("fixture.vbuf");
    let stats = execute_plan_parallel(
        &transport,
        &plan,
        &output,
        &ImportOptions {
            staging_bytes: 64,
            max_retries: 0,
            parallel_requests: 2,
        },
    )
    .unwrap();
    assert_eq!(stats.destination_written_bytes, 8);
    assert!(output.exists());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn retry_is_bounded_and_resume_identity_is_durable() {
    let (metadata, shards, mut transport) = fixture();
    let plan = build_plan(metadata, None, shards, 64).unwrap();
    transport.fail_once = true;
    let root = std::env::temp_dir().join(format!("vbuf-hf-resume-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let output = root.join("fixture.vbuf");
    let stats = execute_plan(
        &mut transport,
        &plan,
        &output,
        &ImportOptions {
            staging_bytes: 64,
            max_retries: 2,
            parallel_requests: 1,
        },
    )
    .unwrap();
    assert_eq!(stats.resume_downloaded_bytes, 8);
    assert!(transport.calls >= 2);
    let mut changed = plan.clone();
    changed.source.resolved_revision = "ffffffffffffffffffffffffffffffffffffffff".into();
    let error =
        execute_plan(&mut transport, &changed, &output, &ImportOptions::default()).unwrap_err();
    assert!(error.to_string().contains("identity") || error.to_string().contains("output"));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn malformed_header_and_space_gate_fail_closed() {
    assert!(parse_safetensors_header("x", 8, &[0; 8], b"{}").is_err());
    assert!(space_gate(1, 1024, 64).is_err());
}

#[test]
fn fp8_plan_streams_bytes_and_reopens_scale_provenance() {
    let header = serde_json::json!({
        "weight": {"dtype":"F8_E4M3", "shape":[2,2], "data_offsets":[0,4]},
        "weight_scale_inv": {"dtype":"F32", "shape":[2,1], "data_offsets":[4,12]}
    })
    .to_string()
    .into_bytes();
    let mut source = (header.len() as u64).to_le_bytes().to_vec();
    source.extend_from_slice(&header);
    source.extend_from_slice(&[0x38, 0x40, 0xb8, 0x00]);
    source.extend_from_slice(&1.0f32.to_le_bytes());
    source.extend_from_slice(&2.0f32.to_le_bytes());
    let first = source[..8].to_vec();
    let header_bytes = source[8..8 + header.len()].to_vec();
    let metadata = RepositoryMetadata {
        source: SourceIdentity {
            repository: "fixture/fp8".into(),
            requested_revision: "main".into(),
            resolved_revision: "0123456789012345678901234567890123456789".into(),
        },
        config: serde_json::json!({
            "architectures":["TinyForCausalLM"], "weight_block_size":[1,2]
        })
        .to_string()
        .into_bytes(),
        index: None,
        metadata_bytes: 0,
    };
    let shard = ShardPlan {
        id: "model.safetensors".into(),
        file_name: "model.safetensors".into(),
        size: source.len() as u64,
        header_length: header.len() as u64,
    };
    let mut transport = FixtureTransport {
        shards: BTreeMap::from([("model.safetensors".into(), source)]),
        fail_once: false,
        calls: 0,
        always_fail: false,
    };
    let plan = build_plan(metadata, None, vec![(shard, first, header_bytes)], 64).unwrap();
    assert!(plan.can_execute);
    assert_eq!(plan.quantization.len(), 1);
    assert_eq!(plan.quantization[0].scale_name, "weight_scale_inv");

    let root = std::env::temp_dir().join(format!("vbuf-hf-fp8-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let output = root.join("fp8.vbuf");
    execute_plan(&mut transport, &plan, &output, &ImportOptions::default()).unwrap();
    let bytes = fs::read(&output).unwrap();
    let validated = vbuf_core::v06::parse_v06(&bytes).unwrap();
    let bootstrap = Bootstrap::discover(&validated).unwrap();
    let directory = TensorDirectory::parse(&validated, &bootstrap).unwrap();
    assert_eq!(
        directory.get("weight").unwrap().representation,
        vbuf_ml::TensorRepresentation::F8_E4M3
    );
    let quantization = F8QuantizationDirectory::parse(&validated, &bootstrap)
        .unwrap()
        .unwrap();
    assert_eq!(quantization.entries(), plan.quantization.as_slice());
    let payload = directory
        .get("weight")
        .unwrap()
        .range
        .as_ref()
        .unwrap()
        .bytes();
    let scales = directory
        .get("weight_scale_inv")
        .unwrap()
        .range
        .as_ref()
        .unwrap()
        .bytes();
    assert_eq!(
        vbuf_ml::dequantize_f8_e4m3(payload, [2, 2], scales, [1, 2]).unwrap(),
        vec![1.0, 2.0, -2.0, 0.0]
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn interrupted_conversion_keeps_incomplete_state_and_resumes() {
    let (metadata, shards, mut transport) = fixture();
    let plan = build_plan(metadata, None, shards, 64).unwrap();
    transport.always_fail = true;
    let root = std::env::temp_dir().join(format!("vbuf-hf-interrupt-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let output = root.join("fixture.vbuf");
    assert!(
        execute_plan(
            &mut transport,
            &plan,
            &output,
            &ImportOptions {
                staging_bytes: 64,
                max_retries: 0,
                parallel_requests: 1,
            }
        )
        .is_err()
    );
    assert!(root.join("fixture.vbuf.partial").exists());
    assert!(root.join("fixture.vbuf.convert-state").exists());
    transport.always_fail = false;
    let stats = execute_plan(&mut transport, &plan, &output, &ImportOptions::default()).unwrap();
    assert_eq!(stats.resume_skipped_bytes, 0);
    assert!(output.exists());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn synthetic_large_layouts_remain_u64_safe_without_allocating_payloads() {
    for size in [1u64 << 32, 1u64 << 36, 1u64 << 37, 1u64 << 38] {
        let request = PlacementLengthRequest {
            class: LayoutClass::TensorPayload,
            order: 1,
            key_id: 0x0200,
            semantic: V06Semantic::Opaque,
            physical: V06Physical::Array,
            bit_width: 8,
            count: size,
            payload_alignment: 16,
            payload_len: size,
        };
        let plan = LayoutPlan::build_lengths(&[request], 4).unwrap();
        assert!(plan.final_size() > size);
        assert!(plan.entries()[0].payload_start % 16 == 0);
        assert!(plan.final_size() < u64::MAX);
    }
}

#[test]
fn sharded_index_ownership_is_validated_against_headers() {
    let (metadata, shards, _) = fixture();
    let index = serde_json::json!({
        "metadata":{"total_size":8},
        "weight_map":{"weight":"model.safetensors"}
    })
    .to_string();
    let plan = build_plan(metadata.clone(), Some(index.as_bytes()), shards.clone(), 64).unwrap();
    assert_eq!(plan.shards.len(), 1);
    assert_eq!(plan.tensors[0].source_shard, "model.safetensors");
    let mismatch = serde_json::json!({"weight_map":{"weight":"other.safetensors"}}).to_string();
    assert!(build_plan(metadata, Some(mismatch.as_bytes()), shards, 64).is_err());
}

#[test]
fn production_http_transport_preserves_range_across_redirect() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        for request_index in 0..2 {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut byte = [0u8; 1];
            while !request.ends_with(b"\r\n\r\n") {
                stream.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
            }
            let request = String::from_utf8(request).unwrap();
            assert!(request.to_ascii_lowercase().contains("range: bytes=0-3"));
            if request_index == 0 {
                stream
                    .write_all(
                        b"HTTP/1.1 302 Found\r\nLocation: /signed\r\nContent-Length: 0\r\n\r\n",
                    )
                    .unwrap();
            } else {
                stream.write_all(b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-3/4\r\nContent-Length: 4\r\n\r\nabcd").unwrap();
            }
        }
    });
    let mut transport = HfTransport::new("repo/model", "0123456789012345678901234567890123456789")
        .with_base_url(format!("http://{address}"));
    let response =
        vbuf_ml::hf_import::RepositoryTransport::range(&mut transport, "model.safetensors", 0, 4)
            .unwrap();
    assert_eq!(response.status, 206);
    assert_eq!(response.body, b"abcd");
    server.join().unwrap();
}
