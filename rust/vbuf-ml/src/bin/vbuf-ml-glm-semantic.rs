//! Build a small semantic sidecar for a qualified GLM Safetensors import.
//!
//! The sidecar contains no model payload. Tensor blocks are zero-length
//! placeholders bound to the already-qualified vBuf artifact through the
//! persistent source profile.

use memmap2::Mmap;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Cursor;
use std::path::PathBuf;
use vbuf_core::v06::{V06Physical, V06Semantic, parse_v06};
use vbuf_core::writer::{BlockOptions, VBufV06Writer};
use vbuf_ml::bootstrap::{
    BOOTSTRAP_KEY_ID, Bootstrap, BootstrapEntry, encode_payload as encode_bootstrap,
};
use vbuf_ml::metadata::ModelMetadata;
use vbuf_ml::moe::{MoeEntry, MoeParameters, encode_payload as encode_moe};
use vbuf_ml::nested::{NestedEntry, encode_payload as encode_nested};
use vbuf_ml::source::{
    SourceDescriptor, SourceHash, SourceId, SourceLocator, SourceRegistry, TensorRef,
    encode_profile,
};
use vbuf_ml::tokenizer::{
    PreTokenizer, TokenizerEntry, TokenizerKind, TokenizerMetadata, TokenizerModel, TokenizerRole,
    encode_payload as encode_tokenizer,
};
use vbuf_ml::{
    BorrowedModel, BorrowedModelView, Gpt2ByteLevelTokenizer, LayoutClass, PlacementRequest,
    RegionRole, TensorDirectory, write_known_size,
};

const TOKENIZER_KEY: u16 = 0x030f;
const TOKEN_DATA_BASE: u16 = 0x0300;
const NESTED_DATA_KEY: u16 = 0x0310;
const NESTED_DIRECTORY_KEY: u16 = 0x0311;
const MOE_DIRECTORY_KEY: u16 = 0x0312;
const SOURCE_METADATA_KEY: u16 = 0xeffe;

#[derive(Debug)]
struct TokenizerData {
    text: Vec<u8>,
    offsets: Vec<u64>,
    left: Vec<u32>,
    right: Vec<u32>,
    flags: Vec<u8>,
    special_ids: Vec<u32>,
    eos: u32,
    pad: u32,
}

fn json_string(value: &Value, path: &str) -> Result<String, String> {
    value
        .get(path)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("missing string field {path}"))
}

fn parse_tokenizer(bytes: &[u8], config_bytes: &[u8]) -> Result<TokenizerData, String> {
    let root: Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    let config: Value = serde_json::from_slice(config_bytes).map_err(|e| e.to_string())?;
    let vocab = root
        .pointer("/model/vocab")
        .and_then(Value::as_object)
        .ok_or("tokenizer model vocab is absent")?;
    let max_id = vocab
        .values()
        .filter_map(Value::as_u64)
        .max()
        .ok_or("tokenizer vocabulary is empty")?;
    let added = root
        .get("added_tokens")
        .and_then(Value::as_array)
        .ok_or("tokenizer added_tokens is absent")?;
    let max_added = added
        .iter()
        .filter_map(|entry| entry.get("id").and_then(Value::as_u64))
        .max()
        .unwrap_or(0);
    let token_count = usize::try_from(max_id.max(max_added) + 1)
        .map_err(|_| "tokenizer vocabulary exceeds host range")?;
    let mut tokens: Vec<Option<String>> = vec![None; token_count];
    for (text, id) in vocab {
        let id = usize::try_from(id.as_u64().ok_or("token ID is invalid")?)
            .map_err(|_| "token ID exceeds host range")?;
        if tokens[id].replace(text.clone()).is_some() {
            return Err("duplicate tokenizer token ID".into());
        }
    }
    let mut flags = vec![0u8; token_count];
    let mut special_ids = Vec::new();
    for entry in added {
        let id = usize::try_from(
            entry
                .get("id")
                .and_then(Value::as_u64)
                .ok_or("added tokenizer token ID is missing")?,
        )
        .map_err(|_| "added tokenizer token ID exceeds host range")?;
        let content = entry
            .get("content")
            .and_then(Value::as_str)
            .ok_or("added tokenizer token content is missing")?;
        if let Some(previous) = tokens[id].as_ref() {
            if previous != content {
                return Err("added tokenizer token disagrees with vocabulary".into());
            }
        } else {
            tokens[id] = Some(content.to_owned());
        }
        flags[id] |= 1;
        if entry.get("special").and_then(Value::as_bool) == Some(true) {
            flags[id] |= 2;
            special_ids.push(u32::try_from(id).map_err(|_| "special token ID is too large")?);
        }
    }
    if tokens.iter().any(Option::is_none) {
        return Err("tokenizer token IDs are not dense".into());
    }
    special_ids.sort_unstable();
    special_ids.dedup();
    let mut text = Vec::new();
    let mut offsets = Vec::with_capacity(token_count + 1);
    offsets.push(0);
    for token in tokens {
        let token = token.unwrap();
        if token.as_bytes().contains(&0) {
            return Err("tokenizer token contains NUL".into());
        }
        text.extend_from_slice(token.as_bytes());
        offsets.push(u64::try_from(text.len()).map_err(|_| "token text exceeds u64")?);
    }
    let mut ids = HashMap::with_capacity(vocab.len() + added.len());
    for (index, start) in offsets.windows(2).enumerate() {
        ids.insert(
            std::str::from_utf8(
                &text[usize::try_from(start[0]).unwrap()..usize::try_from(start[1]).unwrap()],
            )
            .unwrap()
            .to_owned(),
            u32::try_from(index).map_err(|_| "token ID exceeds u32")?,
        );
    }
    let merges = root
        .pointer("/model/merges")
        .and_then(Value::as_array)
        .ok_or("tokenizer merge table is absent")?;
    let mut left = Vec::with_capacity(merges.len());
    let mut right = Vec::with_capacity(merges.len());
    for merge in merges {
        let pair = merge.as_array().ok_or("tokenizer merge is not a pair")?;
        if pair.len() != 2 {
            return Err("tokenizer merge pair does not contain two tokens".into());
        }
        let l = pair[0].as_str().ok_or("tokenizer merge-left is not text")?;
        let r = pair[1]
            .as_str()
            .ok_or("tokenizer merge-right is not text")?;
        left.push(
            *ids.get(l)
                .ok_or("tokenizer merge-left is not in vocabulary")?,
        );
        right.push(
            *ids.get(r)
                .ok_or("tokenizer merge-right is not in vocabulary")?,
        );
    }
    let eos_text = json_string(&config, "eos_token")?;
    let pad_text = json_string(&config, "pad_token")?;
    let eos = *ids.get(&eos_text).ok_or("configured EOS token is absent")?;
    let pad = *ids.get(&pad_text).ok_or("configured PAD token is absent")?;
    Ok(TokenizerData {
        text,
        offsets,
        left,
        right,
        flags,
        special_ids,
        eos,
        pad,
    })
}

fn layer_index(name: &str) -> Option<u32> {
    let parts: Vec<_> = name.split('.').collect();
    (parts.len() > 2 && parts[0] == "model" && parts[1] == "layers")
        .then(|| parts[2].parse().ok())
        .flatten()
}

fn moe_role(name: &str) -> Option<(u32, u32, u16, Option<String>)> {
    let parts: Vec<_> = name.split('.').collect();
    let layer = layer_index(name)?;
    if parts.len() == 8 && parts[3] == "mlp" && parts[4] == "experts" && parts[7] == "weight" {
        let expert = parts[5].parse().ok()?;
        let role = match parts[6] {
            "gate_proj" => 1,
            "up_proj" => 2,
            "down_proj" => 3,
            _ => return None,
        };
        return Some((layer, expert, role, Some(format!("{name}_scale"))));
    }
    if parts.len() == 7 && parts[3] == "mlp" && parts[4] == "shared_experts" && parts[6] == "weight"
    {
        let role = match parts[5] {
            "gate_proj" => 11,
            "up_proj" => 12,
            "down_proj" => 13,
            _ => return None,
        };
        return Some((layer, 0, role, Some(format!("{name}_scale"))));
    }
    if parts.len() == 6 && parts[3] == "mlp" && parts[4] == "gate" && parts[5] == "weight" {
        return Some((layer, 0, 10, None));
    }
    if parts.len() == 6
        && parts[3] == "mlp"
        && parts[4] == "gate"
        && parts[5] == "e_score_correction_bias"
    {
        return Some((layer, 0, 14, None));
    }
    None
}

fn build_moe(directory: &TensorDirectory<'_>) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>), String> {
    let mut entries = Vec::new();
    let mut nested_payload = Vec::new();
    let mut nested_entries = Vec::new();
    let mut layer_count = 0u32;
    for tensor in directory.tensors() {
        let Some((layer, expert, role, scale_name)) = moe_role(&tensor.name) else {
            continue;
        };
        layer_count = layer_count.max(layer + 1);
        let scale = scale_name
            .as_deref()
            .and_then(|name| directory.get(name))
            .map(|tensor| tensor.occurrence);
        let child_name = format!("glm.l{layer:03}.e{expert:03}.r{role:02}");
        let marker = [
            b'G',
            b'L',
            b'M',
            b'X',
            role as u8,
            (tensor.occurrence & 0xff) as u8,
            scale.map_or(0xff, |value| (value & 0xff) as u8),
        ];
        let child = write_known_size(
            Cursor::new(Vec::new()),
            4,
            &[PlacementRequest {
                class: LayoutClass::Auxiliary,
                order: 0,
                key_id: 0x7000,
                semantic: V06Semantic::Opaque,
                physical: V06Physical::Array,
                bit_width: 8,
                count: marker.len() as u64,
                payload_alignment: 16,
                payload: &marker,
            }],
        )
        .map_err(|e| e.to_string())?
        .into_inner();
        let child_offset = nested_payload.len() as u64;
        let child_length = child.len() as u64;
        nested_payload.extend_from_slice(&child);
        nested_entries.push(NestedEntry {
            name: child_name.clone(),
            key_id: NESTED_DATA_KEY,
            occurrence: 0,
            child_offset,
            child_length,
        });
        entries.push(MoeEntry {
            layer_index: layer,
            expert_index: expert,
            role,
            child_name,
            tensor_ordinal: Some(tensor.occurrence),
            scale_ordinal: scale,
        });
    }
    if entries.is_empty() || layer_count == 0 {
        return Err("GLM MoE tensor families are absent".into());
    }
    let parameters = MoeParameters {
        expert_count: 128,
        active_expert_count: 8,
        layer_count,
        shared_experts: true,
        shared_expert_count: 1,
        normalize_topk_prob: true,
        routing_group_count: 1,
        routing_topk_group_count: 1,
        routed_scaling_factor_bits: 1.0f32.to_bits(),
    };
    let nested = encode_nested(&nested_entries).map_err(|e| e.to_string())?;
    let moe = encode_moe(parameters, &entries).map_err(|e| e.to_string())?;
    Ok((nested_payload, nested, moe))
}

fn write_tokenizer(
    writer: &mut VBufV06Writer<Cursor<Vec<u8>>>,
    data: &TokenizerData,
) -> Result<Vec<u8>, String> {
    let entries = vec![
        TokenizerEntry::new(
            TokenizerRole::TokenTextBytes as u16,
            true,
            TOKEN_DATA_BASE,
            0,
        ),
        TokenizerEntry::new(
            TokenizerRole::TokenOffsets as u16,
            true,
            TOKEN_DATA_BASE + 1,
            0,
        ),
        TokenizerEntry::new(
            TokenizerRole::MergeLeftIds as u16,
            true,
            TOKEN_DATA_BASE + 2,
            0,
        ),
        TokenizerEntry::new(
            TokenizerRole::MergeRightIds as u16,
            true,
            TOKEN_DATA_BASE + 3,
            0,
        ),
        TokenizerEntry::new(
            TokenizerRole::TokenizerModelIdentity as u16,
            true,
            TOKEN_DATA_BASE + 4,
            0,
        ),
        TokenizerEntry::new(
            TokenizerRole::PreTokenizerIdentity as u16,
            true,
            TOKEN_DATA_BASE + 5,
            0,
        ),
        TokenizerEntry::new(TokenizerRole::AddBos as u16, true, TOKEN_DATA_BASE + 6, 0),
        TokenizerEntry::new(
            TokenizerRole::IgnoreMerges as u16,
            true,
            TOKEN_DATA_BASE + 7,
            0,
        ),
        TokenizerEntry::new(TokenizerRole::EosId as u16, false, TOKEN_DATA_BASE + 8, 0),
        TokenizerEntry::new(TokenizerRole::PadId as u16, false, TOKEN_DATA_BASE + 9, 0),
        TokenizerEntry::new(
            TokenizerRole::TokenFlags as u16,
            false,
            TOKEN_DATA_BASE + 10,
            0,
        ),
        TokenizerEntry::new(
            TokenizerRole::SpecialTokenIds as u16,
            false,
            TOKEN_DATA_BASE + 11,
            0,
        ),
    ];
    let offsets: Vec<u8> = data.offsets.iter().flat_map(|v| v.to_le_bytes()).collect();
    let left: Vec<u8> = data.left.iter().flat_map(|v| v.to_le_bytes()).collect();
    let right: Vec<u8> = data.right.iter().flat_map(|v| v.to_le_bytes()).collect();
    let special: Vec<u8> = data
        .special_ids
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    writer
        .write_block(
            BlockOptions::array(TOKEN_DATA_BASE),
            V06Semantic::Opaque,
            8,
            data.text.len() as u64,
            &data.text,
        )
        .map_err(|e| e.to_string())?;
    writer
        .write_block(
            BlockOptions::array(TOKEN_DATA_BASE + 1),
            V06Semantic::Unsigned,
            64,
            data.offsets.len() as u64,
            &offsets,
        )
        .map_err(|e| e.to_string())?;
    writer
        .write_block(
            BlockOptions::array(TOKEN_DATA_BASE + 2),
            V06Semantic::Unsigned,
            32,
            data.left.len() as u64,
            &left,
        )
        .map_err(|e| e.to_string())?;
    writer
        .write_block(
            BlockOptions::array(TOKEN_DATA_BASE + 3),
            V06Semantic::Unsigned,
            32,
            data.right.len() as u64,
            &right,
        )
        .map_err(|e| e.to_string())?;
    for (key, value) in [
        (TOKEN_DATA_BASE + 4, TokenizerModel::Gpt2Bpe as u64),
        (TOKEN_DATA_BASE + 5, PreTokenizer::Gpt2ByteLevel as u64),
    ] {
        writer
            .write_block(
                BlockOptions::scalar(key),
                V06Semantic::Unsigned,
                64,
                1,
                &value.to_le_bytes(),
            )
            .map_err(|e| e.to_string())?;
    }
    for (key, value) in [(TOKEN_DATA_BASE + 6, 0u8), (TOKEN_DATA_BASE + 7, 1u8)] {
        writer
            .write_block(
                BlockOptions::scalar(key),
                V06Semantic::Unsigned,
                8,
                1,
                &[value],
            )
            .map_err(|e| e.to_string())?;
    }
    for (key, value) in [
        (TOKEN_DATA_BASE + 8, data.eos),
        (TOKEN_DATA_BASE + 9, data.pad),
    ] {
        writer
            .write_block(
                BlockOptions::scalar(key),
                V06Semantic::Unsigned,
                32,
                1,
                &value.to_le_bytes(),
            )
            .map_err(|e| e.to_string())?;
    }
    writer
        .write_block(
            BlockOptions::array(TOKEN_DATA_BASE + 10),
            V06Semantic::Unsigned,
            8,
            data.flags.len() as u64,
            &data.flags,
        )
        .map_err(|e| e.to_string())?;
    writer
        .write_block(
            BlockOptions::array(TOKEN_DATA_BASE + 11),
            V06Semantic::Unsigned,
            32,
            data.special_ids.len() as u64,
            &special,
        )
        .map_err(|e| e.to_string())?;
    let control =
        encode_tokenizer(TokenizerKind::Gpt2BpeByteLevel, &entries).map_err(|e| e.to_string())?;
    writer
        .write_block(
            BlockOptions::array(TOKENIZER_KEY),
            V06Semantic::Opaque,
            8,
            control.len() as u64,
            &control,
        )
        .map_err(|e| e.to_string())?;
    Ok(control)
}

fn build(
    source_bytes: &[u8],
    metadata_size: u64,
    locator: &str,
    source_hash: &[u8],
    data: &TokenizerData,
) -> Result<Vec<u8>, String> {
    let validated = parse_v06(source_bytes).map_err(|e| e.to_string())?;
    let bootstrap = Bootstrap::discover(&validated).map_err(|e| e.to_string())?;
    let _metadata = ModelMetadata::parse(&validated, &bootstrap).map_err(|e| e.to_string())?;
    let directory = TensorDirectory::parse(&validated, &bootstrap).map_err(|e| e.to_string())?;
    let tensor_blocks: HashSet<usize> = directory
        .tensors()
        .iter()
        .map(|tensor| tensor.block_index)
        .collect();
    let self_source = SourceDescriptor::self_artifact(metadata_size);
    let authoritative = SourceDescriptor {
        id: SourceId::new(1),
        declared_size: Some(source_bytes.len() as u64),
        locator: if locator.starts_with("http://") || locator.starts_with("https://") {
            SourceLocator::Http(locator.to_owned())
        } else {
            SourceLocator::File(locator.to_owned())
        },
        hashes: vec![SourceHash {
            algorithm: 1,
            value: source_hash.to_vec(),
        }],
    };
    let registry = SourceRegistry::new(vec![self_source, authoritative.clone()])
        .map_err(|e| format!("source registry: {e:?}"))?;
    let bindings: Vec<(u16, u16, TensorRef)> = directory
        .tensors()
        .iter()
        .map(|tensor| {
            TensorRef::new(
                &authoritative,
                tensor.payload.offset(),
                tensor.payload.length(),
            )
            .map(|reference| (tensor.key_id, tensor.occurrence, reference))
            .map_err(|e| e.to_string())
        })
        .collect::<Result<_, _>>()?;
    let source_profile = encode_profile(&registry, &bindings).map_err(|e| e.to_string())?;
    let (nested_payload, nested_directory, moe_directory) = build_moe(&directory)?;
    let mut profile_entries: Vec<BootstrapEntry> = bootstrap
        .regions()
        .iter()
        .filter(|region| {
            !matches!(
                region.role,
                RegionRole::TokenizerMetadata
                    | RegionRole::NestedDirectory
                    | RegionRole::MoeDirectory
            )
        })
        .map(|region| {
            BootstrapEntry::new(
                region.role as u16,
                region.role.is_required(),
                region.key_id,
                region.occurrence,
            )
        })
        .collect();
    profile_entries.extend([
        BootstrapEntry::new(
            RegionRole::TokenizerMetadata as u16,
            false,
            TOKENIZER_KEY,
            0,
        ),
        BootstrapEntry::new(
            RegionRole::NestedDirectory as u16,
            false,
            NESTED_DIRECTORY_KEY,
            0,
        ),
        BootstrapEntry::new(RegionRole::MoeDirectory as u16, false, MOE_DIRECTORY_KEY, 0),
        BootstrapEntry::new(
            RegionRole::SourceMetadata as u16,
            false,
            SOURCE_METADATA_KEY,
            0,
        ),
    ]);
    let bootstrap_payload = encode_bootstrap(&profile_entries).map_err(|e| e.to_string())?;
    let original_bootstrap_index = validated
        .blocks()
        .iter()
        .position(|block| block.key_id == BOOTSTRAP_KEY_ID)
        .ok_or("bootstrap block is absent")?;
    let mut writer =
        VBufV06Writer::new_known_size(Cursor::new(Vec::new()), validated.header().base_shift)
            .map_err(|e| e.to_string())?;
    for (index, block) in validated.blocks().iter().enumerate() {
        if index == original_bootstrap_index {
            writer
                .write_block(
                    BlockOptions::array(BOOTSTRAP_KEY_ID),
                    V06Semantic::Opaque,
                    8,
                    bootstrap_payload.len() as u64,
                    &bootstrap_payload,
                )
                .map_err(|e| e.to_string())?;
        } else if tensor_blocks.contains(&index) {
            writer
                .write_block(
                    BlockOptions::array(block.key_id),
                    block.semantic,
                    block.bit_width,
                    0,
                    &[],
                )
                .map_err(|e| e.to_string())?;
        } else {
            let payload = validated
                .payload_range(index)
                .map_err(|e| e.to_string())?
                .bytes();
            writer
                .write_block(
                    BlockOptions {
                        key_id: block.key_id,
                        physical: block.physical,
                        continuation: false,
                        payload_shift: 0,
                    },
                    block.semantic,
                    block.bit_width,
                    block.count,
                    payload,
                )
                .map_err(|e| e.to_string())?;
        }
    }
    write_tokenizer(&mut writer, data)?;
    writer
        .write_block(
            BlockOptions::array(NESTED_DATA_KEY),
            V06Semantic::Opaque,
            8,
            nested_payload.len() as u64,
            &nested_payload,
        )
        .map_err(|e| e.to_string())?;
    writer
        .write_block(
            BlockOptions::array(NESTED_DIRECTORY_KEY),
            V06Semantic::Opaque,
            8,
            nested_directory.len() as u64,
            &nested_directory,
        )
        .map_err(|e| e.to_string())?;
    writer
        .write_block(
            BlockOptions::array(MOE_DIRECTORY_KEY),
            V06Semantic::Opaque,
            8,
            moe_directory.len() as u64,
            &moe_directory,
        )
        .map_err(|e| e.to_string())?;
    writer
        .write_block(
            BlockOptions::array(SOURCE_METADATA_KEY),
            V06Semantic::Opaque,
            8,
            source_profile.len() as u64,
            &source_profile,
        )
        .map_err(|e| e.to_string())?;
    Ok(writer.finish().map_err(|e| e.to_string())?.into_inner())
}

fn main() -> Result<(), String> {
    let value = |name: &str| {
        std::env::args()
            .collect::<Vec<_>>()
            .windows(2)
            .find(|pair| pair[0] == name)
            .map(|pair| PathBuf::from(&pair[1]))
            .ok_or_else(|| format!("missing {name}"))
    };
    let source_path = value("--source")?;
    let tokenizer_path = value("--tokenizer")?;
    let config_path = value("--config")?;
    let target_path = value("--target")?;
    let hash_text = std::env::args()
        .collect::<Vec<_>>()
        .windows(2)
        .find(|pair| pair[0] == "--source-sha256")
        .map(|pair| pair[1].clone())
        .ok_or("missing --source-sha256")?;
    if hash_text.len() != 64 {
        return Err("source SHA-256 must contain 64 hexadecimal characters".into());
    }
    let source_hash = (0..32)
        .map(|index| {
            u8::from_str_radix(&hash_text[index * 2..index * 2 + 2], 16)
                .map_err(|_| "source SHA-256 is not hexadecimal")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let source_file = File::open(&source_path).map_err(|e| e.to_string())?;
    let mapping = unsafe { Mmap::map(&source_file).map_err(|e| e.to_string())? };
    let tokenizer = parse_tokenizer(
        &std::fs::read(tokenizer_path).map_err(|e| e.to_string())?,
        &std::fs::read(config_path).map_err(|e| e.to_string())?,
    )?;
    let locator = source_path.to_string_lossy();
    let first = build(&mapping, 0, &locator, &source_hash, &tokenizer)?;
    let output = build(
        &mapping,
        first.len() as u64,
        &locator,
        &source_hash,
        &tokenizer,
    )?;
    if first.len() != output.len() {
        return Err("semantic sidecar size is not stable across source-profile rebuild".into());
    }
    std::fs::write(&target_path, &output).map_err(|e| e.to_string())?;
    let target_file = File::open(&target_path).map_err(|e| e.to_string())?;
    let target_mapping = unsafe { Mmap::map(&target_file).map_err(|e| e.to_string())? };
    let validated = parse_v06(&target_mapping).map_err(|e| e.to_string())?;
    let bootstrap = Bootstrap::discover(&validated).map_err(|e| e.to_string())?;
    let profile = vbuf_ml::parse_source_profile(&validated, &bootstrap)
        .map_err(|e| e.to_string())?
        .ok_or("sidecar source profile is absent")?;
    let view = BorrowedModelView::parse_with_sources(&target_mapping, &profile.registry, &[])
        .map_err(|e| e.to_string())?;
    let reopened = BorrowedModel::open_persistent(&target_path).map_err(|e| e.to_string())?;
    let tokenizer = TokenizerMetadata::parse(&validated, &bootstrap).map_err(|e| e.to_string())?;
    let runtime = Gpt2ByteLevelTokenizer::build(&tokenizer).map_err(|e| e.to_string())?;
    let encoded = runtime.encode("Hello world").map_err(|e| e.to_string())?;
    if runtime.decode(&encoded).map_err(|e| e.to_string())? != "Hello world" {
        return Err("GLM tokenizer encode/decode round trip failed".into());
    }
    let added_tokens = (0..tokenizer.token_count())
        .filter(|index| {
            tokenizer
                .token_flags(*index)
                .is_some_and(|flags| flags & 1 != 0)
        })
        .count();
    let moe = view.moe.as_ref().ok_or("GLM MoE directory is absent")?;
    moe.validate_tensor_bindings(&view.directory, 0x0200)
        .map_err(|e| e.to_string())?;
    let role_counts = moe
        .entries()
        .iter()
        .fold([0usize; 15], |mut counts, entry| {
            if usize::from(entry.role) < counts.len() {
                counts[usize::from(entry.role)] += 1;
            }
            counts
        });
    let parameters = moe.parameters();
    if tokenizer.token_count() != 151_365
        || tokenizer.merge_count() == 0
        || tokenizer.special_ids().map_or(0, |ids| ids.len()) != 22
        || added_tokens != 36
        || role_counts[1] + role_counts[2] + role_counts[3] != 17_664
        || role_counts[11] + role_counts[12] + role_counts[13] != 138
        || role_counts[10] != 46
        || role_counts[14] != 46
        || parameters.expert_count != 128
        || parameters.active_expert_count != 8
        || parameters.shared_expert_count != 1
        || !parameters.normalize_topk_prob
        || parameters.routing_group_count != 1
        || parameters.routing_topk_group_count != 1
        || f32::from_bits(parameters.routed_scaling_factor_bits) != 1.0
        || reopened.view().directory.tensors().len() != view.directory.tensors().len()
    {
        return Err(format!(
            "GLM semantic sidecar qualification counts are inconsistent: tokens={} merges={} specials={} added={} moe={} reopened_tensors={} view_tensors={}",
            tokenizer.token_count(),
            tokenizer.merge_count(),
            tokenizer.special_ids().map_or(0, |ids| ids.len()),
            added_tokens,
            view.moe.is_some(),
            reopened.view().directory.tensors().len(),
            view.directory.tensors().len(),
        ));
    }
    println!(
        "sidecar_bytes={} tensors={} tokens={} merges={} moe_entries={}",
        output.len(),
        view.directory.tensors().len(),
        tokenizer.token_count(),
        tokenizer.merge_count(),
        moe.entries().len()
    );
    Ok(())
}
