use memmap2::Mmap;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use vbuf_core::v06::{parse_v06, V06Physical, V06Semantic};
use vbuf_ml::bootstrap::{encode_payload as encode_bootstrap, BootstrapEntry, BOOTSTRAP_KEY_ID};
use vbuf_ml::metadata::MetadataEntry;
use vbuf_ml::tensor_directory::{encode_payload as encode_directory, TensorEntry};
use vbuf_ml::tokenizer::{encode_payload as encode_tokenizer, TokenizerEntry, TokenizerKind};
use vbuf_ml::{Bootstrap, MetadataValue, ModelMetadata, ModelMetadataKey, PreTokenizer, RegionRole, TensorDirectory, TensorRepresentation, TokenizerMetadata, TokenizerModel};
use vbuf_ml::{LayoutClass, PlacementRequest};

const PLAN_MAGIC: &[u8; 8] = b"VBUF20PL";
const PLAN_VERSION: u32 = 1;
const TENSOR_KEY: u16 = 0x0200;
const DIRECTORY_KEY: u16 = 0x0201;
const METADATA_KEY: u16 = 0x0202;
const TOKENIZER_KEY: u16 = 0x0203;
const DATA_BASE: u16 = 0x0300;

#[derive(Debug)]
struct Meta { key: u16, required: bool, kind: u8, bytes: Vec<u8> }
#[derive(Debug)]
struct Tensor { name: String, dims: Vec<u64>, repr: TensorRepresentation, offset: u64, bytes: u64, order: u32 }
#[derive(Debug)]
struct Plan { base_shift: u8, source_size: u64, source_hash: [u8; 32], metadata: Vec<Meta>, text: Vec<u8>, offsets: Vec<u64>, types: Vec<u32>, left: Vec<u32>, right: Vec<u32>, add_bos: bool, specials: Vec<(u8, u64)>, chat: Vec<u8>, tensors: Vec<Tensor> }

fn take<'a>(bytes: &'a [u8], cursor: &mut usize, count: usize) -> Result<&'a [u8], String> { let end = cursor.checked_add(count).ok_or("plan cursor overflow")?; let result = bytes.get(*cursor..end).ok_or("truncated conversion plan")?; *cursor = end; Ok(result) }
fn u8v(bytes: &[u8], c: &mut usize) -> Result<u8, String> { Ok(take(bytes, c, 1)?[0]) }
fn u16v(bytes: &[u8], c: &mut usize) -> Result<u16, String> { Ok(u16::from_le_bytes(take(bytes, c, 2)?.try_into().unwrap())) }
fn u32v(bytes: &[u8], c: &mut usize) -> Result<u32, String> { Ok(u32::from_le_bytes(take(bytes, c, 4)?.try_into().unwrap())) }
fn u64v(bytes: &[u8], c: &mut usize) -> Result<u64, String> { Ok(u64::from_le_bytes(take(bytes, c, 8)?.try_into().unwrap())) }
fn blob(bytes: &[u8], c: &mut usize) -> Result<Vec<u8>, String> { let len = usize::try_from(u64v(bytes, c)?).map_err(|_| "plan blob exceeds host range")?; Ok(take(bytes, c, len)?.to_vec()) }
fn text(bytes: &[u8], c: &mut usize) -> Result<String, String> { String::from_utf8(blob(bytes, c)?).map_err(|_| "plan text is not UTF-8".into()) }

fn parse_plan(bytes: &[u8]) -> Result<Plan, String> {
    let mut c = 0; if take(bytes, &mut c, 8)? != PLAN_MAGIC { return Err("conversion plan magic mismatch".into()); }
    if u32v(bytes, &mut c)? != PLAN_VERSION { return Err("unsupported conversion plan version".into()); }
    let base_shift = u8v(bytes, &mut c)?; if u8v(bytes, &mut c)? != 0 || u16v(bytes, &mut c)? != 0 { return Err("conversion plan flags are nonzero".into()); }
    let source_size = u64v(bytes, &mut c)?; let source_hash: [u8; 32] = take(bytes, &mut c, 32)?.try_into().unwrap();
    let mut metadata = Vec::new();
    for _ in 0..usize::from(u16v(bytes, &mut c)?) {
        metadata.push(Meta { key: u16v(bytes, &mut c)?, required: u8v(bytes, &mut c)? != 0, kind: u8v(bytes, &mut c)?, bytes: blob(bytes, &mut c)? });
    }
    let text_pool = blob(bytes, &mut c)?; let offset_count = usize::try_from(u64v(bytes, &mut c)?).map_err(|_| "offset count exceeds host range")?; let mut offsets = Vec::with_capacity(offset_count); for _ in 0..offset_count { offsets.push(u64v(bytes, &mut c)?); }
    let type_count = usize::try_from(u64v(bytes, &mut c)?).map_err(|_| "type count exceeds host range")?; let mut types = Vec::with_capacity(type_count); for _ in 0..type_count { types.push(u32v(bytes, &mut c)?); }
    let merge_count = usize::try_from(u64v(bytes, &mut c)?).map_err(|_| "merge count exceeds host range")?; let mut left = Vec::with_capacity(merge_count); for _ in 0..merge_count { left.push(u32v(bytes, &mut c)?); } let mut right = Vec::with_capacity(merge_count); for _ in 0..merge_count { right.push(u32v(bytes, &mut c)?); }
    let add_bos = u8v(bytes, &mut c)? != 0; let special_count = usize::from(u8v(bytes, &mut c)?); let mut specials = Vec::new(); for _ in 0..special_count { specials.push((u8v(bytes, &mut c)?, u64v(bytes, &mut c)?)); }
    let chat = blob(bytes, &mut c)?; let tensor_count = usize::try_from(u32v(bytes, &mut c)?).map_err(|_| "tensor count exceeds host range")?; let mut tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count { let name = text(bytes, &mut c)?; let rank = usize::from(u8v(bytes, &mut c)?); let mut dims = Vec::with_capacity(rank); for _ in 0..rank { dims.push(u64v(bytes, &mut c)?); } let repr = match u8v(bytes, &mut c)? { 0 => TensorRepresentation::CanonicalPrimitive, 1 => TensorRepresentation::Bf16, 2 => TensorRepresentation::GgmlQ8_0, _ => return Err("unknown tensor representation in plan".into()) }; tensors.push(Tensor { name, dims, repr, offset: u64v(bytes, &mut c)?, bytes: u64v(bytes, &mut c)?, order: u32v(bytes, &mut c)? }); }
    if c != bytes.len() { return Err("conversion plan has trailing bytes".into()); }
    Ok(Plan { base_shift, source_size, source_hash, metadata, text: text_pool, offsets, types, left, right, add_bos, specials, chat, tensors })
}

type ControlSpec = (LayoutClass, u64, u16, V06Semantic, V06Physical, u16, u64, usize);
type RequestSpec = (LayoutClass, u64, u16, V06Semantic, V06Physical, u16, u64, usize, Option<(u64, u64)>);

#[allow(clippy::too_many_arguments)]
fn req<'a>(class: LayoutClass, order: u64, key: u16, semantic: V06Semantic, physical: V06Physical, width: u16, count: u64, payload: &'a [u8], base_shift: u8) -> PlacementRequest<'a> { PlacementRequest { class, order, key_id: key, semantic, physical, bit_width: width, count, payload_alignment: 1u64 << base_shift, payload } }
fn metadata_payload(plan: &Plan, controls: &mut Vec<Vec<u8>>, requests: &mut Vec<ControlSpec>) -> Result<Vec<u8>, String> {
    let mut entries = Vec::new();
    for meta in &plan.metadata { let data_key = 0x0400u16.checked_add(meta.key).ok_or("metadata key overflow")?; let index = controls.len(); controls.push(meta.bytes.clone()); let (semantic, width) = match meta.kind { 1 => (V06Semantic::Opaque, 8), 2 => (V06Semantic::Unsigned, 64), 3 => (V06Semantic::Float, 64), _ => return Err("invalid metadata kind".into()) }; let count = if semantic == V06Semantic::Opaque { controls[index].len() as u64 } else { 1 }; requests.push((LayoutClass::ModelMetadata, 100 + u64::from(meta.key), data_key, semantic, V06Physical::Array, width, count, index)); entries.push(MetadataEntry::new(meta.key, meta.required, data_key, 0)); }
    vbuf_ml::encode_metadata_payload(&entries).map_err(|e| e.to_string())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect(); let value = |name: &str| args.windows(2).find(|pair| pair[0] == name).map(|pair| PathBuf::from(&pair[1])).ok_or_else(|| format!("missing {name}"));
    let source_path = value("--source")?; let plan_path = value("--plan")?; let target_path = value("--target")?; let evidence = value("--evidence")?;
    let plan_bytes = std::fs::read(&plan_path).map_err(|e| e.to_string())?; let plan = parse_plan(&plan_bytes)?; let source_file = File::open(&source_path).map_err(|e| e.to_string())?; let source = unsafe { Mmap::map(&source_file).map_err(|e| e.to_string())? };
    if u64::try_from(source.len()).map_err(|_| "source size exceeds u64")? != plan.source_size { return Err("source size differs from plan".into()); }
    let mut digest = Sha256::new(); digest.update(&source[..]); if digest.finalize().as_slice() != plan.source_hash { return Err("source hash differs from plan".into()); }
    if plan.offsets.len() != plan.types.len().saturating_add(1) { return Err("tokenizer array lengths are inconsistent".into()); }

    let mut controls: Vec<Vec<u8>> = Vec::new();
    let metadata_index = controls.len(); let metadata_payload = metadata_payload(&plan, &mut controls, &mut Vec::new())?; controls.push(metadata_payload); let _ = metadata_index;
    // Rebuild metadata requests with stable control indexes.
    let mut requests: Vec<RequestSpec> = Vec::new();
    for meta in &plan.metadata { let data_key = 0x0400 + meta.key; let index = controls.iter().position(|payload| payload.as_slice() == meta.bytes.as_slice()).ok_or("metadata payload lookup failed")?; let (semantic, width) = match meta.kind { 1 => (V06Semantic::Opaque, 8), 2 => (V06Semantic::Unsigned, 64), 3 => (V06Semantic::Float, 64), _ => return Err("metadata kind invalid".into()) }; let count = if semantic == V06Semantic::Opaque { meta.bytes.len() as u64 } else { 1 }; let physical = if semantic == V06Semantic::Opaque { V06Physical::Array } else { V06Physical::Scalar }; requests.push((LayoutClass::ModelMetadata, 100 + u64::from(meta.key), data_key, semantic, physical, width, count, index, None)); }
    let mut md_entries = Vec::new(); for meta in &plan.metadata { md_entries.push(MetadataEntry::new(meta.key, meta.required, 0x0400 + meta.key, 0)); }
    let md_control = vbuf_ml::encode_metadata_payload(&md_entries).map_err(|e| e.to_string())?; let md_control_index = controls.len(); controls.push(md_control); requests.push((LayoutClass::ModelMetadata, 1, METADATA_KEY, V06Semantic::Opaque, V06Physical::Array, 8, controls[md_control_index].len() as u64, md_control_index, None));
    let mut dir_entries = Vec::new();
    for tensor in &plan.tensors { dir_entries.push(TensorEntry { name: tensor.name.clone(), dimensions: tensor.dims.clone(), representation: tensor.repr, key_id: TENSOR_KEY, occurrence: u16::try_from(tensor.order).map_err(|_| "tensor occurrence exceeds u16")? }); } let dir_control = encode_directory(&dir_entries).map_err(|e| e.to_string())?; let dir_index = controls.len(); controls.push(dir_control); requests.push((LayoutClass::TensorDirectory, 2, DIRECTORY_KEY, V06Semantic::Opaque, V06Physical::Array, 8, controls[dir_index].len() as u64, dir_index, None));
    let mut tok_entries = vec![TokenizerEntry::new(1, true, DATA_BASE, 0), TokenizerEntry::new(2, true, DATA_BASE+1, 0), TokenizerEntry::new(4, false, DATA_BASE+2, 0), TokenizerEntry::new(9, true, DATA_BASE+3, 0), TokenizerEntry::new(10, true, DATA_BASE+4, 0), TokenizerEntry::new(11, true, DATA_BASE+5, 0), TokenizerEntry::new(12, true, DATA_BASE+6, 0), TokenizerEntry::new(13, true, DATA_BASE+7, 0)];
    let text_i=controls.len(); controls.push(plan.text.clone()); requests.push((LayoutClass::TokenizerPayload,200,DATA_BASE,V06Semantic::Opaque,V06Physical::Array,8,controls[text_i].len() as u64,text_i,None));
    let offsets_bytes: Vec<u8> = plan.offsets.iter().flat_map(|v| v.to_le_bytes()).collect(); let offsets_i=controls.len(); controls.push(offsets_bytes); requests.push((LayoutClass::TokenizerPayload,201,DATA_BASE+1,V06Semantic::Unsigned,V06Physical::Array,64,plan.offsets.len() as u64,offsets_i,None));
    let types_bytes: Vec<u8> = plan.types.iter().flat_map(|v| v.to_le_bytes()).collect(); let types_i=controls.len(); controls.push(types_bytes); requests.push((LayoutClass::TokenizerPayload,202,DATA_BASE+2,V06Semantic::Unsigned,V06Physical::Array,32,plan.types.len() as u64,types_i,None));
    let left_bytes: Vec<u8> = plan.left.iter().flat_map(|v| v.to_le_bytes()).collect(); let left_i=controls.len(); controls.push(left_bytes); requests.push((LayoutClass::TokenizerPayload,203,DATA_BASE+3,V06Semantic::Unsigned,V06Physical::Array,32,plan.left.len() as u64,left_i,None));
    let right_bytes: Vec<u8> = plan.right.iter().flat_map(|v| v.to_le_bytes()).collect(); let right_i=controls.len(); controls.push(right_bytes); requests.push((LayoutClass::TokenizerPayload,204,DATA_BASE+4,V06Semantic::Unsigned,V06Physical::Array,32,plan.right.len() as u64,right_i,None));
    for (order,key,value) in [(205,DATA_BASE+5,1u8),(206,DATA_BASE+6,1u8),(207,DATA_BASE+7,u8::from(plan.add_bos))] { let i=controls.len(); controls.push(vec![value]); requests.push((LayoutClass::TokenizerPayload,order,key,V06Semantic::Unsigned,V06Physical::Scalar,8,1,i,None)); }
    let chat_i=controls.len(); controls.push(plan.chat.clone()); requests.push((LayoutClass::TokenizerPayload,208,DATA_BASE+8,V06Semantic::Opaque,V06Physical::Array,8,controls[chat_i].len() as u64,chat_i,None)); tok_entries.push(TokenizerEntry::new(14,false,DATA_BASE+8,0));
    for (index,(role,value)) in plan.specials.iter().enumerate() { let key=DATA_BASE+16+u16::from(*role); let i=controls.len(); controls.push(value.to_le_bytes().to_vec()); requests.push((LayoutClass::TokenizerPayload,220+u64::from(*role),key,V06Semantic::Unsigned,V06Physical::Scalar,64,1,i,None)); tok_entries.push(TokenizerEntry::new(u16::from(*role),false,key,0)); let _=index; }
    let tok_control = encode_tokenizer(TokenizerKind::Gpt2BpeQwen2,&tok_entries).map_err(|e| e.to_string())?; let tok_i=controls.len(); controls.push(tok_control); requests.push((LayoutClass::TokenizerControl,3,TOKENIZER_KEY,V06Semantic::Opaque,V06Physical::Array,8,controls[tok_i].len() as u64,tok_i,None));
    let bootstrap = encode_bootstrap(&[BootstrapEntry::new(RegionRole::TensorDirectory as u16,true,DIRECTORY_KEY,0),BootstrapEntry::new(RegionRole::ModelMetadata as u16,true,METADATA_KEY,0),BootstrapEntry::new(RegionRole::TokenizerMetadata as u16,false,TOKENIZER_KEY,0)]).map_err(|e| e.to_string())?; let boot_i=controls.len(); controls.push(bootstrap); requests.push((LayoutClass::Bootstrap,0,BOOTSTRAP_KEY_ID,V06Semantic::Opaque,V06Physical::Array,8,controls[boot_i].len() as u64,boot_i,None));
    for tensor in &plan.tensors {
        let (semantic, width, count) = match tensor.repr {
            TensorRepresentation::CanonicalPrimitive => (V06Semantic::Float, 32, tensor.bytes / 4),
            TensorRepresentation::Bf16 | TensorRepresentation::GgmlQ8_0 => (V06Semantic::Opaque, 8, tensor.bytes),
        };
        requests.push((LayoutClass::TensorPayload, 1_000_000 + u64::from(tensor.order), TENSOR_KEY, semantic, V06Physical::Array, width, count, controls.len(), Some((tensor.offset, tensor.bytes))));
    }
    let mut placements = Vec::new();
    for (class, order, key, semantic, physical, width, count, index, source_range) in &requests {
        let payload = if let Some((start, len)) = source_range {
            let end = start.checked_add(*len).ok_or("source range overflow")?;
            source.get(usize::try_from(*start).map_err(|_| "source offset host overflow")?..usize::try_from(end).map_err(|_| "source end host overflow")?).ok_or("source range outside file")?
        } else { &controls[*index] };
        placements.push(req(*class, *order, *key, *semantic, *physical, *width, *count, payload, plan.base_shift));
    }
    let out = OpenOptions::new().write(true).create_new(true).open(&target_path).map_err(|e| e.to_string())?; let out = vbuf_ml::layout::write_known_size(out,plan.base_shift,&placements).map_err(|e| e.to_string())?; drop(out);
    validate_target(&target_path,&source,&plan,&evidence)
}

fn validate_target(path: &Path, source: &[u8], plan: &Plan, evidence: &Path) -> Result<(), String> {
    let target_file = File::open(path).map_err(|e| e.to_string())?; let bytes = unsafe { Mmap::map(&target_file).map_err(|e| e.to_string())? }; let validated = parse_v06(&bytes).map_err(|e| e.to_string())?; let bootstrap=Bootstrap::discover(&validated).map_err(|e| e.to_string())?; let metadata=ModelMetadata::parse(&validated,&bootstrap).map_err(|e| e.to_string())?; let directory=TensorDirectory::parse(&validated,&bootstrap).map_err(|e| e.to_string())?; let tokenizer=TokenizerMetadata::parse(&validated,&bootstrap).map_err(|e| e.to_string())?; if directory.tensors().len()!=plan.tensors.len() || tokenizer.token_count()!=u64::try_from(plan.types.len()).unwrap() || tokenizer.merge_count()!=u64::try_from(plan.left.len()).unwrap() { return Err("target semantic counts differ from plan".into()); }
    for meta in &plan.metadata { let key=ModelMetadataKey::from_id(meta.key).ok_or("unknown target metadata key")?; let actual=metadata.get(key).ok_or("target metadata key is missing")?; match (meta.kind, &actual.value) { (1, MetadataValue::Text(value)) if value.as_bytes()==meta.bytes.as_slice() => {}, (2, MetadataValue::Unsigned(value)) if value.to_le_bytes()==meta.bytes.as_slice() => {}, (3, MetadataValue::Float(value)) if value.to_le_bytes()==meta.bytes.as_slice() => {}, _ => return Err(format!("target metadata mismatch for key {}",meta.key)), } }
    if tokenizer.model()!=Some(TokenizerModel::Gpt2Bpe) || tokenizer.pre_tokenizer()!=Some(PreTokenizer::Qwen2) { return Err("target tokenizer identities differ from plan".into()); }
    for index in 0..plan.types.len() { let start=usize::try_from(plan.offsets[index]).map_err(|_| "token offset overflow")?; let end=usize::try_from(plan.offsets[index+1]).map_err(|_| "token offset overflow")?; let expected=plan.text.get(start..end).ok_or("token text range outside pool")?; if tokenizer.token_text(index as u64).map(str::as_bytes)!=Some(expected) { return Err(format!("token text mismatch at {}",index)); } }
    for (role,value) in &plan.specials { let special=match *role { 5=>vbuf_ml::SpecialToken::Bos, 6=>vbuf_ml::SpecialToken::Eos, 7=>vbuf_ml::SpecialToken::Unk, 8=>vbuf_ml::SpecialToken::Pad, _=>return Err("unknown special role".into()) }; if !tokenizer.specials().contains(&(special,*value)) { return Err(format!("special token mismatch for role {}",role)); } }
    std::fs::create_dir_all(evidence).map_err(|e| e.to_string())?; let mut csv=String::from("tensor_name,source_representation,target_representation,source_payload_bytes,target_payload_bytes,source_sha256,target_sha256,match\n");
    for tensor in &plan.tensors { let target=directory.get(&tensor.name).ok_or_else(|| format!("target tensor missing: {}",tensor.name))?; if target.dimensions!=tensor.dims || target.representation!=tensor.repr || target.occurrence!=u16::try_from(tensor.order).unwrap() { return Err(format!("target tensor descriptor mismatch: {}",tensor.name)); } let start=usize::try_from(tensor.offset).map_err(|_| "source offset overflow")?; let end=start.checked_add(usize::try_from(tensor.bytes).map_err(|_| "source length overflow")?).ok_or("source end overflow")?; let src=source.get(start..end).ok_or("source payload outside file")?; let dst=target.range.bytes(); let sh=hex(&digest(src)); let dh=hex(&digest(dst)); let equal=src==dst; csv.push_str(&format!("{},{},{},{},{},{},{},{}\n",tensor.name,vbuf_ml::representation_name(tensor.repr),vbuf_ml::representation_name(target.representation),src.len(),dst.len(),sh,dh,equal)); if !equal { return Err(format!("payload digest mismatch: {}",tensor.name)); } }
    std::fs::write(evidence.join("tensor-payload-parity.csv"),csv).map_err(|e| e.to_string())?; let mut tok=String::from("semantic,count,match\n"); for i in 0..plan.types.len() { if tokenizer.token_type(i as u64)!=Some(u64::from(plan.types[i])) { return Err("token type mismatch".into()); } } for i in 0..plan.left.len() { if tokenizer.merge_pair(i as u64)!=Some((u64::from(plan.left[i]),u64::from(plan.right[i]))) { return Err("merge mismatch".into()); } } if tokenizer.add_bos()!=Some(plan.add_bos) || tokenizer.chat_template().map(str::as_bytes)!=Some(plan.chat.as_slice()) { return Err("tokenizer control mismatch".into()); } tok.push_str(&format!("vocabulary,{},true\n",plan.types.len())); tok.push_str(&format!("token_types,{},true\n",plan.types.len())); tok.push_str(&format!("merges,{},true\n",plan.left.len())); tok.push_str("model_identity,1,true\npre_tokenizer_identity,1,true\nadd_bos,1,true\nchat_template,1,true\n"); std::fs::write(evidence.join("tokenizer-parity.csv"),tok).map_err(|e| e.to_string())?;
    let mut layout=String::from("target_order,tensor_name,block_index,occurrence,payload_start,payload_end\n"); for tensor in &plan.tensors { let target=directory.get(&tensor.name).unwrap(); let block=&validated.blocks()[target.block_index]; layout.push_str(&format!("{},{},{},{},{},{}\n",tensor.order,tensor.name,target.block_index,target.occurrence,block.payload_start,block.payload_end)); } std::fs::write(evidence.join("layout-parity.csv"),layout).map_err(|e| e.to_string())?; Ok(())
}
fn digest(bytes: &[u8])->[u8;32] { let mut h=Sha256::new(); h.update(bytes); h.finalize().into() }
fn hex(bytes: &[u8])->String { bytes.iter().map(|b|format!("{b:02x}")).collect() }
