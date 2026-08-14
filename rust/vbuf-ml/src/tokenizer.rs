use crate::bootstrap::Bootstrap;
use crate::error::{MlError, MlErrorCode};
use crate::region_roles::RegionRole;
use vbuf_core::v06::{V06Physical, V06Semantic, ValidatedV06};
use vbuf_layout::CheckedRange;

pub const TOKENIZER_MAGIC: [u8; 8] = *b"VBTOK\0\0\0";
pub const TOKENIZER_VERSION: u16 = 1;
pub const MAX_TOKENIZER_BYTES: usize = 4096;
pub const MAX_VOCABULARY: u64 = 10_000_000;
pub const MAX_TOKEN_BYTES: u64 = 4096;
pub const MAX_MERGES: u64 = 10_000_000;
pub const MAX_CHAT_TEMPLATE_BYTES: u64 = 1024 * 1024;
const HEADER_BYTES: usize = 20;
const ENTRY_BYTES: usize = 12;
const ROLE_SLOTS: usize = 15;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TokenizerKind {
    VocabularyOnly = 1,
    Gpt2BpeQwen2 = 2,
}

impl TokenizerKind {
    fn from_id(id: u8) -> Result<Self, MlError> {
        match id {
            1 => Ok(Self::VocabularyOnly),
            2 => Ok(Self::Gpt2BpeQwen2),
            _ => Err(MlError::new(MlErrorCode::UnsupportedTokenizerKind, "unsupported tokenizer kind")),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub enum TokenizerRole {
    TokenTextBytes = 1,
    TokenOffsets = 2,
    TokenScores = 3,
    TokenTypes = 4,
    BosId = 5,
    EosId = 6,
    UnkId = 7,
    PadId = 8,
    MergeLeftIds = 9,
    MergeRightIds = 10,
    TokenizerModelIdentity = 11,
    PreTokenizerIdentity = 12,
    AddBos = 13,
    ChatTemplate = 14,
}

impl TokenizerRole {
    fn from_id(id: u16) -> Option<Self> {
        match id { 1 => Some(Self::TokenTextBytes), 2 => Some(Self::TokenOffsets), 3 => Some(Self::TokenScores), 4 => Some(Self::TokenTypes), 5 => Some(Self::BosId), 6 => Some(Self::EosId), 7 => Some(Self::UnkId), 8 => Some(Self::PadId), 9 => Some(Self::MergeLeftIds), 10 => Some(Self::MergeRightIds), 11 => Some(Self::TokenizerModelIdentity), 12 => Some(Self::PreTokenizerIdentity), 13 => Some(Self::AddBos), 14 => Some(Self::ChatTemplate), _ => None }
    }
    const fn is_required(self, kind: TokenizerKind) -> bool {
        matches!(self, Self::TokenTextBytes | Self::TokenOffsets)
            || matches!(kind, TokenizerKind::Gpt2BpeQwen2)
                && matches!(self, Self::MergeLeftIds | Self::MergeRightIds | Self::TokenizerModelIdentity | Self::PreTokenizerIdentity | Self::AddBos)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenizerEntry {
    pub role_id: u16,
    pub required: bool,
    pub generic_key_id: u16,
    pub occurrence: u16,
}

impl TokenizerEntry {
    pub const fn new(role_id: u16, required: bool, generic_key_id: u16, occurrence: u16) -> Self { Self { role_id, required, generic_key_id, occurrence } }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpecialToken {
    Bos,
    Eos,
    Unk,
    Pad,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u64)]
pub enum TokenizerModel {
    Gpt2Bpe = 1,
}

impl TokenizerModel {
    fn from_value(value: u64) -> Result<Self, MlError> {
        match value { 1 => Ok(Self::Gpt2Bpe), _ => Err(MlError::new(MlErrorCode::InvalidTokenizerIdentity, "unsupported tokenizer model identity")) }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u64)]
pub enum PreTokenizer {
    Qwen2 = 1,
}

impl PreTokenizer {
    fn from_value(value: u64) -> Result<Self, MlError> {
        match value { 1 => Ok(Self::Qwen2), _ => Err(MlError::new(MlErrorCode::InvalidTokenizerIdentity, "unsupported pre-tokenizer identity")) }
    }
}

#[derive(Debug)]
pub struct TokenizerMetadata<'a> {
    pub kind: TokenizerKind,
    token_count: u64,
    text_pool: CheckedRange<'a>,
    offsets: CheckedRange<'a>,
    scores: Option<CheckedRange<'a>>,
    types: Option<CheckedRange<'a>>,
    merge_left_ids: Option<CheckedRange<'a>>,
    merge_right_ids: Option<CheckedRange<'a>>,
    model: Option<TokenizerModel>,
    pre_tokenizer: Option<PreTokenizer>,
    add_bos: Option<bool>,
    chat_template: Option<CheckedRange<'a>>,
    specials: Vec<(SpecialToken, u64)>,
}

impl<'a> TokenizerMetadata<'a> {
    pub fn parse(validated: &ValidatedV06<'a>, bootstrap: &Bootstrap<'a>) -> Result<Self, MlError> {
        let region = bootstrap.region(RegionRole::TokenizerMetadata).ok_or_else(|| MlError::new(MlErrorCode::MissingTokenizerReference, "tokenizer metadata role is absent"))?;
        let region_block = validated.blocks().get(region.block_index).ok_or_else(|| MlError::new(MlErrorCode::MissingTokenizerReference, "tokenizer metadata block index is absent"))?;
        if region_block.semantic != V06Semantic::Opaque || region_block.physical != V06Physical::Array || region_block.bit_width != 8 || region_block.continuation { return Err(MlError::new(MlErrorCode::MalformedTokenizerMetadata, "tokenizer metadata must be a non-continuing opaque byte array")); }
        let (kind, entries) = parse_payload(region.range.bytes())?;
        let mut previous = 0u16;
        let mut seen = [false; ROLE_SLOTS];
        let mut resolved: [Option<(usize, CheckedRange<'a>)>; ROLE_SLOTS] = std::array::from_fn(|_| None);
        for (position, entry) in entries.into_iter().enumerate() {
            if position > 0 && entry.role_id <= previous { return Err(MlError::new(if entry.role_id == previous { MlErrorCode::DuplicateTokenizerRole } else { MlErrorCode::MalformedTokenizerMetadata }, "tokenizer roles must be strictly sorted")); }
            previous = entry.role_id;
            let Some(role) = TokenizerRole::from_id(entry.role_id) else { if entry.required { return Err(MlError::new(MlErrorCode::MissingTokenizerReference, "unknown required tokenizer role")); } continue; };
            if entry.required != role.is_required(kind) { return Err(MlError::new(MlErrorCode::MalformedTokenizerMetadata, "tokenizer role requiredness is invalid")); }
            if seen[entry.role_id as usize] { return Err(MlError::new(MlErrorCode::DuplicateTokenizerRole, "tokenizer role occurs more than once")); }
            seen[entry.role_id as usize] = true;
            resolved[entry.role_id as usize] = Some(resolve_value(validated, entry.generic_key_id, entry.occurrence)?);
        }
        let text_pool = resolved[TokenizerRole::TokenTextBytes as usize].take().ok_or_else(|| MlError::new(MlErrorCode::MissingTokenizerReference, "token text bytes are missing"))?;
        let offsets = resolved[TokenizerRole::TokenOffsets as usize].take().ok_or_else(|| MlError::new(MlErrorCode::MissingTokenizerReference, "token offsets are missing"))?;
        validate_text_pool(&validated.blocks()[text_pool.0], text_pool.1.bytes())?;
        let token_count = validate_offsets(&validated.blocks()[offsets.0], offsets.1.bytes(), text_pool.1.bytes())?;
        if token_count > MAX_VOCABULARY { return Err(MlError::new(MlErrorCode::TokenizerArrayLengthMismatch, "token vocabulary exceeds profile maximum")); }
        let scores = optional_array(&mut resolved, TokenizerRole::TokenScores, token_count, validated, true)?;
        let types = optional_array(&mut resolved, TokenizerRole::TokenTypes, token_count, validated, false)?;
        if matches!(kind, TokenizerKind::Gpt2BpeQwen2) && let Some(types) = &types {
            validate_token_types(types, token_count)?;
        }
        let (merge_left_ids, merge_right_ids) = if matches!(kind, TokenizerKind::Gpt2BpeQwen2) {
            let left = required_u32_array(&mut resolved, TokenizerRole::MergeLeftIds, validated, MAX_MERGES)?;
            let right = required_u32_array(&mut resolved, TokenizerRole::MergeRightIds, validated, MAX_MERGES)?;
            if left.1.bytes().len() != right.1.bytes().len() { return Err(MlError::new(MlErrorCode::InvalidMergeTable, "merge arrays have different lengths")); }
            validate_merge_ids(&left.1, &right.1, token_count)?;
            (Some(left.1), Some(right.1))
        } else { (None, None) };
        let model = if matches!(kind, TokenizerKind::Gpt2BpeQwen2) { Some(TokenizerModel::from_value(required_unsigned_scalar(&mut resolved, TokenizerRole::TokenizerModelIdentity, validated)?)?) } else { None };
        let pre_tokenizer = if matches!(kind, TokenizerKind::Gpt2BpeQwen2) { Some(PreTokenizer::from_value(required_unsigned_scalar(&mut resolved, TokenizerRole::PreTokenizerIdentity, validated)?)?) } else { None };
        let add_bos = if matches!(kind, TokenizerKind::Gpt2BpeQwen2) { Some(required_unsigned_scalar(&mut resolved, TokenizerRole::AddBos, validated)? != 0) } else { None };
        let chat_template = optional_utf8_payload(&mut resolved, TokenizerRole::ChatTemplate, validated)?;
        let mut specials = Vec::new();
        for (role, special) in [(TokenizerRole::BosId, SpecialToken::Bos), (TokenizerRole::EosId, SpecialToken::Eos), (TokenizerRole::UnkId, SpecialToken::Unk), (TokenizerRole::PadId, SpecialToken::Pad)] {
            if let Some((index, range)) = resolved[role as usize].take() {
                let value = decode_unsigned_scalar(&validated.blocks()[index], range.bytes())?;
                if value >= token_count { return Err(MlError::new(MlErrorCode::InvalidSpecialTokenId, "special token ID is outside the vocabulary")); }
                specials.push((special, value));
            }
        }
        Ok(Self { kind, token_count, text_pool: text_pool.1, offsets: offsets.1, scores, types, merge_left_ids, merge_right_ids, model, pre_tokenizer, add_bos, chat_template, specials })
    }

    pub fn token_count(&self) -> u64 { self.token_count }
    /// Borrow the validated canonical token text pool without materialization.
    pub fn text_bytes(&self) -> &[u8] { self.text_pool.bytes() }
    /// Borrow the validated little-endian u64 offset array.
    pub fn offset_bytes(&self) -> &[u8] { self.offsets.bytes() }
    /// Borrow optional validated token scores.
    pub fn score_bytes(&self) -> Option<&[u8]> { self.scores.as_ref().map(|range| range.bytes()) }
    /// Borrow optional validated token types.
    pub fn type_bytes(&self) -> Option<&[u8]> { self.types.as_ref().map(|range| range.bytes()) }
    /// Borrow validated little-endian merge-left IDs.
    pub fn merge_left_bytes(&self) -> Option<&[u8]> { self.merge_left_ids.as_ref().map(|range| range.bytes()) }
    /// Borrow validated little-endian merge-right IDs.
    pub fn merge_right_bytes(&self) -> Option<&[u8]> { self.merge_right_ids.as_ref().map(|range| range.bytes()) }
    pub fn specials(&self) -> &[(SpecialToken, u64)] { &self.specials }
    pub fn model(&self) -> Option<TokenizerModel> { self.model }
    pub fn pre_tokenizer(&self) -> Option<PreTokenizer> { self.pre_tokenizer }
    pub fn add_bos(&self) -> Option<bool> { self.add_bos }
    pub fn merge_count(&self) -> u64 { self.merge_left_ids.as_ref().map_or(0, |range| (range.bytes().len() / 4) as u64) }
    pub fn merge_pair(&self, index: u64) -> Option<(u64, u64)> {
        let left = self.merge_left_ids.as_ref()?;
        let right = self.merge_right_ids.as_ref()?;
        let index = usize::try_from(index).ok()?.checked_mul(4)?;
        Some((u64::from(u32::from_le_bytes(left.bytes().get(index..index + 4)?.try_into().ok()?)), u64::from(u32::from_le_bytes(right.bytes().get(index..index + 4)?.try_into().ok()?))))
    }
    pub fn chat_template(&self) -> Option<&str> { std::str::from_utf8(self.chat_template.as_ref()?.bytes()).ok() }
    pub fn token_text(&self, index: u64) -> Option<&str> {
        if index >= self.token_count { return None; }
        let start = usize::try_from(self.offset(index)?).ok()?;
        let end = usize::try_from(self.offset(index + 1)?).ok()?;
        std::str::from_utf8(self.text_pool.bytes().get(start..end)?).ok()
    }
    pub fn score(&self, index: u64) -> Option<f64> {
        let range = self.scores.as_ref()?;
        if index >= self.token_count { return None; }
        let block_width = range.bytes().len().checked_div(usize::try_from(self.token_count).ok()?)?;
        let start = usize::try_from(index).ok()?.checked_mul(block_width)?;
        match block_width { 4 => Some(f32::from_le_bytes(range.bytes().get(start..start + 4)?.try_into().ok()?) as f64), 8 => Some(f64::from_le_bytes(range.bytes().get(start..start + 8)?.try_into().ok()?)), _ => None }
    }
    pub fn token_type(&self, index: u64) -> Option<u64> {
        let range = self.types.as_ref()?;
        if index >= self.token_count { return None; }
        let width = range.bytes().len().checked_div(usize::try_from(self.token_count).ok()?)?;
        let start = usize::try_from(index).ok()?.checked_mul(width)?;
        let bytes = range.bytes().get(start..start + width)?;
        Some(match width { 1 => u64::from(bytes[0]), 2 => u64::from(u16::from_le_bytes(bytes.try_into().ok()?)), 4 => u64::from(u32::from_le_bytes(bytes.try_into().ok()?)), 8 => u64::from_le_bytes(bytes.try_into().ok()?), _ => return None })
    }
    fn offset(&self, index: u64) -> Option<u64> { let start = usize::try_from(index).ok()?.checked_mul(8)?; Some(u64::from_le_bytes(self.offsets.bytes().get(start..start + 8)?.try_into().ok()?)) }
}

pub fn encode_payload(kind: TokenizerKind, entries: &[TokenizerEntry]) -> Result<Vec<u8>, MlError> {
    if entries.len() > 256 { return Err(MlError::new(MlErrorCode::MalformedTokenizerMetadata, "tokenizer entry count exceeds profile maximum")); }
    let mut ordered: Vec<&TokenizerEntry> = entries.iter().collect(); ordered.sort_by_key(|entry| entry.role_id);
    for pair in ordered.windows(2) { if pair[0].role_id == pair[1].role_id { return Err(MlError::new(MlErrorCode::DuplicateTokenizerRole, "tokenizer role occurs more than once")); } }
    let mut output = Vec::with_capacity(HEADER_BYTES + ordered.len() * ENTRY_BYTES);
    output.extend_from_slice(&TOKENIZER_MAGIC); output.extend_from_slice(&TOKENIZER_VERSION.to_le_bytes()); output.extend_from_slice(&0u16.to_le_bytes()); output.push(kind as u8); output.push(0); output.extend_from_slice(&(ordered.len() as u16).to_le_bytes()); output.extend_from_slice(&0u32.to_le_bytes());
    for entry in ordered { output.extend_from_slice(&entry.role_id.to_le_bytes()); output.extend_from_slice(&(if entry.required { 1u16 } else { 0 }).to_le_bytes()); output.extend_from_slice(&entry.generic_key_id.to_le_bytes()); output.extend_from_slice(&entry.occurrence.to_le_bytes()); output.extend_from_slice(&0u32.to_le_bytes()); }
    Ok(output)
}

fn parse_payload(bytes: &[u8]) -> Result<(TokenizerKind, Vec<TokenizerEntry>), MlError> {
    if bytes.len() < HEADER_BYTES || bytes.len() > MAX_TOKENIZER_BYTES || bytes[..8] != TOKENIZER_MAGIC { return Err(MlError::new(MlErrorCode::MalformedTokenizerMetadata, "tokenizer metadata header is invalid")); }
    if u16::from_le_bytes(bytes[8..10].try_into().unwrap()) != TOKENIZER_VERSION || u16::from_le_bytes(bytes[10..12].try_into().unwrap()) != 0 || bytes[13] != 0 || bytes[16..20].iter().any(|byte| *byte != 0) { return Err(MlError::new(MlErrorCode::MalformedTokenizerMetadata, "tokenizer metadata header is invalid")); }
    let kind = TokenizerKind::from_id(bytes[12])?;
    let count = usize::from(u16::from_le_bytes(bytes[14..16].try_into().unwrap()));
    let expected = HEADER_BYTES.checked_add(count.checked_mul(ENTRY_BYTES).ok_or_else(|| MlError::new(MlErrorCode::MalformedTokenizerMetadata, "tokenizer entry count overflows"))?).ok_or_else(|| MlError::new(MlErrorCode::MalformedTokenizerMetadata, "tokenizer metadata length overflows"))?;
    if expected != bytes.len() { return Err(MlError::new(MlErrorCode::MalformedTokenizerMetadata, "tokenizer metadata length does not match entry count")); }
    let mut entries = Vec::with_capacity(count);
    for index in 0..count { let offset = HEADER_BYTES + index * ENTRY_BYTES; let flags = u16::from_le_bytes(bytes[offset + 2..offset + 4].try_into().unwrap()); if flags & !1 != 0 || bytes[offset + 8..offset + 12].iter().any(|byte| *byte != 0) { return Err(MlError::new(MlErrorCode::MalformedTokenizerMetadata, "tokenizer entry flags are invalid")); } entries.push(TokenizerEntry::new(u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap()), flags != 0, u16::from_le_bytes(bytes[offset + 4..offset + 6].try_into().unwrap()), u16::from_le_bytes(bytes[offset + 6..offset + 8].try_into().unwrap()))); }
    Ok((kind, entries))
}

fn resolve_value<'a>(validated: &ValidatedV06<'a>, key_id: u16, occurrence: u16) -> Result<(usize, CheckedRange<'a>), MlError> {
    let index = validated.blocks().iter().enumerate().filter(|(_, block)| block.key_id == key_id).nth(occurrence as usize).map(|(index, _)| index).ok_or_else(|| MlError::new(MlErrorCode::MissingTokenizerReference, "tokenizer canonical reference is absent"))?;
    if validated.blocks()[index].continuation { return Err(MlError::new(MlErrorCode::MissingTokenizerReference, "continuation tokenizer arrays are unsupported")); }
    Ok((index, validated.payload_range(index)?))
}

fn validate_text_pool(block: &vbuf_core::v06::V06Block, bytes: &[u8]) -> Result<(), MlError> {
    if block.semantic != V06Semantic::Opaque || block.physical != V06Physical::Array || block.bit_width != 8 || u64::try_from(bytes.len()).unwrap_or(u64::MAX) > 64 * 1024 * 1024 { return Err(MlError::new(MlErrorCode::MetadataTypeMismatch, "token text pool must be canonical byte data")); }
    if std::str::from_utf8(bytes).is_err() || bytes.contains(&0) { return Err(MlError::new(MlErrorCode::InvalidTokenText, "token text pool is not valid UTF-8")); }
    Ok(())
}

fn validate_offsets(block: &vbuf_core::v06::V06Block, bytes: &[u8], pool: &[u8]) -> Result<u64, MlError> {
    if block.semantic != V06Semantic::Unsigned || block.physical != V06Physical::Array || block.bit_width != 64 || block.count < 2 || !bytes.len().is_multiple_of(8) { return Err(MlError::new(MlErrorCode::MetadataTypeMismatch, "token offsets must be a u64 array")); }
    let token_count = block.count - 1;
    let first = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    if first != 0 { return Err(MlError::new(MlErrorCode::InvalidTokenText, "token offsets must start at zero")); }
    let pool_len = u64::try_from(pool.len()).unwrap_or(u64::MAX);
    let mut start_offset = first;
    let offset_count = usize::try_from(block.count).map_err(|_| MlError::new(MlErrorCode::InvalidTokenText, "token offset count exceeds host range"))?;
    for index in 1..offset_count {
        let start = index.checked_mul(8).ok_or_else(|| MlError::new(MlErrorCode::InvalidTokenText, "token offset index overflows"))?;
        let end_offset = u64::from_le_bytes(bytes[start..start + 8].try_into().unwrap());
        if end_offset < start_offset || end_offset - start_offset > MAX_TOKEN_BYTES || end_offset > pool_len { return Err(MlError::new(MlErrorCode::InvalidTokenText, "token offset is invalid")); }
        let text_start = usize::try_from(start_offset).map_err(|_| MlError::new(MlErrorCode::InvalidTokenText, "token offset exceeds host range"))?;
        let text_end = usize::try_from(end_offset).map_err(|_| MlError::new(MlErrorCode::InvalidTokenText, "token offset exceeds host range"))?;
        let text = pool.get(text_start..text_end).ok_or_else(|| MlError::new(MlErrorCode::InvalidTokenText, "token offset is outside text pool"))?;
        if std::str::from_utf8(text).is_err() || text.contains(&0) { return Err(MlError::new(MlErrorCode::InvalidTokenText, "token text is not valid UTF-8")); }
        start_offset = end_offset;
    }
    Ok(token_count)
}

fn optional_array<'a>(resolved: &mut [Option<(usize, CheckedRange<'a>)>; ROLE_SLOTS], role: TokenizerRole, count: u64, validated: &ValidatedV06<'a>, float: bool) -> Result<Option<CheckedRange<'a>>, MlError> {
    let Some((index, range)) = resolved[role as usize].take() else { return Ok(None); };
    let block = &validated.blocks()[index];
    if block.physical != V06Physical::Array || block.count != count || (float && !matches!(block.semantic, V06Semantic::Float)) || (!float && !matches!(block.semantic, V06Semantic::Unsigned)) || !(float && matches!(block.bit_width, 32 | 64) || !float && matches!(block.bit_width, 8 | 16 | 32 | 64)) { return Err(MlError::new(MlErrorCode::TokenizerArrayLengthMismatch, "tokenizer parallel array is incompatible")); }
    Ok(Some(range))
}

fn validate_token_types(types: &CheckedRange<'_>, token_count: u64) -> Result<(), MlError> {
    let count = usize::try_from(token_count).map_err(|_| MlError::new(MlErrorCode::InvalidTokenType, "token type count exceeds host range"))?;
    let width = types.bytes().len().checked_div(count).ok_or_else(|| MlError::new(MlErrorCode::InvalidTokenType, "token type array is empty"))?;
    for index in 0..count {
        let start = index.checked_mul(width).ok_or_else(|| MlError::new(MlErrorCode::InvalidTokenType, "token type index overflows"))?;
        let bytes = types.bytes().get(start..start + width).ok_or_else(|| MlError::new(MlErrorCode::InvalidTokenType, "token type is truncated"))?;
        let value = match width { 1 => u64::from(bytes[0]), 2 => u64::from(u16::from_le_bytes(bytes.try_into().unwrap())), 4 => u64::from(u32::from_le_bytes(bytes.try_into().unwrap())), 8 => u64::from_le_bytes(bytes.try_into().unwrap()), _ => return Err(MlError::new(MlErrorCode::InvalidTokenType, "token type width is invalid")) };
        if value > 6 { return Err(MlError::new(MlErrorCode::InvalidTokenType, "token type is outside pinned llama.cpp domain")); }
    }
    Ok(())
}

fn required_unsigned_scalar<'a>(resolved: &mut [Option<(usize, CheckedRange<'a>)>; ROLE_SLOTS], role: TokenizerRole, validated: &ValidatedV06<'a>) -> Result<u64, MlError> {
    let (index, range) = resolved[role as usize].take().ok_or_else(|| MlError::new(MlErrorCode::MissingTokenizerReference, "required tokenizer scalar is absent"))?;
    decode_unsigned_scalar(&validated.blocks()[index], range.bytes())
}

fn required_u32_array<'a>(resolved: &mut [Option<(usize, CheckedRange<'a>)>; ROLE_SLOTS], role: TokenizerRole, validated: &ValidatedV06<'a>, max_count: u64) -> Result<(usize, CheckedRange<'a>), MlError> {
    let value = resolved[role as usize].take().ok_or_else(|| MlError::new(MlErrorCode::MissingTokenizerReference, "required merge array is absent"))?;
    let block = &validated.blocks()[value.0];
    if block.semantic != V06Semantic::Unsigned || block.physical != V06Physical::Array || block.bit_width != 32 || block.count > max_count || value.1.bytes().len() != usize::try_from(block.count.checked_mul(4).ok_or_else(|| MlError::new(MlErrorCode::InvalidMergeTable, "merge array length overflows"))?).map_err(|_| MlError::new(MlErrorCode::InvalidMergeTable, "merge array exceeds host range"))? { return Err(MlError::new(MlErrorCode::MetadataTypeMismatch, "merge IDs must be a bounded u32 array")); }
    Ok(value)
}

fn validate_merge_ids(left: &CheckedRange<'_>, right: &CheckedRange<'_>, token_count: u64) -> Result<(), MlError> {
    for index in 0..left.bytes().len() / 4 {
        let offset = index * 4;
        let l = u64::from(u32::from_le_bytes(left.bytes()[offset..offset + 4].try_into().unwrap()));
        let r = u64::from(u32::from_le_bytes(right.bytes()[offset..offset + 4].try_into().unwrap()));
        if l >= token_count || r >= token_count { return Err(MlError::new(MlErrorCode::InvalidMergeTable, "merge ID is outside the vocabulary")); }
    }
    Ok(())
}

fn optional_utf8_payload<'a>(resolved: &mut [Option<(usize, CheckedRange<'a>)>; ROLE_SLOTS], role: TokenizerRole, validated: &ValidatedV06<'a>) -> Result<Option<CheckedRange<'a>>, MlError> {
    let Some((index, range)) = resolved[role as usize].take() else { return Ok(None); };
    let block = &validated.blocks()[index];
    if block.semantic != V06Semantic::Opaque || block.physical != V06Physical::Array || block.bit_width != 8 || u64::try_from(range.bytes().len()).map_err(|_| MlError::new(MlErrorCode::InvalidTokenText, "chat template exceeds host range"))? > MAX_CHAT_TEMPLATE_BYTES || std::str::from_utf8(range.bytes()).is_err() { return Err(MlError::new(MlErrorCode::InvalidTokenText, "chat template must be bounded UTF-8 bytes")); }
    Ok(Some(range))
}

fn decode_unsigned_scalar(block: &vbuf_core::v06::V06Block, bytes: &[u8]) -> Result<u64, MlError> {
    if block.semantic != V06Semantic::Unsigned || block.physical != V06Physical::Scalar || block.count != 1 || !matches!(block.bit_width, 8 | 16 | 32 | 64) { return Err(MlError::new(MlErrorCode::MetadataTypeMismatch, "special token must be an unsigned scalar")); }
    Ok(match block.bit_width { 8 => u64::from(bytes[0]), 16 => u64::from(u16::from_le_bytes(bytes.try_into().unwrap())), 32 => u64::from(u32::from_le_bytes(bytes.try_into().unwrap())), 64 => u64::from_le_bytes(bytes.try_into().unwrap()), _ => unreachable!() })
}
