//! Runtime-local tokenizer indexes over validated canonical tokenizer views.
//!
//! These indexes are deliberately not part of the vBuf representation. They
//! borrow token bytes from `TokenizerMetadata` and own only lookup-table
//! structure. The backing mapping must outlive the indexes.

use crate::{MlError, MlErrorCode, PreTokenizer, TokenizerKind, TokenizerMetadata};
use regex::Regex;
use std::collections::HashMap;

#[derive(Debug)]
pub struct TokenIndex<'a> {
    entries: HashMap<&'a [u8], u32>,
}

impl<'a> TokenIndex<'a> {
    pub fn build(tokenizer: &'a TokenizerMetadata<'a>) -> Result<Self, MlError> {
        let count = usize::try_from(tokenizer.token_count()).map_err(|_| {
            MlError::new(
                MlErrorCode::TokenizerArrayLengthMismatch,
                "token count exceeds host range",
            )
        })?;
        let mut entries = HashMap::with_capacity(count);
        for index in 0..tokenizer.token_count() {
            let bytes = tokenizer.token_bytes(index).ok_or_else(|| {
                MlError::new(
                    MlErrorCode::InvalidTokenText,
                    "validated token span is unavailable",
                )
            })?;
            // Match llama's last-ordinal-wins behavior for duplicate text.
            entries.insert(
                bytes,
                u32::try_from(index).map_err(|_| {
                    MlError::new(
                        MlErrorCode::TokenizerArrayLengthMismatch,
                        "token ID exceeds runtime index width",
                    )
                })?,
            );
        }
        Ok(Self { entries })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn lookup(&self, bytes: &[u8]) -> Option<u32> {
        self.entries.get(bytes).copied()
    }
    pub fn retained_key_bytes(&self) -> usize {
        self.entries.keys().map(|key| key.len()).sum()
    }
}

#[derive(Debug)]
pub struct MergeRankIndex {
    entries: HashMap<u64, u32>,
}

impl MergeRankIndex {
    pub fn build(tokenizer: &TokenizerMetadata<'_>) -> Result<Self, MlError> {
        let count = usize::try_from(tokenizer.merge_count()).map_err(|_| {
            MlError::new(
                MlErrorCode::InvalidMergeTable,
                "merge count exceeds host range",
            )
        })?;
        let mut entries = HashMap::with_capacity(count);
        for rank in 0..tokenizer.merge_count() {
            let (left, right) = tokenizer.merge_pair(rank).ok_or_else(|| {
                MlError::new(
                    MlErrorCode::InvalidMergeTable,
                    "validated merge pair is unavailable",
                )
            })?;
            let key = pack_pair(left, right)?;
            entries.insert(
                key,
                u32::try_from(rank).map_err(|_| {
                    MlError::new(
                        MlErrorCode::InvalidMergeTable,
                        "merge rank exceeds runtime index width",
                    )
                })?,
            );
        }
        Ok(Self { entries })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn lookup(&self, left: u64, right: u64) -> Option<u32> {
        pack_pair(left, right)
            .ok()
            .and_then(|key| self.entries.get(&key).copied())
    }
    pub fn retained_bytes(&self) -> usize {
        self.entries.len() * std::mem::size_of::<(u64, u32)>()
    }
}

fn pack_pair(left: u64, right: u64) -> Result<u64, MlError> {
    if left > u64::from(u32::MAX) || right > u64::from(u32::MAX) {
        return Err(MlError::new(
            MlErrorCode::InvalidMergeTable,
            "merge ID exceeds packed runtime index width",
        ));
    }
    Ok((left << 32) | right)
}

#[derive(Debug)]
pub struct RuntimeTokenizerIndexes<'a> {
    pub token: TokenIndex<'a>,
    pub merge: MergeRankIndex,
}

/// Runtime GPT-2 byte-level BPE view over persistent tokenizer ranges.
///
/// The regex and lookup maps are runtime-only; vocabulary, merge ranks, and
/// special-token identities remain borrowed from the canonical sidecar.
#[derive(Debug)]
pub struct Gpt2ByteLevelTokenizer<'a> {
    metadata: &'a TokenizerMetadata<'a>,
    indexes: RuntimeTokenizerIndexes<'a>,
    pretokenizer: Regex,
}

impl<'a> Gpt2ByteLevelTokenizer<'a> {
    pub fn build(metadata: &'a TokenizerMetadata<'a>) -> Result<Self, MlError> {
        if metadata.kind != TokenizerKind::Gpt2BpeByteLevel
            || metadata.model().is_none()
            || metadata.pre_tokenizer() != Some(PreTokenizer::Gpt2ByteLevel)
        {
            return Err(MlError::new(
                MlErrorCode::UnsupportedTokenizerKind,
                "tokenizer is not the persistent GPT-2 byte-level profile",
            ));
        }
        let pretokenizer = Regex::new(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+",
        )
        .map_err(|_| MlError::new(MlErrorCode::InvalidTokenizerIdentity, "GPT-2 regex is invalid"))?;
        Ok(Self {
            metadata,
            indexes: RuntimeTokenizerIndexes::build(metadata)?,
            pretokenizer,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, MlError> {
        let mut output = Vec::new();
        let mut cursor = 0usize;
        while cursor < text.len() {
            if let Some((id, length)) = self.special_at(text, cursor) {
                output.push(id);
                cursor += length;
                continue;
            }
            let piece = self.pretokenizer.find_at(text, cursor).ok_or_else(|| {
                MlError::new(
                    MlErrorCode::InvalidTokenText,
                    "GPT-2 pre-tokenization made no progress",
                )
            })?;
            if piece.start() != cursor {
                return Err(MlError::new(
                    MlErrorCode::InvalidTokenText,
                    "GPT-2 pre-tokenizer left an uncovered input span",
                ));
            }
            let mut symbols = Vec::new();
            for byte in piece.as_str().as_bytes() {
                let symbol = byte_to_unicode(*byte);
                let mut symbol_bytes = [0u8; 4];
                let symbol_bytes = symbol.encode_utf8(&mut symbol_bytes).as_bytes();
                let id = self.indexes.token.lookup(symbol_bytes).ok_or_else(|| {
                    MlError::new(MlErrorCode::InvalidTokenText, "byte token is absent")
                })?;
                symbols.push(id);
            }
            if !self.metadata.ignore_merges().unwrap_or(false) {
                symbols = self.merge_symbols(symbols)?;
            }
            output.extend(symbols);
            cursor = piece.end();
        }
        if self.metadata.add_bos() == Some(true) {
            if let Some((_, bos)) = self
                .metadata
                .specials()
                .iter()
                .find(|(kind, _)| matches!(kind, crate::SpecialToken::Bos))
            {
                output.insert(
                    0,
                    u32::try_from(*bos).map_err(|_| {
                        MlError::new(
                            MlErrorCode::InvalidSpecialTokenId,
                            "BOS ID exceeds runtime width",
                        )
                    })?,
                );
            }
        }
        Ok(output)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String, MlError> {
        let mut bytes = Vec::new();
        let special_ids = self.metadata.special_ids().unwrap_or_default();
        for token in tokens {
            let index = u64::from(*token);
            let text = self.metadata.token_text(index).ok_or_else(|| {
                MlError::new(
                    MlErrorCode::InvalidSpecialTokenId,
                    "token ID is outside vocabulary",
                )
            })?;
            if special_ids.contains(&index) {
                bytes.extend_from_slice(text.as_bytes());
            } else {
                for character in text.chars() {
                    bytes.push(unicode_to_byte(character).ok_or_else(|| {
                        MlError::new(
                            MlErrorCode::InvalidTokenText,
                            "token is not GPT-2 byte-level text",
                        )
                    })?);
                }
            }
        }
        String::from_utf8(bytes)
            .map_err(|_| MlError::new(MlErrorCode::InvalidTokenText, "decoded bytes are not UTF-8"))
    }

    fn special_at(&self, text: &str, cursor: usize) -> Option<(u32, usize)> {
        self.metadata
            .special_ids()?
            .into_iter()
            .filter_map(|id| {
                let token = self.metadata.token_text(id)?;
                text.get(cursor..)
                    .filter(|rest| rest.starts_with(token))
                    .and_then(|_| u32::try_from(id).ok().map(|id| (id, token.len())))
            })
            .max_by_key(|(_, length)| *length)
    }

    fn merge_symbols(&self, mut symbols: Vec<u32>) -> Result<Vec<u32>, MlError> {
        while symbols.len() > 1 {
            let Some((position, _rank)) = symbols
                .windows(2)
                .enumerate()
                .filter_map(|(position, pair)| {
                    self.indexes
                        .merge
                        .lookup(u64::from(pair[0]), u64::from(pair[1]))
                        .map(|rank| (position, rank))
                })
                .min_by_key(|(_, rank)| *rank)
            else {
                break;
            };
            let left = self
                .metadata
                .token_bytes(u64::from(symbols[position]))
                .ok_or_else(|| {
                    MlError::new(MlErrorCode::InvalidTokenText, "merge-left token is absent")
                })?;
            let right = self
                .metadata
                .token_bytes(u64::from(symbols[position + 1]))
                .ok_or_else(|| {
                    MlError::new(MlErrorCode::InvalidTokenText, "merge-right token is absent")
                })?;
            let mut merged = Vec::with_capacity(left.len() + right.len());
            merged.extend_from_slice(left);
            merged.extend_from_slice(right);
            let id = self.indexes.token.lookup(&merged).ok_or_else(|| {
                MlError::new(MlErrorCode::InvalidTokenText, "merged token is absent")
            })?;
            symbols.splice(position..=position + 1, [id]);
        }
        Ok(symbols)
    }
}

fn byte_to_unicode(byte: u8) -> char {
    let mut extra = 0u32;
    for value in 0..=255u8 {
        if is_direct_byte(value) {
            if value == byte {
                return char::from(value);
            }
        } else {
            if value == byte {
                return char::from_u32(256 + extra).unwrap();
            }
            extra += 1;
        }
    }
    unreachable!()
}

fn unicode_to_byte(character: char) -> Option<u8> {
    let code = character as u32;
    if code <= u32::from(u8::MAX) && is_direct_byte(code as u8) {
        return Some(code as u8);
    }
    let mut extra = 0u32;
    for value in 0..=255u8 {
        if !is_direct_byte(value) {
            if 256 + extra == code {
                return Some(value);
            }
            extra += 1;
        }
    }
    None
}

fn is_direct_byte(value: u8) -> bool {
    (b'!'..=b'~').contains(&value)
        || (0xa1..=0xac).contains(&value)
        || (0xae..=0xff).contains(&value)
}

impl<'a> RuntimeTokenizerIndexes<'a> {
    pub fn build(tokenizer: &'a TokenizerMetadata<'a>) -> Result<Self, MlError> {
        Ok(Self {
            token: TokenIndex::build(tokenizer)?,
            merge: MergeRankIndex::build(tokenizer)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::pack_pair;

    #[test]
    fn packed_pair_is_bounded_and_ordinal_safe() {
        assert_eq!(pack_pair(1, 2).unwrap(), (1u64 << 32) | 2);
        assert!(pack_pair(u64::from(u32::MAX) + 1, 0).is_err());
    }
}
