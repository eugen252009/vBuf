//! Runtime-local tokenizer indexes over validated canonical tokenizer views.
//!
//! These indexes are deliberately not part of the vBuf representation. They
//! borrow token bytes from `TokenizerMetadata` and own only lookup-table
//! structure. The backing mapping must outlive the indexes.

use crate::{MlError, MlErrorCode, TokenizerMetadata};
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
