use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlErrorCode {
    BootstrapNotFound,
    BootstrapDuplicate,
    MalformedBootstrap,
    UnsupportedProfileVersion,
    InvalidBootstrapFlags,
    UnknownRequiredRole,
    DuplicateRole,
    MissingRequiredRole,
    ReferencedRegionMissing,
    RegionTypeMismatch,
    TensorDirectoryMissing,
    MalformedTensorDirectory,
    UnsupportedTensorDirectoryVersion,
    DuplicateTensorName,
    InvalidTensorName,
    InvalidRank,
    InvalidDimension,
    ShapeOverflow,
    TensorReferenceMissing,
    UnsupportedTensorRepresentation,
    TensorRepresentationMismatch,
    TensorPayloadSizeMismatch,
    TensorPayloadAlignmentMismatch,
    InvalidQuantizedShape,
    InvalidQuantizedBlockCount,
    RepresentationArithmeticOverflow,
    InvalidDirectoryOrder,
    MalformedIntegrityMetadata,
    UnsupportedIntegrityAlgorithm,
    DuplicateIntegrityTarget,
    IntegrityTargetMissing,
    IntegrityTargetUnsupported,
    InvalidIntegrityDigest,
    NoIntegrityAvailable,
    IntegrityMismatch,
    MissingModelMetadata,
    MalformedModelMetadata,
    DuplicateMetadataKey,
    MissingRequiredMetadata,
    MetadataTypeMismatch,
    MetadataValueOutOfRange,
    UnsupportedMetadataKey,
    MalformedTokenizerMetadata,
    TokenizerArrayLengthMismatch,
    InvalidTokenText,
    InvalidTokenType,
    InvalidSpecialTokenId,
    UnsupportedTokenizerKind,
    InvalidTokenizerIdentity,
    InvalidMergeTable,
    MissingTokenizerReference,
    DuplicateTokenizerRole,
    NestedDirectoryMissing,
    MalformedNestedDirectory,
    UnsupportedNestedDirectoryVersion,
    NestedChildReferenceMissing,
    NestedChildRangeInvalid,
    NestedChildNotCanonical,
    NestedChildOverlap,
    MoeDirectoryMissing,
    MalformedMoeDirectory,
    UnsupportedMoeDirectoryVersion,
    MoeExpertReferenceMissing,
    MoeExpertIndexInvalid,
    UnsupportedMoeArchitecture,
    MalformedSourceMetadata,
    UnsupportedSourceMetadataVersion,
    DuplicateSourceBinding,
    InvalidSourceLocator,
    InvalidSourceHash,
    Canonical(vbuf_core::v06::V06ErrorCode),
}

#[derive(Debug)]
pub struct MlError {
    pub code: MlErrorCode,
    pub detail: &'static str,
}

impl MlError {
    pub(crate) const fn new(code: MlErrorCode, detail: &'static str) -> Self {
        Self { code, detail }
    }
}

impl fmt::Display for MlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "vBuf-ML {:?}: {}", self.code, self.detail)
    }
}

impl std::error::Error for MlError {}

impl From<vbuf_core::v06::V06Error> for MlError {
    fn from(error: vbuf_core::v06::V06Error) -> Self {
        Self::new(
            MlErrorCode::Canonical(error.code),
            "canonical v0.6 validation failed",
        )
    }
}
