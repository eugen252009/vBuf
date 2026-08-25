#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub enum RegionRole {
    TensorDirectory = 1,
    ModelMetadata = 2,
    TokenizerMetadata = 3,
    IntegrityMetadata = 4,
    NestedDirectory = 5,
    MoeDirectory = 6,
    SourceMetadata = 7,
    QuantizationMetadata = 8,
}

impl RegionRole {
    pub const fn from_id(id: u16) -> Option<Self> {
        match id {
            1 => Some(Self::TensorDirectory),
            2 => Some(Self::ModelMetadata),
            3 => Some(Self::TokenizerMetadata),
            4 => Some(Self::IntegrityMetadata),
            5 => Some(Self::NestedDirectory),
            6 => Some(Self::MoeDirectory),
            7 => Some(Self::SourceMetadata),
            8 => Some(Self::QuantizationMetadata),
            _ => None,
        }
    }

    pub const fn is_required(self) -> bool {
        matches!(self, Self::TensorDirectory | Self::ModelMetadata)
    }
}
