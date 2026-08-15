#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub enum RegionRole {
    TensorDirectory = 1,
    ModelMetadata = 2,
    TokenizerMetadata = 3,
    IntegrityMetadata = 4,
    NestedDirectory = 5,
    MoeDirectory = 6,
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
            _ => None,
        }
    }

    pub const fn is_required(self) -> bool {
        matches!(self, Self::TensorDirectory | Self::ModelMetadata)
    }
}
