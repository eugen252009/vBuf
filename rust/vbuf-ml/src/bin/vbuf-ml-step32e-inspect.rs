//! Qualification-only inspection of the persisted GLM semantic sidecar.
//!
//! This tool uses source names only to print the audited inventory. Execution
//! code must use the resulting tensor identities and MoE catalog identities.

use memmap2::Mmap;
use std::fs::File;
use std::path::PathBuf;
use vbuf_core::v06::parse_v06;
use vbuf_ml::{Bootstrap, BorrowedModelView, ModelMetadataKey, parse_source_profile};

fn main() -> Result<(), String> {
    let sidecar = PathBuf::from(std::env::args().nth(1).ok_or("sidecar path")?);
    let layer = std::env::args()
        .nth(2)
        .ok_or("layer index")?
        .parse::<u32>()
        .map_err(|_| "layer index is invalid")?;
    let file = File::open(&sidecar).map_err(|error| error.to_string())?;
    let mapping = unsafe { Mmap::map(&file).map_err(|error| error.to_string())? };
    let validated = parse_v06(&mapping).map_err(|error| error.to_string())?;
    let bootstrap = Bootstrap::discover(&validated).map_err(|error| error.to_string())?;
    let profile = parse_source_profile(&validated, &bootstrap)
        .map_err(|error| error.to_string())?
        .ok_or("persistent source profile is absent")?;
    let view = BorrowedModelView::parse_with_sources(&mapping, &profile.registry, &[])
        .map_err(|error| error.to_string())?;
    let metadata = &view.metadata;
    println!(
        "architecture={} embedding={} layers={} heads={} kv_heads={} key_dim={} value_dim={} epsilon={} rope_theta={}",
        metadata.architecture().unwrap_or(""),
        metadata
            .unsigned(ModelMetadataKey::EmbeddingLength)
            .unwrap_or(0),
        metadata.unsigned(ModelMetadataKey::LayerCount).unwrap_or(0),
        metadata.unsigned(ModelMetadataKey::HeadCount).unwrap_or(0),
        metadata
            .unsigned(ModelMetadataKey::KVHeadCount)
            .unwrap_or(0),
        metadata
            .unsigned(ModelMetadataKey::KeyHeadDimension)
            .unwrap_or(0),
        metadata
            .unsigned(ModelMetadataKey::ValueHeadDimension)
            .unwrap_or(0),
        metadata
            .float(ModelMetadataKey::NormalizationEpsilon)
            .unwrap_or(0.0),
        metadata.float(ModelMetadataKey::RopeTheta).unwrap_or(0.0),
    );
    println!("layer={layer} tensors:");
    let prefix = format!("{layer:02}");
    for tensor in view
        .directory
        .tensors()
        .iter()
        .filter(|tensor| tensor.name.contains(&prefix))
    {
        println!(
            "tensor key={} occurrence={} name={} repr={:?} dims={:?} source={} offset={} bytes={}",
            tensor.key_id,
            tensor.occurrence,
            tensor.name,
            tensor.representation,
            tensor.dimensions,
            tensor.payload.source_id().value(),
            tensor.payload.offset(),
            tensor.payload.length(),
        );
    }
    let moe = view.moe.as_ref().ok_or("MoE directory is absent")?;
    let entries: Vec<_> = moe.entries_for_layer(layer).collect();
    println!("layer_moe_entries={}", entries.len());
    for entry in entries {
        println!(
            "moe expert={} role={} child={} tensor_ordinal={:?} scale_ordinal={:?}",
            entry.expert_index,
            entry.role,
            entry.child_name,
            entry.tensor_ordinal,
            entry.scale_ordinal,
        );
    }
    Ok(())
}
