//! Plan and execute a pinned Hugging Face Safetensors -> canonical vBuf import.

use std::path::PathBuf;
use vbuf_ml::hf_import::{
    DEFAULT_STAGING_BYTES, HfTransport, ImportOptions, PlanReport, execute_plan, plan_remote,
    serialized_plan, space_gate,
};

fn value(args: &[String], name: &str) -> Result<String, String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
        .ok_or_else(|| format!("missing {name}"))
}

fn has(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

fn print_plan(plan: &PlanReport, planning_requests: u64, planning_bytes: u64, free: Option<u64>) {
    println!(
        "RESOLVED_REPOSITORY_REVISION={}",
        plan.source.resolved_revision
    );
    println!("SAFETENSORS_SHARDS={}", plan.shards.len());
    println!("TENSORS={}", plan.tensors.len());
    println!("SOURCE_PAYLOAD_BYTES={}", plan.source_payload_bytes);
    println!("FINAL_VBUF_PAYLOAD_BYTES={}", plan.final_vbuf_payload_bytes);
    println!("FINAL_VBUF_BYTES={}", plan.final_vbuf_bytes);
    println!("METADATA_BYTES={}", plan.metadata_bytes);
    println!("BOOTSTRAP_BYTES={}", plan.bootstrap_bytes);
    println!("ALIGNMENT_PADDING_BYTES={}", plan.alignment_padding_bytes);
    println!(
        "ALIGNMENT_OVERHEAD_PERCENT={:.6}",
        if plan.source_payload_bytes == 0 {
            0.0
        } else {
            plan.alignment_padding_bytes as f64 * 100.0 / plan.source_payload_bytes as f64
        }
    );
    println!("LARGEST_TENSOR_BYTES={}", plan.largest_tensor);
    println!(
        "DTYPE_DISTRIBUTION={}",
        serde_json::to_string(&plan.dtype_distribution).unwrap_or_default()
    );
    println!("METADATA_FILES_FETCHED={}", plan.metadata_files.join(","));
    println!("DESTINATION_ALIGNMENT_BYTES={}", plan.destination_alignment);
    println!("ESTIMATED_STAGING_BYTES={}", plan.estimated_staging_bytes);
    println!("UNSUPPORTED_DTYPES={}", plan.unsupported_dtypes.join(","));
    println!(
        "UNSUPPORTED_SEMANTICS={}",
        plan.unsupported_semantics.join(";")
    );
    println!(
        "ALL_DESTINATION_OFFSETS_KNOWN={}",
        plan.all_destination_offsets_known
    );
    println!(
        "EXPERT_BANK_REPACK_AFTER_DOWNLOAD_REQUIRED={}",
        plan.expert_bank_repack_required
    );
    println!("PLANNING_HTTP_REQUESTS={planning_requests}");
    println!("PLANNING_BYTES_FETCHED={planning_bytes}");
    println!("FILESYSTEM_FREE_BYTES_BEFORE={}", free.unwrap_or(0));
    let gate = free.is_some_and(|bytes| {
        space_gate(bytes, plan.final_vbuf_bytes, plan.estimated_staging_bytes).is_ok()
    });
    println!("SPACE_GATE_PASS={gate}");
    println!("CAN_EXECUTE={}", plan.can_execute && gate);
}

fn free_bytes(path: &std::path::Path) -> Option<u64> {
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    let result = unsafe {
        libc::statvfs(
            std::ffi::CString::new(path.to_string_lossy().as_bytes())
                .ok()?
                .as_ptr(),
            stat.as_mut_ptr(),
        )
    };
    if result != 0 {
        return None;
    }
    let stat = unsafe { stat.assume_init() };
    u64::try_from(stat.f_bavail)
        .ok()?
        .checked_mul(u64::try_from(stat.f_frsize).ok()?)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let repository = value(&args, "--repo")?;
    let revision = value(&args, "--revision")?;
    let output = PathBuf::from(value(&args, "--output")?);
    let staging = value(&args, "--staging-bytes")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_STAGING_BYTES);
    let mut transport = HfTransport::new(repository.clone(), revision.clone());
    let (plan, stats) =
        plan_remote(&mut transport, &revision, &repository, staging).map_err(|e| e.to_string())?;
    let free = free_bytes(output.parent().unwrap_or(std::path::Path::new(".")));
    print_plan(&plan, stats.requests, stats.bytes, free);
    let plan_path = output.with_extension(format!(
        "{}plan.json",
        output
            .extension()
            .and_then(|v| v.to_str())
            .map(|v| format!("{v}."))
            .unwrap_or_default()
    ));
    std::fs::write(
        &plan_path,
        serialized_plan(&plan).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if has(&args, "--plan-only") {
        return Ok(());
    }
    if !plan.can_execute {
        return Err("CAN_EXECUTE=NO".into());
    }
    let free = free.ok_or_else(|| "filesystem free-space query failed".to_string())?;
    space_gate(free, plan.final_vbuf_bytes, plan.estimated_staging_bytes)
        .map_err(|e| e.to_string())?;
    let stats = execute_plan(
        &mut transport,
        &plan,
        &output,
        &ImportOptions {
            staging_bytes: staging,
            ..ImportOptions::default()
        },
    )
    .map_err(|e| e.to_string())?;
    println!("PAYLOAD_HTTP_REQUESTS={}", stats.payload_requests);
    println!("PAYLOAD_REQUESTED_BYTES={}", stats.payload_requested_bytes);
    println!("PAYLOAD_RETURNED_BYTES={}", stats.payload_returned_bytes);
    println!("SOURCE_OVERFETCH_BYTES={}", stats.source_overfetch_bytes);
    println!(
        "SOURCE_OVERFETCH_PERCENT={:.6}",
        if plan.source_payload_bytes == 0 {
            0.0
        } else {
            stats.source_overfetch_bytes as f64 * 100.0 / plan.source_payload_bytes as f64
        }
    );
    println!(
        "DESTINATION_WRITTEN_BYTES={}",
        stats.destination_written_bytes
    );
    println!("RESUME_SKIPPED_BYTES={}", stats.resume_skipped_bytes);
    println!("RESUME_DOWNLOADED_BYTES={}", stats.resume_downloaded_bytes);
    Ok(())
}
