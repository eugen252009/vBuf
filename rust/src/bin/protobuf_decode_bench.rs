//! Reproducible single-format Prost decode and numeric-aggregation benchmark.

use prost::Message;
use std::hint::black_box;
use std::time::Instant;

const NUM_RECORDS: usize = 1_000_000;
const WARM_UP_RUNS: usize = 5;
const MEASURED_RUNS: usize = 30;
const MAX_RETRIES: usize = 100;

#[derive(Clone, PartialEq, Message)]
struct ProtoRecord {
    #[prost(uint32, tag = "1")]
    id: u32,
    #[prost(double, tag = "2")]
    value: f64,
}

#[derive(Clone, PartialEq, Message)]
struct ProtoRecordList {
    #[prost(message, repeated, tag = "1")]
    records: Vec<ProtoRecord>,
}

#[derive(Clone)]
struct Sample {
    round: usize,
    position: usize,
    duration_secs: f64,
    tsc_ticks: u64,
    cpu_before: Option<i32>,
    cpu_after: Option<i32>,
}

struct Stats {
    median: f64,
    mean: f64,
    population_stddev: f64,
    p5: f64,
    p95: f64,
    min: f64,
    max: f64,
    median_diagnostic_tsc_rate_ghz: f64,
}

struct Target<'a> {
    workload: &'static str,
    operation: &'static str,
    run: Box<dyn Fn() -> f64 + 'a>,
}

struct RunMetadata {
    run_label: String,
    commit: String,
    source_sha256: String,
    rustc: String,
    affinity: String,
    worktree_status: String,
    rustflags: String,
    effective_rustc_flags: String,
}

impl RunMetadata {
    fn required() -> Self {
        fn value(name: &str) -> String {
            std::env::var(name)
                .unwrap_or_else(|_| panic!("required benchmark metadata {name} is unset"))
        }
        Self {
            run_label: value("BENCH_RUN_LABEL"),
            commit: value("BENCH_COMMIT"),
            source_sha256: value("BENCH_SOURCE_SHA256"),
            rustc: value("BENCH_RUSTC"),
            affinity: value("BENCH_AFFINITY"),
            worktree_status: value("BENCH_WORKTREE_STATUS"),
            rustflags: value("BENCH_RUSTFLAGS"),
            effective_rustc_flags: value("BENCH_EFFECTIVE_RUSTC_FLAGS"),
        }
    }
}

#[inline(always)]
fn tsc_start() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::asm!("lfence", options(nomem, nostack));
        let high: u64;
        let low: u64;
        std::arch::asm!("rdtsc", out("rdx") high, out("rax") low, options(nomem, nostack));
        (high << 32) | low
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0
    }
}

#[inline(always)]
fn tsc_end() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let high: u64;
        let low: u64;
        std::arch::asm!("rdtscp", out("rdx") high, out("rax") low, out("rcx") _, options(nomem, nostack));
        std::arch::asm!("lfence", options(nomem, nostack));
        (high << 32) | low
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0
    }
}

fn cpu_from_sched_getcpu(result: i32) -> Option<i32> {
    (result >= 0).then_some(result)
}

fn current_cpu() -> Option<i32> {
    #[cfg(target_os = "linux")]
    unsafe {
        cpu_from_sched_getcpu(libc::sched_getcpu())
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn page_size_from_sysconf(result: libc::c_long) -> Option<usize> {
    (result > 0).then_some(result as usize)
}

#[inline(never)]
fn prefault_pages(bytes: &[u8]) -> Result<(), &'static str> {
    if bytes.is_empty() {
        return Ok(());
    }
    let page_size = page_size_from_sysconf(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
        .ok_or("sysconf(_SC_PAGESIZE) did not return a positive page size")?;
    for offset in (0..bytes.len()).step_by(page_size) {
        unsafe { std::ptr::read_volatile(&bytes[offset]) };
    }
    unsafe { std::ptr::read_volatile(&bytes[bytes.len() - 1]) };
    Ok(())
}

/// The named quantile definition is floor-index: sort n values and select
/// `floor(n * probability)`, clamped to the last index.
fn floor_index_quantile(sorted: &[f64], probability: f64) -> f64 {
    assert!(!sorted.is_empty());
    sorted[((sorted.len() as f64 * probability) as usize).min(sorted.len() - 1)]
}

fn median_sorted(sorted: &[f64]) -> f64 {
    assert!(!sorted.is_empty());
    let midpoint = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[midpoint - 1] + sorted[midpoint]) / 2.0
    } else {
        sorted[midpoint]
    }
}

fn stats(samples: &[Sample]) -> Stats {
    assert!(!samples.is_empty());
    let mut durations: Vec<f64> = samples.iter().map(|sample| sample.duration_secs).collect();
    durations.sort_by(f64::total_cmp);
    let mean = durations.iter().sum::<f64>() / durations.len() as f64;
    let population_stddev = (durations
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / durations.len() as f64)
        .sqrt();
    let mut rates: Vec<f64> = samples
        .iter()
        .map(|sample| sample.tsc_ticks as f64 / sample.duration_secs / 1e9)
        .collect();
    rates.sort_by(f64::total_cmp);
    Stats {
        median: median_sorted(&durations),
        mean,
        population_stddev,
        p5: floor_index_quantile(&durations, 0.05),
        p95: floor_index_quantile(&durations, 0.95),
        min: durations[0],
        max: durations[durations.len() - 1],
        median_diagnostic_tsc_rate_ghz: median_sorted(&rates),
    }
}

/// Even rounds run value-only first; odd rounds run full-record first.  This
/// is a deterministic, balanced two-target rotation, not a general shuffle.
fn order_for_round(round: usize) -> [usize; 2] {
    if round.is_multiple_of(2) {
        [0, 1]
    } else {
        [1, 0]
    }
}

fn report(
    metadata: &RunMetadata,
    targets: &[Target<'_>],
    samples: &[Vec<Sample>],
    rejected: &[usize],
) -> String {
    let cpu_identification = if samples
        .iter()
        .flatten()
        .all(|sample| sample.cpu_before.is_some() && sample.cpu_after.is_some())
    {
        "available"
    } else {
        "unavailable-for-at-least-one-sample"
    };
    let mut output = String::from("# isolated Prost decode raw benchmark report\n");
    for (name, value) in [
        ("run_label", &metadata.run_label),
        ("commit", &metadata.commit),
        ("source_sha256", &metadata.source_sha256),
        ("rustc", &metadata.rustc),
        ("affinity", &metadata.affinity),
        ("worktree_status", &metadata.worktree_status),
        ("rustflags", &metadata.rustflags),
        ("effective_rustc_flags", &metadata.effective_rustc_flags),
    ] {
        output.push_str(&format!("{name}={value}\n"));
    }
    output.push_str(&format!(
        "cpu_identification={cpu_identification}\nrecords={NUM_RECORDS}\nwarm_up_runs={WARM_UP_RUNS}\nmeasured_runs={MEASURED_RUNS}\nexecution_order=deterministic-balanced-two-target-rotation; Value-Only position-0 on even rounds and Full-Record position-0 on odd rounds\ninput=one-prefaulted-Protobuf-byte-buffer-reused-for-all-samples\ntimed_operation=decode-into-owned-Vec<ProtoRecord>; numeric-sum-reduction; local-decoded-value-is-dropped-before-closure-return\ntiming_windows=Instant starts after TSC start and ends before TSC end; diagnostic TSC rate is ticks divided by Instant duration and is not active-core-frequency\nblack_box=returned-numeric-aggregation-is-made-opaque-before-end-timestamp; barrier-overhead-is-timed\nquantiles=floor-index: floor(n*p), clamped to n-1\nstandard_deviation=population: divide by N\n"
    ));
    output.push_str("kind,workload,operation,accepted_samples,rejected_migration_samples,median_ms,mean_ms,population_stddev_ms,p5_ms,p95_ms,min_ms,max_ms,median_diagnostic_tsc_rate_ghz\n");
    for (index, target) in targets.iter().enumerate() {
        let calculated = stats(&samples[index]);
        output.push_str(&format!(
            "summary,{},{},{},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}\n",
            target.workload,
            target.operation,
            samples[index].len(),
            rejected[index],
            calculated.median * 1e3,
            calculated.mean * 1e3,
            calculated.population_stddev * 1e3,
            calculated.p5 * 1e3,
            calculated.p95 * 1e3,
            calculated.min * 1e3,
            calculated.max * 1e3,
            calculated.median_diagnostic_tsc_rate_ghz,
        ));
    }
    output.push_str("kind,workload,operation,round,position,duration_ns,tsc_ticks,diagnostic_tsc_rate_ghz,cpu_before,cpu_after,migrated\n");
    for (index, target) in targets.iter().enumerate() {
        for sample in &samples[index] {
            output.push_str(&format!(
                "sample,{},{},{},{},{:.0},{},{:.9},{},{},false\n",
                target.workload,
                target.operation,
                sample.round,
                sample.position,
                sample.duration_secs * 1e9,
                sample.tsc_ticks,
                sample.tsc_ticks as f64 / sample.duration_secs / 1e9,
                sample
                    .cpu_before
                    .map_or("unavailable".into(), |cpu| cpu.to_string()),
                sample
                    .cpu_after
                    .map_or("unavailable".into(), |cpu| cpu.to_string()),
            ));
        }
    }
    output
}

fn main() {
    let metadata = RunMetadata::required();
    let mut input = ProtoRecordList {
        records: Vec::with_capacity(NUM_RECORDS),
    };
    for id in 0..NUM_RECORDS {
        input.records.push(ProtoRecord {
            id: id as u32,
            value: id as f64 * 1.5,
        });
    }
    let mut bytes = Vec::new();
    input.encode(&mut bytes).expect("encode benchmark input");
    prefault_pages(&bytes).expect("prefault benchmark input");

    let expected_value_sum = 749_999_250_000.0;
    let expected_full_sum = 1_249_998_750_000.0;
    let decoded = ProtoRecordList::decode(&bytes[..]).expect("decode validation input");
    assert_eq!(
        decoded
            .records
            .iter()
            .map(|record| record.value)
            .sum::<f64>(),
        expected_value_sum
    );
    assert_eq!(
        decoded
            .records
            .iter()
            .map(|record| record.id as f64 + record.value)
            .sum::<f64>(),
        expected_full_sum
    );

    let targets = vec![
        Target {
            workload: "Value-Only",
            operation: "Prost decode into owned Vec<ProtoRecord> + sum(value)",
            run: Box::new({
                let bytes = &bytes;
                move || {
                    let decoded =
                        ProtoRecordList::decode(&bytes[..]).expect("decode benchmark input");
                    decoded.records.iter().map(|record| record.value).sum()
                }
            }),
        },
        Target {
            workload: "Full-Record",
            operation: "Prost decode into owned Vec<ProtoRecord> + sum(id as f64 + value)",
            run: Box::new({
                let bytes = &bytes;
                move || {
                    let decoded =
                        ProtoRecordList::decode(&bytes[..]).expect("decode benchmark input");
                    decoded
                        .records
                        .iter()
                        .map(|record| record.id as f64 + record.value)
                        .sum()
                }
            }),
        },
    ];
    for target in &targets {
        for _ in 0..WARM_UP_RUNS {
            black_box((target.run)());
        }
    }

    let mut samples = vec![Vec::new(), Vec::new()];
    let mut rejected = vec![0_usize, 0_usize];
    for round in 0..MEASURED_RUNS {
        for (position, target_index) in order_for_round(round).into_iter().enumerate() {
            for retries in 0..=MAX_RETRIES {
                let cpu_before = current_cpu();
                let cycles_start = tsc_start();
                let start = Instant::now();
                // `decoded` is a closure-local value. Rust drops it at closure scope
                // exit, before this call returns and before the end timestamps below.
                black_box((targets[target_index].run)());
                let duration_secs = start.elapsed().as_secs_f64();
                let cycles_end = tsc_end();
                let cpu_after = current_cpu();
                if cpu_before.is_some() && cpu_before != cpu_after {
                    rejected[target_index] += 1;
                    assert!(retries < MAX_RETRIES, "migration retry limit reached");
                    continue;
                }
                samples[target_index].push(Sample {
                    round,
                    position,
                    duration_secs,
                    tsc_ticks: cycles_end.saturating_sub(cycles_start),
                    cpu_before,
                    cpu_after,
                });
                break;
            }
        }
    }
    let raw_report = report(&metadata, &targets, &samples, &rejected);
    print!("{raw_report}");
    let path = std::env::var("BENCH_REPORT_PATH")
        .expect("required benchmark metadata BENCH_REPORT_PATH is unset");
    std::fs::write(path, raw_report).expect("write raw benchmark report");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_result_wrappers_reject_errors() {
        assert_eq!(cpu_from_sched_getcpu(-1), None);
        assert_eq!(cpu_from_sched_getcpu(0), Some(0));
        assert_eq!(page_size_from_sysconf(-1), None);
        assert_eq!(page_size_from_sysconf(0), None);
        assert_eq!(page_size_from_sysconf(4096), Some(4096));
        assert!(prefault_pages(&[]).is_ok());
    }

    #[test]
    fn deterministic_order_is_balanced_and_alternating() {
        let mut value_positions = [0_usize; 2];
        let mut full_positions = [0_usize; 2];
        for round in 0..MEASURED_RUNS {
            let order = order_for_round(round);
            assert_ne!(order[0], order[1]);
            value_positions[order.iter().position(|&target| target == 0).unwrap()] += 1;
            full_positions[order.iter().position(|&target| target == 1).unwrap()] += 1;
        }
        assert_eq!(value_positions, [15, 15]);
        assert_eq!(full_positions, [15, 15]);
        assert_eq!(order_for_round(0), [0, 1]);
        assert_eq!(order_for_round(1), [1, 0]);
    }

    #[test]
    fn statistics_use_floor_index_quantiles_and_population_standard_deviation() {
        let values: Vec<f64> = (1..=30).map(f64::from).collect();
        assert_eq!(median_sorted(&values), 15.5);
        assert_eq!(floor_index_quantile(&values, 0.05), 2.0);
        assert_eq!(floor_index_quantile(&values, 0.95), 29.0);
        assert_eq!(values[0], 1.0);
        assert_eq!(values[values.len() - 1], 30.0);
        assert_eq!(median_sorted(&[1.0, 3.0, 5.0]), 3.0);
        let samples: Vec<Sample> = values
            .iter()
            .map(|&duration_secs| Sample {
                round: 0,
                position: 0,
                duration_secs,
                tsc_ticks: 1,
                cpu_before: Some(0),
                cpu_after: Some(0),
            })
            .collect();
        let calculated = stats(&samples);
        assert!((calculated.mean - 15.5).abs() < 1e-12);
        assert!((calculated.population_stddev - 8.655_441_448_399_19).abs() < 1e-12);
        assert_eq!(calculated.min, 1.0);
        assert_eq!(calculated.max, 30.0);
    }
}
