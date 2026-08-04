//! Real native and vBuf SoA/AoS baseline. No simulated or external formats.

use std::hint::black_box;
use std::io::Cursor;
use std::time::Instant;
use vbuf_core::{VBufInstance, VBufWriter};

const NUM_RECORDS: usize = 1_000_000;
const WARM_UP_RUNS: usize = 5;
const MEASURED_RUNS: usize = 30;
const MAX_RETRIES: usize = 100;
const ALIGNMENT: usize = 4096;
const ID_COLUMN: u16 = 1;
const VALUE_COLUMN: u16 = 2;
const AOS_COLUMN: u16 = 3;

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeRecord {
    id: u32,
    value: f64,
}

#[derive(Clone, Copy)]
struct TargetResult {
    numeric_sum: f64,
    output_bytes: usize,
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
    name: &'static str,
    layout: &'static str,
    category: &'static str,
    workload: &'static str,
    timed_operation: &'static str,
    expected_sum: f64,
    run: Box<dyn Fn() -> TargetResult + 'a>,
}

struct RunMetadata {
    run_label: String,
    commit: String,
    source_sha256: String,
    vbuf_core_sha256: String,
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
            vbuf_core_sha256: value("BENCH_VBUF_CORE_SHA256"),
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

fn current_cpu() -> Option<i32> {
    #[cfg(target_os = "linux")]
    unsafe {
        let result = libc::sched_getcpu();
        (result >= 0).then_some(result)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn prefault_pages(bytes: &[u8]) -> Result<(), &'static str> {
    if bytes.is_empty() {
        return Ok(());
    }
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return Err("sysconf(_SC_PAGESIZE) did not return a positive page size");
    }
    for offset in (0..bytes.len()).step_by(page_size as usize) {
        unsafe { std::ptr::read_volatile(&bytes[offset]) };
    }
    unsafe { std::ptr::read_volatile(&bytes[bytes.len() - 1]) };
    Ok(())
}

fn encode_vbuf_soa(ids: &[u32], values: &[f64]) -> Vec<u8> {
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer =
            VBufWriter::new(&mut cursor, ALIGNMENT).expect("create real vBuf SoA writer");
        writer
            .write_column(ID_COLUMN, ids)
            .expect("write real vBuf SoA id column");
        writer
            .write_column(VALUE_COLUMN, values)
            .expect("write real vBuf SoA value column");
    }
    cursor.into_inner()
}

fn encode_vbuf_aos(records: &[NativeRecord]) -> Vec<u8> {
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer =
            VBufWriter::new(&mut cursor, ALIGNMENT).expect("create real vBuf AoS writer");
        writer
            .write_column(AOS_COLUMN, records)
            .expect("write real vBuf AoS record column");
    }
    cursor.into_inner()
}

fn write_bytes(path: &str, bytes: &[u8]) {
    std::fs::write(path, bytes).expect("write real vBuf dataset");
}

fn sum_values_soa(values: &[f64]) -> f64 {
    values.iter().copied().sum()
}

fn sum_full_soa(ids: &[u32], values: &[f64]) -> f64 {
    ids.iter()
        .zip(values)
        .map(|(&id, &value)| id as f64 + value)
        .sum()
}

fn sum_values_aos(records: &[NativeRecord]) -> f64 {
    records.iter().map(|record| record.value).sum()
}

fn sum_full_aos(records: &[NativeRecord]) -> f64 {
    records
        .iter()
        .map(|record| record.id as f64 + record.value)
        .sum()
}

fn encode_aos_as_soa(records: &[NativeRecord], full: bool) -> TargetResult {
    let mut ids = Vec::with_capacity(records.len());
    let mut values = Vec::with_capacity(records.len());
    let mut numeric_sum = 0.0;
    for record in records {
        ids.push(record.id);
        values.push(record.value);
        numeric_sum += if full {
            record.id as f64 + record.value
        } else {
            record.value
        };
    }
    TargetResult {
        numeric_sum,
        output_bytes: encode_vbuf_soa(&ids, &values).len(),
    }
}

fn encode_aos_as_aos(records: &[NativeRecord], full: bool) -> TargetResult {
    let numeric_sum = if full {
        sum_full_aos(records)
    } else {
        sum_values_aos(records)
    };
    TargetResult {
        numeric_sum,
        output_bytes: encode_vbuf_aos(records).len(),
    }
}

fn floor_index_quantile(sorted: &[f64], probability: f64) -> f64 {
    sorted[((sorted.len() as f64 * probability) as usize).min(sorted.len() - 1)]
}

fn median_sorted(sorted: &[f64]) -> f64 {
    let midpoint = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[midpoint - 1] + sorted[midpoint]) / 2.0
    } else {
        sorted[midpoint]
    }
}

fn stats(samples: &[Sample]) -> Stats {
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

fn order_for_round(round: usize, count: usize) -> Vec<usize> {
    (0..count)
        .map(|position| (round + position) % count)
        .collect()
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
    let mut output = String::from("# real vBuf SoA/AoS raw baseline report\n");
    for (name, value) in [
        ("run_label", &metadata.run_label),
        ("commit", &metadata.commit),
        ("source_sha256", &metadata.source_sha256),
        ("vbuf_core_sha256", &metadata.vbuf_core_sha256),
        ("rustc", &metadata.rustc),
        ("affinity", &metadata.affinity),
        ("worktree_status", &metadata.worktree_status),
        ("rustflags", &metadata.rustflags),
        ("effective_rustc_flags", &metadata.effective_rustc_flags),
    ] {
        output.push_str(&format!("{name}={value}\n"));
    }
    output.push_str(&format!("cpu_identification={cpu_identification}\nrecords={NUM_RECORDS}\nwarm_up_runs={WARM_UP_RUNS}\nmeasured_runs={MEASURED_RUNS}\ndataset=id:u32(index),value:f64(index*1.5)\nlayouts=Native Rust SoA; Native Rust AoS repr(C); real vBuf SoA (two VBufWriter columns); real vBuf AoS (one VBufWriter NativeRecord column)\nexecution_order=deterministic-cyclic-rotation-of-{}-targets\ntiming_windows=Instant starts after TSC start and ends before TSC end; diagnostic TSC rate is not active core frequency\nquantiles=floor-index: floor(n*p), clamped to n-1\nstandard_deviation=population: divide by N\nblack_box=TargetResult made opaque before end timestamp; barrier overhead is timed\n", targets.len()));
    output.push_str("kind,target,layout,category,workload,timed_operation,accepted_samples,rejected_migration_samples,median_ms,mean_ms,population_stddev_ms,p5_ms,p95_ms,min_ms,max_ms,median_diagnostic_tsc_rate_ghz\n");
    for (index, target) in targets.iter().enumerate() {
        let calculated = stats(&samples[index]);
        output.push_str(&format!(
            "summary,{},{},{},{},{},{},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}\n",
            target.name,
            target.layout,
            target.category,
            target.workload,
            target.timed_operation,
            samples[index].len(),
            rejected[index],
            calculated.median * 1e3,
            calculated.mean * 1e3,
            calculated.population_stddev * 1e3,
            calculated.p5 * 1e3,
            calculated.p95 * 1e3,
            calculated.min * 1e3,
            calculated.max * 1e3,
            calculated.median_diagnostic_tsc_rate_ghz
        ));
    }
    output.push_str("kind,target,layout,category,workload,round,position,duration_ns,tsc_ticks,diagnostic_tsc_rate_ghz,cpu_before,cpu_after,migrated\n");
    for (index, target) in targets.iter().enumerate() {
        for sample in &samples[index] {
            output.push_str(&format!(
                "sample,{},{},{},{},{},{},{:.0},{},{:.9},{},{},false\n",
                target.name,
                target.layout,
                target.category,
                target.workload,
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
                    .map_or("unavailable".into(), |cpu| cpu.to_string())
            ));
        }
    }
    output
}

fn main() {
    let metadata = RunMetadata::required();
    let soa_path = std::env::var("BENCH_VBUF_SOA_PATH").expect("BENCH_VBUF_SOA_PATH is unset");
    let aos_path = std::env::var("BENCH_VBUF_AOS_PATH").expect("BENCH_VBUF_AOS_PATH is unset");
    let mut ids = Vec::with_capacity(NUM_RECORDS);
    let mut values = Vec::with_capacity(NUM_RECORDS);
    let mut aos = Vec::with_capacity(NUM_RECORDS);
    for index in 0..NUM_RECORDS {
        let id = index as u32;
        let value = index as f64 * 1.5;
        ids.push(id);
        values.push(value);
        aos.push(NativeRecord { id, value });
    }
    let expected_value = 749_999_250_000.0;
    let expected_full = 1_249_998_750_000.0;
    assert_eq!(sum_values_soa(&values), expected_value);
    assert_eq!(sum_full_soa(&ids, &values), expected_full);
    assert_eq!(sum_values_aos(&aos), expected_value);
    assert_eq!(sum_full_aos(&aos), expected_full);
    write_bytes(&soa_path, &encode_vbuf_soa(&ids, &values));
    write_bytes(&aos_path, &encode_vbuf_aos(&aos));
    let vbuf_soa = VBufInstance::open(&soa_path).expect("open real vBuf SoA");
    let vbuf_aos = VBufInstance::open(&aos_path).expect("open real vBuf AoS");
    prefault_pages(vbuf_soa.as_raw_slice()).expect("prefault real vBuf SoA");
    prefault_pages(vbuf_aos.as_raw_slice()).expect("prefault real vBuf AoS");
    let vbuf_soa_ids = vbuf_soa
        .get_as::<u32>(ID_COLUMN as u32)
        .expect("resolve real vBuf SoA ids");
    let vbuf_soa_values = vbuf_soa
        .get_as::<f64>(VALUE_COLUMN as u32)
        .expect("resolve real vBuf SoA values");
    let vbuf_aos_records = vbuf_aos
        .get_as::<NativeRecord>(AOS_COLUMN as u32)
        .expect("resolve real vBuf AoS records");
    assert_eq!(vbuf_soa_ids.len(), NUM_RECORDS);
    assert_eq!(vbuf_aos_records.len(), NUM_RECORDS);
    assert_eq!(sum_values_soa(vbuf_soa_values), expected_value);
    assert_eq!(sum_full_soa(vbuf_soa_ids, vbuf_soa_values), expected_full);
    assert_eq!(sum_values_aos(vbuf_aos_records), expected_value);
    assert_eq!(sum_full_aos(vbuf_aos_records), expected_full);
    assert_eq!(encode_aos_as_soa(&aos, true).numeric_sum, expected_full);
    assert_eq!(encode_aos_as_aos(&aos, true).numeric_sum, expected_full);

    let targets = vec![
        Target {
            name: "Native SoA Prepared Value",
            layout: "Native Rust SoA",
            category: "A prepared scan",
            workload: "value-only",
            timed_operation: "sum(value) over prebuilt Vec<f64>",
            expected_sum: expected_value,
            run: Box::new({
                let values = &values;
                move || TargetResult {
                    numeric_sum: sum_values_soa(values),
                    output_bytes: 0,
                }
            }),
        },
        Target {
            name: "Native AoS Prepared Value",
            layout: "Native Rust AoS",
            category: "A prepared scan",
            workload: "value-only",
            timed_operation: "visit every NativeRecord and sum(value)",
            expected_sum: expected_value,
            run: Box::new({
                let aos = &aos;
                move || TargetResult {
                    numeric_sum: sum_values_aos(aos),
                    output_bytes: 0,
                }
            }),
        },
        Target {
            name: "vBuf SoA Prepared Value",
            layout: "real vBuf SoA",
            category: "A prepared scan",
            workload: "value-only",
            timed_operation: "sum(value) over pre-resolved VBufInstance typed f64 slice",
            expected_sum: expected_value,
            run: Box::new(move || TargetResult {
                numeric_sum: sum_values_soa(vbuf_soa_values),
                output_bytes: 0,
            }),
        },
        Target {
            name: "vBuf AoS Prepared Value",
            layout: "real vBuf AoS",
            category: "A prepared scan",
            workload: "value-only",
            timed_operation: "visit every pre-resolved VBufInstance NativeRecord and sum(value)",
            expected_sum: expected_value,
            run: Box::new(move || TargetResult {
                numeric_sum: sum_values_aos(vbuf_aos_records),
                output_bytes: 0,
            }),
        },
        Target {
            name: "Native SoA Prepared Full",
            layout: "Native Rust SoA",
            category: "A prepared scan",
            workload: "full-record",
            timed_operation: "sum(id as f64 + value) over prebuilt SoA vectors",
            expected_sum: expected_full,
            run: Box::new({
                let ids = &ids;
                let values = &values;
                move || TargetResult {
                    numeric_sum: sum_full_soa(ids, values),
                    output_bytes: 0,
                }
            }),
        },
        Target {
            name: "Native AoS Prepared Full",
            layout: "Native Rust AoS",
            category: "A prepared scan",
            workload: "full-record",
            timed_operation: "visit every NativeRecord and sum(id as f64 + value)",
            expected_sum: expected_full,
            run: Box::new({
                let aos = &aos;
                move || TargetResult {
                    numeric_sum: sum_full_aos(aos),
                    output_bytes: 0,
                }
            }),
        },
        Target {
            name: "vBuf SoA Prepared Full",
            layout: "real vBuf SoA",
            category: "A prepared scan",
            workload: "full-record",
            timed_operation: "sum(id as f64 + value) over pre-resolved real vBuf typed slices",
            expected_sum: expected_full,
            run: Box::new(move || TargetResult {
                numeric_sum: sum_full_soa(vbuf_soa_ids, vbuf_soa_values),
                output_bytes: 0,
            }),
        },
        Target {
            name: "vBuf AoS Prepared Full",
            layout: "real vBuf AoS",
            category: "A prepared scan",
            workload: "full-record",
            timed_operation: "visit every pre-resolved real vBuf NativeRecord and sum(id as f64 + value)",
            expected_sum: expected_full,
            run: Box::new(move || TargetResult {
                numeric_sum: sum_full_aos(vbuf_aos_records),
                output_bytes: 0,
            }),
        },
        Target {
            name: "vBuf SoA Setup Value",
            layout: "real vBuf SoA",
            category: "B reader/view setup plus scan",
            workload: "value-only",
            timed_operation: "VBufInstance::get_as<f64> then sum(value)",
            expected_sum: expected_value,
            run: Box::new({
                let instance = &vbuf_soa;
                move || {
                    let values = instance
                        .get_as::<f64>(VALUE_COLUMN as u32)
                        .expect("vBuf SoA view");
                    TargetResult {
                        numeric_sum: sum_values_soa(values),
                        output_bytes: 0,
                    }
                }
            }),
        },
        Target {
            name: "vBuf AoS Setup Value",
            layout: "real vBuf AoS",
            category: "B reader/view setup plus scan",
            workload: "value-only",
            timed_operation: "VBufInstance::get_as<NativeRecord> then visit every record and sum(value)",
            expected_sum: expected_value,
            run: Box::new({
                let instance = &vbuf_aos;
                move || {
                    let records = instance
                        .get_as::<NativeRecord>(AOS_COLUMN as u32)
                        .expect("vBuf AoS view");
                    TargetResult {
                        numeric_sum: sum_values_aos(records),
                        output_bytes: 0,
                    }
                }
            }),
        },
        Target {
            name: "vBuf SoA Setup Full",
            layout: "real vBuf SoA",
            category: "B reader/view setup plus scan",
            workload: "full-record",
            timed_operation: "two VBufInstance::get_as calls then sum(id as f64 + value)",
            expected_sum: expected_full,
            run: Box::new({
                let instance = &vbuf_soa;
                move || {
                    let ids = instance
                        .get_as::<u32>(ID_COLUMN as u32)
                        .expect("vBuf SoA id view");
                    let values = instance
                        .get_as::<f64>(VALUE_COLUMN as u32)
                        .expect("vBuf SoA value view");
                    TargetResult {
                        numeric_sum: sum_full_soa(ids, values),
                        output_bytes: 0,
                    }
                }
            }),
        },
        Target {
            name: "vBuf AoS Setup Full",
            layout: "real vBuf AoS",
            category: "B reader/view setup plus scan",
            workload: "full-record",
            timed_operation: "VBufInstance::get_as<NativeRecord> then visit every record and sum(id as f64 + value)",
            expected_sum: expected_full,
            run: Box::new({
                let instance = &vbuf_aos;
                move || {
                    let records = instance
                        .get_as::<NativeRecord>(AOS_COLUMN as u32)
                        .expect("vBuf AoS view");
                    TargetResult {
                        numeric_sum: sum_full_aos(records),
                        output_bytes: 0,
                    }
                }
            }),
        },
        Target {
            name: "vBuf SoA Encode Value",
            layout: "real vBuf SoA",
            category: "C pack/encode",
            workload: "value-only",
            timed_operation: "extract SoA columns from NativeRecord input; VBufWriter writes two columns; sum(value)",
            expected_sum: expected_value,
            run: Box::new({
                let aos = &aos;
                move || encode_aos_as_soa(aos, false)
            }),
        },
        Target {
            name: "vBuf AoS Encode Value",
            layout: "real vBuf AoS",
            category: "C pack/encode",
            workload: "value-only",
            timed_operation: "VBufWriter writes one NativeRecord column; visit every source record and sum(value)",
            expected_sum: expected_value,
            run: Box::new({
                let aos = &aos;
                move || encode_aos_as_aos(aos, false)
            }),
        },
        Target {
            name: "vBuf SoA Encode Full",
            layout: "real vBuf SoA",
            category: "C pack/encode",
            workload: "full-record",
            timed_operation: "extract SoA columns from NativeRecord input; VBufWriter writes two columns; sum(id as f64 + value)",
            expected_sum: expected_full,
            run: Box::new({
                let aos = &aos;
                move || encode_aos_as_soa(aos, true)
            }),
        },
        Target {
            name: "vBuf AoS Encode Full",
            layout: "real vBuf AoS",
            category: "C pack/encode",
            workload: "full-record",
            timed_operation: "VBufWriter writes one NativeRecord column; visit every source record and sum(id as f64 + value)",
            expected_sum: expected_full,
            run: Box::new({
                let aos = &aos;
                move || encode_aos_as_aos(aos, true)
            }),
        },
    ];
    for target in &targets {
        for _ in 0..WARM_UP_RUNS {
            let result = (target.run)();
            assert_eq!(result.numeric_sum, target.expected_sum);
            assert!(result.output_bytes == 0 || result.output_bytes > 16);
            black_box(result);
        }
    }
    let mut samples = vec![Vec::new(); targets.len()];
    let mut rejected = vec![0_usize; targets.len()];
    for round in 0..MEASURED_RUNS {
        for (position, target_index) in order_for_round(round, targets.len())
            .into_iter()
            .enumerate()
        {
            for retries in 0..=MAX_RETRIES {
                let cpu_before = current_cpu();
                let cycles_start = tsc_start();
                let start = Instant::now();
                let result = black_box((targets[target_index].run)());
                let duration_secs = start.elapsed().as_secs_f64();
                let cycles_end = tsc_end();
                let cpu_after = current_cpu();
                assert_eq!(
                    result.numeric_sum, targets[target_index].expected_sum,
                    "numeric aggregation mismatch for {}",
                    targets[target_index].name
                );
                assert!(result.output_bytes == 0 || result.output_bytes > 16);
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
    std::fs::write(
        std::env::var("BENCH_REPORT_PATH").expect("BENCH_REPORT_PATH is unset"),
        raw_report,
    )
    .expect("write raw benchmark report");
}
