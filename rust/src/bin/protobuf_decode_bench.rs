//! A single-format benchmark for Prost decode plus checksum.
//!
//! It deliberately has no vBuf, FlatBuffers, or Cap'n Proto target.  The two
//! workload variants are randomly ordered in each round so their positions are
//! recorded rather than assumed.

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
    stddev: f64,
    p5: f64,
    p95: f64,
    min: f64,
    max: f64,
    median_tsc_frequency_ghz: f64,
}

struct Target<'a> {
    workload: &'static str,
    operation: &'static str,
    run: Box<dyn Fn() -> f64 + 'a>,
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
        Some(libc::sched_getcpu())
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[inline(never)]
fn prefault_pages(bytes: &[u8]) {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
    for offset in (0..bytes.len()).step_by(page_size) {
        unsafe { std::ptr::read_volatile(&bytes[offset]) };
    }
    if let Some(last) = bytes.last() {
        unsafe { std::ptr::read_volatile(last) };
    }
}

fn stats(samples: &[Sample]) -> Stats {
    assert!(!samples.is_empty());
    let mut durations: Vec<f64> = samples.iter().map(|sample| sample.duration_secs).collect();
    durations.sort_by(f64::total_cmp);
    let mean = durations.iter().sum::<f64>() / durations.len() as f64;
    let stddev = (durations.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
        / durations.len() as f64)
        .sqrt();
    let midpoint = durations.len() / 2;
    let median = if durations.len().is_multiple_of(2) {
        (durations[midpoint - 1] + durations[midpoint]) / 2.0
    } else {
        durations[midpoint]
    };
    let mut frequencies: Vec<f64> = samples
        .iter()
        .map(|sample| sample.tsc_ticks as f64 / sample.duration_secs / 1e9)
        .collect();
    frequencies.sort_by(f64::total_cmp);
    let frequency_midpoint = frequencies.len() / 2;
    let median_tsc_frequency_ghz = if frequencies.len().is_multiple_of(2) {
        (frequencies[frequency_midpoint - 1] + frequencies[frequency_midpoint]) / 2.0
    } else {
        frequencies[frequency_midpoint]
    };
    let last = durations.len() - 1;
    Stats {
        median,
        mean,
        stddev,
        p5: durations[(durations.len() as f64 * 0.05) as usize],
        p95: durations[(durations.len() as f64 * 0.95).min(last as f64) as usize],
        min: durations[0],
        max: durations[last],
        median_tsc_frequency_ghz,
    }
}

fn report(
    targets: &[Target<'_>],
    samples: &[Vec<Sample>],
    rejected: &[usize],
) -> String {
    let mut output = String::from("# protobuf decode raw benchmark report\n");
    for (name, value) in [
        ("run_label", std::env::var("BENCH_RUN_LABEL").unwrap_or_else(|_| "unrecorded".into())),
        ("commit", std::env::var("BENCH_COMMIT").unwrap_or_else(|_| "unrecorded".into())),
        ("source_sha256", std::env::var("BENCH_SOURCE_SHA256").unwrap_or_else(|_| "unrecorded".into())),
        ("rustc", std::env::var("BENCH_RUSTC").unwrap_or_else(|_| "unrecorded".into())),
        ("affinity", std::env::var("BENCH_AFFINITY").unwrap_or_else(|_| "unrecorded".into())),
    ] {
        output.push_str(&format!("{name}={value}\n"));
    }
    output.push_str(&format!(
        "records={NUM_RECORDS}\nwarm_up_runs={WARM_UP_RUNS}\nmeasured_runs={MEASURED_RUNS}\nexecution_order=fixed-seed-lcg-fisher-yates(seed=12345); randomized-every-round\ninput=one-prefaulted-protobuf-byte-buffer-reused-for-all-samples\nallocation=ProtoRecordList::decode-allocates-and-drops-owned-Vec<ProtoRecord>-per-sample\nblack_box=after-end-timestamp\n"
    ));
    output.push_str("kind,workload,operation,accepted_samples,rejected_migration_samples,median_ms,mean_ms,stddev_ms,p5_ms,p95_ms,min_ms,max_ms,median_measured_tsc_frequency_ghz\n");
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
            calculated.stddev * 1e3,
            calculated.p5 * 1e3,
            calculated.p95 * 1e3,
            calculated.min * 1e3,
            calculated.max * 1e3,
            calculated.median_tsc_frequency_ghz,
        ));
    }
    output.push_str("kind,workload,operation,round,position,duration_ns,tsc_ticks,measured_tsc_frequency_ghz,cpu_before,cpu_after,migrated\n");
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
                sample.cpu_before.map_or(-1, |cpu| cpu),
                sample.cpu_after.map_or(-1, |cpu| cpu),
            ));
        }
    }
    output
}

fn main() {
    let mut input = ProtoRecordList { records: Vec::with_capacity(NUM_RECORDS) };
    for id in 0..NUM_RECORDS {
        input.records.push(ProtoRecord { id: id as u32, value: id as f64 * 1.5 });
    }
    let mut bytes = Vec::new();
    input.encode(&mut bytes).expect("encode benchmark input");
    prefault_pages(&bytes);

    let expected_value_sum = 749_999_250_000.0;
    let expected_full_sum = 1_249_998_750_000.0;
    let decoded = ProtoRecordList::decode(&bytes[..]).expect("decode validation input");
    assert_eq!(decoded.records.iter().map(|record| record.value).sum::<f64>(), expected_value_sum);
    assert_eq!(decoded.records.iter().map(|record| record.id as f64 + record.value).sum::<f64>(), expected_full_sum);

    let targets = vec![
        Target {
            workload: "Value-Only",
            operation: "Protobuf Decode + value checksum",
            run: Box::new({
                let bytes = &bytes;
                move || {
                    let decoded = ProtoRecordList::decode(&bytes[..]).expect("decode benchmark input");
                    decoded.records.iter().map(|record| record.value).sum()
                }
            }),
        },
        Target {
            workload: "Full-Record",
            operation: "Protobuf Decode + id/value checksum",
            run: Box::new({
                let bytes = &bytes;
                move || {
                    let decoded = ProtoRecordList::decode(&bytes[..]).expect("decode benchmark input");
                    decoded.records.iter().map(|record| record.id as f64 + record.value).sum()
                }
            }),
        },
    ];

    for target in &targets {
        for _ in 0..WARM_UP_RUNS {
            black_box((target.run)());
        }
    }

    let mut seed = 12_345_u64;
    let mut samples = vec![Vec::new(), Vec::new()];
    let mut rejected = vec![0_usize, 0_usize];
    for round in 0..MEASURED_RUNS {
        let mut order = [0_usize, 1_usize];
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        if seed & 1 == 1 {
            order.swap(0, 1);
        }
        for (position, target_index) in order.into_iter().enumerate() {
            for retries in 0..=MAX_RETRIES {
                let cpu_before = current_cpu();
                let cycles_start = tsc_start();
                let start = Instant::now();
                let checksum = (targets[target_index].run)();
                let duration_secs = start.elapsed().as_secs_f64();
                let cycles_end = tsc_end();
                let cpu_after = current_cpu();
                black_box(checksum);
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

    let raw_report = report(&targets, &samples, &rejected);
    print!("{raw_report}");
    if let Ok(path) = std::env::var("BENCH_REPORT_PATH") {
        std::fs::write(path, raw_report).expect("write raw benchmark report");
    }
}
