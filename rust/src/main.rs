use chrono::Local;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;

const FILENAME: &str = "./dual_test.vbuf";

fn main() -> Result<(), ()> {
    let args: Vec<String> = env::args().collect();

    let total_elements = if args.len() > 1 {
        parse_size(&args[1])
    } else {
        100_000_000
    };

    let n_per_column = total_elements / 6;

    println!("--- vBuf Industrie 5.0 Generator ---");
    println!("Gesamtzielelemente: {}", total_elements);
    println!("Elemente pro Spalte (6 Spalten): {}", n_per_column);
    println!("Generiere vBuf Datei: {}...", FILENAME);

    _ = ensure_test_file(FILENAME, n_per_column);
    Ok(())
}

fn parse_size(size_str: &str) -> usize {
    let multiplier = match size_str.chars().last().unwrap().to_ascii_uppercase() {
        'K' => 1_000,
        'M' => 1_000_000,
        'G' => 1_000_000_000,
        _ => return size_str.parse::<usize>().unwrap_or(0),
    };

    let val_str = &size_str[..size_str.len() - 1];
    val_str.parse::<usize>().unwrap_or(0) * multiplier
}

fn ensure_test_file(filename: &str, n: usize) -> Result<(), &str> {
    if std::path::Path::new(filename).exists() {
        return Ok(());
    }

    println!("Generiere Testdatei {} mit {} Einträgen...", filename, n);

    let mut vbuf_writer = vbuf_core::VBufWriter::<std::fs::File>::create(filename)
        .map_err(|_| "Datei konnte nicht erstellt werden.")?;

    let n_per_col = n; // n ist bereits n_per_column aus main

    let columns = vec![101, 102, 103, 104, 105, 106];
    for id in columns {
        println!("Schreibe Spalte {}...", id);
        let col: Vec<u32> = match id {
            101 => (0..n_per_col as u32).collect(),
            102 => vec![2u32; n_per_col],
            103 => (0..n_per_col as u32).map(|x| x * 2).collect(),
            104 => (0..n_per_col as u32).map(|x| x.wrapping_mul(x)).collect(),
            105 => (0..n_per_col as u32).map(|x| !x).collect(),
            106 => (0..n_per_col as u32).map(|x| (x + 100) % 500).collect(),
            _ => vec![],
        };
        vbuf_writer
            .write_column(id, &col)
            .map_err(|_| "Fehler beim Schreiben")?;
    }

    Ok(())
}

fn get_cycles() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { std::arch::x86_64::_rdtsc() }
    }

    #[cfg(target_arch = "riscv64")]
    {
        let cycles: u64;
        unsafe {
            std::arch::asm!("rdcycle {}", out(reg) cycles);
        }
        cycles
    }

    #[cfg(target_arch = "arm")]
    {
        // Fallback für Cubietruck
        std::time::Instant::now().elapsed().as_nanos() as u64
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "riscv64", target_arch = "arm")))]
    {
        0
    }
}

fn get_current_freq_hz() -> f64 {
    let freq_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq";
    std::fs::read_to_string(freq_path)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map(|khz| khz * 1000.0)
        .unwrap_or(1_200_000_000.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;
    use std::fs::remove_file;
    use std::time::{Duration, Instant};
    use vbuf_core::{VBufInstance, VBufWriter};

    fn log_benchmark_to_md(
        name: &str,
        n: usize,
        speed_gb_s: f64,
        cycles_per_item: f64,
        freq_ghz: f64,
        speed_per_ghz: f64,
        duration: std::time::Duration,
    ) {
        let file_path = "bench.md";
        let file_exists = std::path::Path::new(file_path).exists();
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(file_path)
            .unwrap();

        // 1. Logische Threads (vCores)
        let logical_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // 2. Physische Kerne (Echte Hardware)
        // Wir lesen die echten Kerne direkt aus dem System
        let physical_cores = std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .map(|info| {
                let mut core_ids = info
                    .lines()
                    .filter(|l| l.contains("core id"))
                    .map(|l| l.split(':').nth(1).unwrap_or("").trim())
                    .collect::<Vec<_>>();
                core_ids.sort();
                core_ids.dedup();
                if core_ids.is_empty() {
                    logical_threads / 2
                } else {
                    core_ids.len()
                }
            })
            .unwrap_or(logical_threads / 2)
            .max(1);

        // Metriken berechnen
        let eff_per_thread = speed_per_ghz / (logical_threads as f64);
        let eff_per_core = speed_per_ghz / (physical_cores as f64);

        let arch = std::env::consts::ARCH;
        let cpu_model = std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|i| {
                i.lines()
                    .find(|l| l.starts_with("model name") || l.starts_with("Hardware"))
                    .and_then(|l| l.split(':').nth(1))
                    .map(|s| s.trim().to_string())
            })
            .unwrap_or_else(|| "Unknown CPU".to_string());

        if !file_exists {
            writeln!(file, "| Timestamp | Arch | CPU | P/L Cores | Testname | GB/s | Eff/Thread | Eff/Core | Cyc/Item |").unwrap();
            writeln!(file, "|-----------|------|-----|-----------|----------|------|------------|----------|----------|").unwrap();
        }

        let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        writeln!(
            file,
            "| {} | {} | {} | {}/{} | {:<20} | {:>6.2} | {:>10.4} | {:>8.4} | {:>8.4} |",
            ts,
            arch,
            cpu_model,
            physical_cores,
            logical_threads,
            name,
            speed_gb_s,
            eff_per_thread,
            eff_per_core,
            cycles_per_item
        )
        .unwrap();
    }

    fn print_metrics(
        name: &str,
        n: usize,
        bytes_per_item: usize,
        duration: Duration,
        c_start: u64,
        c_end: u64,
    ) {
        let freq_hz = get_current_freq_hz();
        let freq_ghz = freq_hz / 1e9;
        let total_cycles = if c_end > c_start {
            (c_end.wrapping_sub(c_start)) as f64
        } else {
            duration.as_secs_f64() * freq_hz
        };
        let cycles_per_item = total_cycles / (n as f64);
        let speed_gb_s = ((n * bytes_per_item) as f64 / duration.as_secs_f64()) / 1e9;
        let speed_per_ghz = speed_gb_s / freq_ghz;

        println!("\n--- {} ---", name);
        println!(
            "Speed: {:.2} GB/s | Score: {:.1}",
            speed_gb_s,
            speed_gb_s * speed_per_ghz
        );

        log_benchmark_to_md(
            name,
            n,
            speed_gb_s,
            cycles_per_item,
            freq_ghz,
            speed_per_ghz,
            duration,
        );
    }

    #[test]
    fn benchmark_branch_killer() {
        let inst = VBufInstance::open(FILENAME).expect("Input fehlt");
        let col_a = inst.get_as::<u32>(101).expect("101 fehlt");
        let col_b = inst.get_as::<u32>(102).expect("102 fehlt");
        let start = Instant::now();
        let c_start = get_cycles();
        let count = col_a
            .par_iter()
            .zip(col_b.par_iter())
            .filter(|&(&a, &b)| a > b)
            .count();
        let c_end = get_cycles();
        let duration = start.elapsed();
        println!("Matches: {}", count);
        print_metrics(
            "Branch-Killer (if A > B)",
            col_a.len(),
            8,
            duration,
            c_start,
            c_end,
        );
    }

    #[test]
    fn benchmark_dot_product() {
        let inst = VBufInstance::open(FILENAME).expect("Input fehlt");
        let col_a = inst.get_as::<u32>(101).expect("101 fehlt");
        let col_b = inst.get_as::<u32>(102).expect("102 fehlt");
        let start = Instant::now();
        let c_start = get_cycles();
        let _: f64 = col_a
            .par_iter()
            .zip(col_b.par_iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();
        let c_end = get_cycles();
        print_metrics(
            "Dot-Product (A * B)",
            col_a.len(),
            8,
            start.elapsed(),
            c_start,
            c_end,
        );
    }

    #[test]
    fn benchmark_histogram() {
        let inst = VBufInstance::open(FILENAME).expect("Input fehlt");
        let data = inst.get_as::<u32>(101).expect("101 fehlt");
        let start = Instant::now();
        let c_start = get_cycles();
        let _ = data
            .par_iter()
            .fold(
                || vec![0usize; 256],
                |mut bins, &v| {
                    bins[(v & 0xFF) as usize] += 1;
                    bins
                },
            )
            .reduce(
                || vec![0usize; 256],
                |mut a, b| {
                    for i in 0..256 {
                        a[i] += b[i];
                    }
                    a
                },
            );
        let c_end = get_cycles();
        print_metrics(
            "Histogramm (u32 -> 256)",
            data.len(),
            4,
            start.elapsed(),
            c_start,
            c_end,
        );
    }

    #[test]
    fn benchmark_directory_scan_debug() {
        let inst = VBufInstance::open(FILENAME).expect("Input fehlt");
        let mem = inst.as_raw_slice();
        let start = Instant::now();
        let c_start = get_cycles();

        // Simulierter Scan durch die Header-Struktur
        let mut curr = 16;
        let mut cols_found = 0;
        while curr + 16 <= mem.len() {
            let anchor = u64::from_le_bytes(mem[curr..curr + 8].try_into().unwrap());
            if anchor == 0 {
                curr += 8;
                continue;
            }
            let plen = ((anchor >> 32) & 0xFFFF) as u16;
            let n = u64::from_le_bytes(mem[curr + 8..curr + 16].try_into().unwrap());
            cols_found += 1;
            let data_start = (curr + 16 + 63) & !63;
            curr = (data_start + (n * (plen as u64 / 8)) as usize + 7) & !7;
        }

        let c_end = get_cycles();
        let duration = start.elapsed();
        let freq_ghz = get_current_freq_hz() / 1e9;

        log_benchmark_to_md(
            "Directory Metadata Scan",
            cols_found,
            0.0,
            (c_end.wrapping_sub(c_start)) as f64 / cols_found as f64,
            freq_ghz,
            0.0,
            duration,
        );
        println!("\n--- vBuf Directory Scan ---\nSpalten: {}", cols_found);
    }
    #[test]
    fn benchmark_transform_and_write() {
        const OUTPUT_FILENAME: &str = "./test_out.vbuf";
        let inst = VBufInstance::open(FILENAME).expect("Input fehlt");
        let col_a = inst.get_as::<u32>(101).expect("101 fehlt");
        let col_b = inst.get_as::<u32>(102).expect("102 fehlt");
        let n = col_a.len();

        let start = Instant::now();
        let c_start = get_cycles();

        // Transformation: Durchschnitt bilden
        let result: Vec<u32> = col_a
            .par_iter()
            .zip(col_b.par_iter())
            .map(|(&a, &b)| a / 2 + b / 2)
            .collect();

        // Direktes Wegschreiben in neue Datei
        let mut out_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(OUTPUT_FILENAME)
            .unwrap();
        {
            let mut writer = VBufWriter::new(&mut out_file, 12).unwrap();
            writer.write_column(103, &result).unwrap();
        }

        let c_end = get_cycles();
        let duration = start.elapsed();

        // 12 Bytes: 8 Bytes Read (2x u32) + 4 Bytes Write (1x u32)
        print_metrics(
            "Transform & Write ((A+B)/2)",
            n,
            12,
            duration,
            c_start,
            c_end,
        );
        let _ = remove_file(OUTPUT_FILENAME);
    }

    #[test]
    fn benchmark_copy_and_scale() {
        let inst = VBufInstance::open(FILENAME).expect("Input fehlt");
        let data = inst.get_as::<u32>(101).expect("101 fehlt");
        let start = Instant::now();
        let c_start = get_cycles();

        let result: Vec<u32> = data.par_iter().map(|&x| x.wrapping_mul(42)).collect();

        let c_end = get_cycles();
        let duration = start.elapsed();

        // 8 Bytes: 4 Read + 4 Write
        print_metrics(
            "Copy & Scale (*42)",
            data.len(),
            8,
            duration,
            c_start,
            c_end,
        );
    }

    #[test]
    fn benchmark_weighted_sum() {
        let inst = VBufInstance::open(FILENAME).expect("Input fehlt");
        let data = inst.get_as::<u32>(101).expect("101 fehlt");
        let start = Instant::now();
        let c_start = get_cycles();

        // Gleitendes Fenster: 0.25*a + 0.5*b + 0.25*c
        let result: Vec<u32> = data
            .par_windows(3)
            .map(|w| ((w[0] as f32 * 0.25) + (w[1] as f32 * 0.5) + (w[2] as f32 * 0.25)) as u32)
            .collect();

        let c_end = get_cycles();
        let duration = start.elapsed();

        // 8 Bytes: ca. 4 Read (Sliding) + 4 Write
        print_metrics(
            "Weighted Sum (Window 3)",
            data.len(),
            8,
            duration,
            c_start,
            c_end,
        );
    }

    #[test]
    fn benchmark_find_indices() {
        let inst = VBufInstance::open(FILENAME).expect("Input fehlt");
        let data = inst.get_as::<u32>(101).expect("101 fehlt");
        let start = Instant::now();
        let c_start = get_cycles();

        let indices: Vec<usize> = data
            .par_iter()
            .enumerate()
            .filter(|&(_, &val)| val > 100_000)
            .map(|(idx, _)| idx)
            .collect();

        let c_end = get_cycles();
        let duration = start.elapsed();

        // 4 Bytes Read pro Element
        print_metrics(
            "Find Indices (>100k)",
            data.len(),
            4,
            duration,
            c_start,
            c_end,
        );
    }
}
