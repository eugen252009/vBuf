use std::env;

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
    println!("Generiere vBuf Datei in /dev/shm/test.vbuf...");

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

    // Wir teilen n durch 6, da wir 6 Spalten haben wollen
    let n_per_col = n / 6;

    // --- SPALTE 101 ---
    {
        println!("Schreibe Spalte 101...");
        let col: Vec<u32> = (0..n_per_col as u32).collect();
        vbuf_writer
            .write_column(101, &col)
            .map_err(|_| "Fehler 101")?;
    } // col wird hier gedroppt -> RAM wieder frei

    // --- SPALTE 102 ---
    {
        println!("Schreibe Spalte 102...");
        let col = vec![2u32; n_per_col];
        vbuf_writer
            .write_column(102, &col)
            .map_err(|_| "Fehler 102")?;
    } // RAM wieder frei

    // --- SPALTE 103 ---
    {
        println!("Schreibe Spalte 103...");
        let col: Vec<u32> = (0..n_per_col as u32).map(|x| x * 2).collect();
        vbuf_writer
            .write_column(103, &col)
            .map_err(|_| "Fehler 103")?;
    }

    // --- SPALTE 104 ---
    {
        println!("Schreibe Spalte 104...");
        let col: Vec<u32> = (0..n_per_col as u32).map(|x| x.wrapping_mul(x)).collect();
        vbuf_writer
            .write_column(104, &col)
            .map_err(|_| "Fehler 104")?;
    }

    // --- SPALTE 105 ---
    {
        println!("Schreibe Spalte 105...");
        let col: Vec<u32> = (0..n_per_col as u32).map(|x| !x).collect();
        vbuf_writer
            .write_column(105, &col)
            .map_err(|_| "Fehler 105")?;
    }

    // --- SPALTE 106 ---
    {
        println!("Schreibe Spalte 106...");
        let col: Vec<u32> = (0..n_per_col as u32).map(|x| (x + 100) % 500).collect();
        vbuf_writer
            .write_column(106, &col)
            .map_err(|_| "Fehler 106")?;
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
        // Auf ARMv7 (Cubietruck) nutzen wir den System-Timer/Counter
        // Falls PMU-Zugriff gesperrt ist, nehmen wir die Zeit als Fallback
        let mut cycles: u32;
        unsafe {
            // Achtung: MRC braucht oft Root-Rechte oder Kernel-Modul
            // Wir nutzen hier den Instant-Fallback für Stabilität
            std::time::Instant::now().elapsed().as_nanos() as u64
        }
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
    use chrono::Local;
    use rayon::prelude::*;
    use std::fs::{OpenOptions, remove_file}; // Hier ist OpenOptions jetzt nur einmal drin
    use std::io::Write;
    use std::time::Instant;
    use vbuf_core::{VBufInstance, VBufWriter};

    // Hilfsfunktion für die "Eugen-Metrik"
    // Hilfsfunktion zum Auslesen der CPU-Frequenz auf dem Truck
    fn get_current_freq_hz() -> f64 {
        let freq_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq";
        std::fs::read_to_string(freq_path)
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .map(|khz| khz * 1000.0)
            .unwrap_or(1_200_000_000.0) // Fallback auf 1.2 GHz
    }

    fn print_metrics(
        name: &str,
        n: usize,
        bytes_per_item: usize,
        duration: std::time::Duration,
        c_start: u64,
        c_end: u64,
    ) {
        let secs = duration.as_secs_f64();
        let freq_hz = get_current_freq_hz();
        let freq_ghz = freq_hz / 1e9;

        let total_cycles = if c_end > c_start {
            (c_end.wrapping_sub(c_start)) as f64
        } else {
            duration.as_secs_f64() * freq_hz
        };
        let cycles_per_item = total_cycles / (n as f64);
        let cycles_item = if cycles_per_item > 0.0001 {
            cycles_per_item
        } else {
            (duration.as_secs_f64() * freq_hz) / n as f64
        };
        let total_bytes = (n * bytes_per_item) as f64;
        let speed_gb_s = (total_bytes / secs) / 1e9;
        let speed_per_ghz = speed_gb_s / freq_ghz;

        // Output Konsole
        println!("\n--- {} ---", name);
        println!("Frequenz:    {:.2} GHz", freq_ghz);
        println!("Speed:       \x1b[1;32m{:.2} GB/s\x1b[0m", speed_gb_s);
        println!(
            "Effizienz:   \x1b[1;33m{:.2} GB/s pro GHz\x1b[0m",
            speed_per_ghz
        );
        println!("Zyklen/Item: \x1b[1;33m{:.4}\x1b[0m", cycles_per_item);

        // Ab in die Datei
        log_benchmark_to_md(
            name,
            n,
            speed_gb_s,
            cycles_per_item,
            freq_ghz,
            speed_per_ghz,
        );
    }

    fn log_benchmark_to_md(
        name: &str,
        n: usize,
        speed_gb_s: f64,
        cycles_per_item: f64,
        freq_ghz: f64,
        speed_per_ghz: f64,
    ) {
        let file_path = "bench.md";
        let file_exists = std::path::Path::new(file_path).exists();

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(file_path)
            .expect("Konnte bench.md nicht öffnen");

        // Architektur automatisch erkennen
        let arch = std::env::consts::ARCH;

        // CPU-Modellnamen unter Linux ermitteln (Cubietruck/Orange Pi)
        let cpu_model = std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|info| {
                info.lines()
                    .find(|line| line.starts_with("model name") || line.starts_with("Hardware"))
                    .and_then(|line| line.split(':').nth(1))
                    .map(|s| s.trim().to_string())
            })
            .unwrap_or_else(|| "Unknown CPU".to_string());

        if !file_exists {
            // Header mit neuen Spalten: Arch | CPU
            writeln!(file, "| Timestamp | Arch | CPU | Testname | Elemente | Speed (GB/s) | Freq (GHz) | Speed/GHz | Cycles/Item |").unwrap();
            writeln!(file, "|-----------|------|-----|----------|----------|--------------|------------|-----------|-------------|").unwrap();
        }

        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

        writeln!(
            file,
            "| {} | {} | {} | {:<25} | {:>10} | {:>12.2} | {:>10.2} | {:>9.2} | {:>11.4} |",
            timestamp,
            arch,
            cpu_model,
            name,
            n,
            speed_gb_s,
            freq_ghz,
            speed_per_ghz,
            cycles_per_item
        )
        .expect("Schreibfehler in bench.md");
    }

    #[test]
    fn benchmark_branch_killer() {
        let path = FILENAME;
        let inst = VBufInstance::open(path).expect("Input fehlt");
        let col_a = inst.get_as::<u32>(101).expect("Spalte 101 fehlt");
        let col_b = inst.get_as::<u32>(102).expect("Spalte 102 fehlt");
        let n = col_a.len();

        let start = Instant::now();
        let c_start = get_cycles();

        let match_count: usize = col_a
            .par_iter()
            .zip(col_b.par_iter())
            .filter(|&(&a, &b)| a > b)
            .count();

        let c_end = get_cycles();
        let duration = start.elapsed();

        println!("Matches: {}", match_count);
        print_metrics("Branch-Killer (if A > B)", n, 8, duration, c_start, c_end);
        assert!(n > 0);
    }

    #[test]
    fn benchmark_dot_product() {
        let inst = VBufInstance::open(FILENAME).expect("Input fehlt");
        let col_a = inst.get_as::<u32>(101).expect("Spalte 101 fehlt");
        let col_b = inst.get_as::<u32>(102).expect("Spalte 102 fehlt");
        let n = col_a.len();

        let start = Instant::now();
        let c_start = get_cycles();

        let dot_product: f64 = col_a
            .par_iter()
            .zip(col_b.par_iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();

        let c_end = get_cycles();
        let duration = start.elapsed();

        println!("Ergebnis: {:.2}", dot_product);
        print_metrics("Dot-Product (A * B)", n, 8, duration, c_start, c_end);
    }

    #[test]
    fn benchmark_transform_and_write() {
        const OUTPUT_FILENAME: &str = "./test_out.vbuf";
        let inst = VBufInstance::open(FILENAME).expect("Input fehlt");
        let col_a = inst.get_as::<u32>(101).expect("Spalte 101 fehlt");
        let col_b = inst.get_as::<u32>(102).expect("Spalte 102 fehlt");
        let n = col_a.len();

        let start = Instant::now();
        let c_start = get_cycles();

        let result: Vec<u32> = col_a
            .par_iter()
            .zip(col_b.par_iter())
            .map(|(&a, &b)| a / 2 + b / 2)
            .collect();

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
        let data = inst.get_as::<u32>(101).expect("Spalte 101 fehlt");
        let n = data.len();

        let start = Instant::now();
        let c_start = get_cycles();

        let result: Vec<u32> = data.par_iter().map(|&x| x.wrapping_mul(42)).collect();

        let c_end = get_cycles();
        let duration = start.elapsed();

        println!("Check: Erstes Element = {}", result[0]);
        print_metrics("Copy & Scale (*42)", n, 8, duration, c_start, c_end);
    }

    #[test]
    fn benchmark_histogram() {
        let inst = VBufInstance::open(FILENAME).expect("Input fehlt");
        let data = inst.get_as::<u32>(101).expect("Spalte 101 fehlt");
        let n = data.len();

        let start = Instant::now();
        let c_start = get_cycles();

        let bins = data
            .par_iter()
            .fold(
                || vec![0usize; 256],
                |mut local_bins, &val| {
                    let bucket = (val & 0xFF) as usize;
                    local_bins[bucket] += 1;
                    local_bins
                },
            )
            .reduce(
                || vec![0usize; 256],
                |mut a, b| {
                    for (i, val) in b.iter().enumerate() {
                        a[i] += val;
                    }
                    a
                },
            );

        let c_end = get_cycles();
        let duration = start.elapsed();

        println!("Check: Buckets summiert = {}", bins.iter().sum::<usize>());
        print_metrics("Histogramm (u32 -> 256)", n, 4, duration, c_start, c_end);
    }
    #[test]
    fn benchmark_weighted_sum() {
        let inst = VBufInstance::open(FILENAME).expect("Input fehlt");
        let data = inst.get_as::<u32>(101).expect("Spalte 101 fehlt");
        let n = data.len();

        let start = Instant::now();
        let c_start = get_cycles();

        // 0.25*a + 0.5*b + 0.25*c
        let result: Vec<u32> = data
            .par_windows(3)
            .map(|w| ((w[0] as f32 * 0.25) + (w[1] as f32 * 0.5) + (w[2] as f32 * 0.25)) as u32)
            .collect();

        let c_end = get_cycles();
        let duration = start.elapsed();

        println!("Resultat-Länge: {}", result.len());
        // 4 Bytes Read (Window gleitet) + 4 Bytes Write
        print_metrics("Weighted Sum (Window 3)", n, 8, duration, c_start, c_end);
    }

    #[test]
    fn benchmark_find_indices() {
        let inst = VBufInstance::open(FILENAME).expect("Input fehlt");
        let data = inst.get_as::<u32>(101).expect("Spalte 101 fehlt");
        let n = data.len();

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

        println!("Gefunden: {}", indices.len());
        // Wir lesen u32 (4 Bytes)
        print_metrics("Find Indices (>100k)", n, 4, duration, c_start, c_end);
    }

    #[test]
    fn benchmark_directory_scan_debug() {
        let inst = VBufInstance::open(FILENAME).expect("Input fehlt");
        let mem = inst.as_raw_slice();
        let end = mem.len();
        let alignment = 64; // Auf dein vBuf Alignment anpassen

        let start = Instant::now();
        let c_start = get_cycles();

        let mut curr = 16;
        let mut cols_found = 0;

        while curr + 16 <= end {
            let anchor = u64::from_le_bytes(mem[curr..curr + 8].try_into().unwrap());
            if anchor == 0 {
                curr += 8;
                continue;
            }

            let _id = ((anchor >> 16) & 0xFFFF) as u16;
            let plen = ((anchor >> 32) & 0xFFFF) as u16;
            let n = u64::from_le_bytes(mem[curr + 8..curr + 16].try_into().unwrap());

            cols_found += 1;
            let data_start = (curr + 16 + (alignment - 1)) & !(alignment - 1);
            let data_bytes = (n * (plen as u64 / 8)) as usize;
            curr = (data_start + data_bytes + 7) & !7;
        }

        let c_end = get_cycles();
        let duration = start.elapsed();

        // Logge den Directory Scan als spezialisierten Eintrag
        #[allow(dead_code)]
        let freq_ghz = get_current_freq_hz() / 1e9;
        log_benchmark_to_md(
            "Directory Metadata Scan",
            cols_found,
            0.0,
            (c_end - c_start) as f64 / cols_found as f64,
            freq_ghz,
            0.0,
        );

        println!("\n--- vBuf Directory Scan ---");
        println!("Spalten gefunden: {}", cols_found);
        println!("Zeit:             {:?}", duration);
    }
}
