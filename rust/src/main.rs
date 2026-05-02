const FILENAME: &str = "/dev/shm/dual_test.vbuf";

fn main() -> Result<(), ()> {
    _ = ensure_test_file(FILENAME, 4_000_000);
    Ok(())
}
fn ensure_test_file(filename: &str, n: usize) -> Result<(), &str> {
    if std::path::Path::new(filename).exists() {
        return Ok(());
    }

    println!("Generiere Testdatei {} mit {} Einträgen...", filename, n);

    // 1. Writer erstellen (schreibt automatisch MAGIC, VERSION und Padding)
    let mut vbuf_writer = vbuf_core::VBufWriter::<std::fs::File>::create(filename)
        .map_err(|_| "Datei konnte nicht erstellt werden.")?;

    // 2. Daten schreiben
    // Wir nutzen direkt den internen BufWriter des VBufWriters.
    // Da vbuf_writer.writer ein BufWriter<File> ist, funktioniert write_all.

    // n ist die Anzahl der Elemente (z.B. 1.000.000)
    let n_u32 = n as u32;

    // Spalte A: Einfache Sequenz (0, 1, 2, 3...)
    let col_a: Vec<u32> = (0..n_u32).collect();

    // Spalte B: Konstante 2 (wie gehabt)
    let col_b: Vec<u32> = vec![2u32; n];

    // Spalte C: Verdopplung (0, 2, 4, 6...)
    let col_c: Vec<u32> = (0..n_u32).map(|x| x * 2).collect();

    // Spalte D: Quadratzahlen (0, 1, 4, 9...)
    // Hinweis: Nutze wrapping_mul, falls n sehr groß ist, um Overflows zu vermeiden
    let col_d: Vec<u32> = (0..n_u32).map(|x| x.wrapping_mul(x)).collect();

    // Spalte E: Bitweise invertiert (Alle Bits umdrehen)
    let col_e: Vec<u32> = (0..n_u32).map(|x| !x).collect();

    // Spalte F: Ein Mix aus Addition und Modulo (erzeugt ein Sägezahn-Muster)
    let col_f: Vec<u32> = (0..n_u32).map(|x| (x + 100) % 500).collect();

    // 3. Deine write_column Methode nutzen
    // Ich nehme an, die IDs sind 101 und 102
    vbuf_writer
        .write_column(101, &col_a)
        .map_err(|_| "Fehler beim Schreiben von Spalte 101")?;

    vbuf_writer
        .write_column(102, &col_b)
        .map_err(|_| "Fehler beim Schreiben von Spalte 102")?;
    vbuf_writer
        .write_column(103, &col_c)
        .map_err(|_| "Fehler beim Schreiben von Spalte 102")?;
    vbuf_writer
        .write_column(104, &col_d)
        .map_err(|_| "Fehler beim Schreiben von Spalte 102")?;
    vbuf_writer
        .write_column(105, &col_e)
        .map_err(|_| "Fehler beim Schreiben von Spalte 102")?;
    vbuf_writer
        .write_column(106, &col_f)
        .map_err(|_| "Fehler beim Schreiben von Spalte 102")?;

    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;
    use std::{
        fs::{OpenOptions, remove_file},
        time::Instant,
    };
    use vbuf_core::{VBufInstance, VBufWriter};

    #[test]
    fn benchmark_branch_killer() {
        // Pfad zur Benchmark-Datei (muss existieren)
        let path = FILENAME;

        let inst = match VBufInstance::open(path) {
            Ok(i) => i,
            Err(_) => {
                println!("Überspringe Benchmark: {} nicht gefunden.", path);
                return;
            }
        };

        // 1. Spalten laden (hier als u32, wie in deinem letzten Rust-Stand)
        let col_a = inst.get_as::<u32>(101).expect("Spalte 101 fehlt");
        let col_b = inst.get_as::<u32>(102).expect("Spalte 102 fehlt");

        let n = col_a.len();
        println!("\n--- vBuf Branch-Killer Benchmark (Rust/Rayon) ---");
        println!("Vergleiche A vs B: if(A[i] > B[i]) count++ | n = {}", n);

        let start = Instant::now();
        let cycle_start = unsafe { std::arch::x86_64::_rdtsc() };

        // 2. Parallele Verarbeitung mit Rayon
        // Wir nutzen .zip(), um beide Spalten gleichzeitig zu iterieren
        let match_count: usize = col_a
            .par_iter()
            .zip(col_b.par_iter())
            .filter(|&(&a, &b)| a > b)
            .count();

        let duration = start.elapsed();
        let cycle_end = unsafe { std::arch::x86_64::_rdtsc() };
        let secs = duration.as_secs_f64();

        let total_cycles = (cycle_end - cycle_start) as f64;
        let cycles_per_item = total_cycles / (n as f64);

        // Berechnung: n * 4 Bytes * 2 Spalten
        let total_bytes = (n * std::mem::size_of::<u32>() * 2) as f64;
        let speed_gb_s = (total_bytes / secs) / 1e9;
        println!("Matches:    {}", match_count);
        println!("Speed:      \x1b[1;33m{:.2} GB/s\x1b[0m", speed_gb_s);
        println!("Zeit:       {:.4} s", secs);
        println!("Zyklen/Item: \x1b[1;33m{:.4}\x1b[0m", cycles_per_item);
        println!("------------------------------------");

        // Ein kleiner Assert, damit es ein gültiger Test bleibt
        assert!(n > 0);
    }

    #[test]
    fn benchmark_dot_product() {
        let inst = match VBufInstance::open(FILENAME) {
            Ok(i) => i,
            Err(_) => {
                println!("Überspringe Dot-Product: {} nicht gefunden.", FILENAME);
                return;
            }
        };

        // Spalten laden
        let col_a = inst.get_as::<u32>(101).expect("Spalte 101 fehlt");
        let col_b = inst.get_as::<u32>(102).expect("Spalte 102 fehlt");
        let n = col_a.len();

        println!("\n--- vBuf Dot-Product (A * B) Benchmark (Rust/Rayon) ---");

        // Messung Start
        let time_start = std::time::Instant::now();
        let cycle_start = unsafe { std::arch::x86_64::_rdtsc() };

        // Berechnung
        let dot_product: f64 = col_a
            .par_iter()
            .zip(col_b.par_iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();

        // Messung Ende
        let time_duration = time_start.elapsed();
        let cycle_end = unsafe { std::arch::x86_64::_rdtsc() };

        // Kalkulationen
        let secs = time_duration.as_secs_f64();
        let total_cycles = (cycle_end - cycle_start) as f64;
        let cycles_per_item = total_cycles / (n as f64);

        // n * 4 Bytes (u32) * 2 Spalten
        let total_bytes = (n * 4 * 2) as f64;
        let speed_gb_s = (total_bytes / secs) / 1_000_000_000.0;

        println!("Ergebnis:    {:.2}", dot_product);
        println!("Speed:       \x1b[1;36m{:.2} GB/s\x1b[0m", speed_gb_s);
        println!("Zeit:        {:.4} s", secs);
        println!("Zyklen/Item: \x1b[1;33m{:.4}\x1b[0m", cycles_per_item);
        println!("---------------------------------------");
        assert!(n > 0);
    }
    #[test]
    fn benchmark_transform_and_write() {
        const OUTPUT_FILENAME: &str = "/dev/shm/output_test.vbuf";
        let inst = VBufInstance::open(FILENAME).expect("Input Datei nicht gefunden");
        let col_a = inst.get_as::<u32>(101).expect("Spalte 101 fehlt");
        let col_b = inst.get_as::<u32>(102).expect("Spalte 102 fehlt");
        let n = col_a.len();

        let start = Instant::now();
        let cycle_start = unsafe { std::arch::x86_64::_rdtsc() };

        // 1. Transform (Warning gefixt: Klammern entfernt)
        let result: Vec<u32> = col_a
            .par_iter()
            .zip(col_b.par_iter())
            .map(|(&a, &b)| a / 2 + b / 2)
            .collect();

        // 2. Write
        let mut out_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(OUTPUT_FILENAME)
            .expect("Konnte Output nicht öffnen");

        {
            let a_shift: u8 = 12;
            let mut writer =
                VBufWriter::new(&mut out_file, a_shift.into()).expect("Writer Init Error");

            writer.write_column(103, &result).expect("Write Error");
        }

        let cycle_end = unsafe { std::arch::x86_64::_rdtsc() };
        let secs = start.elapsed().as_secs_f64();

        let total_cycles = (cycle_end - cycle_start) as f64;
        let cycles_per_item = total_cycles / (n as f64);

        // n * 4 Bytes * 2 (Read) + n * 4 Bytes (Write) = n * 12
        println!("\n\x1b[1;35m--- vBuf Transform & Write Benchmark (A+B)/2 ---\x1b[0m");
        println!("Elemente:   {}", n);
        println!(
            "Speed:      \x1b[1;32m{:.2} GB/s\x1b[0m (Read+Write)",
            ((n * 12) as f64 / secs) / 1e9
        );
        println!("Zeit:       {:.4} s", secs);
        println!("Zyklen/Item: \x1b[1;33m{:.4}\x1b[0m", cycles_per_item);

        let _ = remove_file(OUTPUT_FILENAME);
    }
    #[test]
    fn benchmark_weighted_sum() {
        let inst = VBufInstance::open(FILENAME).expect("Input fehlt");
        let data = inst.get_as::<u32>(101).expect("Spalte 101 fehlt");
        let n = data.len();

        println!("\n--- vBuf Weighted Sum (Window Test) ---");
        let start = Instant::now();
        let cycle_start = unsafe { std::arch::x86_64::_rdtsc() };

        // Wir berechnen eine gewichtete Summe über 3 Elemente (0.25*a + 0.5*b + 0.25*c)
        let result: Vec<u32> = data
            .par_windows(3)
            .map(|w| ((w[0] as f32 * 0.25) + (w[1] as f32 * 0.5) + (w[2] as f32 * 0.25)) as u32)
            .collect();

        let cycle_end = unsafe { std::arch::x86_64::_rdtsc() };
        let secs = start.elapsed().as_secs_f64();
        let total_bytes = (n * 4 + result.len() * 4) as f64;

        let total_cycles = (cycle_end - cycle_start) as f64;
        let cycles_per_item = total_cycles / (data.len() as f64);
        println!(
            "Speed:      \x1b[1;32m{:.2} GB/s\x1b[0m (R+W Combined)",
            (total_bytes / secs) / 1e9
        );
        println!("Result:      \x1b[1;32m{:?}\x1b[0m", result.len());
        println!("Zeit:       {:.4} s", secs);
        println!("Zyklen/Item: \x1b[1;33m{:.4}\x1b[0m", cycles_per_item);
    }
    #[test]
    fn benchmark_find_indices() {
        let inst = VBufInstance::open(FILENAME).expect("Input fehlt");
        let data = inst.get_as::<u32>(101).expect("Spalte 101 fehlt");

        println!("\n--- vBuf Search & Index Collect ---");
        let start = Instant::now();
        let cycle_start = unsafe { std::arch::x86_64::_rdtsc() };

        // Finde alle Indizes, wo der Wert > 100000 ist
        let indices: Vec<usize> = data
            .par_iter()
            .enumerate()
            .filter(|&(_, &val)| val > 100_000)
            .map(|(idx, _)| idx)
            .collect();

        let cycle_end = unsafe { std::arch::x86_64::_rdtsc() };
        let secs = start.elapsed().as_secs_f64();
        let total_cycles = (cycle_end - cycle_start) as f64;
        let cycles_per_item = total_cycles / (data.len() as f64);
        println!("Gefunden:   {}", indices.len());
        println!(
            "Speed:      \x1b[1;32m{:.2} GB/s\x1b[0m",
            ((data.len() * 4) as f64 / secs) / 1e9
        );
        println!("Zyklen/Item: \x1b[1;33m{:.4}\x1b[0m", cycles_per_item);
    }
    #[test]
    fn benchmark_directory_scan_debug() {
        let inst = VBufInstance::open(FILENAME).expect("Input Datei nicht gefunden");
        let mem = inst.as_raw_slice(); // Zugriff auf den gesamten mmap-Bereich
        let end = mem.len();
        let alignment = 4096; // Dein Standard-Alignment (anpassen falls nötig)

        println!("\n\x1b[1;36m--- vBuf Datei-Inhaltsverzeichnis (Debug - Rust Version) ---\x1b[0m");

        let mut curr = 16; // Start nach Global Header
        let mut cols_found = 0;

        while curr + 16 <= end {
            // Sicherer Read des u64 Anchors
            let anchor_bytes = &mem[curr..curr + 8];
            let anchor = u64::from_le_bytes(anchor_bytes.try_into().unwrap());

            // 1. Alignment-Padding überspringen
            if anchor == 0 {
                curr += 8;
                continue;
            }

            // 2. Felder extrahieren (identisch zu deiner C-Logik)
            let id = ((anchor >> 16) & 0xFFFF) as u16;
            let plen = ((anchor >> 32) & 0xFFFF) as u16;

            // N aus den nächsten 8 Bytes lesen (da Bit 9 gesetzt ist)
            let n_bytes = &mem[curr + 8..curr + 16];
            let n = u64::from_le_bytes(n_bytes.try_into().unwrap());

            println!(
                "Gefunden: ID {:5} | N = {:10} | {:2}-Bit | Offset = {}",
                id, n, plen, curr
            );

            cols_found += 1;

            // 3. Sprung-Logik (Synchron mit vbuf.c)
            let header_size = 16;

            // Alignment-Berechnung
            let data_start = (curr + header_size + (alignment - 1)) & !(alignment - 1);
            let data_bytes = (n * (plen as u64 / 8)) as usize;

            // Nächster Slot: Ende der Daten + 8-Byte Alignment für den nächsten Header
            curr = data_start + data_bytes;
            curr = (curr + 7) & !7;
        }

        println!(
            "\x1b[1;36m--- Scan beendet. Insgesamt {} Spalten gefunden ---\x1b[0m\n",
            cols_found
        );
    }

    #[test]
    fn benchmark_copy_and_scale() {
        let inst = VBufInstance::open(FILENAME).expect("Input fehlt");
        let data = inst.get_as::<u32>(101).expect("Spalte 101 fehlt");
        let n = data.len();

        println!("\n\x1b[1;33m--- vBuf Copy & Scale (SIMD Check) ---\x1b[0m");
        let start = Instant::now();
        let cycle_start = unsafe { std::arch::x86_64::_rdtsc() };

        // Simuliert das Erstellen einer Arbeitskopie mit Skalierung
        let result: Vec<u32> = data.par_iter().map(|&x| x.wrapping_mul(42)).collect();

        let cycle_end = unsafe { std::arch::x86_64::_rdtsc() };
        let secs = start.elapsed().as_secs_f64();

        let total_cycles = (cycle_end - cycle_start) as f64;
        let cycles_per_item = total_cycles / (n as f64);
        // Lesen + Schreiben
        let total_bytes = (n * 4 * 2) as f64;
        println!(
            "Speed:      \x1b[1;32m{:.2} GB/s\x1b[0m (Read+Write)",
            (total_bytes / secs) / 1e9
        );
        println!("Zeit:       {:.4} s", secs);
        println!("Check:      Erstes Element = {}", result[0]);
        println!("Zyklen/Item: \x1b[1;33m{:.4}\x1b[0m", cycles_per_item);
    }
    #[test]
    fn benchmark_histogram() {
        let inst = VBufInstance::open(FILENAME).expect("Input fehlt");
        let data = inst.get_as::<u32>(101).expect("Spalte 101 fehlt");
        let n = data.len();

        println!("\n\x1b[1;34m--- vBuf Histogramm Benchmark (u32 -> 256 Buckets) ---\x1b[0m");
        let start = Instant::now();
        let cycle_start = unsafe { std::arch::x86_64::_rdtsc() };
        // Rayon fold/reduce braucht korrekte Typ-Annotationen
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

        let cycle_end = unsafe { std::arch::x86_64::_rdtsc() };
        let secs = start.elapsed().as_secs_f64();

        let total_cycles = (cycle_end - cycle_start) as f64;
        let cycles_per_item = total_cycles / (n as f64);
        println!(
            "Speed:      \x1b[1;32m{:.2} GB/s\x1b[0m",
            ((n * 4) as f64 / secs) / 1e9
        );
        println!("Zyklen/Item: \x1b[1;33m{:.4}\x1b[0m", cycles_per_item);
        println!("Zeit:       {:.4} s", secs);
        println!(
            "Check:      Buckets summiert = {}",
            bins.iter().sum::<usize>()
        );
    }
}
