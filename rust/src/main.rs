use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader, BufWriter};
use vbuf_core::{VBuf, Xxh3Strategy};
// optional: use zstd;

fn main() -> std::io::Result<()> {
    // 1. Quelle öffnen
    let file = File::open("openfoodfacts-products.jsonl")?;
    let reader = BufReader::new(file);

    // 2. Ziel vorbereiten
    let out_file = File::create("output.vbuf")?;

    // BufWriter ist wichtig für die Performance beim Schreiben auf SSD/HDD
    let mut writer = BufWriter::new(out_file);

    // 3. VBuf-Konfiguration (ohne dass es selbst Daten speichert)
    let vbuf = VBuf::empty(Box::new(Xxh3Strategy));

    println!("Starte Verarbeitung der 69GB Datei...");

    for (i, line) in reader.lines().enumerate() {
        if let Ok(l) = line {
            vbuf.stream_json(l.as_bytes(), &mut writer)?;
        }

        if i % 100_000 == 0 && i > 0 {
            println!("Verarbeitet: {} Zeilen", i);
        }
    }

    writer.flush()?;

    println!("Fertig! Die Datei wurde gestreamt.");
    Ok(())
}
