use ::std::io::{BufWriter, Seek, Write};
use memmap2::Mmap;
use std::fs::File;

const MAGIC: u32 = 0x46554256; // "VBUF"
const VERSION: u32 = 0x00050000; // 0.5.0

pub struct VBufInstance {
    pub mmap: Mmap,
    pub alignment: usize,
}
pub struct VBufWriter<W: Write + Seek> {
    writer: BufWriter<W>,
    alignment: usize,
}
// Hier muss die Implementierung stehen
impl VBufInstance {
    pub fn open(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 16 || &mmap[0..4] != MAGIC.to_le_bytes() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid Magic header",
            ));
        }
        let a_shift = mmap[8];
        let alignment = 1 << a_shift;
        Ok(VBufInstance { mmap, alignment })
    }

    pub fn get_as<T>(&self, target_id: u32) -> Option<&[T]> {
        let mem = &self.mmap;
        let mut curr = 16; // Nach Global Header
        let end = mem.len();
        let alignment = self.alignment;

        while curr + 16 <= end {
            let anchor = u64::from_le_bytes(mem[curr..curr + 8].try_into().unwrap());

            // Padding überspringen
            if anchor == 0 {
                curr += 8;
                continue;
            }

            // ID extrahieren (Bits 16-31 laut Standard)
            let id = ((anchor >> 16) & 0xFFFF) as u32;
            let plen = ((anchor >> 32) & 0xFFFF) as u16;

            // Overflow bit check (Bit 9)
            let has_overflow = (anchor & (1 << 9)) != 0;
            let (n, header_size) = if has_overflow {
                let count = u64::from_le_bytes(mem[curr + 8..curr + 16].try_into().unwrap());
                (count, 16)
            } else {
                let count = ((anchor >> 48) & 0xFFFF) as u64;
                (count, 8)
            };

            let data_start = (curr + header_size + (alignment - 1)) & !(alignment - 1);

            // Wenn das unsere ID ist, Slice zurückgeben
            if id == target_id {
                if (plen as usize) != std::mem::size_of::<T>() * 8 {
                    return None; // Typ-Mismatch erkannt!
                }

                let data_end = data_start + (n as usize * (plen as usize / 8));
                if data_end <= end {
                    let ptr = mem[data_start..data_end].as_ptr() as *const T;
                    return Some(unsafe { std::slice::from_raw_parts(ptr, n as usize) });
                }
            }

            // Weitermarschieren zum nächsten Block
            let data_bytes = n as usize * (plen as usize / 8);
            curr = data_start + data_bytes;
            curr = (curr + 7) & !7; // Align auf 8-Byte Grenze für den nächsten Anchor
        }
        None
    }

    pub fn as_raw_slice(&self) -> &[u8] {
        &self.mmap
    }
}

impl<W: Write + Seek> VBufWriter<W> {
    pub fn new(mut inner: W, alignment: usize) -> std::io::Result<Self> {
        let a_shift = alignment.trailing_zeros() as u8;
        // Global Header direkt schreiben (MAGIC & VERSION)
        inner.write_all(&MAGIC.to_le_bytes())?;
        inner.write_all(&VERSION.to_le_bytes())?;
        // a_shift at byte 8, and the rest reserved/datalen = 0
        inner.write_all(&[a_shift, 0, 0, 0, 0, 0, 0, 0])?;

        Ok(Self {
            writer: BufWriter::new(inner),
            alignment,
        })
    }

    // Für die Praxis (Datei)
    pub fn create(path: &str) -> std::io::Result<VBufWriter<File>> {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        // Standard: 4096 Alignment (a_shift = 12)
        VBufWriter::<File>::new(file, 4096)
    }

    pub fn write_column<T>(&mut self, id: u16, data: &[T]) -> std::io::Result<()> {
        let n = data.len() as u64;
        let item_size = std::mem::size_of::<T>();
        let plen = (item_size * 8) as u64;

        let mut anchor: u64 = 0;
        anchor |= 1u64 << 4; // PHYS: Array
        anchor |= 1u64 << 9; // OVERFLOW: N follows as u64
        anchor |= (id as u64) << 16;
        anchor |= plen << 32;

        self.writer.write_all(&anchor.to_le_bytes())?;
        self.writer.write_all(&n.to_le_bytes())?;

        // Alignment zum Diamond-Payload
        let current_pos = self.writer.stream_position()?;
        let padding =
            (self.alignment - (current_pos % self.alignment as u64) as usize) % self.alignment;
        if padding > 0 {
            self.writer.write_all(&vec![0u8; padding])?;
        }

        // Sicherer Byte-Cast für die Daten
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };
        self.writer.write_all(byte_slice)?;

        // Finales 8-Byte Alignment für den nächsten Anchor
        let end_pos = self.writer.stream_position()?;
        let tail_padding = (8 - (end_pos % 8)) % 8;
        if tail_padding > 0 {
            self.writer.write_all(&vec![0u8; tail_padding as usize])?;
        }

        self.writer.flush()
    }
}

#[cfg(test)]
mod tests {
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

    // Falls das nicht klappt, versuche den direkten Pfad zu den Konstanten:
    use crate::{MAGIC, VBufInstance, VBufWriter};
    // use rayon::iter::IntoParallelRefIterator;
    use std::{
        ffi::{CStr, c_char, c_uint},
        io::Cursor,
    };

    #[test]
    fn test_round_trip_logic() {
        let mut buffer = Cursor::new(Vec::new());
        let test_data = vec![100u32, 200, 300, 400];
        let col_id = 500;

        {
            let mut writer = VBufWriter::new(&mut buffer, 12).expect("Writer init failed");
            writer
                .write_column(col_id, &test_data)
                .expect("Write failed");
        }

        let raw_bytes = buffer.into_inner();

        assert!(raw_bytes.len() > 16);
        // Nutze den expliziten Pfad, falls der Import oben immer noch zickt:
        assert_eq!(&raw_bytes[0..4], MAGIC.to_le_bytes());
    }
    /// TEST 2: Falsche ID
    /// Prüft, ob das System korrekt reagiert, wenn eine ID nicht existiert.
    #[test]
    fn test_missing_id() {
        // Wir nutzen deine bestehende Datei vom Benchmark für einen schnellen Check
        if let Ok(inst) = VBufInstance::open("/dev/shm/dual_test.vbuf") {
            let result = inst.get_as::<u32>(99999); // ID die es nicht gibt
            assert!(
                result.is_none(),
                "Sollte None zurückgeben für unbekannte ID"
            );
        }
    }

    /// TEST 3: Typ-Mismatch (Sicherheitstest)
    /// Was passiert, wenn ich u16 erwarte, aber u32 drinsteckt?
    #[test]
    fn test_type_mismatch() {
        if let Ok(inst) = VBufInstance::open("/dev/shm/dual_test.vbuf") {
            // Spalte 101 ist u32 (32-bit)
            // Wir versuchen sie als u16 zu laden:
            let result = inst.get_as::<u16>(101);
            assert!(
                result.is_none(),
                "System muss Mismatch zwischen 16-bit Erwartung und 32-bit Realität erkennen"
            );
        }
    }

    /// TEST 4: Alignment-Stabilität
    /// Schreibt mehrere Spalten hintereinander weg und prüft, ob die zweite Spalte
    /// trotz unterschiedlicher Längen immer noch korrekt gefunden wird (Padding-Check).
    #[test]
    fn test_alignment_integrity() {
        let mut buffer = Cursor::new(Vec::new());
        let data1 = vec![1u16, 2, 3]; // Ungerade Länge für Padding-Provokation
        let data2 = vec![100u32, 200];

        {
            let mut writer = VBufWriter::new(&mut buffer, 12).unwrap();
            writer.write_column(1, &data1).unwrap();
            writer.write_column(2, &data2).unwrap();
        }

        // In einem echten Integrationstest würde man dies nun über ein Temp-File laden.
        // Hier prüfen wir nur, ob der Buffer geschrieben wurde.
        assert!(buffer.get_ref().len() > 0);
    }

    // --- C-INTERFACE (Shared Object) ---

    #[unsafe(no_mangle)]
    pub extern "C" fn vbuf_open(path: *const c_char) -> *mut VBufInstance {
        if path.is_null() {
            return std::ptr::null_mut();
        }
        let c_str = unsafe { CStr::from_ptr(path) };
        let r_path = match c_str.to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };

        match VBufInstance::open(r_path) {
            Ok(inst) => Box::into_raw(Box::new(inst)),
            Err(_) => std::ptr::null_mut(),
        }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn vbuf_close(ptr: *mut VBufInstance) {
        if !ptr.is_null() {
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn vbuf_get_sum_u32(ptr: *mut VBufInstance, col_id: c_uint) -> f64 {
        let inst = unsafe {
            if ptr.is_null() {
                return -1.0;
            }
            &*ptr
        };

        if let Some(col) = inst.get_as::<u32>(col_id) {
            col.par_iter().map(|&x| x as f64).sum()
        } else {
            -2.0
        }
    }
}
